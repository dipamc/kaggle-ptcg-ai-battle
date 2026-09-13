#!/usr/bin/env python3
"""Two-stage deck A/B on a fixed converged policy (the decklab test harness).

    tools/decklab_ab.py run decklab/proposals/dlab_0001_x [--replicate]
                        [--games-scale 1.0] [--stage both|targeted|field]
                        [--profile screen]
    tools/decklab_ab.py probe --deck <proposal-dir|deck.csv> [--opps a.csv,b.csv]
                        [--games 60] [--vs-base] [--pilot lane|frozen]
    tools/decklab_ab.py verdict decklab/proposals/dlab_0001_x
    tools/decklab_ab.py setup-arena          # remote lanes only

Methodology (docs/decklab.md):
  * ONE checkpoint pilots both seats (tools/selfeval.py or the batched
    runner), greedy, seats alternating; only the decks differ.
  * Stage 1 targeted: the proposal's PRE-COMMITTED 3-6 opponents,
    games_targeted_per_opp per cell, PASS on delta >= max(min_delta, z*SE) in
    the predicted direction.
  * Stage 2 field: the frozen field gauntlet (field-share weighted), PASS on
    non-inferiority (weighted delta > -z*SE).
  * --replicate reruns on the second checkpoint; with replication_required
    the final accept needs both.

Base-arm cells are cached in decklab/results/base_cache.jsonl keyed by
(ckpt, base deck md5, opponent, games, temp, heads, lane) so testing five
variants of one deck pays the base cost once.

Lanes come from decklab/config.json: "arena" runs confirmatory A/Bs, and
"arena_probe" (optional) runs exploratory probes. A lane's "host" is either
"local" (games run in this checkout with this interpreter; jobs and staged
decks live under decklab/jobs and decklab/staging) or an ssh host whose
"arena_dir" mirrors this checkout (refresh it with setup-arena). Never split
one experiment's two arms across lanes: the lane is part of the instrument.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C

BASE_CACHE = os.path.join(C.RESULTS, "base_cache.jsonl")
SYNC_EXCLUDES = ["data/deckmat", "data/episode_farm*", "data/ladder_replays",
                 "data/runlogs", "*.pt", ".git", "__pycache__"]


def is_local(ln):
    return not ln.get("host") or ln["host"] in ("local", "localhost")


def lane(cfgd, which="ab"):
    """Resolve an arena lane. 'ab' = confirmatory A/Bs (config block 'arena',
    the frozen instrument). 'probe' = the exploratory lane (block 'arena_probe'
    when present, else the same lane as 'ab'). Each block: host, runner
    (batched|serial), device, workers_*, and optionally ckpt/heads (a lane may
    pilot a different checkpoint than the A/B evaluator; heads is not
    recoverable from a state dict, so it travels with the ckpt). Remote lanes
    also need repo (ckpt-relative root on the host), arena_dir and python."""
    ln = dict(cfgd["arena"])
    if which == "probe" and "arena_probe" in cfgd:
        ln = dict(cfgd["arena_probe"])
    # config keys may be present but null: treat null like absent
    ln["device"] = ln.get("device") or cfgd.get("device", "cpu")
    ln["ckpt"] = ln.get("ckpt") or cfgd["ckpt_primary"]
    ln["heads"] = ln.get("heads") or cfgd["heads"]
    if is_local(ln):
        ln["host"] = "local"
        ln["repo"] = C.REPO
        ln["arena_dir"] = C.REPO
        ln["python"] = sys.executable
    else:
        for k in ("repo", "arena_dir"):
            if not ln.get(k):
                sys.exit(f"remote lane {ln['host']}: config needs '{k}'")
        ln["python"] = ln.get("python") or "python3"
    return ln


def check_ckpt(ln, ckpt_abs):
    """A missing checkpoint on a local lane fails before any deck is staged."""
    if is_local(ln) and not os.path.exists(ckpt_abs):
        sys.exit(f"checkpoint not found: {ckpt_abs} (set ckpt_primary / the lane's "
                 f"ckpt in decklab/config.json)")


def dedupe_cells(cells):
    """Order-preserving dedupe of [my, opp, games] cells: a targeted opponent
    that also sits in the field gauntlet at the same game count must be sent
    once, or the batched runner (which merges rows per matchup) returns fewer
    rows than cells and the run is misread as an arena failure."""
    seen, out = set(), []
    for c in cells:
        k = tuple(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def sh(cmd, **kw):
    print(f"$ {cmd}" if isinstance(cmd, str) else "$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def lane_ssh(ln, cmd, **kw):
    return sh(["ssh", "-o", "BatchMode=yes", ln["host"], cmd], **kw)


def md5f(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def stage_file(ln, local_path, rel):
    """Put a local file at <arena_dir>/<rel> on the lane."""
    if is_local(ln):
        dst = os.path.join(C.REPO, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(local_path) != os.path.abspath(dst):
            shutil.copy(local_path, dst)
    else:
        lane_ssh(ln, f"mkdir -p {ln['arena_dir']}/{os.path.dirname(rel)}")
        sh(f"rsync -a {local_path} {ln['host']}:{ln['arena_dir']}/{rel}")


def sync_pool(ln):
    """Remote lanes never trust their own copy of the pool."""
    if not is_local(ln):
        sh(f"rsync -a {C.REPO}/decks/pool/ {ln['host']}:{ln['arena_dir']}/decks/pool/")


def run_job(ln, job, name, workers, lock):
    """Run one selfeval job on the lane, serialized on `lock`; return its
    result rows. Job paths are relative to the lane's arena_dir (the runner's
    cwd); the checkpoint path is resolved against the lane's repo."""
    job = dict(job, out=f"decklab/jobs/{name}.out.json")
    rel = f"decklab/jobs/{name}.json"
    runner = ("tools/selfeval_batched.py" if ln.get("runner") == "batched"
              else "tools/selfeval.py")
    if is_local(ln):
        os.makedirs(os.path.join(C.REPO, "decklab/jobs"), exist_ok=True)
        jp = os.path.join(C.REPO, rel)
        json.dump(job, open(jp, "w"))
        env = {**os.environ, "PYTHONPATH": "data:." + (
            ":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
        with open(os.path.join(C.REPO, f"decklab/jobs/{lock}.lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            sh([ln["python"], runner, rel, str(workers)], cwd=C.REPO, env=env)
        out_local = os.path.join(C.REPO, job["out"])
    else:
        jp = f"/tmp/{name}.json"
        json.dump(job, open(jp, "w"))
        lane_ssh(ln, f"mkdir -p {ln['arena_dir']}/decklab/jobs")
        sh(f"rsync -a {jp} {ln['host']}:{ln['arena_dir']}/{rel}")
        lane_ssh(ln, f"cd {ln['arena_dir']} && flock -w 7200 /tmp/{lock}.lock "
                     f"-c 'PYTHONPATH=data:. {ln['python']} {runner} {rel} {workers}'")
        out_local = f"/tmp/{name}.out.json"
        sh(f"rsync -a {ln['host']}:{ln['arena_dir']}/{job['out']} {out_local}")
    return json.load(open(out_local))


def setup_arena(cfgd):
    """Mirror this checkout's code, data and pool onto every remote lane."""
    ex = " ".join(f"--exclude={e}" for e in SYNC_EXCLUDES)
    done = set()
    for which in ("ab", "probe"):
        ln = lane(cfgd, which)
        if is_local(ln):
            print(f"[{which}] local lane: runs in this checkout, nothing to set up")
            continue
        host, adir = ln["host"], ln["arena_dir"]
        if (host, adir) in done:
            continue
        done.add((host, adir))
        lane_ssh(ln, f"mkdir -p {adir}/decklab/staging {adir}/decklab/jobs "
                     f"{adir}/decks {adir}/tools {adir}/data")
        sh(f"rsync -a --delete {ex} {C.REPO}/ptcg/ {host}:{adir}/ptcg/")
        sh(f"rsync -a {C.REPO}/tools/selfeval.py {C.REPO}/tools/selfeval_batched.py "
           f"{host}:{adir}/tools/")
        sh(f"rsync -a {ex} {C.REPO}/data/ {host}:{adir}/data/")
        sh(f"rsync -a {C.REPO}/decks/pool/ {host}:{adir}/decks/pool/")
        print(f"[{which}] arena ready at {host}:{adir} (checkpoints are NOT "
              f"synced: put {ln['ckpt']} under {ln['repo']} yourself)")


def cache_read():
    if not os.path.exists(BASE_CACHE):
        return {}
    out = {}
    for ln in open(BASE_CACHE):
        e = json.loads(ln)
        out[e["key"]] = e
    return out


def cache_key(cfgd, ckpt, base_md5, opp, games):
    # instrument tag: base cells measured under one lane/runner must never
    # pair with proposal cells from another (host and runner effects would
    # ride the arm contrast). Changing the ab lane invalidates the old keys.
    ab = lane(cfgd, "ab")
    instr = f"{ab['host']}:{ab.get('runner', 'selfeval')}"
    return "|".join([ckpt, base_md5, opp, str(games),
                     str(cfgd["temp"]), str(cfgd["heads"]), instr])


def run(a):
    cfgd = C.cfg()
    if getattr(a, "profile", None):
        # profile overrides game counts / gauntlet only; accept bars unchanged
        cfgd = {**cfgd, **cfgd[a.profile]}
    pdir = a.proposal_dir.rstrip("/")
    pid = os.path.basename(pdir)
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    if meta.get("evidence_class") == "reasoning":
        sys.exit("reasoning-class proposal (evaluator-OOV cards): not game-gated. "
                 "The frozen policy cannot pilot these cards, so A/B results carry "
                 "no information. Gate = adversarial review of the logic; then "
                 "stage. Probes stay available for information only; the "
                 "training run learns the cards post-append and "
                 "decklab_score judges.")
    if meta.get("status") not in ("validated", "tested_pass", "tested_fail",
                                  "accepted"):
        sys.exit(f"proposal status is '{meta.get('status')}' -- validate first "
                 "(tools/decklab_validate.py)")
    ckpt = cfgd["ckpt_replicate"] if a.replicate else cfgd["ckpt_primary"]
    if not ckpt:
        sys.exit("no checkpoint configured for this arm (decklab/config.json)")
    ar = lane(cfgd, "ab")
    ckpt_abs = os.path.join(ar["repo"], ckpt)
    check_ckpt(ar, ckpt_abs)
    tag = "rep" if a.replicate else "pri"
    deck_path = os.path.join(pdir, "deck.csv")
    base_rel = meta["base_deck"]
    base_md5 = md5f(os.path.join(C.REPO, base_rel))
    scale = a.games_scale

    field = []
    if a.stage in ("both", "field"):
        for ln in open(os.path.join(C.REPO, cfgd["field_gauntlet"])):
            if ln.strip() and not ln.startswith("#"):
                shr, nm = ln.split()
                field.append((float(shr), f"decks/pool/{nm}.csv"))
    targeted = [(None, t) for t in meta["target_opponents"]] \
        if a.stage in ("both", "targeted") else []

    gt = max(10, int(cfgd["games_targeted_per_opp"] * scale))
    gf = max(10, int(cfgd["games_field_per_opp"] * scale))
    cache = cache_read()
    cells, need_base = [], []
    prop_arena = f"decklab/staging/{pid}/deck.csv"
    for _, opp in targeted:
        cells.append([prop_arena, opp, gt])
        k = cache_key(cfgd, ckpt, base_md5, opp, gt)
        if k not in cache:
            need_base.append([base_rel, opp, gt])
    for _, opp in field:
        cells.append([prop_arena, opp, gf])
        k = cache_key(cfgd, ckpt, base_md5, opp, gf)
        if k not in cache:
            need_base.append([base_rel, opp, gf])

    # selfeval takes one games count per job -> group cells by games
    all_cells = dedupe_cells(cells + need_base)
    jobs = []
    for games in sorted({c[2] for c in all_cells}):
        sub = [[m, o] for m, o, g in all_cells if g == games]
        jobs.append((games, sub))
    total_games = sum(g * len(s) for g, s in jobs)
    print(f"{pid} [{tag}] {len(cells)} proposal cells + {len(need_base)} uncached "
          f"base cells = {total_games} games on lane {ar['host']}")

    # stage the deck + run the jobs on the lane
    seed = int(hashlib.md5(f"{pid}{tag}".encode()).hexdigest()[:8], 16) % 10**6
    stage_file(ar, deck_path, prop_arena)
    sync_pool(ar)
    results = []
    for gi, (games, sub) in enumerate(jobs):
        job = {"ckpt": ckpt_abs, "cells": sub, "games": games,
               "device": ar["device"], "heads": cfgd["heads"],
               "temp": cfgd["temp"], "seed": seed + gi}
        t0 = time.time()
        got = run_job(ar, job, f"{pid}_{tag}_{gi}", ar.get("workers_ab", 8),
                      "decklab_ab")
        if len(got) < len(sub):
            sys.exit(f"ARENA FAILURE: job {gi} returned {len(got)}/{len(sub)} cells -- "
                     "read the selfeval output above; nothing recorded.")
        results += got
        print(f"  job {gi}: {len(sub)} cells in {time.time()-t0:.0f}s")

    # split results into proposal cells and fresh base cells; update cache
    os.makedirs(C.RESULTS, exist_ok=True)
    prop_cells, base_cells = {}, {}
    with open(BASE_CACHE, "a") as bc:
        for r in results:
            kk = (r["opp_deck"], r["games"])
            if r["my_deck"].startswith("decklab/staging/"):
                prop_cells[kk] = r
            else:
                k = cache_key(cfgd, ckpt, base_md5, r["opp_deck"], r["games"])
                e = {"key": k, "wins": r["wins"], "games": r["games"],
                     "opp": r["opp_deck"]}
                bc.write(json.dumps(e) + "\n")
                base_cells[kk] = r
    cache = cache_read()
    for opp, g in [(o, gt) for _, o in targeted] + [(o, gf) for _, o in field]:
        if (opp, g) not in base_cells:
            e = cache[cache_key(cfgd, ckpt, base_md5, opp, g)]
            base_cells[(opp, g)] = {"wins": e["wins"], "games": e["games"]}

    out = {"id": pid, "ckpt": ckpt, "tag": tag, "seed": seed,
           "targeted": [], "field": [], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for _, opp in targeted:
        p, b = prop_cells[(opp, gt)], base_cells[(opp, gt)]
        out["targeted"].append({"opp": opp, "prop": [p["wins"], p["games"]],
                                "base": [b["wins"], b["games"]]})
    for shr, opp in field:
        p, b = prop_cells[(opp, gf)], base_cells[(opp, gf)]
        out["field"].append({"opp": opp, "share": shr,
                             "prop": [p["wins"], p["games"]],
                             "base": [b["wins"], b["games"]]})
    rdir = os.path.join(C.RESULTS, pid)
    os.makedirs(rdir, exist_ok=True)
    # a partial --stage rerun must not erase the other stage's results;
    # the rerun stage REPLACES its old measurement (higher-n supersedes)
    old_p = os.path.join(rdir, f"ab_{tag}.json")
    if os.path.exists(old_p) and a.stage != "both":
        old = json.load(open(old_p))
        for st in ("targeted", "field"):
            if not out[st]:
                out[st] = old.get(st, [])
    json.dump(out, open(old_p, "w"), indent=1)
    print(f"wrote {rdir}/ab_{tag}.json")
    verdict(argparse.Namespace(proposal_dir=pdir))


def _stage_stats(rows, weighted=False):
    ds, vs, ws = [], [], []
    for r in rows:
        pw, pn = r["prop"]
        bw, bn = r["base"]
        p1, p0 = pw / pn, bw / bn
        ds.append(p1 - p0)
        vs.append(p1 * (1 - p1) / pn + p0 * (1 - p0) / bn)
        ws.append(r.get("share", 1.0) if weighted else 1.0)
    w = np.array(ws) / sum(ws)
    d = float(np.dot(w, ds))
    se = float(np.sqrt(np.dot(w ** 2, vs)))
    return d, se, ds


def probe(a):
    """Exploratory games on the probe lane: fast, concurrent-safe, and NEVER
    evidence for acceptance. A probe result may motivate a proposal's
    rationale; the confirmatory A/B is separate games under the frozen
    protocol. Ledger-tagged 'probe' so nothing downstream mistakes it."""
    cfgd = C.cfg()
    ln = lane(cfgd, "probe")
    if os.path.isdir(a.deck):
        pdir = a.deck.rstrip("/")
        pid = os.path.basename(pdir)
        meta = json.load(open(os.path.join(pdir, "meta.json")))
        deck_local = os.path.join(pdir, "deck.csv")
        opps = a.opps.split(",") if a.opps else meta["target_opponents"]
        base_rel = meta["base_deck"] if a.vs_base else None
    else:
        pid = "probe_" + md5f(a.deck)[:8]
        deck_local = a.deck
        if not a.opps:
            sys.exit("--opps required when --deck is a raw csv")
        opps = a.opps.split(",")
        base_rel = None
    stage_rel = f"decklab/staging/{pid}/deck.csv"
    cells = [[stage_rel, o] for o in opps]
    if base_rel:
        cells += [[base_rel, o] for o in opps]
    stage_file(ln, deck_local, stage_rel)
    sync_pool(ln)
    jn = f"{pid}_probe_{int(time.time()) % 100000}"
    frozen = getattr(a, "pilot", "lane") == "frozen"
    ck = cfgd["ckpt_primary"] if frozen else ln["ckpt"]
    hd = cfgd["heads"] if frozen else ln["heads"]
    check_ckpt(ln, os.path.join(ln["repo"], ck))
    job = {"ckpt": os.path.join(ln["repo"], ck),
           "cells": cells, "games": a.games, "device": ln["device"],
           "heads": hd, "temp": cfgd["temp"],
           "seed": int(time.time()) % 10**6}
    res = run_job(ln, job, jn, ln.get("workers_probe", 3), "decklab_probe")
    print(f"  [pilot {os.path.basename(ck)} heads={hd}]")
    if len(res) < len(cells):
        sys.exit(f"ARENA FAILURE: {len(res)}/{len(cells)} cells returned")
    print(f"\nPROBE {pid} ({a.games} games/cell, EXPLORATORY -- not evidence):")
    rows = []
    for r in res:
        who = "prop" if r["my_deck"].startswith("decklab/staging/") else "base"
        rows.append({"arm": who, "opp": os.path.basename(r["opp_deck"])[:-4],
                     "wr": r["wr"], "games": r["games"]})
        print(f"  {who}  vs {rows[-1]['opp'][:48]:48s} {r['wins']}/{r['games']} = {r['wr']:.3f}")
    C.ledger_append({"id": pid, "event": "probe", "games_per_cell": a.games,
                     "results": rows})


def verdict(a):
    cfgd = C.cfg()
    pdir = a.proposal_dir.rstrip("/")
    pid = os.path.basename(pdir)
    meta = json.load(open(os.path.join(pdir, "meta.json")))
    acc = cfgd["accept"]
    reasoning = meta.get("evidence_class") == "reasoning"
    if reasoning:
        print("reasoning-class: not game-gated; acceptance is the review's "
              "call (status set by the orchestrator), scoring happens "
              "post-append via decklab_score.")
        return {"id": pid, "evidence_class": "reasoning", "status": meta.get("status")}
    report = {"id": pid, "evidence_class": "measured"}
    for tag in ("pri", "rep"):
        p = os.path.join(C.RESULTS, pid, f"ab_{tag}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        entry = {}
        if r["targeted"]:
            d, se, ds = _stage_stats(r["targeted"])
            want = meta["predicted"].get("targeted", "+")
            sign = 1 if want != "-" else -1
            ok = sign * d >= max(acc["targeted_min_delta"],
                                 acc["targeted_min_z"] * se)
            entry["targeted"] = {"delta": round(d, 4), "se": round(se, 4),
                                 "per_opp": [round(x, 3) for x in ds], "pass": bool(ok)}
        if r["field"]:
            d, se, _ = _stage_stats(r["field"], weighted=True)
            du, seu, _ = _stage_stats(r["field"])
            ok = d >= -acc["field_noninferiority_z"] * se
            entry["field"] = {"delta_weighted": round(d, 4), "se": round(se, 4),
                              "delta_unweighted": round(du, 4), "pass": bool(ok)}
        entry["pass"] = all(v["pass"] for v in entry.values()) if entry else False
        report[tag] = entry
    pri_ok = report.get("pri", {}).get("pass", False)
    rep_ok = report.get("rep", {}).get("pass", False)
    need_rep = cfgd.get("replication_required", True)
    if pri_ok and (rep_ok or not need_rep):
        meta["status"] = "accepted"
    elif pri_ok:
        meta["status"] = "tested_pass"          # awaiting replication
    else:
        meta["status"] = "tested_fail" if report.get("pri") else meta.get("status")
    json.dump(meta, open(os.path.join(pdir, "meta.json"), "w"), indent=1)
    report["status"] = meta["status"]
    print(json.dumps(report, indent=1))
    C.ledger_append({"id": pid, "event": "verdict", **report})
    return report


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup-arena")
    r = sub.add_parser("run")
    r.add_argument("proposal_dir")
    r.add_argument("--replicate", action="store_true")
    r.add_argument("--stage", default="both", choices=["both", "targeted", "field"])
    r.add_argument("--games-scale", type=float, default=1.0)
    r.add_argument("--profile", default=None, choices=["screen"],
                   help="screen: the config's 'screen' game counts and "
                        "gauntlet (a faster pool-insertion lane); same accept bars")
    v = sub.add_parser("verdict")
    v.add_argument("proposal_dir")
    p = sub.add_parser("probe")
    p.add_argument("--deck", required=True,
                   help="proposal dir OR raw deck csv path")
    p.add_argument("--opps", default=None,
                   help="comma-separated opponent csv paths (default: the "
                        "proposal's target_opponents)")
    p.add_argument("--games", type=int, default=60)
    p.add_argument("--vs-base", action="store_true",
                   help="also measure the base deck on the same opponents")
    p.add_argument("--pilot", default="lane", choices=["lane", "frozen"],
                   help="which policy PILOTS the probe. 'lane' (default) = the "
                        "probe lane's own checkpoint; 'frozen' = the A/B "
                        "evaluator (ckpt_primary), for comparing a probe "
                        "against historical probe numbers.")
    a = ap.parse_args()
    if a.cmd == "setup-arena":
        setup_arena(C.cfg())
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "probe":
        probe(a)
    else:
        verdict(a)


if __name__ == "__main__":
    main()
