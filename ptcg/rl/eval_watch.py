"""Checkpoint watcher: whenever training saves a new checkpoint, run an
eval vs the random baseline (the objective progress metric self-play
stats can't provide), dump a few games — both vs-random and self-play —
as replay JSON, and log to wandb + runs/eval_watch.jsonl.

Run alongside training:
    nohup python -m ptcg.rl.eval_watch --wandb --run-name selfplay_v1 \
        > runs/eval_watch.log 2>&1 &
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from .buffers import OFFSETS
from .env import CAUSES, PTCGEnv
from .model import PTCGTransformer, arch_from_state_dict

REPLAY_DIR = "runs/replays"
BENCH_OPPS = {}   # dir -> loaded kaggle agent fn (filled once in main)

# deckmeta layer 1: fixed watchlist for the movement panel; promote
# risers here when they break out in the layer-2 table
DECKMETA_WATCHLIST = (
    "mega_kangaskhan_ex", "crustle", "marnie_s_grimmsnarl_ex",
    "dragapult_ex", "ogerpon_meganium_hydrapple", "starmie_froslass",
    "archaludon_dudunsparce")
DECKMETA_WALLS = ("mega_kangaskhan_ex", "crustle")
DECKMETA_SUB_DECK = ""   # optional: a deck name to track in the deckmeta panel


def deckmeta(log_dir, window=50000):
    """Aggregate the RECENT window of the training deck logs into
    (scalars, table_rows): watchlist archetype win rates, submission-
    deck win rate, walls' deck-out win share, plus a full per-archetype
    table (wr/BT/n/cause shares) for the new-riser catcher."""
    import glob as g_
    from collections import Counter, defaultdict, deque

    from .deck_report import bradley_terry, deck_names
    names = deck_names()
    arch = [n.split("__")[0] for n in names]
    files = g_.glob(os.path.join(log_dir, "*.jsonl"))
    if not files:
        return {}, []
    per = max(200, window // len(files))
    rows = []
    for f in files:
        tail = deque(open(f), maxlen=per)
        rows.extend(json.loads(l) for l in tail if l.strip())

    astat = defaultdict(lambda: [0, 0])
    causes = defaultdict(Counter)
    dstat = defaultdict(lambda: [0, 0])
    h2h = Counter()
    for r in rows:
        # idx -1 = external decklist (scripted league anchor) — keep
        # anchor-agent strength out of the deck tallies
        if r["r"] not in (0, 1) or r["d0"] < 0 or r["d1"] < 0:
            continue
        w, l = (r["d0"], r["d1"]) if r["r"] == 0 else (r["d1"], r["d0"])
        if w != l:
            dstat[w][0] += 1
            dstat[w][1] += 1
            dstat[l][1] += 1
        wa, la = arch[w], arch[l]
        if wa == la:
            continue
        astat[wa][0] += 1
        astat[wa][1] += 1
        astat[la][1] += 1
        causes[wa][r["c"]] += 1
        h2h[(wa, la)] += 1
    bt = bradley_terry(h2h) if h2h else {}

    scalars = {}
    for a in DECKMETA_WATCHLIST:
        if astat[a][1]:
            scalars[f"deckmeta/wr_{a}"] = astat[a][0] / astat[a][1]
    sub_idx = names.index(DECKMETA_SUB_DECK) \
        if DECKMETA_SUB_DECK in names else None
    if sub_idx is not None and dstat[sub_idx][1]:
        scalars[f"deckmeta/wr_{DECKMETA_SUB_DECK}"] = \
            dstat[sub_idx][0] / dstat[sub_idx][1]
    wall_w = sum(astat[a][0] for a in DECKMETA_WALLS)
    wall_deck = sum(causes[a]["deck"] for a in DECKMETA_WALLS)
    if wall_w:
        scalars["deckmeta/deckout_share_walls"] = wall_deck / wall_w
    scalars["deckmeta/window_games"] = float(len(rows))

    table_rows = []
    for a, (w, g) in sorted(astat.items(), key=lambda kv: -bt.get(kv[0], 0)):
        if g < 50:
            continue
        cw = causes[a]
        tot = max(1, sum(cw.values()))
        table_rows.append([a, round(w / g, 3), round(bt.get(a, 0), 2), g,
                           round(cw["prize"] / tot, 3),
                           round(cw["deck"] / tot, 3),
                           round(cw["bench"] / tot, 3)])
    return scalars, table_rows


def play(policy, env, episodes, device, rng):
    env.reset(seed=int(rng.integers(1 << 30)))
    while env._stats["episodes"] < episodes:
        obs = torch.as_tensor(env.observations).to(device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs)
        actions = torch.distributions.Categorical(
            logits=logits).sample().cpu().numpy()
        env.step(actions)
    st = dict(env._stats)
    env.close()
    return st


def list_checkpoints(ckpt_dir):
    """Periodic saves live at experiments/<run_id>/model_<epoch>.pt;
    close-time saves at experiments/<id>.pt. Watch both.

    SCOPED TO ONE RUN on purpose. Globbing all of experiments/ is fine
    while a machine holds exactly one run and silently wrong otherwise: a
    checkpoint staged from a *different*, fully-trained run then gets
    scored as the new run's first eval point (win_vs_random 0.990 at
    epoch 0).
    """
    return (glob.glob(f"{ckpt_dir}.pt")
            + glob.glob(os.path.join(ckpt_dir, "model_*.pt")))


def _bench_chunk(units, ckpt_path, device, heads, seed):
    """Run a share of the deck x opponent matrix on one device. Invoked in a
    SUBPROCESS via `--bench-worker job.json` (self-invocation — mp.spawn's
    main-module re-import is unreliable under `python -m` across hosts).

    A UNIT is (deck, opponent, game indices) — a SLICE of a cell, not a whole
    cell. Cell-granular sharding makes wall-clock the max over workers instead
    of the mean: 30 cells over 14 workers hands two workers 3 cells and the
    other twelve 2, so most of the workers idle for the last third of every
    battery. Slices let any worker count balance.

    Each unit seeds its own RNG from (seed, deck, opponent, first index), so
    the numbers do not depend on how the work was split. One RNG per worker,
    consumed across whatever cells that worker drew, would not hold that.
    """
    import random as _random
    from .arena import play as arena_play, load_kaggle_agent
    from .search_agent import SearchAgent
    out = []
    agents = {}
    opps = {}
    for deck_path, opp_dir, idxs in units:
        if deck_path not in agents:
            agents[deck_path] = SearchAgent(ckpt_path, deck_path, cfg=None,
                                            device=device, heads=heads)
        if opp_dir not in opps:
            opps[opp_dir] = load_kaggle_agent(opp_dir)
        rng2 = _random.Random(f"{seed}|{deck_path}|{opp_dir}|{idxs[0]}")
        w = 0
        for i in idxs:
            # seat parity follows the GLOBAL game index, so a slice cannot
            # skew the seat split the way a naive 0..len(idxs) would
            r, steps, dec, cause = arena_play(agents[deck_path],
                                              opps[opp_dir], opp_dir,
                                              i % 2, rng2, [])
            w += r == i % 2
        out.append((deck_path, opp_dir, w, len(idxs)))
    return out


def _bench_worker_main(job_path):
    import json
    with open(job_path) as f:
        job = json.load(f)
    units = [(u[0], u[1], u[2]) for u in job["units"]]
    out = _bench_chunk(units, job["ckpt"], job["device"], job["heads"],
                       job["seed"])
    with open(job["out"], "w") as f:
        json.dump(out, f)


def evaluate(path, args, rng):
    tag = os.path.relpath(path, "experiments").replace("/", "-") \
        .removesuffix(".pt")
    env = PTCGEnv(num_envs=args.games, opponent="random",
                  log_interval=10**9, seed=int(rng.integers(1 << 30)),
                  replay_dir=REPLAY_DIR,
                  replay_every=max(1, args.episodes // args.replays),
                  replay_teams=[f"ckpt-{tag}", "random"])
    sd = torch.load(path, map_location=args.device)
    arch = arch_from_state_dict(sd, heads=args.heads)
    policy = PTCGTransformer(env, **arch).to(args.device)
    policy.load_state_dict(
        {k.replace("module.", ""): v for k, v in sd.items()})
    policy.eval()
    policy.freeze_tables()

    # NAMING CONVENTION (extend this pattern for any future eval
    # opponent <opp>, e.g. a scripted punisher or a frozen anchor):
    #   eval/win_rate_vs_<opp>, eval/win_end_<opp>_<cause>,
    #   eval/lose_end_<opp>_<cause>, eval/ep_len_<opp>,
    #   eval/prize_margin_<opp>, eval/turns_<opp>
    # Mirror games are the one exception: winner-cause == loser-cause by
    # construction, so there is a single eval/end_self_<cause> set.
    # The classifier's 'other' safety bucket goes to misc/ (always 0;
    # showing up on a dashboard at all is the alarm).
    st = play(policy, env, args.episodes, args.device, rng)
    e = st["episodes"]
    metrics = {
        "eval/win_rate_vs_random": st["wins"] / e,
        "eval/ep_len_random": st["ep_len"] / e,
        "eval/prize_margin_random": st["prize_margin"] / e,
        "eval/turns_random": (st["turns_first"] + st["turns_second"]) / e,
        "eval/episodes": e,
        "eval/ckpt_mtime": os.path.getmtime(path),
    }
    # win causes are only meaningful UNMIXED: split by who actually won
    for pre, label in (("wend", "win_end_random"), ("lend", "lose_end_random")):
        tot = max(1, sum(st[f"{pre}_{c}"] for c in CAUSES))
        for c in CAUSES:
            key = f"eval/{label}_{c}" if c != "other" \
                else f"misc/eval_{label}_{c}"
            metrics[key] = st[f"{pre}_{c}"] / tot

    # --- benchmark battery (v4): candidate plays EVERY pinned deck vs
    # every fixed opponent, policy-only. Replaces archetype-eval; the
    # submission deck pick and the promotion gate read this matrix.
    #   <prefix>/win_rate_{deck}_vs_{opp}   per cell
    #   <prefix>/mean5_vs_{opp}             mean over the pinned decks
    #   <prefix>/macro_mean                 unweighted mean over every cell
    # <prefix> is --bench-prefix (default "eval"). gate_mean/gate_min are
    # GONE: they stepped discontinuously whenever the deck set changed and
    # measured nothing the per-opponent means don't.
    if args.bench_opponents:
        import random as _random
        per_opp = {os.path.basename(d): [] for d in BENCH_OPPS}
        cell_wr = {}
        decks = args.bench_decks.split(",")
        gpus = [g.strip() for g in args.bench_gpus.split(",") if g.strip()]
        # PARALLELISM IS INDEPENDENT OF DEVICE. Gating the fan-out on
        # `if args.bench_gpus:` silently ignores --bench-workers whenever no
        # GPU list is passed and runs the whole matrix serially in one
        # process — a ~14x slowdown that looks exactly like an idle machine.
        # --bench-gpus means only "round-robin workers over THESE cuda
        # devices"; with a single GPU leave it empty and the workers inherit
        # the parent's device visibility.
        shards = args.bench_workers or len(gpus) or 1
        if shards > 1:
            # shard the matrix across SUBPROCESSES (self-invocation with
            # --bench-worker); slice cells into game-chunks, aggregate here
            import json as _json
            import subprocess as _sp
            import sys as _sys
            import tempfile as _tf
            cells = [(dp, d) for dp in decks for d in BENCH_OPPS]
            # Cost split per game at d256/h8: policy forward 33% (cuda,
            # 2.78ms/call) or 47% (cpu, 5.11ms), engine+opponent the rest,
            # obs encode ~3%. The forward is cheap next to the engine at
            # d128 but not at d256, so --bench-device is worth setting per
            # machine rather than pinning to cpu. Every call is batch-1
            # (SearchAgent.row is (1, OBS_SIZE)) and cg.game is a per-process
            # singleton, so a process can only ever hold one game: more
            # PROCESSES is the lever, batching would need an inference
            # server shared across workers.
            #
            # Aim for ~8 units per worker so the tail is short: a unit is the
            # granularity at which a straggler can cost us.
            splits = max(1, -(-(shards * 8) // max(1, len(cells))))
            csize = max(1, -(-args.bench_games // splits))
            units = []
            for dp, d in cells:
                for s in range(0, args.bench_games, csize):
                    units.append((dp, d, list(range(
                        s, min(s + csize, args.bench_games)))))
            shards = min(shards, len(units))
            seed = int(rng.integers(1 << 30))
            with _tf.TemporaryDirectory() as td:
                procs = []
                outs = []
                for k in range(shards):
                    # CONTIGUOUS blocks, not units[k::shards]: sizes still
                    # differ by at most one, and a worker draws neighbouring
                    # units so it builds ~2-3 SearchAgents instead of one per
                    # deck in the matrix (each is a torch.load + a full
                    # guess-pool read).
                    chunk = units[k * len(units) // shards:
                                  (k + 1) * len(units) // shards]
                    if not chunk:
                        continue
                    jf = os.path.join(td, f"job{k}.json")
                    of = os.path.join(td, f"out{k}.json")
                    with open(jf, "w") as f:
                        _json.dump({"units": chunk, "ckpt": path,
                                    "device": args.bench_device,
                                    "heads": args.heads,
                                    "seed": seed,
                                    "out": of}, f)
                    # one thread each: torch would otherwise spin up a full
                    # pool per process and oversubscribe the machine
                    env = dict(os.environ, OMP_NUM_THREADS="1",
                               MKL_NUM_THREADS="1")
                    if args.bench_device != "cuda":
                        env["CUDA_VISIBLE_DEVICES"] = ""
                    elif gpus:
                        env["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]
                    procs.append(_sp.Popen(
                        [_sys.executable, "-m", "ptcg.rl.eval_watch",
                         "--bench-worker", jf], env=env))
                    outs.append(of)
                for p in procs:
                    if p.wait() != 0:
                        raise RuntimeError("bench worker failed "
                                           f"(exit {p.returncode})")
                agg = {}
                for of in outs:
                    with open(of) as f:
                        for deck_path, opp_dir, w, n in _json.load(f):
                            a = agg.setdefault((deck_path, opp_dir), [0, 0])
                            a[0] += w
                            a[1] += n
                for key, (w, n) in agg.items():
                    cell_wr[key] = w / n
        else:
            from .arena import play as arena_play
            from .search_agent import SearchAgent
            for deck_path in decks:
                agent = SearchAgent(path, deck_path, cfg=None,
                                    device=args.device, heads=args.heads)
                for d, fn in BENCH_OPPS.items():
                    rng2 = _random.Random(int(rng.integers(1 << 30)))
                    w = 0
                    for i in range(args.bench_games):
                        r, steps, dec, cause = arena_play(agent, fn, d, i % 2,
                                                          rng2, [])
                        w += r == i % 2
                    cell_wr[(deck_path, d)] = w / args.bench_games
        # The battery lives under its OWN namespace (--bench-prefix, default
        # "eval" for back-compat). Point it somewhere new whenever the deck or
        # opponent set changes: these series are keyed by deck and opponent
        # name, so a changed set silently starts new keys next to the old ones
        # and any chart mixing them spans two different measurements.
        pre = args.bench_prefix
        for deck_path in decks:
            dname = os.path.basename(deck_path).removesuffix(".csv")
            dshort = dname.split("__")[0][:24]
            for d in BENCH_OPPS:
                name = os.path.basename(d)
                wr = cell_wr[(deck_path, d)]
                metrics[f"{pre}/win_rate_{dshort}_vs_{name}"] = wr
                per_opp[name].append(wr)
        for name, ws in per_opp.items():
            if ws:
                metrics[f"{pre}/mean5_vs_{name}"] = sum(ws) / len(ws)
        if per_opp:
            allw = [w for ws in per_opp.values() for w in ws]
            metrics[f"{pre}/macro_mean"] = sum(allw) / len(allw)

    n_self = max(args.self_games, args.self_replays)
    env2 = PTCGEnv(num_envs=8, opponent="self", log_interval=10**9,
                   seed=int(rng.integers(1 << 30)),
                   replay_dir=REPLAY_DIR,
                   replay_every=max(1, n_self // max(1, args.self_replays)),
                   replay_teams=[f"ckpt-{tag}", f"ckpt-{tag}"])
    st2 = play(policy, env2, n_self, args.device, rng)
    e2 = max(1, st2["episodes"])
    tot = max(1, sum(st2[f"end_{c}"] for c in CAUSES))
    for c in CAUSES:
        key = f"eval/end_self_{c}" if c != "other" \
            else f"misc/eval_end_self_{c}"
        metrics[key] = st2[f"end_{c}"] / tot
    metrics["eval/turns_self"] = \
        (st2["turns_first"] + st2["turns_second"]) / e2
    metrics["eval/ep_len_self"] = st2["ep_len"] / e2  # BOTH seats' decisions
    # winner-perspective margin, decided games only (negative = won by
    # bench/deck-out while behind on prizes)
    metrics["eval/prize_margin_self"] = \
        st2["prize_margin_dec"] / max(1, st2["eps_dec"])
    return tag, metrics


def _hf_backup(repo, ckpt_path, tag, since):
    """Single batched commit per eval cycle (respect HF rate limits)."""
    from huggingface_hub import HfApi, CommitOperationAdd
    run = ckpt_path.split("/")[1] if "/" in ckpt_path else "misc"
    ops = [CommitOperationAdd(
        f"checkpoints/{run}/{os.path.basename(ckpt_path)}", ckpt_path)]
    if os.path.exists("runs/eval_watch.jsonl"):
        ops.append(CommitOperationAdd("eval_watch.jsonl",
                                      "runs/eval_watch.jsonl"))
    for f in glob.glob("runs/*.log"):
        ops.append(CommitOperationAdd(f"logs/{os.path.basename(f)}", f))
    # deck outcome logs + their offsets manifests (append-only; the
    # batched commit just refreshes them)
    for f in (glob.glob("runs/deck_stats/*/*.jsonl")
              + glob.glob("runs/deck_stats/*_offsets.jsonl")):
        rel = os.path.relpath(f, "runs")
        ops.append(CommitOperationAdd(rel, f))
    for f in glob.glob("runs/replays/*.json"):
        if os.path.getmtime(f) > since:
            ops.append(CommitOperationAdd(
                f"replays/{os.path.basename(f)}", f))
    HfApi().create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                          commit_message=f"backup {tag}")
    print(f"[eval_watch] HF backup: {len(ops)} files, 1 commit ({tag})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--replays", type=int, default=3)
    ap.add_argument("--self-replays", type=int, default=3)
    ap.add_argument("--self-games", type=int, default=24,
                    help="mirror games per cycle for eval/end_self_* stats")
    ap.add_argument("--bench-opponents",
                    default="",
                    help="comma-separated kaggle-agent dirs to benchmark "
                         "each checkpoint against ('' disables)")
    ap.add_argument("--bench-games", type=int, default=16)
    ap.add_argument("--bench-decks",
                    default=",".join(f"decks/pool/{d}.csv" for d in (
                        "marnie_s_grimmsnarl_ex_spikemuth_gym__04",
                        "alakazam_enhanced_hammer__03",
                        "dragapult_ex_crushing_hammer__05",
                        "cynthia_s_garchomp_ex_cynthia_s_roserade_cynth__01",
                        "rillaboom_dipplin_festival_grounds__03",
                        "crustle_jumbo_ice_cream__01")),
                    help="candidate plays EACH of these decks")
    ap.add_argument("--bench-prefix", default="eval",
                    help="wandb namespace for the battery metrics. Change it "
                         "whenever the deck or opponent set changes: series "
                         "are keyed by deck and opponent NAME, so a new set "
                         "silently starts new keys beside the old ones under "
                         "the same prefix.")
    ap.add_argument("--gate-opponent", default="",
                    help="opponent whose mean/min over the pinned decks "
                         "forms the promotion gate ('' disables)")
    ap.add_argument("--bench-workers", type=int, default=0,
                    help="total bench processes; >1 fans the matrix out. This "
                         "is the ONLY knob that turns parallelism on (it no "
                         "longer needs --bench-gpus). Each game is half engine "
                         "and half batch-1 forward, and one process can hold "
                         "one game, so this is the lever. On a dedicated eval "
                         "machine use cores-1; alongside training keep it well "
                         "under the spare core count (training itself wants "
                         "~num-threads * gpus). 0 = one per --bench-gpus "
                         "entry, else serial.")
    ap.add_argument("--bench-device", default="cuda",
                    help="device for bench workers. Set per machine: on a "
                         "dedicated eval machine 'cuda' is ~1.8x faster per "
                         "forward at d256 and frees the cores for engine "
                         "work; alongside training use 'cpu', which costs no "
                         "VRAM (training already holds ~21.8G of each card, "
                         "so cuda workers would OOM).")
    ap.add_argument("--bench-gpus", default="",
                    help="comma-separated cuda device ids (e.g. '4,5,6,7') to "
                         "round-robin --bench-workers over. Empty = workers "
                         "inherit the parent's device visibility, which is "
                         "what you want with a single GPU. This does NOT "
                         "control whether the fan-out happens. "
                         "Results aggregate in the parent — single wandb "
                         "writer either way")
    ap.add_argument("--deckmeta-dir", default=None,
                    help="training deck-log dir (runs/deck_stats/<run>); "
                         "enables deckmeta/ movement metrics + table")
    ap.add_argument("--deckmeta-window", type=int, default=50000)
    # cpu default: the training process typically owns all VRAM
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="ptcg")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--wandb-name", default=None,
                    help="wandb run id/name; default eval-<run-name>. Use a "
                         "NEW one after deleting a run — re-initing a "
                         "just-deleted id hangs on its tombstone")
    ap.add_argument("--ckpt-dir", default=None,
                    help="directory to watch; default "
                         "experiments/train-<run-name>. Scoped to one run "
                         "so a checkpoint staged from another run cannot be "
                         "scored as this one's")
    ap.add_argument("--steps-per-epoch", type=int, default=0,
                    help="rows x bptt (workers x games_per_env x 64). When "
                         "set, eval metrics get a `timesteps` coordinate and "
                         "wandb plots them against it instead of an "
                         "auto-incrementing counter — so eval curves from "
                         "different runs/forks are directly comparable")
    ap.add_argument("--heads", type=int, default=8,
                    help="attention heads the checkpoint was TRAINED with; "
                         "unlike d/layers/ffn this is not recoverable from "
                         "the weights and a wrong value loads silently")
    ap.add_argument("--pool-dir", default=None,
                    help="copy each evaluated checkpoint here (league pool)")
    ap.add_argument("--hf-repo", default=None,
                    help="HF dataset repo for rolling backups — ONE commit "
                         "per eval cycle (checkpoint+jsonl+logs+new replays)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--bench-worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.bench_worker:
        _bench_worker_main(args.bench_worker)
        return

    if args.bench_opponents:
        from .arena import load_kaggle_agent
        for d in args.bench_opponents.split(","):
            if d:
                BENCH_OPPS[d] = load_kaggle_agent(d)

    ckpt_dir = args.ckpt_dir or f"experiments/train-{args.run_name}"
    wb_name = args.wandb_name or f"eval-{args.run_name}"
    print(f"[eval_watch] watching {ckpt_dir} (heads={args.heads}) "
          f"-> wandb {wb_name}", flush=True)

    wb = None
    if args.wandb:
        import wandb
        # stable id: watcher restarts RESUME the same run instead of
        # fragmenting the eval curve across new wandb runs
        wb = wandb.init(project=args.wandb_project,
                        id=wb_name, resume="allow",
                        name=wb_name, group="eval")
        if args.steps_per_epoch:
            # x-axis = timesteps trained, not wandb's auto step counter.
            # define_metric rather than log(step=...) on purpose: the step
            # arg must be monotonic AND fits awkwardly at 2e9 (near int32),
            # while a step_metric is just a logged coordinate.
            wb.define_metric("timesteps")
            for pre in ("eval/*", "misc/*", "deckmeta/*"):
                wb.define_metric(pre, step_metric="timesteps")

    rng = np.random.default_rng(int(time.time()))
    last_mtime = 0.0
    cycle_t0 = 0.0
    fails = 0
    while True:
        ckpts = list_checkpoints(ckpt_dir)
        fresh = [p for p in ckpts
                 if os.path.getmtime(p) > last_mtime
                 and time.time() - os.path.getmtime(p) > 15]
        if fresh:
            path = max(fresh, key=os.path.getmtime)
            last_mtime = os.path.getmtime(path)
            try:
                tag, metrics = evaluate(path, args, rng)
                if args.pool_dir:
                    import shutil
                    os.makedirs(args.pool_dir, exist_ok=True)
                    shutil.copy2(path, os.path.join(
                        args.pool_dir, f"{tag}.pt"))
                dm_scalars, dm_rows = {}, []
                if args.deckmeta_dir:
                    try:
                        dm_scalars, dm_rows = deckmeta(
                            args.deckmeta_dir, args.deckmeta_window)
                        metrics.update(dm_scalars)
                    except Exception as e:
                        print(f"[eval_watch] deckmeta failed: {e!r}",
                              flush=True)
                if args.steps_per_epoch:
                    ep = re.search(r"model_(\d+)\.pt$", path)
                    if ep:
                        metrics["timesteps"] = \
                            int(ep.group(1)) * args.steps_per_epoch
                line = {"ckpt": tag, "time": time.time(), **metrics}
                with open("runs/eval_watch.jsonl", "a") as f:
                    f.write(json.dumps(line) + "\n")
                if wb is not None:
                    payload = dict(metrics)
                    if dm_rows:
                        import wandb
                        payload["deckmeta/archetype_table"] = wandb.Table(
                            columns=["archetype", "wr", "bt", "games",
                                     "prize_share", "deck_share",
                                     "bench_share"], data=dm_rows)
                    wb.log(payload)
                if args.hf_repo:
                    try:
                        _hf_backup(args.hf_repo, path, tag, cycle_t0)
                    except Exception as e:
                        print(f"[eval_watch] HF backup failed: {e!r}",
                              flush=True)
                cycle_t0 = time.time()
                print(f"[eval_watch] {tag}: "
                      f"win_vs_random={metrics['eval/win_rate_vs_random']:.3f} "
                      f"margin={metrics['eval/prize_margin_random']:.2f}",
                      flush=True)
            except Exception as e:  # keep the daemon alive
                # last_mtime already advanced, so this checkpoint is never
                # retried -- a systematic failure (wrong arch, missing
                # opponent dir) therefore costs the run EVERY eval and every
                # HF checkpoint backup while looking like silence. Say so
                # loudly enough that a log watcher can key on it.
                fails += 1
                print(f"[eval_watch] EVAL FAILED ({fails} in a row) on "
                      f"{path}: {e!r}", flush=True)
                if fails >= 2:
                    print("[eval_watch] EVAL FAILING REPEATEDLY — no metrics "
                          "and NO HF BACKUPS are being produced for this run",
                          flush=True)
            else:
                fails = 0
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
