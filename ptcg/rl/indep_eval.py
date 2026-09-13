"""Independent eval harness — a from-scratch cross-check of PTCGEnv.

Deliberately does NOT import ptcg.rl.env. Drives BattleHandle directly
with its own game loop, its own forced-select handling, its own
multi-pick decomposition, its own random opponent, and its own win-cause
classifier — everything computed from the raw engine obs JSON. Shares
with training only what defines the policy itself: encode(), InfoTracker
and PTCGTransformer. If PTCGEnv's row driver / stats had a bug, the
ratios reported here would diverge from eval_watch's.

Reported per mode (mirror self-play and vs-random):
  win causes split by WHO won (learner/opponent per seat for mirror),
  bench behavior straight from engine state (ever-benched, max bench,
  bench size at own turns 1/2), win rate, turns.

    python -m ptcg.rl.indep_eval --ckpt experiments/<run>/model_X.pt \
        --games-self 100 --games-random 200
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch

from . import battle as battle_mod
from .buffers import OBS_SIZE, STOP
from .encoder import encode
from .model import PTCGTransformer
from ..tracker import InfoTracker

# deck loading duplicated from env.py ON PURPOSE: this module must not
# import env (pufferlib/gymnasium), so it stays runnable on machines
# with only torch+numpy installed.
# PTCG_DECK_DIR lets a run point every deck-name lookup at the pool its
# tables blob was actually built from. eval_watch's deckmeta panel maps
# training deck INDICES to names through here, so a run on a different pool
# would silently mislabel every archetype without it.
DECK_DIR = os.environ.get("PTCG_DECK_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "decks", "pool")


def load_deck_pool(deck_dir=DECK_DIR):
    decks = []
    for f in sorted(glob.glob(os.path.join(deck_dir, "*.csv"))):
        with open(f) as fh:
            ids = [int(line) for line in fh if line.strip()]
        if len(ids) == 60:
            decks.append(ids)
    if not decks:
        raise FileNotFoundError(f"no decks in {deck_dir}")
    return decks


def _forced(sel):
    """Independent re-derivation of the forced-select contract."""
    n = len(sel["option"])
    if sel["minCount"] == sel["maxCount"] == n:
        return list(range(n))
    if n == 1 and sel["minCount"] >= 1:
        return [0]
    return None


def _rand_answer(sel, rng):
    n = len(sel["option"])
    k = rng.randint(sel["minCount"], min(sel["maxCount"], n))
    return sorted(rng.sample(range(n), k))


def _end_cause(cur, winner):
    """Independent win-condition classifier from raw engine state."""
    win, lose = cur["players"][winner], cur["players"][1 - winner]
    if len(win["prize"]) == 0:
        return "prize"
    if not any(p for p in lose["active"]) and not any(p for p in lose["bench"]):
        return "bench"
    if lose["deckCount"] == 0:
        return "deck"
    return "other"


def _bench_sizes(cur):
    return [sum(1 for p in pl["bench"] if p) for pl in cur["players"]]


class _Policy:
    def __init__(self, path):
        self.net = PTCGTransformer()
        sd = torch.load(path, map_location="cpu")
        self.net.load_state_dict(
            {k.replace("module.", ""): v for k, v in sd.items()})
        self.net.eval()
        self.net.freeze_tables()
        self.row = np.zeros((1, OBS_SIZE), dtype=np.float32)

    def decide(self, obs, picked, stop_ok, tracker, rng):
        encode(self.row[0], obs, picked, stop_ok, 0, tracker=tracker)
        with torch.no_grad():
            logits = self.net.forward_policy(torch.from_numpy(self.row))[0]
        return int(torch.distributions.Categorical(logits=logits).sample(
            torch.Size([]))), logits


def _policy_answer(policy, obs, tracker, rng, bench_log=None):
    """Full multi-pick decomposition, reimplemented from the engine
    contract: sequential picks then STOP; single-pick submits at once."""
    sel = obs["select"]
    n = len(sel["option"])
    picked = []
    while True:
        stop_ok = (len(picked) >= sel["minCount"]
                   and (sel["maxCount"] > 1 or sel["minCount"] == 0))
        a, _ = policy.decide(obs, picked, stop_ok, tracker, rng)
        if a == STOP:
            if not stop_ok:      # masked out; can't happen — count anyway
                a = next(j for j in range(n) if j not in picked)
            else:
                return picked
        if a >= n or a in picked:
            a = next((j for j in range(n) if j not in picked), None)
            if a is None:
                return picked
        if sel["maxCount"] == 1:
            return [a]
        picked.append(a)
        if len(picked) >= sel["maxCount"]:
            return picked


def play_game(policy, decks, rng, mode, learner_seat):
    """mode: 'self' (policy both seats) or 'random' (opponent random).
    Returns per-game record computed only from raw engine obs."""
    d0, d1 = rng.choice(decks), rng.choice(decks)
    h = battle_mod.BattleHandle(d0, d1)
    trackers = [InfoTracker(d0), InfoTracker(d1)]
    obs = h.obs()
    rec = {"mode": mode, "learner_seat": learner_seat,
           "decisions": [0, 0], "max_bench": [0, 0],
           "bench_t2": [None, None]}   # bench size at own turn 2 decision
    steps = 0
    while True:
        cur = obs["current"]
        if cur["result"] != -1 or steps > 3000:
            break
        seat = cur["yourIndex"]
        trackers[seat].update(obs)
        b = _bench_sizes(cur)
        rec["max_bench"] = [max(rec["max_bench"][k], b[k]) for k in (0, 1)]
        if cur["turn"] >= 3 and rec["bench_t2"][seat] is None:
            rec["bench_t2"][seat] = b[seat]     # own board entering turn>=3
        sel = obs["select"]
        ans = _forced(sel)
        if ans is None:
            if mode == "random" and seat != learner_seat:
                ans = _rand_answer(sel, rng)
            else:
                ans = _policy_answer(policy, obs, trackers[seat], rng)
                rec["decisions"][seat] += 1
        try:
            obs = h.select(ans)
        except RuntimeError:
            if ans:
                raise
            obs = h.select([0])  # engine rejects advertised-legal []
        steps += 1
    cur = obs["current"]
    rec["turns"] = cur["turn"]
    rec["result"] = cur["result"]
    rec["first"] = cur["firstPlayer"]
    if cur["result"] in (0, 1):
        rec["cause"] = _end_cause(cur, cur["result"])
    else:
        rec["cause"] = "draw" if cur["result"] == 2 else "trunc"
    rec["final_bench"] = _bench_sizes(cur)
    h.finish()
    return rec


def _cause_table(recs, won_fn=None):
    causes = ("prize", "bench", "deck", "other")
    sub = [r for r in recs if r["result"] in (0, 1)
           and (won_fn is None or won_fn(r))]
    n = max(1, len(sub))
    return {c: round(sum(r["cause"] == c for r in sub) / n, 3)
            for c in causes} | {"n": len(sub)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games-self", type=int, default=100)
    ap.add_argument("--games-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write full records jsonl")
    args = ap.parse_args()

    torch.set_num_threads(8)
    rng = random.Random(args.seed)
    decks = load_deck_pool(DECK_DIR)
    policy = _Policy(args.ckpt)

    recs = []
    t0 = time.time()
    for i in range(args.games_self):
        recs.append(play_game(policy, decks, rng, "self", None))
        if (i + 1) % 20 == 0:
            print(f"[self {i+1}/{args.games_self}] "
                  f"{time.time()-t0:.0f}s", flush=True)
    for i in range(args.games_random):
        recs.append(play_game(policy, decks, rng, "random", rng.randint(0, 1)))
        if (i + 1) % 40 == 0:
            print(f"[random {i+1}/{args.games_random}] "
                  f"{time.time()-t0:.0f}s", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    selfr = [r for r in recs if r["mode"] == "self"]
    randr = [r for r in recs if r["mode"] == "random"]
    lw = [r for r in randr if r["result"] == r["learner_seat"]]

    def bench_stats(rs, seat_fn):
        seats = [(r, seat_fn(r)) for r in rs]
        ever = np.mean([r["max_bench"][s] > 0 for r, s in seats])
        mx = np.mean([r["max_bench"][s] for r, s in seats])
        t2 = [r["bench_t2"][s] for r, s in seats if r["bench_t2"][s] is not None]
        return {"ever_benched": round(float(ever), 3),
                "mean_max_bench": round(float(mx), 2),
                "mean_bench_at_t3": round(float(np.mean(t2)), 2) if t2 else None}

    report = {
        "ckpt": args.ckpt,
        "self_play": {
            "games": len(selfr),
            "causes_all": _cause_table(selfr),
            "p0_win": round(np.mean([r["result"] == 0
                                     for r in selfr if r["result"] in (0, 1)]), 3),
            "mean_turns": round(np.mean([r["turns"] for r in selfr]), 1),
            "bench_seat0": bench_stats(selfr, lambda r: 0),
            "bench_seat1": bench_stats(selfr, lambda r: 1),
        },
        "vs_random": {
            "games": len(randr),
            "learner_win_rate": round(len(lw) / max(1, len(randr)), 3),
            "causes_learner_won": _cause_table(
                randr, lambda r: r["result"] == r["learner_seat"]),
            "causes_learner_lost": _cause_table(
                randr, lambda r: r["result"] == 1 - r["learner_seat"]),
            "causes_all": _cause_table(randr),
            "mean_turns": round(np.mean([r["turns"] for r in randr]), 1),
            "learner_bench": bench_stats(randr, lambda r: r["learner_seat"]),
            "random_bench": bench_stats(randr, lambda r: 1 - r["learner_seat"]),
        },
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
