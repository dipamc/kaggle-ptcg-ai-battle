"""Aggregate per-episode deck logs into deck / archetype performance.

Reads the jsonl files PTCGEnv writes when deck_log_dir is set (one line
per finished game: deck indices, winner, cause, first player, turns,
opponent kind) and reports:
  - archetype ranking: Bradley-Terry strength over the head-to-head
    matrix (mirror + vs-opponent games where both decks are known),
    win rate, games, win-cause profile
  - top/bottom decks by win rate (same-deck games excluded from win
    rates — they are 1 win + 1 loss by construction)
  - overall first-player advantage

    python -m ptcg.rl.deck_report --logs runs/deck_stats/<run-name>

Deck ids are positions in sorted(decks/pool/*.csv) — the same ordering
load_deck_pool uses; archetype = filename prefix before '__'.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

from .indep_eval import DECK_DIR


def deck_names(deck_dir=DECK_DIR):
    names = []
    for f in sorted(glob.glob(os.path.join(deck_dir, "*.csv"))):
        ids = [l for l in open(f) if l.strip()]
        if len(ids) == 60:
            names.append(os.path.basename(f).removesuffix(".csv"))
    return names


def bradley_terry(wins, iters=300):
    """wins[i][j] = games i beat j. Standard MM updates."""
    ks = sorted({k for ij in wins for k in ij} | {ij[0] for ij in wins}
                | {ij[1] for ij in wins})
    idx = {k: n for n, k in enumerate(ks)}
    n = len(ks)
    w = np.zeros((n, n))
    for (i, j), c in wins.items():
        w[idx[i], idx[j]] = c
    total_w = w.sum(1)
    games = w + w.T
    p = np.ones(n)
    for _ in range(iters):
        denom = (games / (p[:, None] + p[None, :])).sum(1)
        p = np.where(denom > 0, total_w / np.maximum(denom, 1e-12), p)
        p /= max(p.mean(), 1e-12)
    return {k: float(p[idx[k]]) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True,
                    help="dir of deck-log jsonl files (one run)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-games", type=int, default=50)
    ap.add_argument("--json", default=None, help="also dump full stats")
    ap.add_argument("--since", type=int, default=None,
                    help="only records with unix time >= this (needs the "
                         "'t' field; older runs: use --last-frac)")
    ap.add_argument("--last-frac", type=float, default=1.0,
                    help="per log file keep only the last FRAC of lines "
                         "(append order ~ time; works without 't')")
    args = ap.parse_args()

    names = deck_names()
    arch = [n.split("__")[0] for n in names]

    rows = []
    for f in glob.glob(os.path.join(args.logs, "*.jsonl")):
        lines = [l for l in open(f) if l.strip()]
        lines = lines[int(len(lines) * (1 - args.last_frac)):]
        rows.extend(json.loads(l) for l in lines)
    if args.since:
        rows = [r for r in rows if r.get("t", 0) >= args.since]
    decided = [r for r in rows if r["r"] in (0, 1)]
    print(f"{len(rows)} games ({len(decided)} decided) from {args.logs}\n")

    dk = defaultdict(lambda: {"g": 0, "w": 0, "causes": Counter()})
    aw = defaultdict(lambda: {"g": 0, "w": 0, "causes": Counter()})
    h2h = Counter()   # (winner arch, loser arch) -> games
    fp_wins = 0
    for r in decided:
        fp_wins += r["r"] == r["fp"]
        if r["d0"] < 0 or r["d1"] < 0:
            continue  # external decklist (scripted anchor): no deck tally
        win_d, lose_d = (r["d0"], r["d1"]) if r["r"] == 0 else (r["d1"], r["d0"])
        if win_d != lose_d:
            dk[win_d]["g"] += 1
            dk[win_d]["w"] += 1
            dk[lose_d]["g"] += 1
            dk[win_d]["causes"][r["c"]] += 1
        wa, la = arch[win_d], arch[lose_d]
        if wa != la:
            aw[wa]["g"] += 1
            aw[wa]["w"] += 1
            aw[la]["g"] += 1
            h2h[(wa, la)] += 1
        aw[wa]["causes"][r["c"]] += 1

    bt = bradley_terry(h2h) if h2h else {}
    print(f"first-player win rate: {fp_wins / max(1, len(decided)):.3f}\n")

    def cause_str(c):
        t = max(1, sum(c.values()))
        return " ".join(f"{k}:{v / t:.2f}" for k, v in c.most_common(3))

    print(f"=== archetypes (BT strength; {len(aw)} seen) ===")
    ranked = sorted(aw, key=lambda a: -bt.get(a, 0))
    for a in ranked:
        s = aw[a]
        print(f"  {a:34} bt={bt.get(a, 0):5.2f} "
              f"wr={s['w'] / max(1, s['g']):.3f} n={s['g']:>6} "
              f"wins-by[{cause_str(s['causes'])}]")

    eligible = [(d, s) for d, s in dk.items() if s["g"] >= args.min_games]
    ranked_d = sorted(eligible, key=lambda ds: -ds[1]["w"] / ds[1]["g"])
    print(f"\n=== decks: top {args.top} (min {args.min_games} games; "
          f"{len(eligible)}/{len(dk)} eligible) ===")
    for d, s in ranked_d[:args.top]:
        print(f"  {names[d]:40} wr={s['w'] / s['g']:.3f} n={s['g']:>5}")
    print(f"=== decks: bottom {args.top} ===")
    for d, s in ranked_d[-args.top:]:
        print(f"  {names[d]:40} wr={s['w'] / s['g']:.3f} n={s['g']:>5}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"decks": {names[d]: {"wr": s["w"] / max(1, s["g"]),
                                            "n": s["g"]}
                                 for d, s in dk.items()},
                       "archetypes": {a: {"bt": bt.get(a, 0),
                                          "wr": s["w"] / max(1, s["g"]),
                                          "n": s["g"]}
                                      for a, s in aw.items()},
                       "first_player_wr": fp_wins / max(1, len(decided)),
                       "games": len(rows)}, f, indent=1)
        print(f"\nfull stats -> {args.json}")


if __name__ == "__main__":
    main()
