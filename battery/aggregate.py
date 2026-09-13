#!/usr/bin/env python3
"""Pool the arena chunk files a battery run wrote into win-rate tables.

    python3 battery/aggregate.py <outdir> [--json summary.json]

Reads every <deck>--vs--<opponent>--c<k>.jsonl in <outdir> (one game per
line; the trailing {"summary": ...} record is skipped) and prints the
per-cell matrix, the mean over decks for each opponent, the mean over
opponents for each deck, the pooled win rate and the macro mean (unweighted
mean over cells), each with a 95% normal-approximation interval.
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load(outdir):
    cells = defaultdict(lambda: {"w": 0, "n": 0, "draws": 0,
                                 "seat": [[0, 0], [0, 0]], "steps": 0})
    for path in sorted(glob.glob(os.path.join(outdir, "*--vs--*.jsonl"))):
        stem = os.path.basename(path)[:-len(".jsonl")]
        deck, rest = stem.split("--vs--", 1)
        opp = rest.rsplit("--c", 1)[0]
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "summary" in r:
                    continue
                c = cells[(deck, opp)]
                c["n"] += 1
                c["w"] += int(r["won"])
                c["draws"] += int(r.get("result") == 2)
                s = c["seat"][int(r.get("seat", 0))]
                s[0] += int(r["won"])
                s[1] += 1
                c["steps"] += int(r.get("steps", 0))
    return cells


def ci(w, n):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    h = 1.96 * (p * (1 - p) / n) ** 0.5
    return p, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    a = ap.parse_args()
    cells = load(a.outdir)
    if not cells:
        raise SystemExit(f"no *--vs--*.jsonl files under {a.outdir}")
    decks = sorted({d for d, _ in cells})
    opps = sorted({o for _, o in cells})

    w = max(len(d) for d in decks) + 2
    print(f"{'deck':<{w}}" + "".join(f"{o[:16]:>17}" for o in opps) + f"{'mean':>8}")
    per_deck = {}
    for d in decks:
        row = ""
        rates = []
        for o in opps:
            c = cells.get((d, o))
            if c and c["n"]:
                p, h = ci(c["w"], c["n"])
                rates.append(p)
                row += f"{p:>7.3f}±{h:.3f}/{c['n']:<3d}"
            else:
                row += f"{'-':>17}"
        per_deck[d] = sum(rates) / len(rates) if rates else float("nan")
        print(f"{d:<{w}}{row}{per_deck[d]:>8.3f}")
    per_opp = {}
    row = ""
    for o in opps:
        rs = [cells[(d, o)]["w"] / cells[(d, o)]["n"]
              for d in decks if (d, o) in cells and cells[(d, o)]["n"]]
        per_opp[o] = sum(rs) / len(rs) if rs else float("nan")
        row += f"{per_opp[o]:>17.3f}"
    macro = sum(c["w"] / c["n"] for c in cells.values() if c["n"]) / \
        sum(1 for c in cells.values() if c["n"])
    print(f"{'mean':<{w}}{row}{macro:>8.3f}")
    tw = sum(c["w"] for c in cells.values())
    tn = sum(c["n"] for c in cells.values())
    p, h = ci(tw, tn)
    draws = sum(c["draws"] for c in cells.values())
    print(f"\npooled {tw}/{tn} = {p:.3f} ± {h:.3f}   macro mean {macro:.3f}   "
          f"draws {draws}   cells {len(cells)}")
    s0 = [0, 0]; s1 = [0, 0]
    for c in cells.values():
        s0[0] += c["seat"][0][0]; s0[1] += c["seat"][0][1]
        s1[0] += c["seat"][1][0]; s1[1] += c["seat"][1][1]
    print(f"seat 0 {s0[0]}/{s0[1]} = {s0[0] / max(1, s0[1]):.3f}   "
          f"seat 1 {s1[0]}/{s1[1]} = {s1[0] / max(1, s1[1]):.3f}")
    if a.json:
        out = {"cells": {f"{d}|{o}": {"wins": c["w"], "games": c["n"],
                                     "draws": c["draws"]}
                         for (d, o), c in cells.items()},
               "per_deck": per_deck, "per_opponent": per_opp,
               "pooled": p, "macro_mean": macro}
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
