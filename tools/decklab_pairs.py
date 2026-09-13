#!/usr/bin/env python3
"""Mine close deck pairs from a converged deck matrix: the free experiments.

For every pool pair differing by <= --max-swaps cards, report the win-rate
delta overall and per matchup class (classes = deck_generalists profile
clusters, data-driven, no name labels). Each side of a pair has ~10-25k games,
so a 1-swap pair is a natural experiment on those cards at training volume.

    tools/decklab_pairs.py [--matrix <key>] [--max-swaps 2] [--min-cell 150]

Writes decklab/packs/pairs_<matrix>.csv (long format: one row per pair x class
+ an ALL row) and prints the significant single-swap findings.

Read the deltas as: wr(A) - wr(B) piloted by OUR converged policy against OUR
pool field — subject to the same pilotability caveat as every deckmat number.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=None,
                    help="matrix key from decklab/config.json (default: default_matrix)")
    ap.add_argument("--max-swaps", type=int, default=2)
    ap.add_argument("--min-cell", type=int, default=150,
                    help="min combined games for a class cell to be reported")
    ap.add_argument("--cut", type=float, default=0.45)
    a = ap.parse_args()

    a.matrix = a.matrix or C.default_matrix()
    G, W, names, eps = C.load_matrix(a.matrix)
    _, counters = C.load_pool()
    cards, attacks = C.load_carddb()
    idx = {n: i for i, n in enumerate(names)}
    have = [n for n in names if n in counters]

    cls, P, mu = C.clusters_for(G, W, a.cut)
    big = [c for c in cls if len(c) >= 3]
    label = {}
    for ci, mem in enumerate(big):
        label[ci] = "+".join(sorted(names[m][:26] for m in mem[:2]))

    pairs = []
    for x in range(len(have)):
        for y in range(x + 1, len(have)):
            na, nb = have[x], have[y]
            out, inn, k = C.deck_diff(counters[na], counters[nb])
            if 0 < k <= a.max_swaps:
                pairs.append((na, nb, out, inn, k))
    print(f"{a.matrix}: {len(pairs)} pairs within {a.max_swaps} swaps "
          f"({sum(1 for p in pairs if p[4] == 1)} single-swap), {eps} episodes")

    def wr_vs(i, members):
        n = w = 0
        for j in members:
            if j == i:
                continue
            nn, ww = C.pair_cell(G, W, i, j)
            n += nn
            w += ww
        return n, w

    os.makedirs(C.PACKS, exist_ok=True)
    out_path = os.path.join(C.PACKS, f"pairs_{a.matrix}.csv")
    sig = []
    with open(out_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["deck_a", "deck_b", "n_swaps", "cards_a_only", "cards_b_only",
                       "cls", "wr_a", "n_a", "wr_b", "n_b", "delta", "se", "z"])
        for na, nb, out, inn, k in pairs:
            i, j = idx[na], idx[nb]
            co = "; ".join(f"{cards[c]['name']} x{n}" for c, n in sorted(out.items()))
            ci_ = "; ".join(f"{cards[c]['name']} x{n}" for c, n in sorted(inn.items()))
            cells = [("ALL", list(range(len(names))))] + \
                    [(label[ci], mem) for ci, mem in enumerate(big)]
            for lab, mem in cells:
                n1, w1 = wr_vs(i, mem)
                n2, w2 = wr_vs(j, mem)
                if min(n1, n2) < a.min_cell:
                    continue
                p1, p2 = w1 / n1, w2 / n2
                se = float(np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
                d = p1 - p2
                z = d / se if se > 0 else 0.0
                wcsv.writerow([na, nb, k, co, ci_, lab, f"{p1:.4f}", n1,
                               f"{p2:.4f}", n2, f"{d:+.4f}", f"{se:.4f}", f"{z:+.2f}"])
                if k == 1 and lab != "ALL" and abs(z) >= 3:
                    sig.append((abs(d), na, nb, co, ci_, lab, d, n1 + n2, z))
    print(f"wrote {out_path}")
    print(f"\nsingle-swap, |z|>=3 class cells: {len(sig)}")
    for _, na, nb, co, ci_, lab, d, n, z in sorted(sig, reverse=True)[:20]:
        print(f"  {d:+.3f} (z {z:+.1f}, n {n})  [{co}] -> [{ci_}]")
        print(f"      {na}  vs  {nb}   @ {lab[:60]}")


if __name__ == "__main__":
    main()
