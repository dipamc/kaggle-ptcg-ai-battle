#!/usr/bin/env python3
"""Opponent-distribution-corrected deck strength from a trainer deck matrix.

    tools/deck_nash.py <deckmat-dir> --pool decks/pool --names <listing>
    tools/deck_nash.py <dir> --pool decks/pool --epoch-min 3990 --csv out.csv

`tools/deck_generalists.py` reports two numbers: `mean` (= mu, the raw
game-weighted win rate over every opponent file) and `floor` (the worst
matchup class). Those are the two ends of a spectrum, and the interesting
part is that BOTH are answers to the same question under different
assumptions about who you will actually play:

    mean   -> "vs the pool as it happens to be composed"
    macro  -> "vs each strategy class equally often"
    nash   -> "vs the equilibrium field, i.e. an opponent pool that has
               already adapted to you"
    floor  -> "vs the single worst matchup" (maximally adversarial)

THE PROBLEM WITH `mean`. Decks are sampled uniformly over deck FILES, so a
deck's mean is weighted by how many variants of each archetype the pool
happens to ship. On a ~380-deck pool the classes were sized 54/52/51/48/48/40/34/
16/15/7/5/4/4/2/2 — the largest is 14% of the field and the smallest is
0.5%. In a game with rock-paper-scissors structure that is a real bias, not
a rounding error: a deck that happens to counter the 54-deck class is paid
28x more for it than a deck that counters a 2-deck class. Pool composition
is an artifact of how we assembled the pool, so any statistic that inherits
it is measuring our file list as much as the deck.

`macro` fixes the duplication by averaging the shrunk per-class rates with
equal weight per class. That is the right correction if you expect to meet
each STRATEGY equally often. It is still a choice, not a fact — it treats a
2-deck class as being as likely as a 54-deck one.

`nash` makes no such choice. The deck matrix is a symmetric zero-sum game,
so it has an equilibrium mixture p* over decks; `nash_wr` is a deck's win
rate against p*. Opponents that no rational field would play get weight ~0
regardless of how many files of them the pool ships, and the decks that
survive are exactly the ones in the support of p*. This is the principled
answer to "normalize for RPS skew" — it is the only weighting here that is
derived from the payoff structure rather than imposed on it.

Solved by multiplicative-weights self-play on the symmetrized shrunk
matrix, which converges to Nash in zero-sum games; the printed
exploitability is the convergence check (should be ~0.000).

CAVEATS. (1) All of this still measures deck strength UNDER OUR POLICY
against OUR pool - the equilibrium is over the decks we happen to have, so
`nash` corrects for how the pool is WEIGHTED but not for what it is
MISSING. (2) The classes come from profile correlation on the data, not
from names; names are only used to label a class after the fact.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_matrix import deck_names, fit_to_names, load  # noqa: E402
from deck_generalists import fit_shrinkage, matchup_classes  # noqa: E402


def nash(A, iters=40000, eta=1.0, seed=0):
    """Equilibrium mixture of a symmetric zero-sum game via multiplicative
    weights self-play. A[i,j] = win rate of i vs j, A + A.T == 1.

    Returns (p, exploitability). p is the average iterate; exploitability is
    max_j (M p)_j with M = A - 0.5, which is 0 exactly at equilibrium.
    """
    M = A - 0.5
    n = len(M)
    p = np.full(n, 1.0 / n)
    acc = np.zeros(n)
    for t in range(iters):
        g = M @ p
        p = p * np.exp(eta * g)
        p /= p.sum()
        if t >= iters // 2:          # average the second half only
            acc += p
    p = acc / acc.sum()
    return p, float((M @ p).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pool", default="decks/pool")
    ap.add_argument("--alt-pool", default=None)
    ap.add_argument("--names", default=None,
                    help="pinned listing this matrix was built from")
    ap.add_argument("--epoch-min", type=int, default=None)
    ap.add_argument("--epoch-max", type=int, default=None)
    ap.add_argument("--cluster-from-all", action="store_true")
    ap.add_argument("--cut", type=float, default=0.45)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--deck", default=None, help="substring: show these rows")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    G, W, eps, nfiles = load(a.paths, a.epoch_min, a.epoch_max)
    if a.names:
        names = [l.strip() for l in open(a.names) if l.strip()]
    else:
        names = deck_names(a.pool, a.alt_pool)
    # constant-width matrices: a shorter listing is normal, see fit_to_names
    G, W, names = fit_to_names(G, W, names,
                        src=(f"--names {a.names!r}" if a.names else f"pool {a.pool!r}"))
    n = len(G)

    N = G + G.T
    K = W + (G.T - W.T)
    np.fill_diagonal(N, 0)
    np.fill_diagonal(K, 0)
    mu = K.sum(1) / np.maximum(N.sum(1), 1)
    kEB, obs, noise, true = fit_shrinkage(N, K, mu)
    print(f"{nfiles} file(s), {eps:,} episodes, {n} decks")
    print(f"matchup spread: TRUE sd {true**.5:.3f} "
          f"({100*true/obs:.0f}% real), EB prior {kEB:.1f} games")

    # shrunk deck-vs-deck rate, then symmetrized so A + A.T == 1 exactly
    # (each row shrinks toward its own mu, which breaks antisymmetry).
    P = (K + mu[:, None] * kEB) / (N + kEB)
    A = (P + (1.0 - P.T)) / 2.0
    np.fill_diagonal(A, 0.5)

    if a.cluster_from_all and (a.epoch_min is not None or a.epoch_max is not None):
        Ga, Wa, epsa, _ = load(a.paths)
        Na = Ga + Ga.T
        Ka = Wa + (Ga.T - Wa.T)
        np.fill_diagonal(Na, 0)
        np.fill_diagonal(Ka, 0)
        mua = Ka.sum(1) / np.maximum(Na.sum(1), 1)
        ka, _, _, _ = fit_shrinkage(Na, Ka, mua)
        cl = matchup_classes((Ka + mua[:, None] * ka) / (Na + ka), mua, a.cut)
        print(f"classes derived from the full matrix ({epsa:,} episodes)")
    else:
        cl = matchup_classes(P, mu, a.cut)
    C = len(cl)
    NC = np.zeros((n, C))
    KC = np.zeros((n, C))
    for c, mem in enumerate(cl):
        NC[:, c] = N[:, mem].sum(1)
        KC[:, c] = K[:, mem].sum(1)
    PC = (KC + mu[:, None] * kEB) / (NC + kEB)
    # Classes are profile clusters, NOT archetypes. Naming one after its
    # plurality member is convenient and MISLEADING: 8 of 15 classes here
    # have a plurality under 50%, and one labelled kangaskhan is 17%
    # kangaskhan and holds all of its equilibrium mass in hydrapple decks.
    # So carry the purity and the mass-holding archetype alongside the name.
    lab, purity, comp = [], [], []
    for mem in cl:
        cnt = {}
        for i in mem:
            key = names[i].split("__")[0]
            cnt[key] = cnt.get(key, 0) + 1
        best = max(cnt, key=cnt.get)
        lab.append(best)
        purity.append(cnt[best] / len(mem))
        comp.append(cnt)
    print(f"{C} classes, sizes {[len(c) for c in cl]}")
    impure = sum(1 for x in purity if x < 0.5)
    if impure:
        print(f"NOTE: {impure}/{C} class labels are a MINORITY of their class "
              f"(profile clusters, not archetypes) — read the label as "
              f"'the cluster around X', never as 'the X decks'")

    macro = PC.mean(1)
    floor = PC.min(1)
    p, expl = nash(A, a.iters)
    nash_wr = A @ p
    print(f"nash: exploitability {expl:+.4f} after {a.iters} iters, "
          f"support (p > 1/{10*n}) = {int((p > 0.1/n).sum())} decks")

    # class-level equilibrium mass, so the mixture is readable
    cmass = np.zeros(C)
    for c, mem in enumerate(cl):
        cmass[c] = p[mem].sum()
    print("\n== EQUILIBRIUM FIELD (class mass vs pool share) ==")
    print(f"{'class (plurality label)':34s}{'pool%':>7}{'nash%':>7}{'ratio':>7}"
          f"{'pure':>6}  archetype actually holding the mass")
    for c in np.argsort(-cmass):
        pool = 100.0 * len(cl[c]) / n
        if cmass[c] * 100 < 0.5 and pool < 3:
            continue
        r = (100 * cmass[c] / pool) if pool else float("nan")
        hold, hm = "-", 0.0
        for k in comp[c]:
            sub = [i for i in cl[c] if names[i].split("__")[0] == k]
            if p[sub].sum() > hm:
                hm, hold = p[sub].sum(), k
        note = f"{hold[:30]} {100*hm:.0f}%"
        if hold.split("_")[0] != lab[c].split("_")[0]:
            note += "  != LABEL"
        print(f"{lab[c][:32]:34s}{pool:7.1f}{100*cmass[c]:7.1f}{r:6.2f}x"
              f"{100*purity[c]:5.0f}%  {note}")

    rk = lambda v: np.argsort(np.argsort(-v)) + 1
    r_mu, r_ma, r_na, r_fl = rk(mu), rk(macro), rk(nash_wr), rk(floor)

    def row(i):
        return (f"{names[i][:42]:42s}"
                f"{mu[i]:7.3f}{r_mu[i]:5d}"
                f"{macro[i]:8.3f}{r_ma[i]:5d}"
                f"{nash_wr[i]:8.3f}{r_na[i]:5d}"
                f"{floor[i]:8.3f}{r_fl[i]:5d}")

    hdr = (f"{'deck':42s}{'mean':>7}{'#':>5}{'macro':>8}{'#':>5}"
           f"{'nash':>8}{'#':>5}{'floor':>8}{'#':>5}")
    print(f"\n== TOP {a.top} BY NASH (vs the equilibrium field) ==\n{hdr}")
    for i in np.argsort(-nash_wr)[:a.top]:
        print(row(i))

    print(f"\n== TOP {a.top} BY RAW MEAN, for contrast ==\n{hdr}")
    for i in np.argsort(-mu)[:a.top]:
        print(row(i))

    if a.deck:
        print(f"\n== decks matching {a.deck!r} ==\n{hdr}")
        for i in [k for k, x in enumerate(names) if a.deck.lower() in x.lower()]:
            print(row(i))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["deck", "mean", "mean_rank", "macro", "macro_rank",
                        "nash", "nash_rank", "floor", "floor_rank",
                        "nash_weight"])
            for i in np.argsort(-nash_wr):
                w.writerow([names[i], f"{mu[i]:.4f}", r_mu[i],
                            f"{macro[i]:.4f}", r_ma[i], f"{nash_wr[i]:.4f}",
                            r_na[i], f"{floor[i]:.4f}", r_fl[i],
                            f"{p[i]:.5f}"])
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
