#!/usr/bin/env python3
"""Pick an eval OPPONENT set from a trainer deck matrix.

    tools/deck_evalset.py <deckmat-dir> --pool decks/pool --names <listing> \
        --epoch-min 3990 --cluster-from-all --generalists 3 --specialists 3

Choosing decks to PLAY and decks to PLAY AGAINST are different problems.
A deck you ship wants a high floor - it must never be blown out. An eval
opponent wants the opposite property from the SET: between them the
opponents should be able to blow out anything that has a hole, or the
battery cannot see the hole. A battery of six decks that all punish the
same weakness measures one number six times.

So this selects on COVERAGE of the matchup classes, in two blocks:

GENERALISTS - strong everywhere, from distinct classes. These measure
whether the learner is broadly competent, and their win rates move
smoothly with checkpoint quality, which is what makes a battery able to
rank adjacent checkpoints. Ranked by macro (equal weight per class, so
the pool's archetype duplication does not leak in - see deck_nash.py),
gated on floor and on membership in the equilibrium support.

SPECIALISTS - chosen by greedy set cover over the classes the
generalists do NOT already punish. A specialist is deliberately NOT a
good all-round deck; it exists to make one specific weakness visible.
Its value is `best class rate`, not its mean.

The printed coverage matrix is the actual deliverable: it says, for each
of the classes, whether ANY opponent in the set beats it, and by how
much. An uncovered class is a blind spot in the battery, stated
explicitly rather than discovered later.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_matrix import deck_names, fit_to_names, load  # noqa: E402
from deck_generalists import fit_shrinkage, matchup_classes  # noqa: E402
from deck_nash import nash  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pool", default="decks/pool")
    ap.add_argument("--alt-pool", default=None)
    ap.add_argument("--names", default=None)
    ap.add_argument("--epoch-min", type=int, default=None)
    ap.add_argument("--epoch-max", type=int, default=None)
    ap.add_argument("--cluster-from-all", action="store_true")
    ap.add_argument("--cut", type=float, default=0.45)
    ap.add_argument("--generalists", type=int, default=3)
    ap.add_argument("--specialists", type=int, default=3)
    ap.add_argument("--cover", type=float, default=0.60,
                    help="a class counts as covered at this win rate")
    ap.add_argument("--floor-bar", type=float, default=0.35,
                    help="minimum floor for a generalist")
    ap.add_argument("--min-cell", type=int, default=25)
    ap.add_argument("--weight", choices=("nash", "pool", "blend"),
                    default="blend",
                    help="how much each class counts when scoring coverage: "
                         "by equilibrium mass, by pool share, or half each")
    a = ap.parse_args()

    G, W, eps, nfiles = load(a.paths, a.epoch_min, a.epoch_max)
    names = ([l.strip() for l in open(a.names) if l.strip()] if a.names
             else deck_names(a.pool, a.alt_pool))
    # constant-width matrices: a shorter listing is normal, see fit_to_names
    G, W, names = fit_to_names(G, W, names, src=f"pool {a.pool!r}")
    n = len(G)

    N = G + G.T
    K = W + (G.T - W.T)
    np.fill_diagonal(N, 0)
    np.fill_diagonal(K, 0)
    mu = K.sum(1) / np.maximum(N.sum(1), 1)
    kEB, _, _, _ = fit_shrinkage(N, K, mu)
    P = (K + mu[:, None] * kEB) / (N + kEB)
    print(f"{nfiles} file(s), {eps:,} episodes, {n} decks, EB prior {kEB:.1f}")

    if a.cluster_from_all and (a.epoch_min is not None or a.epoch_max is not None):
        Ga, Wa, epsa, _ = load(a.paths)
        Na, Ka = Ga + Ga.T, Wa + (Ga.T - Wa.T)
        np.fill_diagonal(Na, 0)
        np.fill_diagonal(Ka, 0)
        mua = Ka.sum(1) / np.maximum(Na.sum(1), 1)
        ka, _, _, _ = fit_shrinkage(Na, Ka, mua)
        cl = matchup_classes((Ka + mua[:, None] * ka) / (Na + ka), mua, a.cut)
        print(f"classes from the full matrix ({epsa:,} episodes)")
    else:
        cl = matchup_classes(P, mu, a.cut)
    C = len(cl)
    NC = np.zeros((n, C))
    KC = np.zeros((n, C))
    for c, mem in enumerate(cl):
        NC[:, c] = N[:, mem].sum(1)
        KC[:, c] = K[:, mem].sum(1)
    PC = (KC + mu[:, None] * kEB) / (NC + kEB)
    lab = []
    for mem in cl:
        cnt = {}
        for i in mem:
            k = names[i].split("__")[0]
            cnt[k] = cnt.get(k, 0) + 1
        lab.append(max(cnt, key=cnt.get))
    of = np.array([next(c for c, m in enumerate(cl) if i in m) for i in range(n)])

    macro, floor = PC.mean(1), PC.min(1)
    A = (P + (1.0 - P.T)) / 2.0
    np.fill_diagonal(A, 0.5)
    p, expl = nash(A)
    cmass = np.array([p[m].sum() for m in cl])
    ok = NC.min(1) >= a.min_cell
    print(f"{C} classes; nash exploitability {expl:+.4f}; "
          f"{int((~ok).sum())} deck(s) excluded for thin cells\n")

    sel, why = [], []
    # --- generalists: best macro, floor bar, equilibrium support, distinct class
    cand = np.where(ok & (floor >= a.floor_bar) & (p > 0.1 / n))[0]
    if len(cand) < a.generalists:
        cand = np.where(ok & (floor >= a.floor_bar))[0]
        print(f"!! fewer than {a.generalists} supported decks clear the floor "
              f"bar; dropped the equilibrium-support requirement")
    used = set()
    for i in cand[np.argsort(-macro[cand])]:
        if len(sel) >= a.generalists:
            break
        if of[i] in used:
            continue
        used.add(of[i])
        sel.append(i)
        why.append(f"generalist: macro {macro[i]:.3f}, floor {floor[i]:.3f}, "
                   f"nash {100*p[i]:.1f}%")

    # --- specialists: greedy MARGINAL-GAIN cover.
    # A hard "is this class covered yet" test deadlocks: if no deck in the
    # pool beats a class at the bar, that gap never closes and the greedy
    # spends every pick on it (it chose 3 anti-dragapult decks in a row).
    # Scoring each candidate by how much it RAISES the set's best rate,
    # summed over classes and weighted by how much we care, diversifies on
    # its own and degrades gracefully when a class is simply unbeatable.
    share = np.array([len(m) for m in cl], float) / n
    w = {"nash": cmass, "pool": share,
         "blend": 0.5 * cmass + 0.5 * share}[a.weight]
    for _ in range(a.specialists):
        best_c = np.array([max(PC[i, c] for i in sel) for c in range(C)])
        head = np.maximum(0.0, np.minimum(PC, a.cover) - best_c)   # cap: no
        gain = (head * w).sum(1)                                   # credit
        gain[sel] = -1                                             # past bar
        gain[~ok] = -1
        i = int(np.argmax(gain))
        if gain[i] > 1e-9:
            tgt = int(np.argmax(head[i] * w))
            sel.append(i)
            why.append(f"specialist vs {lab[tgt][:26]}: {PC[i, tgt]:.3f} there "
                       f"vs {best_c[tgt]:.3f} for the set so far "
                       f"(macro {macro[i]:.3f} — not a good all-round deck)")
            continue
        # No class can be pushed further (some are simply unbeatable by any
        # deck in the pool). Deepening an already-covered class buys nothing,
        # so fall back to the other axis of a good battery: pick the deck
        # whose matchup PROFILE is least like anything already in the set, so
        # the extra cell is a new measurement rather than a louder copy of an
        # old one. Restricted to decks of at least median strength, else this
        # happily adds a punching bag whose profile is unusual only because
        # it loses to everything.
        Z = PC - PC.mean(1, keepdims=True)
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        rho = (Z @ Z[sel].T).max(1)
        elig = ok & (macro >= np.median(macro[ok]))
        elig[sel] = False
        if not elig.any():
            print("(no eligible deck left; quota unused)")
            break
        i = int(np.argmin(np.where(elig, rho, 9)))
        sel.append(i)
        why.append(f"diversifier: profile correlation only {rho[i]:+.2f} with "
                   f"the closest deck already in the set "
                   f"(macro {macro[i]:.3f}) — every class is already covered "
                   f"as far as the pool allows, so this buys a new angle")

    print("== SELECTED OPPONENT SET ==")
    for i, w in zip(sel, why):
        print(f"  {names[i]}")
        print(f"      {w}")

    print(f"\n== COVERAGE: win rate of each opponent vs each class "
          f"(>= {a.cover:.2f} = covered) ==")
    w = max(len(lab[c][:26]) for c in range(C))
    print(" " * (w + 2) + "".join(f"{k:>7d}" for k in range(1, len(sel) + 1))
          + "     best  pool%  nash%")
    for c in np.argsort(-cmass):
        best = max(PC[i, c] for i in sel)
        mark = " " if best >= a.cover else "  <-- BLIND"
        print(f"{lab[c][:26]:{w}s}  "
              + "".join(f"{PC[i, c]:7.2f}" for i in sel)
              + f"{best:9.2f}{100*len(cl[c])/n:7.1f}{100*cmass[c]:7.1f}{mark}")
    blind = [c for c in range(C) if max(PC[i, c] for i in sel) < a.cover]
    print(f"\ncovered {C - len(blind)}/{C} classes "
          f"({100*(1 - cmass[blind].sum()):.1f}% of equilibrium mass)")
    if blind:
        print("blind spots (no opponent in the set beats these):")
        for c in sorted(blind, key=lambda c: -cmass[c]):
            print(f"    {lab[c][:40]:42s} nash {100*cmass[c]:.1f}%  "
                  f"best in set {max(PC[i, c] for i in sel):.2f}")


if __name__ == "__main__":
    main()
