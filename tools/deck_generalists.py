#!/usr/bin/env python3
"""Find decks with no decisive weakness, from a trainer deck matrix.

    tools/deck_generalists.py <deckmat-dir> --pool decks/pool [--epoch-min N]
    tools/deck_generalists.py data/deckmat/<run> --pool decks/pool --csv out.csv

`tools/deck_matrix.py` ranks decks by mean win rate against the training
field. That is the wrong statistic for picking a deck to ship: the field is a
counter-cycle, so the highest mean usually belongs to a deck that folds to one
archetype. `mega_lopunny_ex__lf_30e496` had the 12th-best mean in one
training pool, a 0.189 floor, and scored 984 on the ladder - the worst of
five arms.

This ranks by the FLOOR instead: the worst matchup a deck has anywhere in the
field. Three things have to be handled for that number to mean anything.

1. SHRINKAGE. At ~60 games per pair, a deck that is truly 0.50 against
   everyone still shows a minimum near 0.34 across 200 opponents, purely from
   noise. The prior strength is fitted from the data (variance decomposition:
   observed spread minus the binomial part), not chosen - on one measured
   pool that gave k=7.3 games, i.e. 90% of the observed spread is real matchup
   structure and the correction is mild. If a future pool has weaker matchup
   structure the fit will shrink harder on its own.

2. OPPONENTS ARE NOT INDEPENDENT. Garchomp shipped 7 variants in pool v1 and
   crustle 3, so counting deck files penalises one archetype-level weakness
   seven times while a singleton counter barely registers. Decks are grouped
   into matchup CLASSES by profile correlation - how they do against each
   opponent relative to their own mean - so decks countered by the same things
   group together. This is done on the data, not on names: it put
   `cynthia_s_garchomp_ex__ladder` with the other six garchomps and merged
   lucario_hariyama into that class because they share a profile.

3. TRUNCATED GAMES NEVER ENTER THE MATRIX (see deck_matrix.py's `res`
   column). A stall deck's win rate is conditional on the game ending, so its
   floor is not comparable. Anything below --res-warn is excluded and listed.

Validation (split the matrix files into two independent halves):
Spearman rho 0.947 on the floor, 0.972 on the mean of the 3 worst classes.
The ranking replicates; it is not fitting noise.

Two decks can share a floor and behave differently, so `spread` (the
noise-corrected sd of win rate across classes) is reported too. Read it WITH
the mean: the flattest decks in any pool are usually the ones that lose to
everything equally.
"""
import argparse
import csv
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from deck_matrix import deck_names, fit_to_names, load  # noqa: E402


def fit_shrinkage(N, K, mu):
    """Prior strength in games, from how much of the spread is not binomial."""
    m = N > 0
    p = np.where(m, K / np.maximum(N, 1), np.nan)
    obs = np.nanvar((p - mu[:, None])[m])
    noise = np.nanmean((p[m] * (1 - p[m])) / N[m])
    true = max(obs - noise, 1e-9)
    return 0.25 / true - 1, obs, noise, true


def matchup_classes(P, mu, cut):
    """Average-linkage clustering on 1 - correlation of matchup profiles."""
    n = len(mu)
    prof = P - mu[:, None]
    np.fill_diagonal(prof, 0)
    Z = prof - prof.mean(1, keepdims=True)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    D = 1 - Z @ Z.T
    np.fill_diagonal(D, 0)
    mem = {i: [i] for i in range(n)}
    Dc = D.copy()
    np.fill_diagonal(Dc, np.inf)
    active = set(range(n))
    while len(active) > 1:
        a = np.array(sorted(active))
        sub = Dc[np.ix_(a, a)]
        ij = np.unravel_index(np.argmin(sub), sub.shape)
        if sub[ij] > cut:
            break
        i, j = a[ij[0]], a[ij[1]]
        ni, nj = len(mem[i]), len(mem[j])
        new = (Dc[i] * ni + Dc[j] * nj) / (ni + nj)
        Dc[i] = new
        Dc[:, i] = new
        Dc[i, i] = np.inf
        Dc[j] = np.inf
        Dc[:, j] = np.inf
        mem[i] += mem[j]
        active.discard(j)
    return [mem[i] for i in sorted(active)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pool", default="decks/pool")
    ap.add_argument("--alt-pool", default=None)
    ap.add_argument("--names", default=None,
                    help="file of deck names, one per line, in id order. Use "
                         "this to read an ARCHIVED matrix: ids are positions "
                         "in the pool's sorted glob, so adding a deck to the "
                         "pool dir silently renames every id after it. "
                         "Keep a names file per run to pin the listing it "
                         "was built with.")
    ap.add_argument("--epoch-min", type=int, default=None)
    ap.add_argument("--epoch-max", type=int, default=None)
    ap.add_argument("--cluster-from-all", action="store_true",
                    help="derive the matchup classes from the FULL matrix, "
                         "then measure inside the epoch window. Use this "
                         "whenever comparing successive windows of one run: "
                         "classes are a property of the field, and letting a "
                         "thin window re-cluster makes the numbers "
                         "incomparable between reads.")
    ap.add_argument("--cut", type=float, default=0.45,
                    help="dendrogram cut; lower = more, finer classes")
    ap.add_argument("--res-warn", type=float, default=0.9,
                    help="exclude decks resolving below this fraction")
    ap.add_argument("--min-cell", type=int, default=25,
                    help="exclude decks whose thinnest class cell is smaller")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--deck", default=None, help="substring: profile these decks")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    G, W, eps, nfiles = load(a.paths, a.epoch_min, a.epoch_max)
    # Matrices are pre-sized to a constant width (docs/deck-pool.md),
    # so "narrower listing than matrix" is the normal case and must NOT be an
    # error — the trailing rows are all-zero. fit_to_names still refuses to trim
    # over recorded games, which is the case that really means a wrong listing:
    # deck ids are positions in the pool's sorted glob, so a reordered or
    # shrunk pool dir gives a mapping that is WRONG, not merely incomplete.
    if a.names:
        names = [ln.strip() for ln in open(a.names) if ln.strip()]
        src = f"--names {a.names!r}"
    else:
        names = deck_names(a.pool, a.alt_pool)
        src = f"pool {a.pool!r}"
    G, W, names = fit_to_names(G, W, names, src=src)
    n = G.shape[0]

    # symmetric: N games between i and j, K = i's wins
    N = G + G.T
    K = W + (G.T - W.T)
    np.fill_diagonal(N, 0)
    np.fill_diagonal(K, 0)
    played = N.sum(1)
    mu = K.sum(1) / np.maximum(played, 1)
    res = played / (2.0 * eps / n)

    kEB, obs, noise, true = fit_shrinkage(N, K, mu)
    print(f"{nfiles} file(s), {eps:,} episodes, {n} decks")
    print(f"matchup spread: observed sd {obs**.5:.3f}, binomial {noise**.5:.3f}, "
          f"TRUE {true**.5:.3f} -> {100*true/obs:.0f}% real structure")
    print(f"empirical-Bayes prior: {kEB:.1f} games")

    if a.cluster_from_all and (a.epoch_min is not None or a.epoch_max is not None):
        Ga, Wa, epsa, _ = load(a.paths)
        # The full-history load is untrimmed, but the window matrix was cut to
        # the names listing above. Matrices are pre-sized to a constant width
        # (1024), so without this the class members carry ids past the end of
        # N and indexing dies. Trailing ids hold no games — coverage decks live
        # at PT_DECK_ALT_BASE and are excluded from the matrix entirely.
        Ga, Wa = Ga[:n, :n], Wa[:n, :n]
        Na = Ga + Ga.T
        Ka = Wa + (Ga.T - Wa.T)
        np.fill_diagonal(Na, 0)
        np.fill_diagonal(Ka, 0)
        mua = Ka.sum(1) / np.maximum(Na.sum(1), 1)
        ka, _, _, _ = fit_shrinkage(Na, Ka, mua)
        cl = matchup_classes((Ka + mua[:, None] * ka) / (Na + ka), mua, a.cut)
        print(f"classes derived from the full matrix ({epsa:,} episodes)")
    else:
        cl = matchup_classes((K + mu[:, None] * kEB) / (N + kEB), mu, a.cut)

    # Drop decks with NO games in the window from every class before the cells
    # are counted. A deck sampled at weight 0 (docs/deck-pool.md)
    # has an all-zero matchup profile, so it never merges with anything and
    # survives clustering as its own singleton class -- and then EVERY deck's
    # thinnest class cell is its 0-game cell against that singleton, so
    # --min-cell excludes the entire pool and the ranking silently degenerates
    # into id order. Masking rather than renumbering keeps deck ids intact.
    dead = int((played == 0).sum())
    cl = [[i for i in mem if played[i] > 0] for mem in cl]
    cl = [mem for mem in cl if mem]
    if dead:
        print(f"dropped {dead} deck(s) with no games in this window "
              f"(weight 0, or appended after it) -> {len(cl)} classes")
    C = len(cl)
    NC = np.zeros((n, C))
    KC = np.zeros((n, C))
    for c, mem in enumerate(cl):
        NC[:, c] = N[:, mem].sum(1)
        KC[:, c] = K[:, mem].sum(1)

    # A class too small to measure must be dropped from the FLOOR, not used to
    # exclude every deck. A 1-2 deck class in a thin window gives every deck a
    # ~5-game cell, and since the floor is a worst-class statistic, --min-cell
    # then excludes the entire pool and the "ranking" degenerates into id order
    # with a nan median. Dropping the class instead costs coverage, so say so
    # rather than truncating silently.
    keep = np.array([np.median(NC[played > 0, c]) >= a.min_cell for c in range(C)])
    if not keep.all():
        lost = sum(len(cl[c]) for c in range(C) if not keep[c])
        print(f"dropped {int((~keep).sum())} of {C} classes too thin to measure "
              f"(median cell < {a.min_cell} games), covering {lost} deck(s) — "
              f"those matchups are NOT in the floor below")
        cl = [cl[c] for c in range(C) if keep[c]]
        NC, KC = NC[:, keep], KC[:, keep]
        C = len(cl)
    PC = (KC + mu[:, None] * kEB) / (NC + kEB)

    lab = []
    for mem in cl:
        cnt = {}
        for i in mem:
            key = names[i].split("__")[0]
            cnt[key] = cnt.get(key, 0) + 1
        lab.append(max(cnt, key=cnt.get))
    print(f"{C} matchup classes, sizes "
          f"{sorted((len(c) for c in cl), reverse=True)}")

    srt = np.sort(PC, 1)
    floor, worst3 = srt[:, 0], srt[:, :3].mean(1)
    raw = KC / np.maximum(NC, 1)
    spread = np.sqrt(np.maximum(
        raw.var(1) - (raw * (1 - raw) / np.maximum(NC, 1)).mean(1), 0))
    worst = PC.argmin(1)

    stall = res < a.res_warn
    thin = NC.min(1) < a.min_cell
    ok = ~stall & ~thin
    if stall.any():
        print(f"\nEXCLUDED, resolved < {a.res_warn:.0%} (win rate is conditional "
              f"on the game ending): {int(stall.sum())} deck(s)")
        for i in np.where(stall)[0][:10]:
            print(f"    res {res[i]:.2f}  {names[i][:56]}")
    if thin.any():
        print(f"EXCLUDED, thinnest class cell < {a.min_cell} games: "
              f"{int(thin.sum())} deck(s)")

    def row(i):
        return (f"{names[i][:46]:46s} {mu[i]:6.3f}{floor[i]:7.3f}"
                f"{worst3[i]:7.3f}{spread[i]:7.3f}  {lab[worst[i]][:24]}")

    hdr = f"{'deck':46s} {'mean':>6}{'floor':>7}{'worst3':>7}{'spread':>7}  worst class"
    print(f"\n== BEST FLOOR (no decisive weakness) ==\n{hdr}")
    for i in np.argsort(np.where(ok, floor, -9))[::-1][:a.top]:
        print(row(i))

    print(f"\n== FLATTEST, among decks above the field median "
          f"({np.median(mu[ok]):.3f}) ==\n{hdr}")
    sel = ok & (mu > np.median(mu[ok]))
    for i in np.argsort(np.where(sel, spread, 9))[:a.top]:
        if not sel[i]:
            break
        print(row(i))

    if a.deck:
        print(f"\n== decks matching {a.deck!r} ==\n{hdr}")
        for i in [k for k, x in enumerate(names) if a.deck.lower() in x.lower()]:
            print(row(i))
            o = np.argsort(PC[i])
            print("      worst: " + "  ".join(
                f"{lab[c]} {PC[i,c]:.2f}(n={int(NC[i,c])})" for c in o[:4]))

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["deck", "mean_wr", "floor", "worst3", "spread",
                        "worst_class", "worst_class_games", "resolved_frac"])
            for i in np.argsort(-floor):
                w.writerow([names[i], f"{mu[i]:.4f}", f"{floor[i]:.4f}",
                            f"{worst3[i]:.4f}", f"{spread[i]:.4f}",
                            lab[worst[i]], int(NC[i, worst[i]]), f"{res[i]:.3f}"])
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
