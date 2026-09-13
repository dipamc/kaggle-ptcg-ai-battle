#!/usr/bin/env python3
"""Read the trainer's deck-vs-deck result matrices.

The native env writes `<run_dir>/deckmat/r<rank>_e<epoch>_<seq>.bin` every N finished
episodes (`--deck-matrix-every`): a full games/wins matrix over the entire deck
pool, from the LEARNER seat's perspective. Each file covers one interval and is
reset after writing, so summing files gives any window you like.

    tools/deck_matrix.py <dir-or-files...> [--top 25] [--min-games 200]
    tools/deck_matrix.py experiments/<run>/deckmat --pair marnie

Why this exists: wandb can carry a handful of deck series, not hundreds, and the
eval battery only ever scores 5 pinned decks at ~300 games each. This is every
deck at training volume — the only view that can surface a strong variant or
archetype we are not already watching.
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

MAGIC = 0x4D445450   # 'PTDM'


# Must equal PT_DECK_ALT_BASE in native/ptcg/ptcg_env.h. The two are NOT
# generated from each other — same hand-mirrored-constant hazard as the model
# size macros, so change both together.
DECK_ALT_BASE = 1024


def deck_names(pool="decks/pool", alt_pool=None):
    """Deck ids are positions in the sorted-glob 60-line filter of the pool —
    the same construction export_tables.py uses to build the blob.

    Training decks occupy [0, n). The coverage pool starts at a FIXED base
    (DECK_ALT_BASE), not at n, so that appending a training deck mid-run cannot
    renumber it; the gap between is padded here so list index == deck id. Only
    the random opponent's seat ever holds a coverage id, and coverage names are
    marked so a row that can only appear as an opponent is obvious in a table.
    """
    def _scan(d):
        out = []
        for fn in sorted(glob.glob(os.path.join(d, "*.csv"))):
            if sum(1 for line in open(fn) if line.strip()) == 60:
                out.append(os.path.basename(fn)[: -len(".csv")])
        return out
    names = _scan(pool)
    if alt_pool:
        if len(names) > DECK_ALT_BASE:
            sys.exit(f"pool {pool!r} has {len(names)} decks, which reaches the "
                     f"coverage id base ({DECK_ALT_BASE})")
        names = names + [""] * (DECK_ALT_BASE - len(names))
        names += [f"[cov] {n}" for n in _scan(alt_pool)]
    return names


def read_one(path):
    """v1 header: magic, version, n_decks, episodes, seq.
    v2 adds a trailing epoch field (0 when the trainer had not stamped one)."""
    with open(path, "rb") as f:
        magic, ver, n, eps, seq = struct.unpack("<5I", f.read(20))
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic:#x}")
        epoch = struct.unpack("<I", f.read(4))[0] if ver >= 2 else 0
        games = np.frombuffer(f.read(n * n * 4), dtype="<u4").reshape(n, n)
        wins = np.frombuffer(f.read(n * n * 4), dtype="<u4").reshape(n, n)
    return n, eps, seq, epoch, games.astype(np.int64), wins.astype(np.int64)


def _pad_to(m, width):
    """Widen a (n,n) count matrix into (width,width), top-left aligned. Deck id
    i means the same deck at any width (append-only), so this is just zero-fill."""
    if m.shape[0] == width:
        return m
    out = np.zeros((width, width), np.int64)
    n = m.shape[0]
    out[:n, :n] = m
    return out


def fit_to_names(G, W, names, src="matrices"):
    """Reconcile a matrix against a deck-name listing.

    Matrices are pre-sized to a CONSTANT width (PTCG_DECK_MATRIX_MAX, default
    1024) so they stay summable as the pool grows mid-run. The rows past the
    real pool are all-zero by construction, so a names listing that is SHORTER
    than the matrix is the normal case, not an error — trim to it and keep the
    real archetype names, rather than falling back to deck_0..deck_N and losing
    every label.

    Returns (games, wins, names) all agreeing on one width.

    Never trims over live data: if any game was recorded for an id the listing
    does not cover, that means the WRONG listing was passed (the pool grew, or
    the dir was reordered), and quietly dropping those games would understate
    exactly the decks someone just added.
    """
    n, width = len(names), G.shape[0]
    if n > width:
        # Normal in two cases: a listing that includes the coverage pool (whose
        # ids sit above the matrix width by construction), and an older, narrower
        # matrix read against a since-grown pool. Under append-only, id i means
        # the same deck at every width, so dropping the uncovered tail is safe.
        names = names[:width]
        n = width
    if n < width:
        lost = int(G[n:, :].sum()) + int(G[:n, n:].sum())
        if lost:
            sys.exit(f"{src}: {lost} games recorded for deck ids >= {n}, which "
                     f"this {n}-name listing does not cover. The pool grew after "
                     f"these matrices were written — pass the listing they were "
                     f"built from via --names.")
        G, W = G[:n, :n], W[:n, :n]
    return G, W, names


def _expand(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.bin"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            sys.exit(f"no deck matrix at {p}: point it at a <run_dir>/deckmat directory "
                     f"written by a run with --deck-matrix-every (decklab/config.json "
                     f"\"matrices\" for the decklab tools)")
    if not files:
        sys.exit("no matrix files found")
    return files


def load(paths, epoch_min=None, epoch_max=None):
    G = W = None
    eps_total, used = 0, 0
    for f in _expand(paths):
        n, eps, _, epoch, g, w = read_one(f)
        if epoch_min is not None and epoch < epoch_min:
            continue
        if epoch_max is not None and epoch > epoch_max:
            continue
        if G is None:
            G, W = np.zeros((n, n), np.int64), np.zeros((n, n), np.int64)
        # Intervals written either side of a mid-run pool append have different
        # widths. Deck ids are stable under append-only, so widen the narrower
        # side and sum. Before this, mixed widths raised a broadcast error.
        if n > G.shape[0]:
            G, W = _pad_to(G, n), _pad_to(W, n)
        elif n < G.shape[0]:
            g, w = _pad_to(g, G.shape[0]), _pad_to(w, G.shape[0])
        G += g
        W += w
        eps_total += eps
        used += 1
    if G is None:
        sys.exit("no files in the requested epoch range")
    return G, W, eps_total, used


def load_by_epoch(paths):
    """-> {epoch: (games, wins, episodes)}, summing the per-rank files that
    share an epoch stamp."""
    out = {}
    for f in _expand(paths):
        n, eps, _, epoch, g, w = read_one(f)
        if epoch not in out:
            out[epoch] = [np.zeros((n, n), np.int64), np.zeros((n, n), np.int64), 0]
        cur = out[epoch][0].shape[0]
        if n > cur:                       # same width reconciliation as load()
            out[epoch][0] = _pad_to(out[epoch][0], n)
            out[epoch][1] = _pad_to(out[epoch][1], n)
        elif n < cur:
            g, w = _pad_to(g, cur), _pad_to(w, cur)
        out[epoch][0] += g
        out[epoch][1] += w
        out[epoch][2] += eps
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-games", type=int, default=100,
                    help="ignore decks with fewer games than this")
    ap.add_argument("--res-warn", type=float, default=0.9,
                    help="flag decks whose resolved fraction is below this")
    ap.add_argument("--pair", default=None,
                    help="substring of a deck name: show its best/worst matchups")
    ap.add_argument("--pool", default="decks/pool")
    ap.add_argument("--alt-pool", default=None,
                    help="coverage pool, if the run used one; needed to name "
                         "the extra rows/cols a two-pool blob produces")
    ap.add_argument("--epoch-min", type=int, default=None)
    ap.add_argument("--epoch-max", type=int, default=None)
    ap.add_argument("--by-epoch", default=None,
                    help="substring of a deck name: print its win rate per "
                         "epoch bucket, to see how a deck moves as training "
                         "progresses")
    a = ap.parse_args()

    if a.by_epoch:
        names = deck_names(a.pool)
        hits = [i for i, n in enumerate(names) if a.by_epoch.lower() in n.lower()]
        if not hits:
            sys.exit(f"no deck matching {a.by_epoch!r}")
        i = hits[0]
        print(f"== {names[i]} over training ==")
        print(f"{'epoch':>8} {'win rate':>9} {'games':>8} {'+-95%':>7} {'episodes':>10}")
        for epoch, (G, W, eps) in load_by_epoch(a.paths).items():
            played = int(G[i].sum() + G[:, i].sum())
            won = int(W[i].sum() + (G[:, i].sum() - W[:, i].sum()))
            if played < a.min_games:
                continue
            wr = won / played
            ci = 1.96 * (wr * (1 - wr) / played) ** 0.5
            print(f"{epoch:>8} {wr:9.3f} {played:8d} {ci:7.3f} {eps:10,}")
        return

    G, W, eps, nfiles = load(a.paths, a.epoch_min, a.epoch_max)
    names = deck_names(a.pool)
    # Matrices are pre-sized to a constant width, so being wider than the pool
    # is normal. fit_to_names trims to the real decks and KEEPS their names;
    # it only errors if games were recorded outside the listing's range.
    G, W, names = fit_to_names(G, W, names, src=f"pool {a.pool!r}")

    # Row marginal = that deck's record when the learner piloted it.
    played, won = G.sum(1), W.sum(1)
    # Column marginal = games where it was the OPPONENT deck; the learner's
    # losses there are that deck's wins, so add them for a two-sided record.
    played_tot = played + G.sum(0)
    won_tot = won + (G.sum(0) - W.sum(0))

    print(f"{nfiles} file(s), {eps:,} episodes, {int(G.sum()):,} recorded games")
    ok = played_tot >= a.min_games
    print(f"{int(ok.sum())}/{len(names)} decks with >= {a.min_games} games\n")

    # `res` = resolved fraction. Decks are drawn UNIFORMLY, so every deck
    # should land 2*episodes/n games; only DECIDED games reach the matrix
    # (env.c finish_episode records nothing for a draw or a max_engine_steps
    # truncation). A deck below 1.0 therefore stalls out that often, and its
    # win rate is conditional on the game resolving -- not comparable to a
    # deck at 1.0, and the +-95% column does not cover that gap. Stall/heal
    # archetypes are the ones that show up here.
    expected = 2.0 * eps / len(names)
    res = played_tot / expected if expected else np.zeros(len(names))

    wr = np.where(played_tot > 0, won_tot / np.maximum(played_tot, 1), np.nan)
    order = np.argsort(np.where(ok, wr, -1))[::-1]
    print(f"{'rank':>4} {'deck':44s} {'win rate':>9} {'games':>8} {'+-95%':>7} {'res':>6}")
    for r, i in enumerate(order[: a.top], 1):
        if not ok[i]:
            break
        ci = 1.96 * (wr[i] * (1 - wr[i]) / played_tot[i]) ** 0.5
        mark = " *" if res[i] < a.res_warn else ""
        print(f"{r:>4} {names[i][:44]:44s} {wr[i]:9.3f} {played_tot[i]:8d} "
              f"{ci:7.3f} {res[i]:6.2f}{mark}")

    low = np.where(ok & (res < a.res_warn))[0]
    if len(low):
        print(f"\n* {len(low)} deck(s) resolved < {a.res_warn:.0%} of their drawn "
              f"games -- win rate is conditional on the game ENDING, so it is "
              f"NOT comparable to the rest of the field:")
        for i in low[np.argsort(res[low])]:
            print(f"    res {res[i]:.2f}  wr {wr[i]:.3f}  {names[i][:52]}")

    if a.pair:
        hits = [i for i, n in enumerate(names) if a.pair.lower() in n.lower()]
        if not hits:
            sys.exit(f"no deck matching {a.pair!r}")
        i = hits[0]
        print(f"\n== matchups for {names[i]} (as learner) ==")
        g, w = G[i], W[i]
        m = g >= max(5, a.min_games // 20)
        if not m.any():
            print("  (not enough games yet)")
            return
        rate = np.where(m, w / np.maximum(g, 1), np.nan)
        idx = np.argsort(np.where(m, rate, np.nan))
        idx = [j for j in idx if m[j]]
        for label, sel in (("worst", idx[:8]), ("best", idx[-8:][::-1])):
            print(f"  {label}:")
            for j in sel:
                print(f"    {rate[j]:5.2f}  ({int(w[j])}/{int(g[j])})  vs {names[j][:48]}")


if __name__ == "__main__":
    main()
