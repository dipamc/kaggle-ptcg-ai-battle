#!/usr/bin/env python3
"""Deck-vs-deck matchup coverage and win rates, straight from the event log.

The C env can write a deck matrix of its own (PTCG_DECK_MATRIX), but the event
log already carries deck_learner / deck_opponent / result in every episode
header, so a dataset dumped without that flag is not missing anything -- it
just needs reading. Headers only: events and the fp16 payload are seeked past,
so this runs over tens of GB in seconds.

Deck ids are positions in the sorted-glob pool the tables blob was built from
(docs/training.md). Pass --names to label them.

Usage:
  python3 tools/event_log_deckmat.py <dir-or-file> [--names pool_names.txt]
                                     [--top N] [--csv out.csv]
"""
import argparse
import collections
import glob
import os
import struct
import sys

MAGIC = 0x50544556


def iter_headers(path):
    """Yield episode headers, skipping event and payload bytes."""
    b = open(path, "rb").read()
    magic, ver, recsz, epoch = struct.unpack_from("<4I", b, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: bad magic {magic:#x}")
    ehdr = 16 if ver == 1 else 24
    off, n = 16, len(b)
    while off + ehdr <= n:
        ep, dl, dop, seat, res, ok, first, nev, nseen = struct.unpack_from(
            "<IHHbbBBHH", b, off)
        pbytes = struct.unpack_from("<I", b, off + 16)[0] if ver == 2 else 0
        nxt = off + ehdr + nev * recsz + pbytes
        if nxt > n:
            break                       # partial tail of a .tmp
        yield dl, dop, seat, res, first
        off = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--names", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    paths = ([args.path] if os.path.isfile(args.path)
             else sorted(glob.glob(os.path.join(args.path, "*.bin"))
                         + glob.glob(os.path.join(args.path, "*.tmp"))))
    if not paths:
        raise SystemExit(f"no PTEV files under {args.path}")

    names = []
    if args.names:
        names = [l.strip() for l in open(args.names) if l.strip()]

    # ordered (learner, opponent) -> games; and learner wins
    games = collections.Counter()
    wins = collections.Counter()
    per_deck = collections.Counter()
    per_deck_w = collections.Counter()
    results = collections.Counter()
    first_win = [0, 0]
    n = 0
    for p in paths:
        for dl, dop, seat, res, first in iter_headers(p):
            n += 1
            results[res] += 1
            games[(dl, dop)] += 1
            per_deck[dl] += 1
            per_deck[dop] += 1
            if res in (0, 1):
                lw = (res == seat)
                wins[(dl, dop)] += lw
                per_deck_w[dl] += lw
                per_deck_w[dop] += (not lw)
                if first in (0, 1):
                    first_win[res == first] += 1

    n_decks = (max(max(a, b) for a, b in games) + 1) if games else 0
    if names and len(names) != n_decks:
        print(f"WARNING: --names has {len(names)} entries but the log uses "
              f"{n_decks} deck ids", file=sys.stderr)
    cells = n_decks * n_decks

    # unordered pair coverage: which deck is "learner" is just a seat label
    unord = collections.Counter()
    for (a, b), c in games.items():
        unord[(min(a, b), max(a, b))] += c
    n_unord_cells = n_decks * (n_decks + 1) // 2

    print(f"episodes        {n}")
    print(f"decks           {n_decks}")
    print(f"results         {dict(sorted(results.items()))}  "
          f"(0/1 = seat won, 2 = draw, -2 = truncated)")
    if sum(first_win):
        print(f"first-player wr {first_win[1] / sum(first_win):.4f}"
              f"   ({sum(first_win)} decided games)")
    print()
    print(f"ordered   (learner,opp) cells covered  {len(games)}/{cells} "
          f"({100.0*len(games)/max(1,cells):.1f}%)")
    print(f"unordered  matchup      cells covered  {len(unord)}/{n_unord_cells} "
          f"({100.0*len(unord)/max(1,n_unord_cells):.1f}%)")

    if unord:
        v = sorted(unord.values())
        exp = n / n_unord_cells
        print(f"\ngames per unordered matchup: min {v[0]}  p05 {v[len(v)//20]}  "
              f"median {v[len(v)//2]}  p95 {v[-max(1,len(v)//20)]}  max {v[-1]}")
        print(f"  expected under a uniform draw: {exp:.1f}  "
              f"(and {n_unord_cells - len(unord)} cells still have 0)")

    def label(i):
        return names[i] if i < len(names) else f"deck_{i}"

    print(f"\nfewest games (deck totals, {args.top}):")
    for i, c in sorted(per_deck.items(), key=lambda kv: kv[1])[:args.top]:
        wr = per_deck_w[i] / c if c else 0
        print(f"   {label(i):<44} {c:>7} games  wr {wr:.3f}")
    print(f"\nmost games:")
    for i, c in per_deck.most_common(3):
        wr = per_deck_w[i] / c if c else 0
        print(f"   {label(i):<44} {c:>7} games  wr {wr:.3f}")

    print(f"\nstrongest decks by win rate (>= {max(50, n // 2000)} games):")
    floor = max(50, n // 2000)
    rank = [(per_deck_w[i] / c, c, i) for i, c in per_deck.items() if c >= floor]
    for wr, c, i in sorted(rank, reverse=True)[:args.top]:
        print(f"   {label(i):<44} wr {wr:.3f}  ({c} games)")
    print(f"weakest:")
    for wr, c, i in sorted(rank)[:args.top]:
        print(f"   {label(i):<44} wr {wr:.3f}  ({c} games)")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("learner_deck,opponent_deck,games,learner_wins\n")
            for (a, b), c in sorted(games.items()):
                f.write(f"{label(a)},{label(b)},{c},{wins[(a,b)]}\n")
        print(f"\nwrote {args.csv} ({len(games)} rows)")


if __name__ == "__main__":
    main()
