#!/usr/bin/env python3
"""Deck-id-ordered listing of a deck pool.

    python3 tools/pool_names.py [--pool decks/pool] [--alt-pool decks/pool_alt]
                                [--write data/pool_names.txt]

Deck id = position in the sorted glob of 60-line .csv files in the pool dir,
which is the construction export_tables.py, train_native.py and
tools/deck_matrix.py all share. Coverage decks (--alt-pool) occupy ids from
DECK_ALT_BASE (1024) upward; they are printed after the training decks.

The listing is the "pinned names" file that `export_tables.py --append-to`
and `validate_deck_batch.py --pinned` check new decks against, and the
`--names` file that `event_log_deckmat.py` labels ids with.
"""
import argparse
import glob
import os
import sys

DECK_ALT_BASE = 1024


def pool_names(pool):
    names = []
    for fn in sorted(glob.glob(os.path.join(pool, "*.csv"))):
        try:
            ids = [int(x) for x in open(fn).read().split() if x.strip()]
        except ValueError:
            continue
        if len(ids) == 60:
            names.append(os.path.basename(fn)[:-4])
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="decks/pool")
    ap.add_argument("--alt-pool", default=None)
    ap.add_argument("--write", default=None, help="write the training-pool listing here")
    a = ap.parse_args()
    names = pool_names(a.pool)
    if not names:
        sys.exit(f"no 60-line decks in {a.pool}")
    if a.write:
        with open(a.write, "w") as f:
            f.write("\n".join(names) + "\n")
        print(f"wrote {a.write}: {len(names)} decks")
    else:
        for i, n in enumerate(names):
            print(f"{i}\t{n}")
    if a.alt_pool:
        alt = pool_names(a.alt_pool)
        if not a.write:
            for i, n in enumerate(alt):
                print(f"{DECK_ALT_BASE + i}\t{n}")
        print(f"# {len(names)} training decks (ids 0..{len(names) - 1}), "
              f"{len(alt)} coverage decks (ids {DECK_ALT_BASE}..{DECK_ALT_BASE + len(alt) - 1})",
              file=sys.stderr)


if __name__ == "__main__":
    main()
