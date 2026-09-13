#!/usr/bin/env python3
"""Pre-flight a batch of decklists before it is exported into an append blob.

    python3 tools/validate_deck_batch.py <batch_dir> [--pool decks/pool]
                                         [--pinned <listing>]
                                         [--cap 1024]

Everything here is a check that catches a real failure, or that
`export_tables.py` performs SILENTLY (in which case the deck simply vanishes and
the count comes up short):

  * exactly 60 non-empty lines -- anything else is dropped without a word
  * ids parse, and exist in the card DB when it can be imported
  * filename sorts after every pinned name -- deck id is the position in the
    sorted 60-line glob, so a name landing mid-list renumbers everything after
    it and pt_reload_decks rejects the blob
  * not already in the pool (an identical list is a wasted id, and ids are capped)
  * pool + batch stays under the training-id cap

Exit code is nonzero if any check fails, so it can gate a build.
See docs/deck-pool.md.
"""
import argparse
import glob
import hashlib
import os
import sys


def lines60(path):
    return [l.strip() for l in open(path) if l.strip()]


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    ap.add_argument("--pool", default="decks/pool")
    ap.add_argument("--pinned", default=None,
                    help="pinned deck listing (tools/pool_names.py --write); "
                         "default: the live sorted glob of --pool")
    ap.add_argument("--cap", type=int, default=1024)
    a = ap.parse_args()

    fails, warns = [], []
    all_csv = sorted(glob.glob(os.path.join(a.batch_dir, "*.csv")))
    # Leading-underscore files are METADATA, not decklists: `decks/pool` uses
    # the same convention for its `_manifest.csv`. They sort first and are
    # dropped by the 60-line filter anyway, so they never take a deck id --
    # but they must not be copied into the pool or counted toward the cap.
    batch = [f for f in all_csv if not os.path.basename(f).startswith("_")]
    meta = [f for f in all_csv if os.path.basename(f).startswith("_")]
    if not batch:
        sys.exit(f"no decklist .csv in {a.batch_dir}")
    for m in meta:
        print(f"note: treating {os.path.basename(m)} as metadata, not a decklist "
              f"(leading underscore) — do NOT copy it into the pool")

    if a.pinned:
        pinned = [l.strip() for l in open(a.pinned) if l.strip()]
    else:
        print(f"note: no --pinned listing given; using the live sorted glob of {a.pool}")
        pinned = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(a.pool, "*.csv"))
                        if len(lines60(f)) == 60)
    if not pinned:
        sys.exit("empty pinned listing / pool: nothing to append to")
    last_pinned = pinned[-1]

    # existing pool contents, by md5, for duplicate detection
    pool_md5 = {}
    for f in glob.glob(os.path.join(a.pool, "*.csv")):
        if len(lines60(f)) == 60:
            pool_md5.setdefault(md5(f), os.path.basename(f)[:-4])

    cards = None
    try:                                   # best effort: needs the cg lib
        sys.path.insert(0, "data")
        from cg.api import all_card_data
        cards = {c.cardId for c in all_card_data()}
    except Exception as e:
        warns.append(f"card DB not importable here ({type(e).__name__}); "
                     f"skipped the card-id existence check")

    print(f"batch: {len(batch)} files from {a.batch_dir}")
    print(f"pinned listing: {len(pinned)} decks, last = {last_pinned}")
    print(f"{'deck':<52}{'lines':>6}{'sorts':>7}  md5")
    seen = {}
    for f in batch:
        name = os.path.basename(f)[:-4]
        ls = lines60(f)
        h = md5(f)
        ok_len = len(ls) == 60
        ok_sort = name > last_pinned
        print(f"{name:<52}{len(ls):>6}{'OK' if ok_sort else 'BAD':>7}  {h[:12]}")

        if not ok_len:
            fails.append(f"{name}: {len(ls)} non-empty lines, must be exactly 60 "
                         f"(export_tables drops it silently)")
        bad = [x for x in ls if not x.lstrip('-').isdigit()]
        if bad:
            fails.append(f"{name}: {len(bad)} line(s) are not integers, e.g. {bad[0]!r}")
        elif cards is not None:
            missing = sorted({int(x) for x in ls} - cards)
            if missing:
                fails.append(f"{name}: card ids not in the DB: {missing[:5]}")
        if not ok_sort:
            fails.append(f"{name}: sorts BEFORE '{last_pinned}' -- this inserts "
                         f"mid-list and renumbers existing deck ids; rename it "
                         f"so it sorts last (see the zz<BB>_ scheme)")
        if h in pool_md5:
            fails.append(f"{name}: identical to pool deck '{pool_md5[h]}' (md5 {h[:8]})")
        if h in seen:
            fails.append(f"{name}: identical to '{seen[h]}' in this same batch")
        seen[h] = name

    # batch must also be internally ordered and land after the pin
    names = [os.path.basename(f)[:-4] for f in batch]
    if names != sorted(names):
        fails.append("batch filenames are not in sorted order relative to each other")

    total = len(pinned) + len(batch)
    print(f"\npool after append: {len(pinned)} + {len(batch)} = {total} "
          f"(cap {a.cap}, headroom {a.cap - total})")
    if total > a.cap:
        fails.append(f"{total} exceeds the {a.cap} training-id cap -- coverage sits "
                     f"at that fixed base, so going past it is a rebuild + stop")

    for w in warns:
        print(f"WARN  {w}")
    if fails:
        print(f"\nFAILED ({len(fails)}):")
        for x in fails:
            print(f"  - {x}")
        sys.exit(1)
    print("\nALL CHECKS PASSED — safe to export with --append-to")


if __name__ == "__main__":
    main()
