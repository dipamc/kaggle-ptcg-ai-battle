#!/usr/bin/env python3
"""Stage ACCEPTED decklab proposals as an append batch for a running run.

    tools/decklab_batch.py [--batch 3] [--dry-run]

Collects every proposal with status "accepted" that is not yet staged, names
it zz0<batch>_<base-stem>__<proposal-id>.csv (sorts after every existing pool
name, so ids stay append-only), copies into decklab/staged/vNNN/ (or --out), runs
tools/validate_deck_batch.py over the result, and prints the runbook steps.

THE APPEND ITSELF IS DELIBERATELY MANUAL — it touches a running trainer. This
tool only prepares and validates the batch. Follow docs/deck-pool.md
(pool dir FIRST, then the request file; repoint PTCG_TABLES after).
New decks enter sampling at weight 1.0 (the mean) automatically.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=3,
                    help="zz prefix number for deck names")
    ap.add_argument("--out", default=None,
                    help="staging dir; default = next decklab/staged/vNNN "
                         "(decklab-owned, versioned — never the shared "
                         "data/append_batch_* dir, which other tooling "
                         "writes to)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.out:
        bdir = os.path.join(C.REPO, a.out)
    else:
        vs = [int(os.path.basename(d)[1:]) for d in
              glob.glob(os.path.join(C.DECKLAB, "staged", "v*")) if
              os.path.basename(d)[1:].isdigit()]
        bdir = os.path.join(C.DECKLAB, "staged", f"v{(max(vs) + 1) if vs else 1:03d}")

    staged = []
    for mp in sorted(glob.glob(os.path.join(C.PROPOSALS, "*", "meta.json"))):
        meta = json.load(open(mp))
        if meta.get("status") not in ("accepted", "accepted_reasoning",
                                      "accepted_user") \
                or meta.get("staged_as"):
            continue
        pdir = os.path.dirname(mp)
        base_stem = os.path.basename(meta["base_deck"])[:-4].split("__")[0]
        # a variant of a previously-appended deck carries its zzNN_ prefix in
        # the stem — strip it so the new name gets exactly one batch prefix
        base_stem = re.sub(r"^zz\d+_", "", base_stem)
        name = f"zz{a.batch:02d}_{base_stem}__{meta['id']}"
        staged.append((pdir, meta, mp, name))

    if not staged:
        print("no unstaged accepted proposals")
        return
    print(f"staging {len(staged)} accepted proposals into {bdir}:")
    for pdir, meta, mp, name in staged:
        print(f"  {meta['id']} -> {name}.csv")
    if a.dry_run:
        return
    os.makedirs(bdir, exist_ok=True)
    copied = []
    for pdir, meta, mp, name in staged:
        dst = os.path.join(bdir, name + ".csv")
        shutil.copy(os.path.join(pdir, "deck.csv"), dst)
        copied.append(dst)
    # validate BEFORE anything is stamped: a failed batch must leave the
    # proposals unstaged and the staging dir clean
    r = subprocess.run([sys.executable, os.path.join(C.REPO, "tools/validate_deck_batch.py"),
                        bdir], cwd=C.REPO)
    if r.returncode:
        for dst in copied:
            os.remove(dst)
        sys.exit("validate_deck_batch FAILED — nothing staged; fix before any append")
    for pdir, meta, mp, name in staged:
        meta["staged_as"] = name
        json.dump(meta, open(mp, "w"), indent=1)
        C.ledger_append({"id": meta["id"], "event": "staged",
                         "batch": a.batch, "as": name})
    print(f"\nBatch validated. Next steps are MANUAL (docs/deck-pool.md):")
    print(f"  1. copy {bdir}/*.csv into decks/pool/ and refresh the pinned listing")
    print(f"     (tools/pool_names.py --pool decks/pool --write data/pool_names.txt)")
    print(f"  2. sync decks/pool to the training machine, export the grown blob,")
    print(f"     publish the pool request (version N+1); verify the trainer log line")
    print(f"  3. repoint PTCG_TABLES at the new blob; archive weights+meta")
    print(f"  4. after ~300 epochs: tools/decklab_score.py --since <append epoch>")


if __name__ == "__main__":
    main()
