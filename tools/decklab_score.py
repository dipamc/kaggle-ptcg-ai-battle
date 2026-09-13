#!/usr/bin/env python3
"""Score inserted decklab decks against their parents at training volume.

    tools/decklab_score.py --since 2500 [--matrix <key>]

For every staged proposal (meta.staged_as set), read the training deck matrix from
the given epoch window and compare child vs parent overall wr — the training-
volume readout of "how did the inserted deck do". Flags:

  * ANOMALY: child wr < 0.15 over >= 2000 games
    (broken interaction or unpilotable); consider zeroing its weight.
  * The comparison is vs the same field the parent sees, so parent-child
    delta is directly interpretable; CIs are binomial.

Appends an event per scored deck to the ledger.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, required=True,
                    help="epoch the append landed (trainer log line)")
    ap.add_argument("--matrix", default=None,
                    help="matrix key from decklab/config.json (default: default_matrix)")
    a = ap.parse_args()
    G, W, names, eps = C.load_matrix(a.matrix, epoch_min=a.since)
    idx = {n: i for i, n in enumerate(names)}
    wr, games = C.per_deck_wr(G, W)

    rows = []
    for mp in sorted(glob.glob(os.path.join(C.PROPOSALS, "*", "meta.json"))):
        meta = json.load(open(mp))
        child = meta.get("staged_as")
        if not child or child not in idx:
            continue
        parent = os.path.basename(meta["base_deck"])[:-4]
        ci, pi = idx[child], idx.get(parent)
        cw, cn = wr[ci], int(games[ci])
        se_c = float(np.sqrt(max(cw * (1 - cw), 1e-9) / max(cn, 1)))
        line = {"id": meta["id"], "event": "scored", "epoch_min": a.since,
                "child": child, "child_wr": round(float(cw), 4), "child_games": cn}
        msg = f"{child[:52]:52s} wr {cw:.3f} ({cn}g)"
        if pi is not None:
            pw, pn = wr[pi], int(games[pi])
            d = float(cw - pw)
            se = float(np.sqrt(cw * (1 - cw) / max(cn, 1) + pw * (1 - pw) / max(pn, 1)))
            line.update({"parent": parent, "parent_wr": round(float(pw), 4),
                         "delta": round(d, 4), "se": round(se, 4)})
            msg += f"  vs parent {pw:.3f}  delta {d:+.3f} +-{2*se:.3f}"
        if cn >= 2000 and cw < 0.15:
            line["anomaly"] = True
            msg += "  *** ANOMALY (near-zero win rate) — consider zero weight"
        rows.append(line)
        print(msg)
    if not rows:
        print("no staged decks found in the matrix window — check --since and "
              "whether the append actually landed")
    for line in rows:
        C.ledger_append(line)


if __name__ == "__main__":
    main()
