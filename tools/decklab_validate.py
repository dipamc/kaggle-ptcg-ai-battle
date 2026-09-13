#!/usr/bin/env python3
"""Validate a decklab proposal directory (pre-test gate).

    tools/decklab_validate.py decklab/proposals/dlab_0001_...

A proposal dir holds:
    deck.csv    60 card ids, one per line (same format as decks/pool)
    meta.json   the PRE-REGISTRATION: what changed, why, what is predicted,
                and the exact targeted gauntlet — written BEFORE any games run
                (the multiple-testing guard, see docs/decklab.md).

meta.json schema:
{
  "id": "dlab_0001_hydrapple_ld03_2judge",
  "base_deck": "decks/pool/....csv",          # must exist
  "rationale": "why these swaps (cites evidence pack lines)",
  "target_opponents": ["decks/pool/a.csv", ...],   # 3-6, pre-committed
  "predicted": {"targeted": "+", "field": "0"},  # direction of expected effect
  "status": "proposed"
}

Checks (fail): 60 cards; ids exist; <=4 copies per card NAME except Basic
Energy; <=1 ACE SPEC; swaps vs base <= max_swaps (config); not identical to
any pool deck or previous proposal; target opponents exist and are pool decks;
meta fields present. Checks (warn only): evolution line without its base —
warning not failure, because that rule miscalls real decks (Rare Candy
lines, attack selectors).
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C

REQUIRED = ["id", "base_deck", "rationale", "target_opponents", "predicted", "status"]


def validate(pdir, quiet=False):
    errs, warns = [], []
    deck_path = os.path.join(pdir, "deck.csv")
    meta_path = os.path.join(pdir, "meta.json")
    if not os.path.exists(deck_path):
        return [f"missing {deck_path}"], []
    if not os.path.exists(meta_path):
        return [f"missing {meta_path}"], []
    try:
        deck = C.read_deck(deck_path)
    except ValueError as e:
        return [str(e)], []
    meta = json.load(open(meta_path))
    for k in REQUIRED:
        if k not in meta:
            errs.append(f"meta.json missing field '{k}'")
    cards, attacks = C.load_carddb()

    for cid in deck:
        if cid not in cards:
            errs.append(f"unknown card id {cid}")
    by_name = defaultdict(int)
    ace = 0
    for cid, n in deck.items():
        if cid not in cards:
            continue
        c = cards[cid]
        if not C.is_basic_energy(cid, cards):
            by_name[c["name"]] += n
        if c.get("aceSpec"):
            ace += n
    for nm, n in by_name.items():
        if n > 4:
            errs.append(f"{n} copies of '{nm}' (max 4)")
    if ace > 1:
        errs.append(f"{ace} ACE SPEC cards (max 1)")

    cfgd = C.cfg()
    base_rel = meta.get("base_deck", "")
    base_path = os.path.join(C.REPO, base_rel)
    if not os.path.exists(base_path):
        errs.append(f"base_deck not found: {base_rel}")
    else:
        base = C.read_deck(base_path)
        out, inn, k = C.deck_diff(base, deck)
        if k == 0:
            errs.append("identical to base deck")
        if k > cfgd.get("max_swaps", 3):
            errs.append(f"{k} swaps from base (max {cfgd.get('max_swaps', 3)}) — "
                        "not looking for radically different decks")
        meta["_swaps_measured"] = {
            "n": k,
            "out": {str(c): {"name": cards[c]["name"], "n": n} for c, n in out.items()},
            "in": {str(c): {"name": cards[c]["name"], "n": n} for c, n in inn.items()},
        }

    names_pool, counters = C.load_pool()
    for nm in names_pool:
        if counters[nm] == deck:
            errs.append(f"identical to pool deck {nm}")
            break
    for other in glob.glob(os.path.join(C.PROPOSALS, "*", "deck.csv")):
        if os.path.abspath(other) == os.path.abspath(deck_path):
            continue
        om = os.path.join(os.path.dirname(other), "meta.json")
        try:
            if os.path.exists(om) and json.load(open(om)).get("status") in (
                    "superseded", "dropped", "withdrawn"):
                continue          # retired lines don't block their successors
            if C.read_deck(other) == deck:
                errs.append(f"identical to prior proposal {os.path.dirname(other)}")
                break
        except ValueError:
            pass

    tg = meta.get("target_opponents", [])
    if not (3 <= len(tg) <= 6):
        errs.append(f"target_opponents must be 3-6 pre-committed decks (got {len(tg)})")
    for t in tg:
        if not os.path.exists(os.path.join(C.REPO, t)):
            errs.append(f"target opponent not found: {t}")

    have_names = {cards[c]["name"] for c in deck if c in cards}
    for cid, n in deck.items():
        c = cards.get(cid, {})
        ef = c.get("evolvesFrom")
        if ef and ef not in have_names:
            warns.append(f"'{c['name']}' evolves from '{ef}' which is not in the deck "
                         "(may be intentional: Rare Candy lines, attack selectors)")

    # OOV vs the evaluator's training vocabulary: the A/B carries no
    # information about cards the evaluator never piloted, so such proposals
    # switch to the reasoning evidence class (review gates the logic and
    # decklab_ab.py refuses to game-gate them).
    vocab = C.evaluator_vocab()
    oov = sorted(cid for cid in deck if cid in cards and cid not in vocab)
    if oov:
        meta["oov_cards"] = {str(c): cards[c]["name"] for c in oov}
        meta["evidence_class"] = "reasoning"
        warns.append("evaluator-OOV cards (reasoning class, A/B is a floor check "
                     "only): " + ", ".join(f"[{c}] {cards[c]['name']}" for c in oov))
    else:
        meta.setdefault("evidence_class", "measured")

    if not errs:
        meta["status"] = "validated" if meta.get("status") == "proposed" else meta.get("status")
        json.dump(meta, open(meta_path, "w"), indent=1)
    if not quiet:
        for e in errs:
            print(f"  FAIL {e}")
        for w in warns:
            print(f"  warn {w}")
        print(f"{os.path.basename(pdir)}: "
              f"{'OK' if not errs else 'REJECTED'} ({len(warns)} warnings)")
    return errs, warns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposal_dirs", nargs="+")
    a = ap.parse_args()
    bad = 0
    for p in a.proposal_dirs:
        errs, _ = validate(p.rstrip("/"))
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
