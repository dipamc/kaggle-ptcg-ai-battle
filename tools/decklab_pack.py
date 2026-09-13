#!/usr/bin/env python3
"""Build LLM-readable evidence packs for decklab.

    tools/decklab_pack.py field                      -> decklab/packs/field.md
    tools/decklab_pack.py swapstats                  -> decklab/packs/swapstats.md
    tools/decklab_pack.py deck --name <pool-deck>    -> decklab/packs/deck_<name>.md
    tools/decklab_pack.py cards --deck <csv> [--extra 12,34]
                                                     -> decklab/packs/cards_<name>.md

`deck` is the main dossier: composition, ladder/implied standing, matchup
profile by data-driven cluster (with how much each cluster matters on the
measured field), the close cohort (nearest pool decks and what they run
differently), and the relevant single-swap natural experiments.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C


def _field(cardsdb=None):
    fc = C.field_cache()
    tot = max(fc["total_sides"], 1)
    rows = sorted(fc["decks"].items(), key=lambda kv: -kv[1]["sides"])
    return fc, tot, rows


def cmd_field(a):
    fc, tot, rows = _field()
    imp = C.implied_table()
    lines = [f"# Ladder field ({fc['window']}, {tot} sides, built {fc['built']})", "",
             "| # | deck | field share | its ladder wr (n) | implied rank |",
             "|---|---|---|---|---|"]
    for k, (d, e) in enumerate(rows[:a.top]):
        lw = f"{e['lad_w']/e['lad_n']:.3f} ({e['lad_n']})" if e["lad_n"] >= 30 else "-"
        ir = imp.get(d)
        irs = f"#{ir[0]} ({ir[1]:.3f})" + (" UNRANKABLE" if ir and ir[2] else "") if ir else "-"
        lines.append(f"| {k+1} | {d} | {e['sides']/tot:.3f} | {lw} | {irs} |")
    _write("field.md", lines)


def cmd_swapstats(a):
    fc, tot, rows = _field()
    _, counters = C.load_pool()
    cards, attacks = C.load_carddb()
    use = defaultdict(lambda: [0.0, 0, 0])          # card -> [field-share mass, decks, copies]
    for d, e in rows:
        if d not in counters:
            continue
        sh = e["sides"] / tot
        for cid, n in counters[d].items():
            u = use[cid]
            u[0] += sh
            u[1] += 1
            u[2] += n
    lines = ["# Card usage across the field (weighted by field share)", "",
             "| card | field mass | decks | avg copies |", "|---|---|---|---|"]
    for cid, (mass, nd, cp) in sorted(use.items(), key=lambda kv: -kv[1][0]):
        if mass < 0.001:
            continue
        lines.append(f"| [{cid}] {cards[cid]['name']} | {mass:.3f} | {nd} | {cp/nd:.1f} |")
    _write("swapstats.md", lines)


def cmd_cards(a):
    cards, attacks = C.load_carddb()
    ids = sorted(C.read_deck(a.deck).items()) if a.deck else []
    extra = [int(x) for x in a.extra.split(",")] if a.extra else []
    name = os.path.basename(a.deck)[:-4] if a.deck else "extra"
    lines = [f"# Card texts: {name}", ""]
    for cid, n in ids:
        lines.append(f"## x{n} {C.card_text(cid, cards, attacks)}")
        lines.append("")
    for cid in extra:
        lines.append(f"## {C.card_text(cid, cards, attacks)}")
        lines.append("")
    _write(f"cards_{name}.md", lines)


def cmd_deck(a):
    target = a.name
    names_pool, counters = C.load_pool()
    if target not in counters:
        sys.exit(f"{target} not in decks/pool")
    cards, attacks = C.load_carddb()
    fc, tot, rows = _field()
    imp = C.implied_table()
    fe = fc["decks"].get(target, {"sides": 0, "lad_n": 0, "lad_w": 0})

    which = a.matrix or C.default_matrix()
    G, W, names, eps = C.load_matrix(which)
    idx = {n: i for i, n in enumerate(names)}
    if target not in idx:
        sys.exit(f"{target} not in {which} matrix listing")
    ti = idx[target]
    cls, P, mu = C.clusters_for(G, W)
    wr_all, games_all = C.per_deck_wr(G, W)

    lines = [f"# Deck dossier: {target}", "",
             f"Matrix: {which} ({eps} episodes). Deckmat wr {wr_all[ti]:.3f} over "
             f"{int(games_all[ti])} games. Field share {fe['sides']/tot:.4f} "
             f"({fe['sides']} sides)."]
    if fe["lad_n"] >= 30:
        lines.append(f"Ladder record with this EXACT list: "
                     f"{fe['lad_w']/fe['lad_n']:.3f} over {fe['lad_n']} sides.")
    ir = imp.get(target)
    if ir:
        note = f" — {ir[2]}" if ir[2] else ""
        lines.append(f"Field-implied wr {ir[1]:.3f}, rank #{ir[0]}/{len(imp)}{note}.")
    lines += ["", "## Composition", ""]
    for cid, n in sorted(counters[target].items(),
                         key=lambda kv: (cards[kv[0]].get("cardType") or "", kv[0])):
        c = cards[cid]
        lines.append(f"- x{n} [{cid}] {c['name']} ({c.get('stageText') or c.get('cardType')})")

    # matchup profile by cluster, with field relevance
    lines += ["", "## Matchup profile (data-driven clusters, worst first)", "",
              "| wr | games | cluster (2 sample members) | cluster field share |",
              "|---|---|---|---|"]
    prof = []
    for mem in cls:
        if len(mem) < 3:
            continue
        n = w = 0
        for j in mem:
            if j == ti:
                continue
            nn, ww = C.pair_cell(G, W, ti, j)
            n += nn
            w += ww
        if n < 100:
            continue
        fsh = sum(fc["decks"].get(names[j], {}).get("sides", 0) for j in mem) / tot
        lab = " + ".join(sorted(names[j] for j in mem)[:2])
        prof.append((w / n, n, lab, fsh, mem))
    for wr, n, lab, fsh, _ in sorted(prof):
        lines.append(f"| {wr:.3f} | {n} | {lab[:70]} | {fsh:.3f} |")

    # suggested targeted gauntlet: worst cluster with real field share
    tg = [p for p in sorted(prof) if p[3] >= 0.005]
    if tg:
        wr, n, lab, fsh, mem = tg[0]
        opp = sorted(mem, key=lambda j: -fc["decks"].get(names[j], {}).get("sides", 0))[:5]
        lines += ["", "## Suggested targeted gauntlet (worst field-relevant cluster)", ""]
        for j in opp:
            e = fc["decks"].get(names[j], {"sides": 0})
            nn, ww = C.pair_cell(G, W, ti, j)
            lines.append(f"- decks/pool/{names[j]}.csv  (our wr vs it "
                         f"{ww/max(nn,1):.3f} over {nn}; field {e['sides']/tot:.4f})")

    # close cohort: nearest pool decks + what they run differently
    lines += ["", "## Close cohort (nearest pool decks by card distance)", ""]
    dists = []
    for nm in names_pool:
        if nm == target:
            continue
        _, _, k = C.deck_diff(counters[target], counters[nm])
        if k <= a.cohort_dist:
            dists.append((k, nm))
    for k, nm in sorted(dists)[:12]:
        out, inn, _ = C.deck_diff(counters[target], counters[nm])
        ir2 = imp.get(nm)
        irs = f"implied {ir2[1]:.3f} #{ir2[0]}" if ir2 else "no implied"
        e = fc["decks"].get(nm, {"sides": 0, "lad_n": 0, "lad_w": 0})
        lw = f", ladder {e['lad_w']/e['lad_n']:.3f}({e['lad_n']})" if e["lad_n"] >= 30 else ""
        lines.append(f"- d={k} {nm} ({irs}{lw})")
        lines.append(f"    they lack: " + ("; ".join(f"{cards[c]['name']} x{n}"
                     for c, n in sorted(out.items())) or "-"))
        lines.append(f"    they run:  " + ("; ".join(f"{cards[c]['name']} x{n}"
                     for c, n in sorted(inn.items())) or "-"))

    # single-swap experiments touching this deck or cohort
    pcsv = os.path.join(C.PACKS, f"pairs_{which}.csv")
    if os.path.exists(pcsv):
        cohort = {nm for _, nm in dists} | {target}
        rel = [r for r in csv.DictReader(open(pcsv))
               if r["deck_a"] in cohort or r["deck_b"] in cohort]
        sig = [r for r in rel if r["cls"] != "ALL" and abs(float(r["z"])) >= 2.5
               and r["n_swaps"] == "1"]
        if sig:
            lines += ["", "## Nearby single-swap natural experiments (|z|>=2.5)", ""]
            for r in sorted(sig, key=lambda r: -abs(float(r["delta"])))[:15]:
                lines.append(f"- {r['delta']} (z {r['z']}) [{r['cards_a_only']}] vs "
                             f"[{r['cards_b_only']}] @ {r['cls'][:55]}")
                lines.append(f"    {r['deck_a']} vs {r['deck_b']}")

    try:
        import dlab
        deck_notes = dlab.notes_for("deck", target)
        card_notes = [(cards[cid]["name"], n) for cid in counters[target]
                      for n in dlab.notes_for("card", cards[cid]["name"])]
        if deck_notes or card_notes:
            lines += ["", "## Lab notes (decklab/notes.jsonl — read these)", ""]
            for n in deck_notes:
                lines.append(f"- [{n['ts']}] {n['note']}")
            for nm, n in card_notes:
                lines.append(f"- [{n['ts']}] {nm}: {n['note']}")
    except Exception as e:
        lines += ["", f"(lab notes unavailable: {e})"]

    lines += ["", "## Caveats", "",
              "- All deck-matrix numbers are the training policy on both seats: a deck "
              "the policy pilots badly reads falsely low.",
              "- Deltas here are correlational except the single-swap pairs, which "
              "are near-causal at training volume.",
              f"- Field: {fc['window']}."]
    _write(f"deck_{target}.md", lines)


def _write(name, lines):
    os.makedirs(C.PACKS, exist_ok=True)
    p = os.path.join(C.PACKS, name)
    open(p, "w").write("\n".join(lines) + "\n")
    print(f"wrote {p} ({len(lines)} lines)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("field")
    f.add_argument("--top", type=int, default=40)
    sub.add_parser("swapstats")
    c = sub.add_parser("cards")
    c.add_argument("--deck", default=None)
    c.add_argument("--extra", default=None)
    d = sub.add_parser("deck")
    d.add_argument("--name", required=True)
    d.add_argument("--matrix", default=None,
                   help="matrix key from decklab/config.json (default: default_matrix)")
    d.add_argument("--cohort-dist", type=int, default=10)
    a = ap.parse_args()
    {"field": cmd_field, "swapstats": cmd_swapstats,
     "cards": cmd_cards, "deck": cmd_deck}[a.cmd](a)


if __name__ == "__main__":
    main()
