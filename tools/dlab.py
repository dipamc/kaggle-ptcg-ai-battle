#!/usr/bin/env python3
"""dlab — quick-query CLI for the decklab (decks, cards, notes, deckmat).

The interactive complement to the heavyweight evidence packs: one-shot
lookups that a proposal/review loop runs constantly.

    tools/dlab.py deck <name|path|dlab_NNNN> [--full]
    tools/dlab.py diff <a> <b>                     # multiset swap diff
    tools/dlab.py near <deck> [--k 8]              # closest pool decks
    tools/dlab.py card <query> [--brief]           # text + vocab + usage + notes
    tools/dlab.py find <query>                     # decks + cards matching
    tools/dlab.py note deck <deck> <text...>       # short note (<=240 chars)
    tools/dlab.py note card <card> <text...>
    tools/dlab.py notes [query]                    # browse notes
    tools/dlab.py history <query>                  # ledger + proposals mentioning it
    tools/dlab.py mat cell <a> <b> [--which <key>]  # one matchup, both seats
    tools/dlab.py mat row <deck> [--best] [--n 12] # worst/best matchups w/ field share
    tools/dlab.py mat rank <deck>                  # raw + implied rank
    tools/dlab.py mat check [--which <key>]         # matrix <-> decklist sync report

Deck args resolve against decks/pool stems (substring ok if unique), a
.csv path, or a proposal id (dlab_NNNN -> its deck.csv). Card args resolve
against carddb names. Notes live in decklab/notes.jsonl (append-only, short,
shown by every deck/card fetch).
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decklab_common as C

NOTES = os.path.join(C.DECKLAB, "notes.jsonl")
NOTE_MAX = 240
GROUPS = ["Pokemon", "Supporter", "Item", "Tool", "Stadium",
          "Special Energy", "Basic Energy"]


# ---------- notes ----------

def notes_read():
    out = []
    if os.path.exists(NOTES):
        for ln in open(NOTES):
            if ln.strip():
                out.append(json.loads(ln))
    return out


def notes_for(kind, key):
    return [n for n in notes_read() if n["kind"] == kind and n["key"] == key]


def notes_add(kind, key, text, by=None):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > NOTE_MAX:
        sys.exit(f"note is {len(text)} chars — max {NOTE_MAX}. Shorten it "
                 f"(notes are for one durable fact, not a report).")
    e = {"kind": kind, "key": key, "note": text,
         "ts": time.strftime("%Y-%m-%d")}
    if by:
        e["by"] = by
    with open(NOTES, "a") as f:
        f.write(json.dumps(e) + "\n")
    return e


def print_notes(kind, key, indent="  "):
    ns = notes_for(kind, key)
    if ns:
        print(f"{indent}notes:")
        for n in ns:
            by = f" ({n['by']})" if n.get("by") else ""
            print(f"{indent}  [{n['ts']}]{by} {n['note']}")


# ---------- resolution ----------

def resolve_deck(q, names=None):
    """-> (stem, path). Accepts pool stem/substring, csv path, proposal id."""
    if q.endswith(".csv") and os.path.exists(q):
        return os.path.basename(q)[:-4], q
    if q.startswith("dlab_"):
        hits = sorted(glob.glob(os.path.join(C.PROPOSALS, q + "*", "deck.csv")))
        if len(hits) == 1:
            return os.path.basename(os.path.dirname(hits[0])), hits[0]
        if hits:
            sys.exit("ambiguous proposal id:\n  " +
                     "\n  ".join(os.path.dirname(h) for h in hits))
    if names is None:
        names, _ = C.load_pool()
    if q in names:
        return q, os.path.join(C.REPO, "decks/pool", q + ".csv")
    sub = [n for n in names if q.lower() in n.lower()]
    if len(sub) == 1:
        return sub[0], os.path.join(C.REPO, "decks/pool", sub[0] + ".csv")
    if sub:
        sys.exit(f"ambiguous deck '{q}' ({len(sub)} matches):\n  " +
                 "\n  ".join(sub[:15]) + ("\n  ..." if len(sub) > 15 else ""))
    sys.exit(f"no pool deck, path, or proposal matches '{q}'")


def resolve_cards(q, cards):
    """-> {name: [ids]} for names matching q (exact name wins)."""
    byname = {}
    for cid, c in cards.items():
        byname.setdefault(c["name"], []).append(cid)
    exact = [n for n in byname if n.lower() == q.lower()]
    if exact:
        return {exact[0]: sorted(byname[exact[0]])}
    hits = {n: sorted(ids) for n, ids in byname.items()
            if q.lower() in n.lower()}
    return dict(sorted(hits.items()))


# ---------- field / implied helpers ----------

def field_line(stem):
    try:
        fc = C.field_cache()
    except Exception:
        return None
    d = fc["decks"].get(stem)
    if not d:
        return "field: absent"
    share = 100.0 * d["sides"] / max(fc["total_sides"], 1)
    s = f"field: {d['sides']} sides ({share:.2f}%)"
    if d.get("lad_n"):
        s += f"; ladder wr {d['lad_w'] / d['lad_n']:.3f} (n={d['lad_n']})"
    return s


def implied_line(stem):
    t = C.implied_table().get(stem)
    if not t:
        return None
    rank, wr, flag = t
    s = f"implied (field-weighted): wr {wr:.3f}, rank #{rank}"
    if flag:
        s += f"  [UNRANKABLE: {flag}]"
    return s


# ---------- commands ----------

def cmd_deck(a):
    stem, path = resolve_deck(a.query)
    cards, attacks = C.load_carddb()
    deck = C.read_deck(path)
    vocab = C.evaluator_vocab()
    print(f"{stem}  ({os.path.relpath(path, C.REPO)})")
    for ln in (field_line(stem), implied_line(stem)):
        if ln:
            print(f"  {ln}")
    oov = sorted(cid for cid in deck if cid not in vocab)
    if oov:
        print(f"  evaluator-OOV cards ({len(oov)}): " +
              ", ".join(cards[c]["name"] for c in oov))
    bygroup = {}
    for cid, n in deck.items():
        bygroup.setdefault(cards[cid].get("cardType"), []).append((cid, n))
    for g in GROUPS + sorted(set(bygroup) - set(GROUPS)):
        if g not in bygroup:
            continue
        items = sorted(bygroup[g], key=lambda t: cards[t[0]]["name"])
        total = sum(n for _, n in items)
        print(f"  -- {g} ({total})")
        for cid, n in items:
            c = cards[cid]
            extra = ""
            if c.get("hp"):
                et = C._ENERGY.get(c.get("energyType"), "?")
                extra = f"  {c.get('stageText') or ''} HP{c['hp']} {et}"
                if c.get("weakness") is not None:
                    extra += f" weak:{C._ENERGY.get(c['weakness'], '?')}"
            print(f"    {n}x [{cid:>4}] {c['name']}{extra}")
            if a.full:
                print("       " + C.card_text(cid, cards, attacks)
                      .replace("\n", "\n       "))
    print_notes("deck", stem)
    for cid in deck:
        for n in notes_for("card", cards[cid]["name"]):
            print(f"  card note [{cards[cid]['name']}]: {n['note']}")


def cmd_diff(a):
    names, _ = C.load_pool()
    sa, pa = resolve_deck(a.a, names)
    sb, pb = resolve_deck(a.b, names)
    cards, _ = C.load_carddb()
    out, inn, k = C.deck_diff(C.read_deck(pa), C.read_deck(pb))
    print(f"{sa}  ->  {sb}   ({k} swap{'s' if k != 1 else ''})")
    for cid, n in sorted(out.items(), key=lambda t: cards[t[0]]["name"]):
        print(f"  -{n} [{cid:>4}] {cards[cid]['name']}")
    for cid, n in sorted(inn.items(), key=lambda t: cards[t[0]]["name"]):
        print(f"  +{n} [{cid:>4}] {cards[cid]['name']}")
    if k == 0:
        print("  identical lists")


def cmd_near(a):
    names, counters = C.load_pool()
    stem, path = resolve_deck(a.query, names)
    me = C.read_deck(path)
    imp = C.implied_table()
    try:
        fc = C.field_cache()["decks"]
    except Exception:
        fc = {}
    ds = []
    for n in names:
        if n == stem:
            continue
        _, _, k = C.deck_diff(me, counters[n])
        ds.append((k, n))
    ds.sort()
    print(f"nearest pool decks to {stem}:")
    print(f"  {'swaps':>5}  {'field sides':>11}  {'implied':>9}  deck")
    for k, n in ds[:a.k]:
        sides = fc.get(n, {}).get("sides", 0)
        t = imp.get(n)
        istr = f"#{t[0]} {t[1]:.3f}" if t else "-"
        print(f"  {k:>5}  {sides:>11}  {istr:>9}  {n}")


def cmd_card(a):
    cards, attacks = C.load_carddb()
    hits = resolve_cards(a.query, cards)
    if not hits:
        sys.exit(f"no card name matches '{a.query}'")
    if len(hits) > 8 and not a.all:
        print(f"{len(hits)} names match — showing 8 (use --all):")
        hits = dict(list(hits.items())[:8])
    vocab = C.evaluator_vocab()
    names, counters = C.load_pool()
    try:
        fc = C.field_cache()["decks"]
    except Exception:
        fc = {}
    for name, ids in hits.items():
        for cid in ids:
            if a.brief:
                c = cards[cid]
                print(f"[{cid}] {name} — {c.get('stageText') or c.get('cardType')}")
            else:
                print(C.card_text(cid, cards, attacks))
            iv = cid in vocab
            print(f"  evaluator vocab: {'YES' if iv else 'NO -> OOV: A/B meaningless, reasoning-class path'}")
            users = [(counters[n][cid], n) for n in names if cid in counters[n]]
            if users:
                copies = {}
                for cnt, _ in users:
                    copies[cnt] = copies.get(cnt, 0) + 1
                dose = ", ".join(f"{k}x in {v}" for k, v in sorted(copies.items()))
                top = sorted(users, key=lambda t: -fc.get(t[1], {}).get("sides", 0))[:3]
                ex = "; ".join(f"{n} ({cnt}x)" for cnt, n in top)
                print(f"  pool usage: {len(users)}/{len(names)} decks ({dose}) e.g. {ex}")
            else:
                print("  pool usage: none")
        print_notes("card", name)
        print()


def cmd_find(a):
    q = a.query.lower()
    names, _ = C.load_pool()
    dh = [n for n in names if q in n.lower()]
    cards, _ = C.load_carddb()
    ch = sorted({c["name"] for c in cards.values() if q in c["name"].lower()})
    for n in dh:
        print(f"deck  {n}")
    for n in ch:
        print(f"card  {n}")
    if not dh and not ch:
        print("no matches")


def cmd_note(a):
    text = " ".join(a.text)
    if a.kind == "deck":
        key, _ = resolve_deck(a.key)
    else:
        cards, _ = C.load_carddb()
        hits = resolve_cards(a.key, cards)
        if len(hits) != 1:
            sys.exit(f"card '{a.key}' matches {len(hits)} names — use the exact name:\n  "
                     + "\n  ".join(list(hits)[:15]))
        key = next(iter(hits))
    e = notes_add(a.kind, key, text, by=a.by)
    print(f"noted [{a.kind}] {key}: {e['note']}")


def cmd_notes(a):
    q = (a.query or "").lower()
    for n in notes_read():
        if q and q not in json.dumps(n).lower():
            continue
        by = f" ({n['by']})" if n.get("by") else ""
        print(f"[{n['ts']}]{by} {n['kind']}:{n['key']}\n    {n['note']}")


def cmd_history(a):
    q = a.query.lower()
    shown = 0
    for e in C.ledger_read():
        if q in json.dumps(e).lower():
            bits = [e.get("ts", "")[:16], e.get("id", "-"), e.get("event", "?")]
            extra = e.get("reason") or e.get("note") or e.get("verdict") or ""
            if isinstance(extra, str) and extra:
                bits.append(extra[:120])
            print("  ".join(str(b) for b in bits))
            shown += 1
    for mp in sorted(glob.glob(os.path.join(C.PROPOSALS, "*", "meta.json"))):
        m = json.load(open(mp))
        if q in json.dumps(m).lower():
            print(f"proposal {m['id']}: status={m.get('status')} "
                  f"base={os.path.basename(m.get('base_deck', '?'))} "
                  f"staged={m.get('staged_as') or '-'}")
            shown += 1
    if not shown:
        print("nothing in ledger or proposals")


# ---------- mat ----------

def _mat(which, epoch_min=None):
    which = which or C.default_matrix()
    G, W, names, eps = C.load_matrix(which, epoch_min=epoch_min)
    return G, W, names, eps


def _resolve_in(names, q):
    if q in names:
        return names.index(q)
    sub = [i for i, n in enumerate(names) if q.lower() in n.lower()]
    if len(sub) == 1:
        return sub[0]
    if sub:
        sys.exit(f"ambiguous in matrix listing ({len(sub)}):\n  " +
                 "\n  ".join(names[i] for i in sub[:15]))
    sys.exit(f"'{q}' not in the {len(names)}-deck matrix listing "
             f"(new appends have no cells until shards catch up — mat check)")


def cmd_mat_cell(a):
    G, W, names, eps = _mat(a.which)
    i, j = _resolve_in(names, a.a), _resolve_in(names, a.b)
    n, w = C.pair_cell(G, W, i, j)
    print(f"{names[i]}  vs  {names[j]}   [{a.which}, {eps} eps in window]")
    for lab, nn, ww in [("as learner", int(G[i, j]), int(W[i, j])),
                        ("as opponent", int(G[j, i]), int(G[j, i] - W[j, i]))]:
        if nn:
            print(f"  {lab:>12}: {ww}/{nn} = {ww / nn:.3f}")
    if n == 0:
        print("  no games in this window")
        return
    p = (w + 2.0) / (n + 4.0)
    z = (w / n - 0.5) / math.sqrt(0.25 / n)
    print(f"  {'combined':>12}: {w}/{n} = {w / n:.3f}   shrunk p^ {p:.3f}   z {z:+.1f}")


def cmd_mat_row(a):
    G, W, names, eps = _mat(a.which)
    i = _resolve_in(names, a.deck)
    try:
        cache = C.field_cache()
        fc, tot = cache["decks"], max(cache["total_sides"], 1)
    except Exception:
        fc, tot = {}, 1
    rows = []
    for j in range(len(names)):
        if j == i:
            continue
        n, w = C.pair_cell(G, W, i, j)
        if n < a.min_games:
            continue
        p = (w + 2.0) / (n + 4.0)
        share = 100.0 * fc.get(names[j], {}).get("sides", 0) / tot
        rows.append((p, n, share, names[j]))
    rows.sort(reverse=a.best)
    wr, games = C.per_deck_wr(G, W)
    print(f"{names[i]}  [{a.which}, {eps} eps in window] "
          f"raw wr {wr[i]:.3f} over {int(games[i])} games — "
          f"{'best' if a.best else 'worst'} {a.n} (min {a.min_games} games/cell):")
    print(f"  {'p^':>6} {'games':>6} {'field%':>7}  opponent")
    for p, n, share, nm in rows[:a.n]:
        print(f"  {p:>6.3f} {n:>6} {share:>6.2f}%  {nm}")


def cmd_mat_rank(a):
    G, W, names, eps = _mat(a.which)
    i = _resolve_in(names, a.deck)
    wr, games = C.per_deck_wr(G, W)
    order = sorted(range(len(names)),
                   key=lambda k: -(wr[k] if wr[k] == wr[k] else -1))
    rank = order.index(i) + 1
    print(f"{names[i]}  [{a.which}, {eps} eps in window]")
    print(f"  raw pool wr {wr[i]:.3f} over {int(games[i])} games — "
          f"rank #{rank}/{len(names)} (raw = uniform-pool, overstates tail-farmers)")
    ln = implied_line(names[i])
    print(f"  {ln}" if ln else
          "  no implied entry (no data/deck_implied.csv)")


def cmd_mat_check(a):
    a.which = a.which or C.default_matrix()
    m = C.matrices()[a.which]
    disk = set(n for n in C.load_pool()[0])
    G, W, names, eps = _mat(a.which)
    listed = set(names)
    _, games = C.per_deck_wr(G, W)
    zero = [names[i] for i in range(len(names)) if games[i] == 0]
    print(f"matrix {a.which}: {len(names)} decks listed "
          f"({'pinned ' + os.path.basename(m['names']) if m['names'] else 'live pool glob'}), "
          f"{eps} eps in window, {int(G.sum())} matrix games")
    print(f"decks on disk: {len(disk)}")
    only_disk = sorted(disk - listed)
    only_mat = sorted(listed - disk)
    if only_disk:
        print(f"on disk but NOT in this matrix listing ({len(only_disk)}):")
        for n in only_disk[:25]:
            print(f"  {n}")
        if len(only_disk) > 25:
            print(f"  ... +{len(only_disk) - 25}")
    if only_mat:
        print(f"in matrix listing but NOT on disk ({len(only_mat)}) — ids may have drifted, INVESTIGATE:")
        for n in only_mat[:25]:
            print(f"  {n}")
    if zero:
        print(f"listed with ZERO games in window ({len(zero)}) — appended but no shards yet:")
        for n in zero[:25]:
            print(f"  {n}")
    if not (only_disk or only_mat or zero):
        print("fully synced: every listed deck on disk, every deck has games")


def main():
    ap = argparse.ArgumentParser(prog="dlab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("deck");  p.add_argument("query")
    p.add_argument("--full", action="store_true", help="full card text per card")
    p.set_defaults(f=cmd_deck)

    p = sub.add_parser("diff");  p.add_argument("a"); p.add_argument("b")
    p.set_defaults(f=cmd_diff)

    p = sub.add_parser("near");  p.add_argument("query")
    p.add_argument("--k", type=int, default=8)
    p.set_defaults(f=cmd_near)

    p = sub.add_parser("card");  p.add_argument("query")
    p.add_argument("--brief", action="store_true")
    p.add_argument("--all", action="store_true")
    p.set_defaults(f=cmd_card)

    p = sub.add_parser("find");  p.add_argument("query")
    p.set_defaults(f=cmd_find)

    p = sub.add_parser("note")
    p.add_argument("kind", choices=["deck", "card"])
    p.add_argument("key")
    p.add_argument("text", nargs="+")
    p.add_argument("--by", default=None)
    p.set_defaults(f=cmd_note)

    p = sub.add_parser("notes"); p.add_argument("query", nargs="?")
    p.set_defaults(f=cmd_notes)

    p = sub.add_parser("history"); p.add_argument("query")
    p.set_defaults(f=cmd_history)

    pm = sub.add_parser("mat")
    ms = pm.add_subparsers(dest="matcmd", required=True)

    p = ms.add_parser("cell"); p.add_argument("a"); p.add_argument("b")
    p.add_argument("--which", default=None, help="matrix key from decklab/config.json")
    p.set_defaults(f=cmd_mat_cell)

    p = ms.add_parser("row"); p.add_argument("deck")
    p.add_argument("--which", default=None, help="matrix key from decklab/config.json")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--best", action="store_true")
    p.add_argument("--min-games", type=int, default=20)
    p.set_defaults(f=cmd_mat_row)

    p = ms.add_parser("rank"); p.add_argument("deck")
    p.add_argument("--which", default=None, help="matrix key from decklab/config.json")
    p.set_defaults(f=cmd_mat_rank)

    p = ms.add_parser("check")
    p.add_argument("--which", default=None, help="matrix key from decklab/config.json")
    p.set_defaults(f=cmd_mat_check)

    a = ap.parse_args()
    if getattr(a, "which", None) is None and hasattr(a, "which"):
        a.which = C.default_matrix()
    a.f(a)


if __name__ == "__main__":
    main()
