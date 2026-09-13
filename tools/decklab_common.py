#!/usr/bin/env python3
"""Shared plumbing for the decklab harness (automated deck proposal + testing).

Everything here is mechanical: pool/card/matrix/field loading, distances,
ledger. The cognitive layer (proposers, reviewers, orchestration) is
described in AGENTS.md and docs/decklab.md.

Conventions this module enforces:
* Deck ids are positions in the sorted-glob 60-line filter of a pool dir,
  always resolved through the same construction as export_tables/deck_matrix.
* A matrix produced against an older pool is read against that pool's
  PINNED listing (config "matrices" -> "names"), never a live glob.
* The ladder field (decklab/packs/field_cache.json) is optional; without
  it a uniform field over the pool is assumed.
"""
import csv
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import deck_matrix as dm                                   # noqa: E402
from deck_generalists import fit_shrinkage, matchup_classes  # noqa: E402

DECKLAB = os.path.join(REPO, "decklab")
PACKS = os.path.join(DECKLAB, "packs")
PROPOSALS = os.path.join(DECKLAB, "proposals")
RESULTS = os.path.join(DECKLAB, "results")
LEDGER = os.path.join(DECKLAB, "LEDGER.jsonl")
CONFIG = os.path.join(DECKLAB, "config.json")

def cfg():
    return json.load(open(CONFIG))


def matrices():
    """Deck-matrix sources from decklab/config.json ("matrices" block).

    Each entry: {"paths": [dirs of r<rank>_e<epoch>_<seq>.bin files, repo-relative],
                 "names": optional listing file that pins deck ids (the pool
                          the matrix's run was built from) or null for the
                          live sorted glob of decks/pool,
                 "epoch_min": optional lower epoch bound}
    """
    out = {}
    for k, v in cfg().get("matrices", {}).items():
        out[k] = {"paths": [os.path.join(REPO, p) for p in v["paths"]],
                  "names": os.path.join(REPO, v["names"]) if v.get("names") else None,
                  "epoch_min": v.get("epoch_min")}
    return out


def default_matrix():
    return cfg().get("default_matrix", "primary")


def load_pool(pool=None):
    """(names in id order, {name: Counter of card ids})."""
    pool = pool or os.path.join(REPO, "decks/pool")
    names, counters = [], {}
    for fn in sorted(glob.glob(os.path.join(pool, "*.csv"))):
        try:
            ids = [int(x) for x in open(fn).read().split() if x.strip()]
        except ValueError:
            continue
        if len(ids) == 60:
            nm = os.path.basename(fn)[:-4]
            names.append(nm)
            counters[nm] = Counter(ids)
    return names, counters


def read_deck(path):
    ids = [int(x) for x in open(path).read().split() if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{path}: {len(ids)} cards, want 60")
    return Counter(ids)


def deck_diff(a, b):
    """(cards only in a, cards only in b, n_swaps). Counters in, Counters out."""
    out, inn = a - b, b - a
    assert sum(out.values()) == sum(inn.values())
    return out, inn, sum(out.values())


def load_carddb():
    p = os.path.join(REPO, "data/carddb.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing: build it with "
                         f"`PYTHONPATH=data python3 tools/build_carddb.py`")
    db = json.load(open(p))
    cards = {int(k): v for k, v in db["cards"].items()}
    attacks = {int(k): v for k, v in db["attacks"].items()}
    return cards, attacks


_ENERGY = {0: "C", 1: "G", 2: "R", 3: "W", 4: "L", 5: "P", 6: "F", 7: "D", 8: "M", 9: "Y"}


def card_text(cid, cards, attacks):
    """Compact single-block engine-exact description of one card."""
    c = cards[cid]
    bits = [f"[{cid}] {c['name']} — {c.get('stageText') or c.get('cardType')}"]
    if c.get("hp"):
        et = _ENERGY.get(c.get("energyType"), c.get("energyType"))
        bits.append(f"HP {c['hp']} type {et} retreat {c.get('retreat')}")
        if c.get("weakness") is not None:
            bits.append(f"weak {_ENERGY.get(c['weakness'], c['weakness'])}")
        if c.get("evolvesFrom"):
            bits.append(f"evolves from {c['evolvesFrom']}")
    for s in c.get("skills") or []:
        sk = s if isinstance(s, dict) else {}
        bits.append(f"ABILITY {sk.get('name','?')}: {sk.get('text','')}")
    for a in c.get("attacks") or []:
        if not isinstance(a, dict):
            a = attacks.get(a) or {}
        if not a.get("name"):
            continue
        cost = "".join(_ENERGY.get(e, str(e)) for e in a.get("energies", []))
        dmg = a.get("damage") or 0
        bits.append(f"ATTACK {a['name']} [{cost}] {dmg}: {a.get('text','')}")
    if c.get("cardEffect"):
        bits.append(f"EFFECT: {c['cardEffect']}")
    if c.get("rule"):
        bits.append(f"RULE: {c['rule']}")
    if c.get("aceSpec"):
        bits.append("ACE SPEC (max 1 per deck)")
    if c.get("tera"):
        bits.append("TERA (takes no damage while benched; Tera-taxing texts "
                    "like Nighttime Mine are LIVE against it)")
    return "\n  ".join(bits)


def is_basic_energy(cid, cards):
    return cards[cid].get("cardType") == "Basic Energy"


def load_matrix(which=None, epoch_min=None, epoch_max=None):
    """(G, W, names, episodes) reconciled to the matrix's deck listing."""
    which = which or default_matrix()
    ms = matrices()
    if which not in ms:
        raise SystemExit(f"no matrix {which!r} in {CONFIG} (have {sorted(ms)})")
    m = ms[which]
    names = ([ln.strip() for ln in open(m["names"]) if ln.strip()]
             if m["names"] else dm.deck_names(os.path.join(REPO, "decks/pool")))
    G, W, eps, _ = dm.load(m["paths"], epoch_min or m["epoch_min"], epoch_max)
    G, W, names = dm.fit_to_names(G, W, names, src=which)
    return G, W, names, eps


def per_deck_wr(G, W):
    games = G.sum(1) + G.sum(0)
    wins = W.sum(1) + (G - W).sum(0)
    return np.divide(wins, games, out=np.full(len(games), np.nan),
                     where=games > 0), games


def pair_cell(G, W, i, j):
    """(games, wins) for deck i vs deck j, both orientations combined."""
    n = int(G[i, j] + G[j, i])
    w = int(W[i, j] + (G[j, i] - W[j, i]))
    return n, w


def clusters_for(G, W, cut=0.45):
    """Matchup classes via deck_generalists on the shrunk pairwise matrix.
    Returns (list of member-index lists, P shrunk, mu)."""
    N = G + G.T
    K = W + (G.T - W.T)
    wr, _ = per_deck_wr(G, W)
    mu = np.nan_to_num(wr, nan=0.5)
    k, *_ = fit_shrinkage(N, K, mu)
    P = (K + k * mu[:, None]) / np.maximum(N + k, 1e-9)
    cls = matchup_classes(P, mu, cut)
    return cls, P, mu


def field_cache(refresh=False):
    """Ladder field: how often each pool deck is played by the opponents you
    care about, plus each deck's own ladder record.

    File: decklab/packs/field_cache.json --
      {"total_sides": n, "decks": {pool_name: {"sides", "lad_n", "lad_w"}},
       "built": ts, "window": str}
    Build it from whatever ladder data you have (deck lists joined to
    decks/pool by exact multiset). Without the file every pool deck gets an
    equal share, so field-weighted numbers reduce to uniform-field numbers.
    """
    path = os.path.join(PACKS, "field_cache.json")
    if os.path.exists(path):
        return json.load(open(path))
    names, _ = load_pool()
    print(f"note: {path} not found; assuming a UNIFORM field over "
          f"{len(names)} pool decks", file=sys.stderr)
    return {"total_sides": max(len(names), 1),
            "decks": {n: {"sides": 1, "lad_n": 0, "lad_w": 0} for n in names},
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "window": "uniform (no field cache)", "unmatched": {}}


def implied_table():
    """{deck: (rank, implied_wr, note)} from data/deck_implied.csv, if present.

    Columns: deck, rank_implied, implied_wr, unrankable_note.
    Optional: a field-weighted ranking of the pool produced by whatever
    ladder analysis you run; every consumer tolerates its absence."""
    p = os.path.join(REPO, "data/deck_implied.csv")
    out = {}
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(open(p)):
        out[r["deck"]] = (int(r["rank_implied"]), float(r["implied_wr"]),
                          r.get("unrankable_note", ""))
    return out


def evaluator_vocab():
    """Card ids the A/B evaluator checkpoint was trained on: the union of the
    decklists in config "evaluator_pool_names" (a listing file), or of the
    whole current pool when unset. Cards outside this set were never piloted
    by the evaluator, so its games carry no information about them;
    proposals using such cards follow the reasoning-class path."""
    listing = cfg().get("evaluator_pool_names")
    if listing:
        names = [ln.strip() for ln in open(os.path.join(REPO, listing)) if ln.strip()]
    else:
        names, _ = load_pool()
    vocab = set()
    for n in names:
        p = os.path.join(REPO, "decks/pool", n + ".csv")
        if os.path.exists(p):
            vocab |= set(read_deck(p))
    return vocab


def ledger_append(entry):
    os.makedirs(DECKLAB, exist_ok=True)
    entry = dict(entry)
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")


def ledger_read():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(ln) for ln in open(LEDGER) if ln.strip()]


def next_proposal_id():
    used = set()
    for d in glob.glob(os.path.join(PROPOSALS, "dlab_*")):
        try:
            used.add(int(os.path.basename(d).split("_")[1]))
        except (IndexError, ValueError):
            pass
    for e in ledger_read():
        pid = e.get("id", "")
        if pid.startswith("dlab_"):
            try:
                used.add(int(pid.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return f"dlab_{(max(used) + 1) if used else 1:04d}"
