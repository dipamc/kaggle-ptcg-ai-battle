"""Export every static table the native (C/CUDA) backend needs into one
binary blob: ptcg_tables.bin.

Sections (all little-endian):
  ENV (CPU, encoder + effects + decks):
    atk_damage   f32 (N_ATTACKS,)
    atk_cost     i8  (N_ATTACKS, 12)
    energy_type  i8  (N_CARDS,)
    weakness     i8  (N_CARDS,)
    resistance   i8  (N_CARDS,)
    retreat      i8  (N_CARDS,)
    top_attacks  i32 (N_CARDS, 4)
    supporters   u8  (N_CARDS,)   membership bitmask arrays
    shields      u8  (N_CARDS,)
    pokemon_ids  u8  (N_CARDS,)
    attack_fx    i32 (n, FX_FIELDS)   flattened lingering-effect records
    play_fx      i32 (n, FX_FIELDS)
    fx_magnitude f32 (n_attack_fx + n_play_fx,)
    decks        i32 (n_decks, 60)    sorted-glob pool order (deck_idx ids)
  MODEL (GPU, static feature tables the CUDA model gathers from):
    card_static  f32 (N_CARDS, 58)
    card_attacks i32 (N_CARDS, 4)
    card_skills  i32 (N_CARDS, 2)
    att_static   f32 (N_ATTACKS, 15)
    card_text    f32 (N_CARDS, 128)
    att_text     f32 (N_ATTACKS, 64)
    skill_text   f32 (N_SKILLS, 64)

Flattened effect record fields (FX_FIELDS = 8):
  [0] key        attackId (attack_fx) or cardId (play_fx)
  [1] kind       0 plain-max flag | 1 dmg_taken signed | 2 dmg_dealt signed
                 | 3 delayed_dmg | 4 delayed_ko  (flag records)  | -1 lock
  [2] flag_tokf  TokF index for flag records (44..51), else 0
  [3] lock       0 none | 1 ITEM | 2 SUPPORTER | 3 EVOLVE
  [4] sign       +1 / -1 (dmg mods)
  [5] target     0 SELF | 1 DEFENDING | 2 ALL_MY | 3 ALL_OPP | 4 OPP_PLAYER | 5 OTHER
  [6] condition  0 NONE | 1 COIN_HEADS | 2 COIN_TAILS | 3 OTHER
  [7] duration   0 SAME_TURN | 1 MY_NEXT | 2 OPP_NEXT | 3 UNTIL_END_MY_NEXT
                 | 4 UNTIL_END_OPP_NEXT | 5 WHILE_CONDITION | 6 OTHER

Run from repo root:  PYTHONPATH=data:. python native/tools/export_tables.py
"""
import glob
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

from ptcg.rl.buffers import (N_CARDS, N_ATTACKS, OBS_SIZE, MAX_TOKENS,
                             MAX_OPTIONS, TOK_INT, TOK_F, OPT_INT, OPT_F,
                             GLOBAL_F, DEC_INT, DEC_F, ORACLE_F, TokF)
from ptcg.rl.cards import build_tables, build_env_tables, load_text_tables, \
    N_SKILLS

MAGIC = b"PTCGTAB1"
FX_FIELDS = 8

_TARGET = {"SELF": 0, "DEFENDING": 1, "ALL_MY_POKEMON": 2,
           "ALL_OPP_POKEMON": 3, "OPP_PLAYER": 4, "OTHER": 5}
_COND = {"NONE": 0, "COIN_HEADS": 1, "COIN_TAILS": 2}
_DUR = {"SAME_TURN": 0, "MY_NEXT_TURN": 1, "OPP_NEXT_TURN": 2,
        "UNTIL_END_OF_MY_NEXT_TURN": 3, "UNTIL_END_OF_OPP_NEXT_TURN": 4,
        "WHILE_CONDITION": 5}
_LOCK = {"ITEM": 1, "SUPPORTER": 2, "EVOLVE": 3}


def _flatten_fx(fx_map):
    """attack_fx/play_fx {key: [rec]} -> (records i32 (n,8), magnitudes f32)."""
    rows, mags = [], []
    for key in sorted(fx_map):
        for rec in fx_map[key]:
            flag = rec["flag"]
            lock = _LOCK.get(rec["lock"] or "", 0)
            if lock:
                kind, tokf, sign = -1, 0, 1
            elif flag is None:
                # registered but no obs slot: python _register returns
                # early for flag None & lock None -> skip entirely
                continue
            elif flag == "EFF_DMG_TAKEN_MOD":
                kind, tokf = 1, TokF.EFF_DMG_TAKEN_MOD
                sign = 1 if rec["effect_class"] == "INCREASE_DAMAGE_TAKEN" else -1
            elif flag == "EFF_DMG_DEALT_MOD":
                kind, tokf = 2, TokF.EFF_DMG_DEALT_MOD
                sign = -1 if rec["effect_class"] == "REDUCE_DAMAGE_DEALT" else 1
            elif flag == "EFF_DELAYED_DMG":
                kind, tokf, sign = 3, TokF.EFF_DELAYED_DMG, 1
            elif flag == "EFF_DELAYED_KO":
                kind, tokf, sign = 4, TokF.EFF_DELAYED_KO, 1
            else:
                kind, tokf, sign = 0, getattr(TokF, flag), 1
            rows.append([key, kind, tokf, lock, sign,
                         _TARGET.get(rec["target"], 5),
                         _COND.get(rec["condition"], 3),
                         _DUR.get(rec["duration"], 6)])
            mags.append(float(rec.get("magnitude") or 0.0))
    if not rows:
        return (np.zeros((0, FX_FIELDS), dtype=np.int32),
                np.zeros(0, dtype=np.float32))
    return (np.asarray(rows, dtype=np.int32),
            np.asarray(mags, dtype=np.float32))


def _load_decks(deck_dir):
    """60-line card-id files in sorted order — that order IS the deck id space.

    Non-deck .csv files are skipped rather than fatal (a pool directory may
    carry a manifest .csv), but every skip is printed: a deck silently dropped here
    shifts every id after it, which would silently invalidate anchors, the
    deck matrix and every report built against an older blob.
    """
    decks, skipped = [], []
    for f in sorted(glob.glob(os.path.join(deck_dir, "*.csv"))):
        with open(f) as fh:
            try:
                ids = [int(line) for line in fh if line.strip()]
            except ValueError:
                skipped.append((os.path.basename(f), "not card ids"))
                continue
        if len(ids) == 60:
            decks.append(ids)
        else:
            skipped.append((os.path.basename(f), f"{len(ids)} lines"))
    assert decks, f"no decks in {deck_dir}"
    for name, why in skipped:
        print(f"  [decks] skipped {deck_dir}/{name} ({why})")
    return np.asarray(decks, dtype=np.int32)


def _check_names(pool, alt_pool, expect_path):
    """Fail loudly if the pool's sorted glob is not EXACTLY the pinned listing.

    Deck ids are positions in that glob, so a blob rebuilt from a drifted pool
    is not merely incomplete, it silently RENAMES every id after the first
    difference — invalidating anchor indices, deck matrices and any checkpoint
    read against it. Drift is easy to cause: dropping a deck into the pool
    shifts every id after it, and removing decks leaves two machines building
    different blobs under the same version. Pin a listing with --expect-names.
    """
    import glob as _g, os as _o
    want = [l.strip() for l in open(expect_path) if l.strip()]
    got = sorted(_o.path.basename(f)[: -len(".csv")]
                 for f in _g.glob(_o.path.join(pool, "*.csv"))
                 if sum(1 for line in open(f) if line.strip()) == 60)
    if got == want:
        print(f"  [decks] pool matches {expect_path} ({len(want)} decks)")
        return
    missing = [n for n in want if n not in got]
    extra = [n for n in got if n not in want]
    msg = [f"POOL MISMATCH vs {expect_path}: expected {len(want)} decks, "
           f"found {len(got)}"]
    if missing:
        msg.append(f"  missing ({len(missing)}): " + ", ".join(missing[:6])
                   + (" ..." if len(missing) > 6 else ""))
    if extra:
        msg.append(f"  unexpected ({len(extra)}): " + ", ".join(extra[:6])
                   + (" ..." if len(extra) > 6 else ""))
    if not missing and not extra:
        msg.append("  same set, DIFFERENT ORDER — ids would still shift")
    raise SystemExit("\n".join(msg))


def _check_append(pool, append_path):
    """Fail unless the pool is the pinned listing plus NEW NAMES AT THE END.

    This is the writer half of the mid-run append contract
    (docs/deck-pool.md). `pt_reload_decks` rejects a blob whose existing
    rows moved, but by then GPU-hours are already staged on a bad file; this
    catches it at build time and says which deck did it.

    The trap this exists for: deck ids are positions in the pool's SORTED glob,
    so a new deck only appends if its FILENAME sorts after every existing one.
    Every deck in the pool is named by archetype, so a genuinely new archetype
    usually does NOT sort last — dropping in `aaa_new.csv` inserts at id 0 and
    renames all 382 ids. Name appended decks so they sort last.

    Returns the new-deck names so the caller can report them.
    """
    import glob as _g, os as _o
    old = [l.strip() for l in open(append_path) if l.strip()]
    got = sorted(_o.path.basename(f)[: -len(".csv")]
                 for f in _g.glob(_o.path.join(pool, "*.csv"))
                 if sum(1 for line in open(f) if line.strip()) == 60)
    if len(got) < len(old):
        raise SystemExit(
            f"APPEND CHECK vs {append_path}: pool has {len(got)} decks, the "
            f"pinned listing has {len(old)} — that is a REMOVAL. Removing a "
            f"deck must be a stop/resume, never a live append.")
    for i, (a, b) in enumerate(zip(old, got)):
        if a != b:
            raise SystemExit(
                f"APPEND CHECK vs {append_path}: deck id {i} changed from "
                f"{a!r} to {b!r}. New decks must sort AFTER every existing "
                f"one, or they insert and rename every id from {i} up. "
                f"Rename the new file(s) so they sort last.")
    added = got[len(old):]
    print(f"  [decks] append OK: {len(old)} unchanged + {len(added)} new "
          f"({', '.join(added[:4])}{' ...' if len(added) > 4 else ''})")
    return added


def main(out_path="native/ptcg_tables.bin", pool="decks/pool", alt_pool=None,
         expect_names=None, append_to=None):
    """Build the tables blob.

    `pool` is the TRAINING pool: its sorted 60-line glob order defines deck
    ids, which anchors, the deck matrix and every report index into, so it
    must not be reordered casually. `alt_pool` is optional and holds decks
    that only the RANDOM opponent's seat draws from (coverage decks) — they
    live in a separate `decks_alt` section so adding or removing one cannot
    shift a single training deck id. Omitting it writes no section at all,
    which older blobs also do, and the C side reads that as zero.
    """
    if expect_names:
        _check_names(pool, alt_pool, expect_names)
    if append_to:
        _check_append(pool, append_to)

    from ptcg.effects import _build_tables as fx_tables
    attack_fx, play_fx, supporters, shields, pokemon_ids = fx_tables()

    env = build_env_tables()
    card_static, card_attacks, card_skills, att_static = build_tables()
    card_text, att_text, skill_text = load_text_tables()

    def member(s):
        a = np.zeros(N_CARDS, dtype=np.uint8)
        for cid in s:
            if 0 <= cid < N_CARDS:
                a[cid] = 1
        return a

    a_fx, a_mag = _flatten_fx(attack_fx)
    p_fx, p_mag = _flatten_fx(play_fx)

    sections = [
        ("atk_damage", env["atk_damage"].astype(np.float32)),
        ("atk_cost", env["atk_cost"].astype(np.int8)),
        ("energy_type", env["energy_type"].astype(np.int8)),
        ("weakness", env["weakness"].astype(np.int8)),
        ("resistance", env["resistance"].astype(np.int8)),
        ("retreat", env["retreat"].astype(np.int8)),
        ("top_attacks", env["top_attacks"].astype(np.int32)),
        ("supporters", member(supporters)),
        ("shields", member(shields)),
        ("pokemon_ids", member(pokemon_ids)),
        ("attack_fx", a_fx),
        ("attack_fx_mag", a_mag),
        ("play_fx", p_fx),
        ("play_fx_mag", p_mag),
        ("decks", _load_decks(pool)),
        ("card_static", card_static.astype(np.float32)),
        ("card_attacks", card_attacks.astype(np.int32)),
        ("card_skills", card_skills.astype(np.int32)),
        ("att_static", att_static.astype(np.float32)),
        ("card_text", card_text.astype(np.float32)),
        ("att_text", att_text.astype(np.float32)),
        ("skill_text", skill_text.astype(np.float32)),
    ]
    if alt_pool:
        sections.append(("decks_alt", _load_decks(alt_pool)))

    _DT = {np.dtype(np.float32): 0, np.dtype(np.int32): 1,
           np.dtype(np.int8): 2, np.dtype(np.uint8): 3}
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        # layout constants the C side static-asserts against
        for v in (OBS_SIZE, MAX_TOKENS, MAX_OPTIONS, TOK_INT, TOK_F,
                  OPT_INT, OPT_F, GLOBAL_F, DEC_INT, DEC_F, ORACLE_F,
                  N_CARDS, N_ATTACKS, N_SKILLS, len(sections)):
            f.write(struct.pack("<i", v))
        for name, arr in sections:
            arr = np.ascontiguousarray(arr)
            nb = name.encode()
            assert len(nb) < 24
            f.write(nb + b"\0" * (24 - len(nb)))
            f.write(struct.pack("<i", _DT[arr.dtype]))
            f.write(struct.pack("<i", arr.ndim))
            for i in range(4):
                f.write(struct.pack("<q", arr.shape[i] if i < arr.ndim else 0))
            f.write(arr.tobytes())
    total = os.path.getsize(out_path)
    print(f"wrote {out_path}: {len(sections)} sections, {total/1e6:.1f} MB")
    for name, arr in sections:
        print(f"  {name:16s} {arr.dtype.str:4s} {arr.shape}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default="native/ptcg_tables.bin")
    ap.add_argument("--pool", default="decks/pool",
                    help="training pool; its sorted 60-line glob order IS the "
                         "deck id space")
    ap.add_argument("--alt-pool", default=None,
                    help="coverage pool drawn only by the random opponent's "
                         "seat (separate id space; omit for none)")
    ap.add_argument("--append-to", default=None,
                    help="file of deck names, one per line, that the pool must "
                         "START WITH verbatim; anything beyond it is treated as "
                         "an append. Use this to build a blob for a MID-RUN "
                         "pool growth: it fails at build time if a new deck "
                         "would insert rather than append (ids are sorted-glob "
                         "positions, so a name that does not sort last renames "
                         "every id after it). See docs/deck-pool.md")
    ap.add_argument("--expect-names", default=None,
                    help="file of deck names, one per line, that the pool's "
                         "sorted glob must match EXACTLY. Ids are glob "
                         "positions, so a drifted pool silently renames every "
                         "id after the first difference. Use the pinned "
                         "listing for whatever run/matrix the blob is for.")
    _a = ap.parse_args()
    main(_a.out, _a.pool, _a.alt_pool, _a.expect_names, _a.append_to)
