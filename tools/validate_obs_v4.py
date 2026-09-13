"""End-to-end validation of the v4 obs bump (torch+numpy only, no pufferlib).

Drives BattleHandle directly with random legal answers, feeding per-seat
InfoTrackers exactly like env._ingest (acting seat only), encoding every
decision, and checking:

  A. every encoded row is finite
  B. LEGALITY CROSS-CHECK: a hard (1.0) EFF_CANT_ATTACK on my active on
     a main-menu select => no OT_ATTACK option offered by the engine;
     same for EFF_CANT_RETREAT => no OT_RETREAT (one-directional:
     flag => absent; absence without flag is fine — energy etc.)
  C. player locks: hard MY_LOCK_ITEM/_SUPPORTER => no OT_PLAY option
     for a hand card of that trainer type
  D. prize-zone truth: whenever tracker.my_prize_known is set, it must
     equal the OracleLedger's exact my_prize multiset
  E. EffectTracker turn counter tracks the engine turn

Run: PYTHONPATH=data:. python3 tools/validate_obs_v4.py [n_games]
"""
import glob
import os
import random
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from ptcg.rl.battle import BattleHandle                  # noqa: E402
from ptcg.rl.buffers import (                            # noqa: E402
    OFFSETS, MAX_TOKENS, TOK_INT, TOK_F, Zone, TokF, GlobF, OBS_SIZE)
from ptcg.rl.encoder import encode                       # noqa: E402
from ptcg.rl.oracle import OracleLedger                  # noqa: E402
from ptcg.tracker import InfoTracker                     # noqa: E402

OT_PLAY, OT_RETREAT, OT_ATTACK, OT_END = 7, 12, 13, 14
FLAG_NAMES = [n for n in vars(TokF) if n.startswith("EFF_")]


def load_decks():
    decks = []
    for f in sorted(glob.glob(os.path.join(ROOT, "decks", "pool", "*.csv"))):
        ids = [int(x) for x in open(f).read().split()]
        if len(ids) == 60:
            decks.append(ids)
    return decks


def rand_answer(sel, rng):
    n = len(sel["option"])
    lo, hi = sel["minCount"], min(sel["maxCount"], n)
    k = rng.randint(lo, max(lo, hi))
    return rng.sample(range(n), min(k, n))


def main(n_games=120, seed=7):
    rng = random.Random(seed)
    decks = load_decks()
    from cg.api import all_card_data, CardType
    ctype = {c.cardId: c.cardType for c in all_card_data()}

    row = np.zeros(OBS_SIZE, dtype=np.float32)
    ti_a, ti_b = OFFSETS["tok_int"]
    tf_a, tf_b = OFFSETS["tok_float"]
    g_a, g_b = OFFSETS["global_f"]

    stats = Counter()
    flag_hits = Counter()
    violations = []

    for game_i in range(n_games):
        d0, d1 = rng.choice(decks), rng.choice(decks)
        h = BattleHandle(d0, d1)
        oracle = OracleLedger((d0, d1), h)
        trackers = [InfoTracker(d0), InfoTracker(d1)]
        obs = h.obs()
        steps = 0
        while obs["current"]["result"] == -1 and steps < 3000:
            cur = obs["current"]
            seat = cur["yourIndex"]
            tr = trackers[seat]
            tr.update(obs)
            oracle.on_seat_obs(obs)
            sel = obs["select"]
            stats["decisions"] += 1

            encode(row, obs, [], False, 0, tr)
            if not np.isfinite(row).all():
                stats["nonfinite_rows"] += 1
            tok_i = row[ti_a:ti_b].reshape(MAX_TOKENS, TOK_INT)
            tok_f = row[tf_a:tf_b].reshape(MAX_TOKENS, TOK_F)
            g = row[g_a:g_b]

            fx = tr.effects

            for t in range(MAX_TOKENS):
                if tok_i[t, 1] == 0:
                    break
                for name in FLAG_NAMES:
                    if tok_f[t, getattr(TokF, name)] != 0.0:
                        flag_hits[name] += 1

            opts = sel["option"]
            main_menu = any(o.get("type") == OT_END for o in opts)
            act_row = next((t for t in range(MAX_TOKENS)
                            if tok_i[t, 1] == Zone.MY_ACTIVE), None)
            advanced = False
            if main_menu and act_row is not None:
                for flag, ot, tag in ((TokF.EFF_CANT_ATTACK, OT_ATTACK,
                                       "cant_attack"),
                                      (TokF.EFF_CANT_RETREAT, OT_RETREAT,
                                       "cant_retreat")):
                    if tok_f[act_row, flag] == 1.0:
                        stats[f"checked_{tag}"] += 1
                        oi = next((j for j, o in enumerate(opts)
                                   if o.get("type") == ot), None)
                        if oi is not None:
                            # offered — but does the engine ACCEPT it?
                            # (option lists can include selects the
                            # engine rejects: known runner footgun)
                            try:
                                obs = h.select([oi])
                                accepted = True
                            except RuntimeError:
                                accepted = False
                            if not accepted:
                                stats[f"offered_but_rejected_{tag}"] += 1
                                continue
                            advanced = True
                            stats[f"VIOLATION_{tag}"] += 1
                            violations.append(
                                (game_i, cur["turn"], tag,
                                 int(tok_i[act_row, 0])))
                            act = next(p for p in cur["players"][seat]
                                       ["active"] if p is not None)
                            print(f"\n-- VIOLATION {tag} game {game_i} "
                                  f"turn {cur['turn']} seat {seat}")
                            print("   active:", act["id"],
                                  "serial", act.get("serial"))
                            for inst in fx.serial_fx.get(
                                    act.get("serial"), ()):
                                r = inst["rec"]
                                print(f"   inst: {r['card_name']}/"
                                      f"{r['source_name']} "
                                      f"{r['effect_class']} "
                                      f"dur={r['duration']} "
                                      f"bind=[{inst['from']},{inst['to']}]"
                                      f" soft={inst['soft']}")
                                print(f"   clause: {r['clause_text'][:100]}")
                            break
                if advanced:
                    steps += 1
                    continue
                for gf, tt, tag in ((GlobF.MY_LOCK_ITEM, 1, "item_lock"),
                                    (GlobF.MY_LOCK_SUPPORTER, 3,
                                     "supporter_lock")):
                    if g[gf] == 1.0:
                        stats[f"checked_{tag}"] += 1
                        me = cur["players"][seat]
                        for o in opts:
                            if o.get("type") == OT_PLAY:
                                hand = me.get("hand") or []
                                idx = o.get("index")
                                if idx is not None and idx < len(hand) and \
                                        ctype.get(hand[idx]["id"]) == tt:
                                    stats[f"VIOLATION_{tag}"] += 1
                                    violations.append(
                                        (game_i, cur["turn"], tag,
                                         hand[idx]["id"]))
                                    break

            if tr.my_prize_known is not None:
                stats["prize_known_decisions"] += 1
                truth = oracle.emit(obs)[3]
                if Counter(tr.my_prize_known) != Counter(truth):
                    stats["VIOLATION_prize"] += 1

            ans = rand_answer(sel, rng)
            ok = False
            for _ in range(6):
                try:
                    obs = h.select(ans)
                    ok = True
                    break
                except RuntimeError:
                    stats["rejected_answers"] += 1
                    ans = rand_answer(sel, rng)
            if not ok:
                try:
                    obs = h.select([0])
                except RuntimeError:
                    stats["aborted_games"] += 1
                    break
            steps += 1
        stats["games"] += 1
        stats["turn_drift"] += sum(t.effects.turn_drift for t in trackers)
        h.finish()

    print("== validate_obs_v4 ==")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print("  flag activations (token-decisions):")
    for k, v in flag_hits.most_common():
        print(f"    {k}: {v}")
    print("  example violations:", violations[:10])
    bad = [k for k in stats if k.startswith("VIOLATION") and stats[k]] + \
        (["nonfinite"] if stats["nonfinite_rows"] else [])
    print("RESULT:", "FAIL " + str(bad) if bad else "PASS")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
