"""Raw engine obs dict -> frozen buffer row (numpy, no dataclass overhead).

Class-level option dedup happens here (docs/model.md): the
written opt_mask keeps ONE representative index per equivalence class, so
the policy's Categorical is natively over classes. Board-entity targets
are never collapsed (dynamic state differs); hand/deck/discard/looking
copies of the same card id are exactly symmetric and collapse; facedown
prizes collapse (information-set symmetric).

Token coverage (complete per §3, minus deliberately-later items):
board (with derived attack affordability/matchup features), hand,
stadium, discards, looking/deck pools, and the tracker zones — my-unseen
pool, opp known hand, opp revealed-hidden (opp_known_deck). SkillVec
in-play-ability tokens ride with the model's CardVec skill pooling for
now (own 3b tokens: later increment with the skill channels).

The oracle block (§6b) is written by write_oracle() from the env's
OracleLedger — truth for the critic, structurally separate from the
tracker (which carries only legitimately-known info into blind tokens).
"""
from collections import Counter

import numpy as np

from .buffers import (
    OFFSETS, MAX_TOKENS, TOK_INT, TOK_F, MAX_OPTIONS, STOP, OPT_INT, OPT_F,
    ORACLE_HAND_SLOTS, ORACLE_SPLIT_SLOTS, ORACLE_PRIZE_SLOTS, OT_STOP,
    Zone, Owner, TokF, OptF, GlobF, DecInt, DecF,
)

# AreaType values (cg.api)
DECK, HAND, DISCARD, ACTIVE, BENCH, PRIZE, STADIUM_A, ENERGY_A, TOOL_A = \
    1, 2, 3, 4, 5, 6, 7, 8, 9
LOOKING_A = 12

# OptionType values (cg.api)
OT_NUMBER, OT_YES, OT_NO, OT_CARD, OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY, \
    OT_PLAY, OT_ATTACH, OT_EVOLVE, OT_ABILITY, OT_DISCARD, OT_RETREAT, \
    OT_ATTACK, OT_END, OT_SKILL, OT_SPECIAL = range(17)

_CARD_ZONES = {DECK, HAND, DISCARD, PRIZE, LOOKING_A}

_T = None


def _tables():
    global _T
    if _T is None:
        from .cards import build_env_tables
        _T = build_env_tables()
    return _T


def _slice(row, name):
    a, b = OFFSETS[name]
    return row[a:b]


def _energy_counts(energies):
    att = np.zeros(12, dtype=np.int32)
    for e in energies:
        if 0 <= e < 12:
            att[e] += 1
    return att


def _deficit(T, att, total, aid):
    """Missing energy count for attack aid given attached counts.
    RAINBOW is a full wildcard; TEAM_ROCKET covers psychic/darkness."""
    cost = T["atk_cost"][aid]
    typed_missing = used = 0
    for t in range(1, 12):
        need, have = int(cost[t]), int(att[t])
        used += min(need, have)
        typed_missing += max(0, need - have)
    tr_need = max(0, int(cost[5]) - int(att[5])) + \
        max(0, int(cost[7]) - int(att[7]))
    tr_used = min(int(att[11]), tr_need)
    typed_missing -= tr_used
    used += tr_used
    wild_used = min(int(att[10]), typed_missing)
    typed_missing -= wild_used
    used += wild_used
    colorless_missing = max(0, int(cost[0]) - (total - used))
    return typed_missing + colorless_missing


def _attack_feats(f, card_id, energies, opp_active_id):
    """Derived §3 features: per top-4 attack [affordable, dmg, deficit],
    weakness/resistance vs opp active, retreat affordability."""
    T = _tables()
    att = _energy_counts(energies)
    total = len(energies)
    for k in range(4):
        aid = T["top_attacks"][card_id, k]
        if aid == 0:
            continue
        deficit = _deficit(T, att, total, aid)
        f[TokF.ATK0 + 3 * k] = float(deficit == 0)
        f[TokF.ATK0 + 3 * k + 1] = T["atk_damage"][aid] / 300.0
        f[TokF.ATK0 + 3 * k + 2] = min(deficit, 3) / 3.0
    my_type = T["energy_type"][card_id]
    if opp_active_id:
        f[TokF.WEAK_HIT] = float(T["weakness"][opp_active_id] == my_type)
        f[TokF.RES_HIT] = float(T["resistance"][opp_active_id] == my_type)
    f[TokF.RETREAT_OK] = float(total >= T["retreat"][card_id])


def _threat(att_id, att_energies, def_id, def_hp, extra_dmg):
    """Static one-ply lethal estimate: can att_id KO def_id with a top-4
    attack (base damage + weakness x2 / resistance -30 + known effect
    mods)? Returns (lethal_now, lethal_with_one_more_energy). Heuristic:
    variable/conditional damage is not modeled."""
    T = _tables()
    att = _energy_counts(att_energies)
    total = len(att_energies)
    my_type = T["energy_type"][att_id]
    now = p1 = 0.0
    for k in range(4):
        aid = T["top_attacks"][att_id, k]
        if aid == 0:
            continue
        dmg = float(T["atk_damage"][aid])
        if dmg <= 0:
            continue
        if T["weakness"][def_id] == my_type:
            dmg *= 2
        if T["resistance"][def_id] == my_type:
            dmg -= 30
        dmg += extra_dmg
        if dmg >= def_hp:
            d = _deficit(T, att, total, aid)
            if d == 0:
                now = 1.0
            if d <= 1:
                p1 = 1.0
    return now, p1


def encode(row: np.ndarray, obs: dict, picked: list[int],
           stop_allowed: bool, forced_run: int, tracker=None) -> None:
    """Fill one flat float32 row in place. obs must have a live select.
    tracker: this seat's InfoTracker (already updated with obs)."""
    row[:] = 0.0
    cur = obs["current"]
    sel = obs["select"]
    seat = cur["yourIndex"]
    me, opp = cur["players"][seat], cur["players"][1 - seat]
    fx = tracker.effects if tracker is not None else None
    turn_now = cur["turn"]
    opp_active_id, opp_active_serial, opp_active_pk = 0, None, None
    for p in opp["active"]:
        if p is not None:
            opp_active_id = p["id"]
            opp_active_serial = p.get("serial")
            opp_active_pk = p

    tok_int = _slice(row, "tok_int").reshape(MAX_TOKENS, TOK_INT)
    tok_f = _slice(row, "tok_float").reshape(MAX_TOKENS, TOK_F)
    nt = 0  # next token slot
    ptr = {}  # (playerIndex, area, index) -> token idx

    def add(card_id, zone, owner):
        nonlocal nt
        if nt >= MAX_TOKENS:
            return None
        t = nt
        tok_int[t, 0] = card_id
        tok_int[t, 1] = zone
        tok_int[t, 2] = owner
        nt += 1
        return t

    def add_pokemon(p, pidx, area, idx, zone, owner, status5, vs_active_id):
        t = add(p["id"], zone, owner)
        if t is None:
            return
        ptr[(pidx, area, idx)] = t
        f = tok_f[t]
        max_hp = p["maxHp"] or 1
        f[TokF.HP_FRAC] = p["hp"] / max_hp
        f[TokF.HP] = p["hp"] / 300.0
        f[TokF.MAX_HP] = max_hp / 300.0
        f[TokF.DMG] = (max_hp - p["hp"]) / 10.0 / 30.0
        en = p["energies"]
        f[TokF.N_ENERGY] = len(en) / 5.0
        for e in en:
            if 0 <= e < 12:
                f[TokF.ENERGY0 + e] += 1 / 3.0
        f[TokF.N_TOOLS] = len(p["tools"]) / 2.0
        f[TokF.STACK] = len(p["preEvolution"]) / 3.0
        f[TokF.APPEAR] = float(p["appearThisTurn"])
        if status5 is not None:
            f[TokF.POISONED:TokF.POISONED + 5] = status5
            f[TokF.IS_ACTIVE] = 1.0
        for k, tc in enumerate(p["tools"][:2]):
            tok_int[t, 3 + k] = tc["id"]
        if fx is not None:
            for name, val in fx.token_flags(p.get("serial"),
                                            turn_now).items():
                f[getattr(TokF, name)] = val
            if p.get("serial") in fx.evolved_now:
                f[TokF.EVOLVED_THIS_TURN] = 1.0
            if p.get("serial") in fx.healed_now:
                f[TokF.HEALED_THIS_TURN] = 1.0
        _attack_feats(f, p["id"], en, vs_active_id)

    def status(pl):
        return np.array([pl["poisoned"], pl["burned"], pl["asleep"],
                         pl["paralyzed"], pl["confused"]], dtype=np.float32)

    my_active_id, my_active_serial, my_active_pk = 0, None, None
    for p in me["active"]:
        if p is not None:
            my_active_id = p["id"]
            my_active_serial = p.get("serial")
            my_active_pk = p
    for pl, pidx, owner, az, bz, vs_id in (
            (me, seat, Owner.MINE, Zone.MY_ACTIVE, Zone.MY_BENCH, opp_active_id),
            (opp, 1 - seat, Owner.OPP, Zone.OPP_ACTIVE, Zone.OPP_BENCH, my_active_id)):
        for i, p in enumerate(pl["active"]):
            if p is not None:
                add_pokemon(p, pidx, ACTIVE, i, az, owner, status(pl), vs_id)
        for i, p in enumerate(pl["bench"]):
            add_pokemon(p, pidx, BENCH, i, bz, owner, None, vs_id)

    for i, c in enumerate(me["hand"] or []):
        t = add(c["id"], Zone.MY_HAND, Owner.MINE)
        if t is not None:
            ptr[(seat, HAND, i)] = t

    for c in cur["stadium"]:
        add(c["id"], Zone.STADIUM,
            Owner.MINE if c["playerIndex"] == seat else Owner.OPP)

    # count-collapsed zones: one token per distinct card id
    def add_collapsed(cards, zone, owner, pidx, area):
        reps = {}
        for i, c in enumerate(cards):
            cid = 0 if c is None else c["id"]
            if cid in reps:
                t = reps[cid]
                tok_f[t, TokF.COPIES] += 1 / 4.0
            else:
                t = add(cid, zone, owner)
                if t is None:
                    continue
                reps[cid] = t
                tok_f[t, TokF.COPIES] = 1 / 4.0
            if area is not None:
                ptr[(pidx, area, i)] = t

    def add_counter(counter, zone, owner):
        for cid, n in sorted(counter.items()):
            t = add(cid, zone, owner)
            if t is not None:
                tok_f[t, TokF.COPIES] = n / 4.0

    add_collapsed(me["discard"], Zone.MY_DISCARD, Owner.MINE, seat, DISCARD)
    add_collapsed(opp["discard"], Zone.OPP_DISCARD, Owner.OPP, 1 - seat, DISCARD)
    if sel.get("deck"):
        add_collapsed(sel["deck"], Zone.LOOKING, Owner.MINE, seat, DECK)
    if cur.get("looking"):
        add_collapsed(cur["looking"], Zone.LOOKING, Owner.NEUTRAL, seat, LOOKING_A)

    # tracker zones (blind, legitimately-known info only)
    n_known_hand = 0
    prizes_known = False
    if tracker is not None:
        unseen = tracker.my_unseen()
        if tracker.my_prize_known is not None:
            add_counter(tracker.my_prize_known, Zone.MY_PRIZE_KNOWN,
                        Owner.MINE)
            unseen = unseen - tracker.my_prize_known
        add_counter(unseen, Zone.MY_UNSEEN, Owner.MINE)
        known_hand = Counter(tracker.opp_known_hand.values())
        n_known_hand = sum(known_hand.values())
        add_counter(known_hand, Zone.OPP_HAND_KNOWN, Owner.OPP)
        add_counter(Counter(tracker.opp_known_deck.values()),
                    Zone.OPP_REVEALED, Owner.OPP)
        prizes_known = tracker.my_prize_known is not None

    # ---- options ----
    opt_int = _slice(row, "opt_int").reshape(MAX_OPTIONS, OPT_INT)
    opt_f = _slice(row, "opt_float").reshape(MAX_OPTIONS, OPT_F)
    mask = _slice(row, "opt_mask")
    options = sel["option"]
    picked_set = set(picked)

    def resolve_card(pidx, area, idx):
        """Card id at (player, area, index); 0 if unknown/facedown."""
        try:
            if area == HAND:
                cards = cur["players"][pidx]["hand"]
            elif area == DISCARD:
                cards = cur["players"][pidx]["discard"]
            elif area == PRIZE:
                cards = cur["players"][pidx]["prize"]
            elif area == DECK:
                cards = sel.get("deck")
            elif area == LOOKING_A:
                cards = cur.get("looking")
            elif area == ACTIVE:
                c = cur["players"][pidx]["active"][idx]
                return 0 if c is None else c["id"]
            elif area == BENCH:
                return cur["players"][pidx]["bench"][idx]["id"]
            elif area == STADIUM_A:
                return cur["stadium"][idx]["id"]
            else:
                return 0
            c = cards[idx] if cards else None
            return 0 if c is None else c["id"]
        except (IndexError, TypeError, KeyError):
            return 0

    classes = {}
    n = min(len(options), MAX_OPTIONS - 1)
    for j in range(n):
        o = options[j]
        ot = o.get("type", 0)
        area = o.get("area")
        idx = o.get("index")
        pidx = o.get("playerIndex")
        opt_int[j, 0] = ot

        card_id = o.get("cardId") or 0
        p1 = p2 = None
        if ot == OT_PLAY:
            card_id = resolve_card(seat, HAND, idx)
            p1 = ptr.get((seat, HAND, idx))
        elif ot in (OT_CARD, OT_DISCARD, OT_ABILITY):
            if not card_id:
                card_id = resolve_card(pidx if pidx is not None else seat,
                                       area, idx)
            p1 = ptr.get((pidx if pidx is not None else seat, area, idx))
        elif ot in (OT_ATTACH, OT_EVOLVE):
            card_id = resolve_card(seat, area, idx)
            p1 = ptr.get((seat, area, idx))
            p2 = ptr.get((seat, o.get("inPlayArea"), o.get("inPlayIndex")))
        elif ot in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY):
            p1 = ptr.get((pidx if pidx is not None else seat, area, idx))
        elif ot == OT_ATTACK:
            p1 = ptr.get((seat, ACTIVE, 0))
        opt_int[j, 1] = 0 if p1 is None else p1 + 1
        opt_int[j, 2] = 0 if p2 is None else p2 + 1
        opt_int[j, 3] = card_id
        opt_int[j, 4] = o.get("attackId") or 0
        opt_int[j, 5] = o.get("number") or 0

        opt_f[j, OptF.NUMBER] = (o.get("number") or 0) / 30.0
        opt_f[j, OptF.COUNT] = (o.get("count") or 0) / 4.0
        opt_f[j, OptF.SPECIAL] = (o.get("specialConditionType") or 0) / 5.0
        opt_f[j, OptF.ENERGY_IDX] = (o.get("energyIndex") or 0) / 5.0
        if j in picked_set:
            opt_f[j, OptF.PICKED] = 1.0
            if p1 is not None:
                tok_f[p1, TokF.PICKED] = 1.0
            continue  # picked indices are never legal again (engine bans dups)
        if p1 is not None:
            tok_f[p1, TokF.REFERENCED] = 1.0

        # equivalence class key
        board_target = (area in (ACTIVE, BENCH)) or o.get("inPlayArea") is not None
        if ot == OT_SKILL:
            key = (ot, card_id, o.get("serial"))
        elif board_target or ot in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY):
            key = (ot, area, idx, pidx, o.get("inPlayArea"), o.get("inPlayIndex"),
                   card_id, o.get("count"),
                   resolve_card(pidx if pidx is not None else seat, area, idx)
                   if ot in (OT_TOOL_CARD, OT_ENERGY_CARD, OT_ENERGY) else None)
        elif area in _CARD_ZONES or ot == OT_PLAY:
            key = (ot, area, pidx, card_id)
        else:
            key = (ot, o.get("number"), o.get("attackId"),
                   o.get("specialConditionType"))
        if key not in classes:
            classes[key] = j
            mask[j] = 1.0

    if stop_allowed:
        mask[STOP] = 1.0
        opt_int[STOP, 0] = OT_STOP   # own type embedding, not NUMBER's (id 0)
        opt_f[STOP, OptF.IS_STOP] = 1.0

    # ---- global ----
    g = _slice(row, "global_f")
    first = cur["firstPlayer"]
    g[GlobF.TURN] = cur["turn"] / 50.0
    g[GlobF.PARITY] = cur["turn"] % 2
    g[GlobF.I_AM_FIRST] = float(first == seat)
    g[GlobF.FIRST_DECIDED] = float(first != -1)
    g[GlobF.ACTIONS] = cur["turnActionCount"] / 20.0
    g[GlobF.SUPPORTER] = float(cur["supporterPlayed"])
    g[GlobF.STADIUM_PLAYED] = float(cur["stadiumPlayed"])
    g[GlobF.ENERGY_ATTACHED] = float(cur["energyAttached"])
    g[GlobF.RETREATED] = float(cur["retreated"])
    g[GlobF.MY_PRIZES] = len(me["prize"]) / 6.0
    g[GlobF.OPP_PRIZES] = len(opp["prize"]) / 6.0
    g[GlobF.MY_DECK] = me["deckCount"] / 60.0
    g[GlobF.OPP_DECK] = opp["deckCount"] / 60.0
    g[GlobF.MY_HAND] = me["handCount"] / 20.0
    g[GlobF.OPP_HAND] = opp["handCount"] / 20.0
    g[GlobF.MY_BENCH] = len(me["bench"]) / 5.0
    g[GlobF.OPP_BENCH] = len(opp["bench"]) / 5.0
    g[GlobF.BENCH_MAX] = me["benchMax"] / 8.0
    g[GlobF.STADIUM_PRESENT] = float(bool(cur["stadium"]))
    g[GlobF.MY_STATUS0:GlobF.MY_STATUS0 + 5] = status(me)
    g[GlobF.OPP_STATUS0:GlobF.OPP_STATUS0 + 5] = status(opp)
    g[GlobF.PICKS_MIN_LEFT] = max(0, sel["minCount"] - len(picked)) / 5.0
    g[GlobF.PICKS_MAX_LEFT] = max(0, sel["maxCount"] - len(picked)) / 10.0
    g[GlobF.ACTOR_SEAT] = float(seat)
    g[GlobF.OPP_HAND_UNKNOWN] = max(0, opp["handCount"] - n_known_hand) / 20.0
    g[GlobF.MY_PRIZES_KNOWN] = float(prizes_known)
    if fx is not None:
        my_locks, my_boost = fx.player_flags(seat, turn_now)
        op_locks, op_boost = fx.player_flags(1 - seat, turn_now)
        g[GlobF.MY_LOCK_ITEM] = my_locks.get("ITEM", 0.0)
        g[GlobF.MY_LOCK_SUPPORTER] = my_locks.get("SUPPORTER", 0.0)
        g[GlobF.MY_LOCK_EVOLVE] = my_locks.get("EVOLVE", 0.0)
        g[GlobF.OPP_LOCK_ITEM] = op_locks.get("ITEM", 0.0)
        g[GlobF.OPP_LOCK_SUPPORTER] = op_locks.get("SUPPORTER", 0.0)
        g[GlobF.OPP_LOCK_EVOLVE] = op_locks.get("EVOLVE", 0.0)
        g[GlobF.MY_DMG_BOOST] = my_boost
        g[GlobF.OPP_DMG_BOOST] = op_boost
        g[GlobF.MY_ENERGY_REMOVED] = fx.energy_removed[seat] / 10.0
        g[GlobF.OPP_ENERGY_REMOVED] = fx.energy_removed[1 - seat] / 10.0
        g[GlobF.MY_TOOLS_REMOVED] = fx.tools_removed[seat] / 5.0
        g[GlobF.OPP_TOOLS_REMOVED] = fx.tools_removed[1 - seat] / 5.0
        g[GlobF.MY_FORCED_SWITCHES] = fx.forced_switches[seat] / 5.0
        g[GlobF.OPP_FORCED_SWITCHES] = fx.forced_switches[1 - seat] / 5.0
        g[GlobF.MY_KOD_LAST_TURN] = float(
            fx.last_ko_turn[seat] >= turn_now - 1)
        g[GlobF.OPP_KOD_LAST_TURN] = float(
            fx.last_ko_turn[1 - seat] >= turn_now - 1)
    if my_active_pk is not None and opp_active_pk is not None:
        myf = fx.token_flags(my_active_serial, turn_now) if fx else {}
        opf = fx.token_flags(opp_active_serial, turn_now) if fx else {}
        g[GlobF.MY_ACTIVE_IN_DANGER], g[GlobF.MY_ACTIVE_IN_DANGER_P1] = \
            _threat(opp_active_id, opp_active_pk["energies"],
                    my_active_id, my_active_pk["hp"],
                    (opf.get("EFF_DMG_DEALT_MOD", 0.0)
                     + myf.get("EFF_DMG_TAKEN_MOD", 0.0)) * 100.0)
        g[GlobF.OPP_ACTIVE_IN_DANGER], g[GlobF.OPP_ACTIVE_IN_DANGER_P1] = \
            _threat(my_active_id, my_active_pk["energies"],
                    opp_active_id, opp_active_pk["hp"],
                    (myf.get("EFF_DMG_DEALT_MOD", 0.0)
                     + opf.get("EFF_DMG_TAKEN_MOD", 0.0)) * 100.0)

    # ---- decision ----
    di = _slice(row, "dec_int")
    df = _slice(row, "dec_float")
    di[DecInt.SELECT_TYPE] = sel["type"]
    di[DecInt.CONTEXT] = sel["context"]
    eff, cc = sel.get("effect"), sel.get("contextCard")
    di[DecInt.EFFECT_CARD] = eff["id"] if eff else 0
    di[DecInt.CONTEXT_CARD] = cc["id"] if cc else 0
    if fx is not None:
        di[DecInt.MY_LAST_ATTACK] = fx.last_attack_id(seat, my_active_serial)
        di[DecInt.OPP_LAST_ATTACK] = fx.last_attack_id(
            1 - seat, opp_active_serial)
        di[DecInt.MY_SUPPORTER] = fx.supporter_now[seat]
        di[DecInt.OPP_SUPPORTER] = fx.supporter_prev[1 - seat]
    df[DecF.MIN_COUNT] = sel["minCount"] / 5.0
    df[DecF.MAX_COUNT] = sel["maxCount"] / 10.0
    df[DecF.N_OPTIONS] = len(options) / 30.0
    df[DecF.REMAIN_DMG] = sel["remainDamageCounter"] / 10.0
    df[DecF.REMAIN_ENERGY] = sel["remainEnergyCost"] / 5.0
    df[DecF.PICKED] = len(picked) / 5.0
    df[DecF.FORCED_RUN] = forced_run / 5.0


def write_oracle(row: np.ndarray, opp_hand: Counter, opp_deck: Counter,
                 opp_prize: Counter, my_prize: Counter) -> None:
    """Fill the §6b oracle block (critic-only truth from the env ledger):
    opp hand (id, count), opp deck/prize split (id, deck_n, prize_n),
    my exact prizes (id, count). Call AFTER encode() (which zeroes the row)."""
    o = _slice(row, "oracle")
    i = 0
    for cid, n in sorted(opp_hand.items())[:ORACLE_HAND_SLOTS]:
        o[i], o[i + 1] = cid, n / 4.0
        i += 2
    i = ORACLE_HAND_SLOTS * 2
    ids = sorted(set(opp_deck) | set(opp_prize))[:ORACLE_SPLIT_SLOTS]
    for cid in ids:
        o[i], o[i + 1], o[i + 2] = cid, opp_deck[cid] / 4.0, opp_prize[cid] / 4.0
        i += 3
    i = ORACLE_HAND_SLOTS * 2 + ORACLE_SPLIT_SLOTS * 3
    for cid, n in sorted(my_prize.items())[:ORACLE_PRIZE_SLOTS]:
        o[i], o[i + 1] = cid, n / 4.0
        i += 2
