"""Lingering-effect tracker: hidden temporal state from card text.

The engine never exposes effects that persist after an action resolves
("During your next turn, this Pokémon can't attack"); they only show up
as options silently missing from selects. This module registers that
state from the log stream using data/lingering_effects.json (parsed
card text, docs/model.md) so the encoder can expose it.

One EffectTracker per seat, embedded in InfoTracker (tracker.effects)
and fed the same per-seat obs stream. Both seats see the same
ATTACK/PLAY/COIN/SWITCH logs, so each reconstructs the same objective
state; perspective is applied at encode time.

Scope (v1): attack-sourced records (both seats) + item/supporter
records (PLAY log). Ability-sourced records are SKIPPED — the Log enum
has no ability-used event. Records needing an owner-chosen target and
classes with no obs slot (cost mods, ability locks, energy-attach
locks) are also skipped. Counts of everything skipped are in
TABLE_STATS after _build_tables().

Timing convention: an instance binds on turns [bind_from, bind_to]
(engine turn numbers, owner-relative durations resolved at
registration). Flag value is 1.0 while binding, 0.5 while registered
but not yet binding (the net sees it coming). COIN_HEADS conditions
resolve from the first COIN log inside the same attack window;
COIN_TAILS marks per-attempt blocks (Sand Attack family) — registered
unconditionally at half strength, never legality-exact.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

# LogType / AreaType ints (data/cg/api.py)
_TURN_START = 2
_TURN_END = 3
_MOVE_CARD = 6
_SWITCH = 8
_CHANGE = 9
_PLAY = 10
_EVOLVE = 12
_DEVOLVE = 13
_ATTACK = 15
_COIN = 22
_A_DECK, _A_HAND, _A_DISCARD, _A_ACTIVE, _A_BENCH = 1, 2, 3, 4, 5
_A_ENERGY, _A_TOOL = 8, 9

_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "lingering_effects.json")

# effect_class -> (kind, token-flag name | player-lock name, sign)
_TOKEN_CLASSES = {
    "CANT_ATTACK": "EFF_CANT_ATTACK",
    "CANT_USE_SPECIFIC_ATTACK": "EFF_ATTACK_LOCKED",
    "CANT_RETREAT": "EFF_CANT_RETREAT",
    "PREVENT_ALL_DAMAGE_AND_EFFECTS": "EFF_PREVENT_ALL",
    "PREVENT_DAMAGE": "EFF_PREVENT_ALL",
    "REDUCE_DAMAGE_TAKEN": "EFF_DMG_TAKEN_MOD",
    "INCREASE_DAMAGE_TAKEN": "EFF_DMG_TAKEN_MOD",
    "BOOST_DAMAGE_DEALT": "EFF_DMG_DEALT_MOD",
    "REDUCE_DAMAGE_DEALT": "EFF_DMG_DEALT_MOD",
    "DELAYED_DAMAGE": "EFF_DELAYED_DMG",
}
_PLAYER_CLASSES = {
    "CANT_PLAY_ITEM": "ITEM",
    "CANT_PLAY_SUPPORTER": "SUPPORTER",
    "CANT_PLAY_POKEMON_FROM_HAND": "EVOLVE",
    "CANT_EVOLVE": "EVOLVE",
}

_tables = None
TABLE_STATS: dict = {}


def _norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"[’‘]", "'", s).strip().lower()


def _build_tables():
    """attack_fx: attackId -> [record]; play_fx: cardId -> [record];
    supporters: set of supporter cardIds (for within-turn memory)."""
    global _tables
    if _tables is not None:
        return _tables
    from cg.api import all_card_data, all_attack, CardType

    recs = json.load(open(_JSON))["records"]
    cards = {c.cardId: c for c in all_card_data()}
    attacks = {a.attackId: a for a in all_attack()}
    by_card_attack = {
        c.cardId: {_norm(attacks[aid].name): aid for aid in c.attacks}
        for c in cards.values()}

    attack_fx, play_fx = {}, {}
    stats = {"attack": 0, "play": 0, "skip_ability": 0, "skip_class": 0,
             "skip_target": 0, "unmatched_attack": []}
    for r in recs:
        cls = r["effect_class"]
        if r["source_kind"] == "ability":
            stats["skip_ability"] += 1
            continue
        is_ko = cls == "OTHER" and "knocked out" in _norm(
            r["clause_text"] + r.get("notes", ""))
        if cls not in _TOKEN_CLASSES and cls not in _PLAYER_CLASSES \
                and not is_ko:
            stats["skip_class"] += 1
            continue
        if r["target"] in ("OTHER", "ALL_MY_POKEMON", "ALL_OPP_POKEMON") \
                and cls in _TOKEN_CLASSES and cls not in (
                    "BOOST_DAMAGE_DEALT", "REDUCE_DAMAGE_DEALT"):
            stats["skip_target"] += 1
            continue
        rec = dict(r)
        rec["flag"] = "EFF_DELAYED_KO" if is_ko else _TOKEN_CLASSES.get(cls)
        rec["lock"] = _PLAYER_CLASSES.get(cls)
        if r["source_kind"] == "attack":
            aid = by_card_attack.get(r["card_id"], {}).get(
                _norm(r["source_name"] or ""))
            if aid is None:
                stats["unmatched_attack"].append(
                    (r["card_id"], r["source_name"]))
                continue
            attack_fx.setdefault(aid, []).append(rec)
            stats["attack"] += 1
        else:  # item / supporter / stadium / tool / energy via PLAY log
            play_fx.setdefault(r["card_id"], []).append(rec)
            stats["play"] += 1

    supporters = {c.cardId for c in cards.values()
                  if c.cardType == CardType.SUPPORTER}
    pokemon_ids = {c.cardId for c in cards.values()
                   if c.cardType == CardType.POKEMON}
    # standing effect-prevention shields (e.g. TR Articuno "Repelling
    # Veil"): while one is on the defender's board, incoming attack
    # effects MAY be nullified — we don't model scope, so registrations
    # against that side demote to soft (validated: every observed hard
    # cant-retreat violation was a shielded TR Pokemon)
    shields = {c.cardId for c in cards.values()
               if any("prevent all effects of" in _norm(s.text)
                      or "prevent all damage from and effects" in
                      _norm(s.text) for s in c.skills)}
    stats["shield_cards"] = len(shields)
    TABLE_STATS.update(stats)
    _tables = (attack_fx, play_fx, supporters, shields, pokemon_ids)
    return _tables


class EffectTracker:
    def __init__(self):
        self.turn = 0
        self.turn_drift = 0     # times log-replayed turn != engine turn
        self.first_player = -1
        self.actives = [None, None]        # running active serial per seat
        self.serial_fx: dict[int, list] = {}   # serial -> [instance]
        self.player_locks: list[list] = [[], []]  # per seat: [instance]
        self.dmg_boost: list[list] = [[], []]     # per seat (magnitude insts)
        self.last_attack = [(0, None), (0, None)]  # (attackId, serial)
        self.supporter_now = [0, 0]        # supporter played this turn
        self.supporter_prev = [0, 0]       # supporter played on last own turn
        self.energy_removed = [0, 0]       # mine removed during opp turns
        self.tools_removed = [0, 0]
        self.forced_switches = [0, 0]
        self.evolved_now: set = set()      # serials evolved this turn
        self.healed_now: set = set()       # serials healed this turn
        self.last_ko_turn = [-9, -9]       # turn each side last lost a mon

    def clone(self) -> "EffectTracker":
        t = EffectTracker.__new__(EffectTracker)
        t.turn, t.first_player = self.turn, self.first_player
        t.turn_drift = self.turn_drift
        t.actives = list(self.actives)
        t.serial_fx = {s: list(v) for s, v in self.serial_fx.items()}
        t.player_locks = [list(v) for v in self.player_locks]
        t.dmg_boost = [list(v) for v in self.dmg_boost]
        t.last_attack = list(self.last_attack)
        t.supporter_now = list(self.supporter_now)
        t.supporter_prev = list(self.supporter_prev)
        t.energy_removed = list(self.energy_removed)
        t.tools_removed = list(self.tools_removed)
        t.forced_switches = list(self.forced_switches)
        t.evolved_now = set(self.evolved_now)
        t.healed_now = set(self.healed_now)
        t.last_ko_turn = list(self.last_ko_turn)
        return t

    # ------------------------------------------------------------ timing
    def _turn_player(self, t):
        if self.first_player < 0:
            return -1
        return self.first_player if t % 2 == 1 else 1 - self.first_player

    def _next_turn_of(self, seat, t):
        return t + 1 if self._turn_player(t + 1) == seat else t + 2

    def _bind_range(self, rec, owner, t):
        d = rec["duration"]
        if d == "SAME_TURN":
            return t, t
        if d == "MY_NEXT_TURN":
            n = self._next_turn_of(owner, t)
            return n, n
        if d == "OPP_NEXT_TURN":
            n = self._next_turn_of(1 - owner, t)
            return n, n
        if d == "UNTIL_END_OF_MY_NEXT_TURN":
            return t, self._next_turn_of(owner, t)
        if d == "UNTIL_END_OF_OPP_NEXT_TURN":
            return t, self._next_turn_of(1 - owner, t)
        if d == "WHILE_CONDITION":
            return t, None
        return t, t + 2                      # OTHER: coarse

    # ---------------------------------------------------------- registration
    def _register(self, rec, owner, defender, t, soft=False, src=None):
        a, b = self._bind_range(rec, owner, t)
        inst = {"rec": rec, "from": a, "to": b, "soft": soft, "src": src}
        if rec["lock"]:
            tgt = 1 - owner if rec["target"] == "OPP_PLAYER" else owner
            self.player_locks[tgt].append(dict(inst, kind=rec["lock"]))
            return
        if rec["flag"] is None:
            return
        if rec["target"] == "SELF":
            serial = self.last_attack[owner][1] if rec["source_kind"] == \
                "attack" else None
        elif rec["target"] == "DEFENDING":
            serial = defender
        elif rec["target"] in ("ALL_MY_POKEMON", "ALL_OPP_POKEMON") \
                and rec["flag"] == "EFF_DMG_DEALT_MOD":
            tgt = owner if rec["target"] == "ALL_MY_POKEMON" else 1 - owner
            self.dmg_boost[tgt].append(inst)
            return
        else:
            serial = None
        if serial is not None:
            self.serial_fx.setdefault(serial, []).append(inst)

    def _clear_serial(self, serial):
        self.serial_fx.pop(serial, None)

    # ---------------------------------------------------------------- update
    def update(self, obs: dict) -> None:
        cur = obs.get("current")
        if cur is None:
            return
        if self.first_player < 0:
            self.first_player = cur.get("firstPlayer", -1)
        logs = obs.get("logs") or []
        attack_fx, play_fx, supporters, shields, pokemon_ids = \
            _build_tables()
        # side-wide: a board Pokemon whose ABILITY is an effect shield
        # (scope unparsed -> whole side soft); per-serial: a shield
        # ATTACHMENT (Mist/Rock Fighting Energy) protects its holder
        shielded = [False, False]
        holder_shield = set()
        for s in (0, 1):
            for pk in (cur["players"][s].get("active") or []) + \
                    (cur["players"][s].get("bench") or []):
                if pk is None:
                    continue
                if pk.get("id") in shields:
                    shielded[s] = True
                if any(c.get("id") in shields for c in
                       (pk.get("energyCards") or []) +
                       (pk.get("tools") or [])):
                    holder_shield.add(pk.get("serial"))
        t = self.turn

        for i, lg in enumerate(logs):
            ty = lg.get("type")
            p = lg.get("playerIndex")
            if ty == _TURN_START:
                t += 1
                self.evolved_now.clear()
                self.healed_now.clear()
            elif ty == _TURN_END:
                ended = self._turn_player(t)
                if ended in (0, 1) and self.supporter_now[ended]:
                    self.supporter_prev[ended] = self.supporter_now[ended]
                    self.supporter_now[ended] = 0
            elif ty == _MOVE_CARD:
                serial, fa, ta = lg.get("serial"), lg.get("fromArea"), \
                    lg.get("toArea")
                if ta == _A_ACTIVE:
                    self.actives[p] = serial
                elif fa == _A_ACTIVE:
                    if self.actives[p] == serial:
                        self.actives[p] = None
                    self._clear_serial(serial)
                elif fa == _A_BENCH and ta in (_A_DISCARD, _A_HAND, _A_DECK):
                    self._clear_serial(serial)
                if fa in (_A_ENERGY, _A_TOOL) and ta == _A_DISCARD \
                        and p in (0, 1) and self._turn_player(t) == 1 - p:
                    if fa == _A_ENERGY:
                        self.energy_removed[p] += 1
                    else:
                        self.tools_removed[p] += 1
                if fa in (_A_ACTIVE, _A_BENCH) and ta == _A_DISCARD \
                        and p in (0, 1) and lg.get("cardId") in pokemon_ids:
                    self.last_ko_turn[p] = t
            elif ty == _SWITCH:
                old, new = lg.get("serialActive"), lg.get("serialBench")
                self.actives[p] = new
                self._clear_serial(old)      # standard rule: switching sheds
                if self._turn_player(t) == 1 - p:
                    self.forced_switches[p] += 1
            elif ty == _CHANGE:
                self.actives[p] = lg.get("serialAfter")
                self._clear_serial(lg.get("serialBefore"))
            elif ty in (_EVOLVE, _DEVOLVE):
                old, new = lg.get("serialTarget"), lg.get("serial")
                if self.actives[p] == old:
                    self.actives[p] = new
                self._clear_serial(old)      # evolving sheds attack effects
                if ty == _EVOLVE:
                    self.evolved_now.add(new)
            elif ty == 16 and (lg.get("value") or 0) > 0:   # HP_CHANGE heal
                self.healed_now.add(lg.get("serial"))
            elif ty == _ATTACK:
                aid, serial = lg.get("attackId"), lg.get("serial")
                self.last_attack[p] = (aid, serial)
                for rec in attack_fx.get(aid, ()):
                    cond, ok, soft = rec["condition"], True, False
                    if cond == "COIN_HEADS":
                        ok = self._coin_after(logs, i)
                    elif cond in ("COIN_TAILS", "OTHER"):
                        soft = True
                    defender = self.actives[1 - p]
                    if rec["target"] == "DEFENDING" and (
                            shielded[1 - p] or defender in holder_shield):
                        soft = True     # possible shield nullify
                    if ok:
                        self._register(rec, p, self.actives[1 - p], t,
                                       soft=soft, src=serial)
            elif ty == _PLAY:
                cid = lg.get("cardId")
                if cid in supporters:
                    self.supporter_now[p] = cid
                for rec in play_fx.get(cid, ()):
                    soft = rec["condition"] != "NONE"
                    self._register(rec, p, self.actives[1 - p], t, soft=soft)

        if t != cur["turn"]:
            self.turn_drift += 1
        self.turn = cur["turn"]              # engine truth at decision time
        self._expire(self.turn)
        # re-seed running actives from obs ground truth so any mid-batch
        # replay error cannot persist past this decision point
        for seat in (0, 1):
            for pk in cur["players"][seat].get("active") or []:
                if pk is not None:
                    self.actives[seat] = pk.get("serial")

    @staticmethod
    def _coin_after(logs, i):
        for lg in logs[i + 1:]:
            ty = lg.get("type")
            if ty == _COIN:
                return bool(lg.get("head"))
            if ty in (_ATTACK, _TURN_END, _PLAY):
                return False
        return False

    def _expire(self, t):
        for serial in list(self.serial_fx):
            keep = [x for x in self.serial_fx[serial]
                    if x["to"] is None or t <= x["to"]]
            if keep:
                self.serial_fx[serial] = keep
            else:
                del self.serial_fx[serial]
        for seat in (0, 1):
            self.player_locks[seat] = [
                x for x in self.player_locks[seat]
                if x["to"] is None or t <= x["to"]]
            self.dmg_boost[seat] = [
                x for x in self.dmg_boost[seat]
                if x["to"] is None or t <= x["to"]]

    # ---------------------------------------------------------------- queries
    def _strength(self, inst, t):
        v = 1.0 if inst["from"] <= t and (inst["to"] is None
                                          or t <= inst["to"]) else 0.5
        return v * (0.5 if inst["soft"] else 1.0)

    def token_flags(self, serial, t) -> dict:
        """flag-name -> value for one board Pokemon at turn t."""
        out = {}
        for inst in self.serial_fx.get(serial, ()):
            rec, s = inst["rec"], self._strength(inst, t)
            f, mag = rec["flag"], rec.get("magnitude") or 0
            if f == "EFF_DMG_TAKEN_MOD":
                sign = 1 if rec["effect_class"] == "INCREASE_DAMAGE_TAKEN" \
                    else -1
                out[f] = out.get(f, 0.0) + sign * s * mag / 100.0
            elif f == "EFF_DMG_DEALT_MOD":
                sign = -1 if rec["effect_class"] == "REDUCE_DAMAGE_DEALT" \
                    else 1
                out[f] = out.get(f, 0.0) + sign * s * mag / 100.0
            elif f == "EFF_DELAYED_DMG":
                out[f] = max(out.get(f, 0.0), s * mag / 10.0)
            else:
                out[f] = max(out.get(f, 0.0), s)
        return out

    def player_flags(self, seat, t) -> tuple[dict, float]:
        """(lock-kind -> value, dmg_boost value) for one seat at turn t."""
        locks = {}
        for inst in self.player_locks[seat]:
            locks[inst["kind"]] = max(locks.get(inst["kind"], 0.0),
                                      self._strength(inst, t))
        boost = sum(self._strength(x, t) * (x["rec"].get("magnitude") or 0)
                    / 100.0 for x in self.dmg_boost[seat])
        return locks, boost

    def last_attack_id(self, seat, active_serial):
        aid, serial = self.last_attack[seat]
        return aid if serial is not None and serial == active_serial else 0
