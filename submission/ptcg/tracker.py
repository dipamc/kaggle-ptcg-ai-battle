"""Public-information tracker: the inferred card lists.

Maintains, from the blind observation stream of ONE game, everything that is
deducible without cheating:
  - my_unseen:      multiset of my cards in (deck ∪ prizes) = my 60 minus seen
  - opp_visible:    multiset of opponent cards visible in open zones
  - opp_known_hand: serial -> cardId for opponent hand cards whose identity
                    was revealed by logs (effect reveals; normal draws are
                    DRAW_REVERSE and stay unknown)
  - opp_known_deck: serial -> cardId known to be in opponent's deck
                    (revealed cards shuffled/returned to deck)

Serial-exact where possible: every physical card has a unique serial and
logs carry serials, so hand knowledge never goes stale.

Usage (agent-side): tr = InfoTracker(my_deck_60); tr.update(obs) each call.
"""
from __future__ import annotations

from collections import Counter

from .effects import EffectTracker

# LogType ints (data/cg/api.py)
_MOVE_CARD = 6
_AREA_HAND = 2
_AREA_DECK = 1


def _visible_player_cards(player: dict) -> Counter:
    """Multiset of card ids visible in a player's open zones."""
    seen: Counter = Counter()
    for pk in (player.get("active") or []) + (player.get("bench") or []):
        if pk is None:
            continue
        seen[pk["id"]] += 1
        for c in (pk.get("energyCards") or []) + (pk.get("tools") or []) \
                + (pk.get("preEvolution") or []):
            seen[c["id"]] += 1
    for c in player.get("discard") or []:
        seen[c["id"]] += 1
    # prizes: only face-up (non-null) entries are knowledge
    for c in player.get("prize") or []:
        if c is not None:
            seen[c["id"]] += 1
    return seen


class InfoTracker:
    def __init__(self, my_deck: list[int]):
        assert len(my_deck) == 60
        self.my_deck_list = list(my_deck)
        self.opp_known_hand: dict[int, int] = {}   # serial -> cardId
        self.opp_known_deck: dict[int, int] = {}   # serial -> cardId
        # exact prize multiset, learned the first time a full deck search
        # reveals my deck (prizes = unseen pool minus shown deck); None
        # until learned or after an unknown card enters the prize zone
        self.my_prize_known: Counter | None = None
        self.effects = EffectTracker()      # lingering-effect state (v4 obs)
        self._last_obs: dict | None = None

    def clone(self) -> "InfoTracker":
        """Cheap copy for simulation rollouts: knowledge dicts are
        copied, obs references shared (obs dicts are replaced on
        update, never mutated)."""
        t = InfoTracker.__new__(InfoTracker)
        t.my_deck_list = self.my_deck_list
        t.opp_known_hand = dict(self.opp_known_hand)
        t.opp_known_deck = dict(self.opp_known_deck)
        t.my_prize_known = None if self.my_prize_known is None \
            else Counter(self.my_prize_known)
        t.effects = self.effects.clone()
        t._last_obs = self._last_obs
        return t

    # ------------------------------------------------------------------ logs
    def _consume_logs(self, logs: list[dict], opp_index: int) -> None:
        for lg in logs:
            if lg.get("type") != _MOVE_CARD or lg.get("playerIndex") != opp_index:
                continue
            serial, card_id = lg.get("serial"), lg.get("cardId")
            if serial is None or not card_id:
                continue
            to_area, from_area = lg.get("toArea"), lg.get("fromArea")
            if to_area == _AREA_HAND:
                self.opp_known_hand[serial] = card_id
                self.opp_known_deck.pop(serial, None)
            elif from_area == _AREA_HAND:
                self.opp_known_hand.pop(serial, None)
            if to_area == _AREA_DECK:
                self.opp_known_deck[serial] = card_id
            elif from_area == _AREA_DECK:
                self.opp_known_deck.pop(serial, None)

    # ---------------------------------------------------------------- update
    def update(self, obs: dict) -> None:
        """Consume one observation (dict form, as the agent receives it)."""
        cur = obs.get("current")
        if cur is None:                      # initial deck-selection call
            return
        you = cur["yourIndex"]
        self._consume_logs(obs.get("logs") or [], 1 - you)
        self._maintain_prize_knowledge(obs.get("logs") or [], you)
        self.effects.update(obs)
        self._last_obs = obs
        self._learn_from_deck_reveal(obs, you)

    def _maintain_prize_knowledge(self, logs: list[dict], you: int) -> None:
        """Keep my_prize_known current: prizes leave via prize-taking
        (logged with cardId); unknown cards entering the prize zone
        invalidate the knowledge."""
        if self.my_prize_known is None:
            return
        _PRIZE = 6
        for lg in logs:
            if lg.get("playerIndex") != you:
                continue
            t = lg.get("type")
            if t == _MOVE_CARD and lg.get("fromArea") == _PRIZE and lg.get("cardId"):
                self.my_prize_known[lg["cardId"]] -= 1
                self.my_prize_known += Counter()          # drop zeros
            elif lg.get("toArea") == _PRIZE:
                if t == _MOVE_CARD and lg.get("cardId"):
                    self.my_prize_known[lg["cardId"]] += 1
                else:                                     # face-down / unknown
                    self.my_prize_known = None
                    return

    def _learn_from_deck_reveal(self, obs: dict, you: int) -> None:
        """A select that shows a full deck list pins down exact contents.
        Mine: prizes = unseen pool minus shown deck (persists). Opponent's:
        every shown card enters opp_known_deck (serial-exact)."""
        sel = obs.get("select") or {}
        shown = sel.get("deck")
        if not shown:
            return
        mine = [c for c in shown if c["playerIndex"] == you]
        theirs = [c for c in shown if c["playerIndex"] != you]
        for c in theirs:
            if c.get("serial") is not None:
                self.opp_known_deck[c["serial"]] = c["id"]
        me = obs["current"]["players"][you]
        if mine and len(mine) == me["deckCount"]:         # full deck shown
            inferred = self.my_unseen() - Counter(c["id"] for c in mine)
            n_down = sum(1 for c in (me.get("prize") or []) if c is None)
            if sum(inferred.values()) == n_down:          # sanity guard
                self.my_prize_known = inferred

    # ------------------------------------------------------------ properties
    @property
    def obs(self) -> dict:
        assert self._last_obs is not None, "update() never called"
        return self._last_obs

    @property
    def you(self) -> int:
        return self.obs["current"]["yourIndex"]

    def my_unseen(self) -> Counter:
        """My (deck ∪ prizes) multiset: 60-list minus everything visible.

        Mid-effect limbo: while a played trainer resolves, it has left the
        hand but not yet reached the discard — invisible to every zone. If
        the resolving card (select.effect) is mine and its serial is not
        visible anywhere, subtract one copy (verified: without this, prize
        inference gains exactly one phantom card)."""
        cur = self.obs["current"]
        me = cur["players"][self.you]
        seen = _visible_player_cards(me)
        for c in me.get("hand") or []:
            seen[c["id"]] += 1
        stadium = cur.get("stadium") or []
        for c in stadium:
            if c.get("playerIndex") == self.you:
                seen[c["id"]] += 1
        unseen = Counter(self.my_deck_list) - seen

        eff = (self.obs.get("select") or {}).get("effect")
        if eff and eff.get("playerIndex") == self.you and unseen.get(eff["id"], 0) > 0 \
                and not self._serial_visible(me, cur, eff.get("serial")):
            unseen[eff["id"]] -= 1
            unseen += Counter()
        return unseen

    def _serial_visible(self, me: dict, cur: dict, serial) -> bool:
        if serial is None:
            return False
        for pk in (me.get("active") or []) + (me.get("bench") or []):
            if pk is None:
                continue
            if pk.get("serial") == serial:
                return True
            for c in (pk.get("energyCards") or []) + (pk.get("tools") or []) \
                    + (pk.get("preEvolution") or []):
                if c.get("serial") == serial:
                    return True
        for c in (me.get("hand") or []) + (me.get("discard") or []):
            if c.get("serial") == serial:
                return True
        for c in cur.get("stadium") or []:
            if c.get("serial") == serial:
                return True
        return False

    def opp_visible(self) -> Counter:
        """Opponent cards visible in open zones (their board/discard/prize-ups)."""
        cur = self.obs["current"]
        opp = cur["players"][1 - self.you]
        seen = _visible_player_cards(opp)
        for c in cur.get("stadium") or []:
            if c.get("playerIndex") == 1 - self.you:
                seen[c["id"]] += 1
        return seen

    def opp_revealed(self) -> Counter:
        """Everything the opponent has shown this game: visible + known hand
        + known-in-deck. Input to archetype matching."""
        out = self.opp_visible()
        for cid in self.opp_known_hand.values():
            out[cid] += 1
        for cid in self.opp_known_deck.values():
            out[cid] += 1
        return out

    def opp_counts(self) -> tuple[int, int, int]:
        """(hand_count, deck_count, prize_facedown_count) for the opponent."""
        opp = self.obs["current"]["players"][1 - self.you]
        n_prize_down = sum(1 for c in (opp.get("prize") or []) if c is None)
        return opp["handCount"], opp["deckCount"], n_prize_down
