"""Env-side oracle ledger: ground-truth hidden state for the critic (§6b).

Truth sources, in order of cost:
- The env sampled both 60-card decks — full multisets known a priori.
- One VisualizeData parse at game start (fast: history is only setup)
  pins the initial prize six for both seats. Per-step VisualizeData was
  measured at 26-38 ms/call and rejected.
- Each seat's own observation stream: whenever seat S acts, S's hand is
  exact; S's prize multiset is maintained from S's own logs (prize
  takes carry cardId in the owner's stream).

Between a seat's decisions its hand can change invisibly (turn-start
draw, effect draws): the delta vs the public handCount is emitted as
UNK_CARD rows and self-corrects at that seat's next decision. Deck is
derived: 60 - hand - visible zones - prizes (multiset-exact whenever
hand is exact, since shuffles never move cards across zone boundaries).

Structurally separate from InfoTracker: the tracker feeds BLIND tokens
(legitimate knowledge only); this ledger feeds ONLY the oracle block.
"""
from collections import Counter

from .buffers import UNK_CARD


def _visible_out_of_deck(cur: dict, pidx: int) -> Counter:
    """Cards of player pidx visibly outside deck/hand/prizes: board
    stacks (with attachments), discard, and their stadium if in play.
    Prizes are deliberately EXCLUDED — the ledger tracks them separately
    (importing the tracker's version would double-subtract face-up
    prizes and miss the stadium)."""
    pl = cur["players"][pidx]
    seen: Counter = Counter()
    for pk in (pl.get("active") or []) + (pl.get("bench") or []):
        if pk is None:
            continue
        seen[pk["id"]] += 1
        for c in (pk.get("energyCards") or []) + (pk.get("tools") or []) \
                + (pk.get("preEvolution") or []):
            seen[c["id"]] += 1
    for c in pl.get("discard") or []:
        seen[c["id"]] += 1
    for c in cur.get("stadium") or []:
        if c["playerIndex"] == pidx:
            seen[c["id"]] += 1
    return seen

_MOVE_CARD = 6
_PRIZE = 6


def _frames(viz) -> list:
    if isinstance(viz, list):
        return viz
    for key in ("entry", "frames", "visualize"):
        if isinstance(viz, dict) and key in viz:
            return viz[key]
    raise ValueError(f"unrecognized visualize payload: {type(viz)}")


class OracleLedger:
    def __init__(self, decks: tuple[list[int], list[int]], handle):
        self.decks = [Counter(decks[0]), Counter(decks[1])]
        self.hand = [Counter(), Counter()]      # last-exact snapshot per seat
        self.prize = [Counter(), Counter()]     # exact once frozen
        self.handle = handle
        # VisualizeData is EMPTY at battle start and prizes are dealt (and
        # possibly re-dealt around mulligans) during setup — so re-parse the
        # omniscient last frame while turn==0 (<1 ms that early) and freeze
        # at turn 1, after which prize moves are log-maintained.
        self._frozen = False

    def _parse_prizes(self) -> bool:
        frames = _frames(self.handle.visualize())
        if not frames:
            return False
        players = frames[-1]["current"]["players"]
        ok = False
        for s in (0, 1):
            ids = [c["id"] for c in players[s]["prize"] if c is not None]
            if len(ids) == len(players[s]["prize"]) and ids:
                self.prize[s] = Counter(ids)
                ok = True
        return ok

    def on_seat_obs(self, obs: dict) -> None:
        """Called whenever a seat receives an obs (forced or not): refresh
        that seat's exact hand; maintain/learn prize multisets."""
        cur = obs["current"]
        s = cur["yourIndex"]
        if not self._frozen:
            pls = cur["players"]
            if len(pls[0]["prize"]) == 6 and len(pls[1]["prize"]) == 6:
                if self._parse_prizes() and cur["turn"] >= 1:
                    self._frozen = True
        else:
            for lg in obs.get("logs") or []:
                if lg.get("playerIndex") != s or lg.get("type") != _MOVE_CARD:
                    continue
                if lg.get("fromArea") == _PRIZE and lg.get("cardId"):
                    self.prize[s][lg["cardId"]] -= 1
                    self.prize[s] += Counter()
                elif lg.get("toArea") == _PRIZE and lg.get("cardId"):
                    self.prize[s][lg["cardId"]] += 1
        hand = cur["players"][s].get("hand")
        if hand is not None:
            self.hand[s] = Counter(c["id"] for c in hand)

    def emit(self, obs: dict):
        """(opp_hand, opp_deck, opp_prize, my_prize) Counters for the
        acting seat's critic. Unknown identities -> UNK_CARD rows."""
        cur = obs["current"]
        me = cur["yourIndex"]
        opp = 1 - me
        opp_pl = cur["players"][opp]

        opp_hand = Counter(self.hand[opp])
        n_now = opp_pl["handCount"]
        n_snap = sum(opp_hand.values())
        if n_now > n_snap:                       # hidden draws since snapshot
            opp_hand[UNK_CARD] += n_now - n_snap
        elif n_now < n_snap:                     # public exits not yet folded
            for cid in sorted(opp_hand, reverse=True):
                drop = min(opp_hand[cid], n_snap - n_now)
                opp_hand[cid] -= drop
                n_snap -= drop
                if n_snap == n_now:
                    break
            opp_hand += Counter()

        visible = _visible_out_of_deck(cur, opp)
        opp_deck = self.decks[opp] - visible - self.prize[opp] \
            - (opp_hand - Counter({UNK_CARD: opp_hand[UNK_CARD]}))
        # reconcile against the public deck count (UNK hand cards came
        # from the deck; Counter subtraction clamps at zero)
        deck_total = opp_pl["deckCount"]
        n_deck = sum(opp_deck.values())
        if n_deck > deck_total:
            opp_deck[UNK_CARD] = 0
            excess = n_deck - deck_total
            for cid in sorted(opp_deck, reverse=True):
                drop = min(opp_deck[cid], excess)
                opp_deck[cid] -= drop
                excess -= drop
                if not excess:
                    break
            opp_deck += Counter()
        elif n_deck < deck_total:
            opp_deck[UNK_CARD] = deck_total - n_deck

        # prize emission: exact when the ledger matches the public count
        # (pre-freeze setup steps, or any drift, degrade to UNK rows —
        # never emit wrong truth)
        def prize_of(seat):
            n = len(cur["players"][seat]["prize"])
            p = Counter(self.prize[seat])
            return p if sum(p.values()) == n else Counter({UNK_CARD: n})

        return opp_hand, opp_deck, prize_of(opp), prize_of(me)
