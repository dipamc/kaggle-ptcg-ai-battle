"""Engine-exact win-line prover: find a line of OUR plays that ends the
game as a WIN *this turn*, proven across K determinizations of the
hidden zones.

This is deliberately NOT the v1.2 sequencer: no policy rollouts, no
value leaf, no uncertainty gate. It enumerates our remaining turn
through the real sim (bounded DFS with transposition dedup over the
SDK's persistent search tree) and reports a line only when the sim
itself says result==us. The intended use is an ungated override that
fires on proofs and never otherwise, so it composes with a confident
policy instead of second-guessing it:
- a KO that does not win is never forced (the declined-lethal lesson:
  attacking ends the turn, ~6pp per wrongly forced attack);
- a win the policy was about to walk past IS forced (ladder mining:
  proven-win states get squandered at a measurable rate, including
  games that are then LOST).

WHAT COUNTS AS A PROOF (v3 `certify=1`, the default)
----------------------------------------------------
Hidden information enters through `begin()`: our deck ORDER, our prize
CONTENTS and every opponent card are invented per world (only our
unseen multiset is exact), and coin flips resolve on the engine's own
RNG, freshly per branch. So a line found in one world is a fact about
that world, not about the game.

v1/v2 (`certify=0`) handled this by requiring all K worlds to find a
win -- but each world searched independently and only world 0's line
was ever executed. That gate is on the EXISTENCE of a win, not on the
line we play, and existence is nearly free in a position good enough to
have a proof at all. Audit over 24 real proven states at these
settings: 22 fire, only 15 execute a line that actually wins in fresh
worlds (188/264 replays, 71%); every sampled line of 15+ actions fails.

v3 certifies the line itself: world 0 ENUMERATES candidate wins, and a
candidate ships only if replaying it VERBATIM reaches result==us in
every one of the remaining K-1 worlds. A line that needed a particular
draw, prize or coin flip diverges there and is rejected.

`strict=1` additionally demands the replay pass through world 0's exact
visible states (`state_key`) -- a structural test (hidden info provably
never reached an observable) rather than a statistical one, which also
makes the cached plan a prediction rather than a per-world guess. It is
OFF by default on measurement, not principle: see the sweep below.

Verification is far cheaper than the search it replaces: K-1 full DFS
passes become K-1 replays of <=len(line) steps. `worlds` therefore costs
almost nothing HERE and is expensive under certify=0 (K full searches).
Note it costs nothing at all on a no-proof decision either way -- world 0
finds nothing and returns before any verification, which is why total
agent time barely moves (measured 101 vs 105 ms/decision over 81 real
eligible decisions, 79 of them no-proof).

Sweep over 60 random proof states, each chosen line then probed in 32
held-out worlds the prover never touched (p = its true win rate):

    setting                fires    ships p<1   worst p   s/prove
    certify=0 (v1/v2)      ~all     many        0.00      --
    loose worlds=12        55/60    1           0.81      0.03
    loose worlds=32        54/60    0           1.00      0.04
    loose worlds=64        54/60    0           1.00      0.06
    strict worlds=12/32/64 43/60    0           1.00      0.03-0.06

worlds=32 is the default: it costs one firing over worlds=12 and that
firing was exactly the p=0.81 line (0.81^11 = 10% survival at K=12,
0.1% at K=32). strict buys no measured soundness beyond that and gives
up 11 more firings. What survives at any setting: this is sampling, so
p is bounded not proven (32 probes resolve to about p>=0.9), and engine
drift -- we vendor 1.32.2, the ladder runs 1.32.6 -- biases every world
in the same direction and no K can detect it.
"""
import hashlib
import json
import time
from itertools import combinations

OT_ATTACK, OT_ABILITY, OT_END = 13, 10, 14


def forced(sel):
    n = len(sel["option"])
    if sel["minCount"] == sel["maxCount"] == n:
        return list(range(n))
    if n == 1 and sel["minCount"] >= 1:
        return [0]
    return None


def state_key(obs):
    """Transposition key: same position reached via a different play
    order collapses (attach A then B == B then A)."""
    cur = obs["current"]
    sel = obs["select"]

    def cid(x):
        return x if isinstance(x, int) else (x or {}).get("id", 0)

    def side(p):
        def mon(m):
            if not m:
                return None
            sc = m.get("specialConditions")
            return (m.get("id"), m.get("hp"),
                    tuple(sorted(cid(e) for e in (m.get("energies") or []))),
                    tuple(sorted(cid(t) for t in (m.get("tools") or []))),
                    tuple(sorted(sc)) if isinstance(sc, list) else sc)
        hand = p.get("hand")
        hand_k = tuple(sorted(cid(h) for h in hand)) if hand is not None \
            else p.get("handCount")
        return (mon((p.get("active") or [None])[0]),
                tuple(mon(b) for b in (p.get("bench") or [])),
                hand_k, p.get("deckCount"), len(p.get("prize") or []))
    me = cur["yourIndex"]
    k = (side(cur["players"][me]), side(cur["players"][1 - me]),
         json.dumps(cur.get("stadium"), sort_keys=True),
         sel.get("type"), sel.get("minCount"), sel.get("maxCount"),
         tuple((o.get("type"), o.get("cardId"), o.get("attackId"),
                o.get("area"), o.get("index"), o.get("number"))
               for o in sel.get("option") or []))
    return hashlib.md5(repr(k).encode()).hexdigest()


class WinLine:
    """Prover over a SearchAgent-compatible host: the host supplies
    `begin(obs) -> state` (search_begin_raw with a fresh determinization)
    and `step(state, ans) -> state` (search_step_raw), plus search_end
    handling around prove()."""

    def __init__(self, begin, step, node_cap=3000, multi_cap=36,
                 depth_cap=50, worlds=32, deadline_s=None,
                 certify=1, strict=0, cand_cap=3):
        self._begin = begin
        self._step = step
        self.node_cap = node_cap
        self.multi_cap = multi_cap
        self.depth_cap = depth_cap
        self.worlds = worlds
        self.deadline_s = deadline_s
        self.certify = certify      # 1 = verify the LINE, 0 = legacy v2
        self.strict = strict        # certify=1 only: also key-identical
        self.cand_cap = cand_cap    # candidate lines tried per decision
        self.nodes_spent = 0          # telemetry, cumulative
        self.cands_tried = 0
        self.vfail_win = 0            # candidates killed: no win in a world
        self.vfail_key = 0            # candidates killed: state diverged
        self.verify_steps = 0

    @staticmethod
    def _order(sel):
        opts = sel.get("option") or []
        rank = {OT_ATTACK: 0, OT_ABILITY: 1}
        return sorted(range(len(opts)),
                      key=lambda j: (2 if opts[j].get("type") == OT_END
                                     else rank.get(opts[j].get("type"), 1),
                                     j))

    def _answers(self, sel):
        fa = forced(sel)
        if fa is not None:
            return [fa]
        n = len(sel["option"])
        lo, hi = sel["minCount"], sel["maxCount"]
        if hi == 1:
            outs = [[j] for j in self._order(sel)]
            if lo == 0:
                outs.append([])
            return outs
        outs = []
        for k in range(lo, min(hi, n) + 1):
            for combo in combinations(range(n), k):
                outs.append(list(combo))
                if len(outs) >= self.multi_cap:
                    return outs
        return outs

    # -- world 0: enumerate candidate wins -------------------------------
    def _gen(self, state, me, root_turn, budget, memo, depth, t_end):
        """Yield (line, keys) for every winning line the bounded DFS
        reaches, in search order. keys[i] is the state_key of the select
        answered by line[i], recorded from the states actually visited --
        so the cached plan describes world 0 exactly, with no second
        determinization to disagree with it."""
        obs = state["observation"]
        cur = obs["current"]
        if cur["result"] != -1:
            if cur["result"] == me:
                yield [], []
            return
        if cur["turn"] != root_turn:
            return
        if depth >= self.depth_cap or budget[0] <= 0:
            return
        if t_end is not None and time.time() >= t_end:
            budget[0] = 0
            return
        sel = obs.get("select")
        if not sel:
            return
        k = state_key(obs)
        if k in memo:
            return
        memo.add(k)
        for ans in self._answers(sel):
            if budget[0] <= 0:
                return
            budget[0] -= 1
            try:
                st = self._step(state, ans)
            except Exception:
                continue
            for sub, subk in self._gen(st, me, root_turn, budget, memo,
                                       depth + 1, t_end):
                yield [ans] + sub, [k] + subk

    def _dfs(self, state, me, root_turn, budget, memo, depth, t_end):
        """First winning line only (legacy certify=0 path)."""
        for line, _keys in self._gen(state, me, root_turn, budget, memo,
                                     depth, t_end):
            return line
        return None

    # -- worlds 1..K-1: replay the candidate verbatim ---------------------
    def _verify(self, obs, line, keys):
        """True iff replaying `line` answer-for-answer in ONE fresh
        determinization reaches result==us. Any divergence -- an option
        index that no longer exists, the turn ending first, an engine
        rejection, or (strict) a visible state that differs from world 0
        -- is a failed proof, not a retry."""
        me = obs["current"]["yourIndex"]
        root_turn = obs["current"]["turn"]
        try:
            st = self._begin(obs)
        except Exception:
            return False
        for i, ans in enumerate(line):
            o = st["observation"]
            cur = o["current"]
            if cur["result"] != -1:
                return cur["result"] == me       # won early: still a win
            if cur["turn"] != root_turn:
                self.vfail_win += 1
                return False
            sel = o.get("select")
            if not sel:
                self.vfail_win += 1
                return False
            n = len(sel["option"])
            if any(a >= n for a in ans) or \
                    not sel["minCount"] <= len(ans) <= max(1, sel["maxCount"]):
                self.vfail_win += 1
                return False
            if self.strict and i < len(keys) and state_key(o) != keys[i]:
                self.vfail_key += 1
                return False
            try:
                st = self._step(st, ans)
            except Exception:
                self.vfail_win += 1
                return False
            self.verify_steps += 1
        ok = st["observation"]["current"]["result"] == me
        if not ok:
            self.vfail_win += 1
        return ok

    def prove(self, obs, want_keys=False):
        """A winning line, or None.

        certify=1: world 0 enumerates candidates; the first one that
        replays to a win in all remaining worlds is returned.
        certify=0: legacy v1/v2 -- every world must find SOME win and
        world 0's line is returned unverified.
        """
        me = obs["current"]["yourIndex"]
        root_turn = obs["current"]["turn"]
        t_end = None if self.deadline_s is None \
            else time.time() + self.deadline_s

        if not self.certify:
            line0 = None
            for w in range(self.worlds):
                budget = [self.node_cap]
                memo = set()
                root = self._begin(obs)
                line = self._dfs(root, me, root_turn, budget, memo, 0, t_end)
                self.nodes_spent += self.node_cap - budget[0]
                if line is None:
                    return None
                if w == 0:
                    line0 = line
                if t_end is not None and time.time() >= t_end and \
                        w < self.worlds - 1:
                    return None   # could not finish verification: no proof
            return (line0, None) if want_keys else line0

        budget = [self.node_cap]
        memo = set()
        try:
            root = self._begin(obs)
        except Exception:
            return None
        tried = 0
        for line, keys in self._gen(root, me, root_turn, budget, memo, 0,
                                    t_end):
            if not line:
                continue
            tried += 1
            self.cands_tried += 1
            certified = True
            for _w in range(1, self.worlds):
                if t_end is not None and time.time() >= t_end:
                    certified = False       # unverified == unproven
                    break
                if not self._verify(obs, line, keys):
                    certified = False
                    break
            if certified:
                self.nodes_spent += self.node_cap - budget[0]
                return (line, keys) if want_keys else line
            if tried >= self.cand_cap or budget[0] <= 0:
                break
            if t_end is not None and time.time() >= t_end:
                break
        self.nodes_spent += self.node_cap - budget[0]
        return None

    def plan(self, obs):
        """prove() + expected-state keys for cached execution: returns
        [(answer, key_of_select_state_BEFORE_answer), ...] or None.

        The keys let the caller execute the certified line while checking
        each real select against the state the sim predicted, replanning
        only on divergence (v1 re-proved at every decision and stranded
        the agent mid-plan when a re-proof missed -- -2.2pp on
        proof-heavy decks). Under certify=1 the keys come from the very
        states world 0 visited; under strict=1 every other world was
        required to reproduce them, so a live mismatch means reality
        diverged from ALL of them, not that we sampled a stray world."""
        r = self.prove(obs, want_keys=True)
        if not r:
            return None
        line, keys = r
        if keys is None:                       # legacy path: re-derive
            keys = [state_key(obs)]
            try:
                st = self._begin(obs)
                for ans in line[:-1]:
                    st = self._step(st, ans)
                    o = st["observation"]
                    if o["current"]["result"] != -1 or not o.get("select"):
                        break
                    keys.append(state_key(o))
            except Exception:
                pass
        return [(line[i], keys[i] if i < len(keys) else None)
                for i in range(len(line))]
