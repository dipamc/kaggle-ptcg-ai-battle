"""Rollout-search agent over the competition SDK's search API.

Two modes (cfg.mode):

- "rollout" (v0): at gated single-pick decisions, top-A root actions x
  n_det determinizations, roll out with the policy for BOTH seats up
  to `horizon` sim decisions; terminal ±1 else value-head leaf.
- "turn" (own-turn sequencer): roll out only to the END OF OUR TURN
  (turn counter changes) and score there. The opponent never moves
  inside the horizon, so the determinized opponent deck is nearly
  irrelevant to the outcome — the search optimizes the ordering of OUR
  plays over OUR exactly-known unseen multiset (draws/searches/flips),
  which is the high-confidence part of the hidden state.

Epistemics: our zones are determinized from InfoTracker knowledge
(exact unseen multiset, exact prizes once known). The opponent's
unknown cards come from a pluggable GUESS pool (default: a random deck
from decks/pool per determinization) — never treated as knowledge, and
in "turn" mode barely consulted. Rollout policies carry trackers
(fix 0): our sim seat a clone() of the live tracker, the opponent seat
a fresh tracker over its guessed deck, so each fictional world is at
least played coherently.

Override discipline (fix 1, min-evidence): search may only override
the policy argmax if the winner has >= min_ev completed rollouts AND
beats the policy choice's mean by `margin`; otherwise the policy
plays. Gate: search activates only when the policy is torn at the root
(top-2 prob gap < gate_gap). Multi-pick selects and forced selects
always bypass. cfg=None degrades to the pure policy — byte-identical
behavior to the deployed no-search agent.

The search runs on the SDK's separate agent_ptr, so it coexists with a
live hosted battle (locally) or the Kaggle runner (remotely). Raw-dict
wrappers around lib.SearchBegin/SearchStep are used instead of the
dataclass API so observations keep the exact shape encode() expects.
"""
import ctypes
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from cg.sim import lib
from .buffers import OBS_SIZE, STOP
from .encoder import encode
from .model import PTCGTransformer, arch_from_state_dict
from ..tracker import InfoTracker

_agent_ptr = None
_DMG_TABLE = None
# validation-only: make every model eval cost this many ms of wall
# clock (sleep-injection kaggle-speed emulation; 0 = off)
_EVAL_MS = float(os.environ.get("PTCG_EVAL_MS", "0"))


def _dmg_table(data_dir=None):
    """card_id -> [(energy_cost, damage), ...] parsed from the official
    card CSV (engine id space). Damage strings like '130+'/'50x' keep
    their base number. Missing file -> {} (threat model falls back to a
    generic energies*40 estimate per mon)."""
    global _DMG_TABLE
    if _DMG_TABLE is None:
        import csv
        import re
        tbl = {}
        here = os.path.dirname(os.path.abspath(__file__))
        paths = ([os.path.join(data_dir, "EN_Card_Data.csv")]
                 if data_dir else []) + \
            [os.path.join(here, "..", "..", "data", "EN_Card_Data.csv")]
        for path in paths:
            try:
                for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                    digits = re.sub(r"[^0-9]", "", r.get("Damage") or "")
                    if not digits:
                        continue
                    cost = len(re.findall(r"\{", r.get("Cost") or ""))
                    tbl.setdefault(int(r["Card ID"]), []).append(
                        (cost, int(digits)))
                break
            except (OSError, KeyError, ValueError):
                tbl = {}
        _DMG_TABLE = tbl
    return _DMG_TABLE


def _ptr():
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = lib.AgentStart()
    return _agent_ptr


def _raw(bs):
    r = json.loads(ctypes.string_at(bs).decode())
    if r["error"] != 0:
        raise RuntimeError(f"search api error {r['error']}")
    return r["state"]      # {"observation": {...}, "searchId": int}


def search_begin_raw(obs, your_deck, your_prize, opp_deck, opp_prize,
                     opp_hand, opp_active):
    sbi = obs["search_begin_input"]
    arr = lambda xs: (ctypes.c_int * len(xs))(*xs)
    return _raw(lib.SearchBegin(
        _ptr(), sbi.encode("ascii"), len(sbi),
        arr(your_deck), arr(your_prize), arr(opp_deck), arr(opp_prize),
        arr(opp_hand), arr(opp_active), 0))


def search_step_raw(search_id, select):
    return _raw(lib.SearchStep(
        _ptr(), search_id, (ctypes.c_int * len(select))(*select),
        len(select)))


def search_end():
    lib.SearchEnd(_ptr())


@dataclass
class SearchCfg:
    actions: int = 4      # top-A root candidates by policy prior
    dets: int = 2         # determinizations per candidate
    horizon: int = 20     # max sim decisions per rollout (rollout mode)
    min_gap: float = 0.0  # skip search if prior top-1 prob >= 1-min_gap
    mode: str = "rollout"   # "rollout" (v0) | "turn" (own-turn sequencer)
    gate_gap: float = 1.0   # search only if P(top1)-P(top2) < gate_gap
    min_ev: int = 0         # min completed rollouts to override policy
    margin: float = 0.0     # required mean advantage over policy choice
    margin_v: float = 0.0   # stiffer margin when the winner's evidence
                            # is ALL value-leaves (no engine-true
                            # terminal); 0 = use `margin` for both
    crn: int = 1            # common random numbers: same sampling seed
                            # across candidates within a determinization
    turn_cap: int = 40      # safety cap on sim decisions in turn mode
    dec_evals: int = 0      # max model evals per searched decision
                            # (0 = unlimited); checked between rollouts,
                            # so grindy-turn decks can't spike a single
                            # decision no matter the wall clock
    guess_match: int = 0    # pick the opponent guess deck by revealed-
                            # card containment (evidence-based) instead
                            # of uniformly at random
    turn_depth: int = 1     # turn-mode horizon in turn boundaries: 1 =
                            # leaf at handoff; 2 = roll THROUGH the
                            # opponent's simulated turn and leaf at the
                            # start of OUR next turn (leaf obs is our
                            # own perspective — much less fiction)
    threat_pen: float = 0.0  # leaf haircut when the opponent has lethal
                             # on board next turn (engine-true, public
                             # info only); losing-lethal leaves clamp to
                             # -0.8. 0 = off
    ucb: int = 0            # UCB1 rollout allocation over candidates
                            # instead of the uniform grid


class SearchAgent:
    def __init__(self, model_path, deck_path, cfg=None, basics=None,
                 seed=0, device="cpu", data_dir=None, heads=8, temp=1.0):
        self.cfg = cfg
        # Real-play sampling temperature for the move we actually SUBMIT.
        # 1.0 = the on-policy sampling every bundle has shipped with
        # (default keeps existing bundles/evals byte-identical); 0 =
        # greedy argmax. Rollouts do NOT read this — _rollout pins its
        # own temp — so this knob moves only the played move, never the
        # search's internal continuation estimate. The two are separate
        # questions and were conflated once already.
        self.temp = temp
        self.data_dir = data_dir   # bundle dir: guess_pool/ + card CSV
        self.rng = random.Random(seed)
        self.device = torch.device(device)
        sd = torch.load(model_path, map_location="cpu")
        # width/depth come off the weights; `heads` cannot (see
        # arch_from_state_dict) so callers evaluating a non-8-head run must
        # pass it. Default 8 keeps every existing bundle byte-compatible.
        self.policy = PTCGTransformer(**arch_from_state_dict(sd, heads=heads))
        self.policy.load_state_dict(
            {k.replace("module.", ""): v for k, v in sd.items()})
        self.policy.eval()
        self.policy.to(self.device)
        self.policy.freeze_tables()
        self.deck = [int(l) for l in open(deck_path) if l.strip()][:60]
        self.row = np.zeros((1, OBS_SIZE), dtype=np.float32)
        # filler for OUR zones only (padding when tracker knowledge runs
        # short) — epistemically ours to use
        self.filler = list(self.deck)
        from cg.api import all_card_data
        cards = all_card_data()
        basic_ids = {c.cardId for c in cards if c.basic}
        if basics is None:
            basics = [c for c in self.deck if c in basic_ids]
        self.basics = basics or self.filler[:1]
        # prizes surrendered on KO (Legacy Energy etc. edge cases ignored)
        self._prize_risk = {c.cardId: 3 if c.megaEx else 2 if c.ex else 1
                            for c in cards}
        # opponent-deck GUESS pool: one deck drawn per determinization.
        # Their real list is private on kaggle — this is a guess setup,
        # never our own multiset projected onto them. Later upgrades
        # (revealed-card matching, replay-harvested ladder decks) plug
        # in here.
        self.guess_pool = self._load_guess_pool(data_dir)
        if not self.guess_pool:   # no local pool (stripped bundle):
            self.guess_pool = [(self.basics * 60)[:60]]  # legal, generic
        self._guess_basics = [
            [c for c in gd if c in basic_ids] or [gd[0]]
            for gd in self.guess_pool]
        self.tracker = None
        self.forced_run = 0
        self.evals = 0        # model forwards since reset (telemetry)

    @staticmethod
    def _load_guess_pool(data_dir=None):
        import glob
        here = os.path.dirname(os.path.abspath(__file__))
        dirs = ([os.path.join(data_dir, "guess_pool")] if data_dir else []) \
            + [os.path.join(here, "..", "..", "decks", "pool")]
        for deck_dir in dirs:
            pool = []
            for f in sorted(glob.glob(os.path.join(deck_dir, "*.csv"))):
                ids = [int(l) for l in open(f) if l.strip()]
                if len(ids) == 60:
                    pool.append(ids)
            if pool:
                return pool
        return []

    # -- policy primitives ------------------------------------------------
    def _logits(self, obs, picked, stop_ok, forced_run, tracker):
        import time as _t
        t0 = _t.time()
        encode(self.row[0], obs, picked, stop_ok, forced_run,
               tracker=tracker)
        with torch.no_grad():
            out = self.policy.forward_eval(
                torch.from_numpy(self.row).to(self.device))
        self.evals += 1
        if _EVAL_MS:   # kaggle-speed emulation: pad to target cost
            dt = _t.time() - t0
            if dt < _EVAL_MS / 1000.0:
                _t.sleep(_EVAL_MS / 1000.0 - dt)
        return out[0][0].cpu(), float(out[1][0])

    def _decompose(self, obs, tracker, forced_run, sample=True, temp=None):
        """Full answer for a select via sequential pick+STOP.

        temp: None = use self.temp (the real-play knob); callers that must
        pin their own behaviour pass it explicitly. temp<=0 is argmax.
        Note this is greedy PER PICK of the sequential decomposition, so
        on multi-picks it can lock a joint combination that sampling
        would have escaped — that is part of what the sweep measures."""
        t = self.temp if temp is None else temp
        sel = obs["select"]
        n = len(sel["option"])
        picked = []
        while True:
            stop_ok = (len(picked) >= sel["minCount"]
                       and (sel["maxCount"] > 1 or sel["minCount"] == 0))
            logits, _ = self._logits(obs, picked, stop_ok, forced_run,
                                     tracker)
            a = int(logits.argmax()) if (not sample or t <= 0) else \
                int(torch.distributions.Categorical(logits=logits / t).sample())
            if a == STOP:
                if stop_ok:
                    return picked
                a = next((j for j in range(n) if j not in picked), 0)
            if a >= n or a in picked:
                a = next((j for j in range(n) if j not in picked), None)
                if a is None:
                    return picked
            if sel["maxCount"] == 1:
                return [a]
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                return picked

    @staticmethod
    def _forced(sel):
        n = len(sel["option"])
        if sel["minCount"] == sel["maxCount"] == n:
            return list(range(n))
        if n == 1 and sel["minCount"] >= 1:
            return [0]
        return None

    # -- determinization --------------------------------------------------
    def _determinize(self, obs):
        cur = obs["current"]
        me = cur["yourIndex"]
        mepl, opl = cur["players"][me], cur["players"][1 - me]
        pool = list(self.tracker.my_unseen().elements())
        self.rng.shuffle(pool)
        n_prize = len(mepl["prize"])
        if self.tracker.my_prize_known is not None:
            my_prize = list(self.tracker.my_prize_known.elements())[:n_prize]
            rest = list((self.tracker.my_unseen()
                         - self.tracker.my_prize_known).elements())
            self.rng.shuffle(rest)
            pool = rest
        else:
            my_prize = pool[:n_prize]
            pool = pool[n_prize:]
        my_deck = (pool + self.filler)[:max(mepl["deckCount"], 0)]
        my_prize += self.filler[:n_prize - len(my_prize)]

        # opponent unknowns from a guessed deck (their list is private);
        # revealed cards (tracker) always take precedence over the guess
        if self.cfg is not None and self.cfg.guess_match:
            gi = self._match_guess()
        else:
            gi = self.rng.randrange(len(self.guess_pool))
        guess = self.guess_pool[gi]
        known_hand = list(self.tracker.opp_known_hand.values())
        known_deck = list(self.tracker.opp_known_deck.values())
        fill = guess * 2
        self.rng.shuffle(fill)
        opp_hand = (known_hand + fill)[:max(opl["handCount"], 0)]
        opp_deck = (known_deck + fill)[:max(opl["deckCount"], 0)]
        if opp_deck:   # sim requires >=1 Basic in the opp deck at setup
            opp_deck[-1] = self.rng.choice(self._guess_basics[gi])
        opp_prize = fill[:len(opl["prize"])]
        opp_active = [self.rng.choice(self._guess_basics[gi])] \
            if (opl["active"] and opl["active"][0] is None) else []
        return (my_deck, my_prize, opp_deck, opp_prize, opp_hand,
                opp_active), guess

    def _match_guess(self) -> int:
        """Evidence-based guess: rank guess-pool decks by how fully
        they CONTAIN the opponent's revealed cards (tracker knowledge +
        board), sample among the best. Converges onto the right
        archetype (e.g. the mirror) within a few reveals; ties stay
        random so we don't overcommit early."""
        from collections import Counter
        seen = Counter(self.tracker.opp_known_hand.values())
        seen += Counter(self.tracker.opp_known_deck.values())
        obs = self.tracker._last_obs
        if obs is not None:
            opp = obs["current"]["players"][
                1 - obs["current"]["yourIndex"]]
            for p in (opp.get("active") or []) + (opp.get("bench") or []):
                if p:
                    seen[p["id"]] += 1
        if not seen:
            return self.rng.randrange(len(self.guess_pool))
        scores = []
        for gd in self.guess_pool:
            gc = Counter(gd)
            scores.append(sum((seen & gc).values()))
        hi = max(scores)
        best = [i for i, s in enumerate(scores) if s == hi]
        return self.rng.choice(best)

    def opp_match(self):
        """Matchup evidence for adaptive configs: (best guess deck,
        matched reveal count, total reveal count). None before any
        reveals. Same containment scoring as guess_match."""
        if self.tracker is None or self.tracker._last_obs is None:
            return None
        from collections import Counter
        seen = Counter(self.tracker.opp_known_hand.values())
        seen += Counter(self.tracker.opp_known_deck.values())
        obs = self.tracker._last_obs
        opp = obs["current"]["players"][1 - obs["current"]["yourIndex"]]
        for p in (opp.get("active") or []) + (opp.get("bench") or []):
            if p:
                seen[p["id"]] += 1
        total = sum(seen.values())
        if not total:
            return None
        best, score = None, -1
        for gd in self.guess_pool:
            s = sum((seen & Counter(gd)).values())
            if s > score:
                best, score = gd, s
        return best, score, total

    # -- rollout ----------------------------------------------------------
    def _sim_trackers(self, me, guess):
        """Fix 0: our sim seat carries a clone of the live tracker (our
        deck is truly known); the opponent seat a fresh tracker over its
        GUESSED deck — a fiction, but a coherently-played one."""
        return {me: self.tracker.clone(), 1 - me: InfoTracker(guess)}

    def _leaf(self, obs, me, trs):
        """Non-terminal leaf: value head from the deciding seat's
        perspective, sign-flipped to ours, optionally corrected by the
        one-ply lethal-threat check. (score, is_terminal=False)"""
        seat = obs["current"]["yourIndex"]
        trs[seat].update(obs)
        _, v = self._logits(obs, [], False, 0, trs[seat])
        v = v if seat == me else -v
        if self.cfg is not None and self.cfg.threat_pen and seat != me:
            v = self._threat_adjust(obs, seat, v)
        return v, False

    def _threat_adjust(self, obs, mover, v):
        """Engine-true one-ply threat check over PUBLIC info only (their
        board, energies, known attack table): if the seat about to move
        can KO our active next turn, the value head's optimism gets a
        haircut; if that KO also clears their prizes, the leaf is a
        near-loss no matter what V thinks. Idea borrowed from the
        'probabilistic' public agent's evaluator."""
        cur = obs["current"]
        them, us = cur["players"][mover], cur["players"][1 - mover]
        act = (us["active"] or [None])[0]
        if not act:
            return v
        tbl = _dmg_table(self.data_dir)
        mx = 0
        for p in (them["active"] or []) + (them["bench"] or []):
            if not p:
                continue
            ne = len(p.get("energies") or []) + 1  # assume next attach
            atks = tbl.get(p["id"])
            dmg = max((d for c, d in atks if c <= ne), default=0) \
                if atks else ne * 40
            mx = max(mx, dmg)
        if mx < act["hp"]:
            return v
        if len(them["prize"]) <= self._prize_risk.get(act["id"], 1):
            return min(v, -0.8)   # losing lethal on board
        return max(-1.0, v - self.cfg.threat_pen)

    @staticmethod
    def _terminal(cur, me):
        """Engine-true outcome. (score, is_terminal=True)"""
        if cur["result"] == 2:
            return 0.0, True
        return (1.0 if cur["result"] == me else -1.0), True

    def _rollout(self, state, me, trs, horizon, stop_turn=None):
        """Play the sim forward with the (tracker-carrying) policy;
        return (score in MY perspective, reached_terminal). stop_turn:
        leaf out as soon as cur['turn'] leaves this value (turn mode —
        the opponent's turn is never simulated)."""
        obs = state["observation"]
        for _ in range(horizon):
            cur = obs["current"]
            if cur["result"] != -1:
                return self._terminal(cur, me)
            if stop_turn is not None and cur["turn"] > stop_turn:
                return self._leaf(obs, me, trs)
            seat = cur["yourIndex"]
            trs[seat].update(obs)
            sel = obs["select"]
            fa = self._forced(sel)
            ans = fa if fa is not None \
                else self._decompose(obs, trs[seat], 0, sample=True, temp=1.0)
            try:
                state = search_step_raw(state["searchId"], ans)
            except RuntimeError:
                if ans:
                    return 0.0, False  # sim rejected a legal-looking answer
                state = search_step_raw(state["searchId"], [0])
            obs = state["observation"]
        cur = obs["current"]
        if cur["result"] != -1:
            return self._terminal(cur, me)
        return self._leaf(obs, me, trs)

    # -- the agent --------------------------------------------------------
    def agent(self, obs_dict, budget_s=None):
        """budget_s: soft wall-clock allowance for THIS decision. The
        rollout loop stops opening new rollouts once exceeded; with no
        completed rollouts (or no budget for even one) it degrades to
        the pure policy answer."""
        import time as _t
        if obs_dict.get("select") is None:
            self.tracker = InfoTracker(self.deck)
            self.forced_run = 0
            return list(self.deck)
        sel = obs_dict["select"]
        if self.tracker is None:
            self.tracker = InfoTracker(self.deck)
        self.tracker.update(obs_dict)
        fa = self._forced(sel)
        if fa is not None:
            self.forced_run += 1
            return fa
        cfg = self.cfg
        n = len(sel["option"])
        if (cfg is None or sel["maxCount"] != 1 or n < 2
                or obs_dict.get("search_begin_input") is None
                or (budget_s is not None and budget_s <= 0)):
            ans = self._decompose(obs_dict, self.tracker, self.forced_run)
            self.forced_run = 0
            return ans
        deadline = None if budget_s is None else _t.time() + budget_s
        # --- search over top-A single-pick candidates
        logits, _ = self._logits(obs_dict, [], False, self.forced_run,
                                 self.tracker)
        probs = torch.softmax(logits, dim=-1)
        cands = sorted(range(n), key=lambda i: -float(probs[i]))[:cfg.actions]
        if cfg.min_gap and float(probs[cands[0]]) >= 1.0 - cfg.min_gap:
            self.forced_run = 0
            return [cands[0]]
        # uncertainty gate: only search where the policy is torn
        if len(cands) > 1 and \
                float(probs[cands[0]]) - float(probs[cands[1]]) >= cfg.gate_gap:
            self.forced_run = 0
            return [cands[0]]
        me = obs_dict["current"]["yourIndex"]
        turn_mode = cfg.mode == "turn"
        depth = max(1, cfg.turn_depth)
        root_turn = obs_dict["current"]["turn"]
        horizon = cfg.turn_cap * depth if turn_mode else cfg.horizon
        evals0 = self.evals
        scores = {a: [] for a in cands}
        try:
            # worlds are shared across candidates; visit k of any
            # candidate uses worlds[k % dets] with its seed — common
            # random numbers make every comparison paired
            worlds = [(self._determinize(obs_dict),
                       self.rng.randrange(1 << 31))
                      for _ in range(cfg.dets)]

            def _pick():
                unv = [c for c in cands if not scores[c]]
                if unv:
                    return unv[0]
                if not cfg.ucb:   # uniform grid: fewest visits first
                    return min(cands, key=lambda c: len(scores[c]))
                import math
                mean_ = {c: sum(x for x, _ in scores[c]) / len(scores[c])
                         for c in cands}
                lo, hi = min(mean_.values()), max(mean_.values())
                span = (hi - lo) or 1.0
                n_tot = sum(len(scores[c]) for c in cands)
                return max(cands, key=lambda c:
                           (mean_[c] - lo) / span
                           + 0.5 * math.sqrt(math.log(n_tot)
                                             / len(scores[c])))

            for _ in range(cfg.actions * cfg.dets):
                if ((deadline is not None and _t.time() >= deadline)
                    or (cfg.dec_evals
                        and self.evals - evals0 >= cfg.dec_evals)) \
                        and scores[cands[0]]:
                    break
                a = _pick()
                (det, guess), det_seed = worlds[len(scores[a]) % cfg.dets]
                root = search_begin_raw(obs_dict, *det)
                try:
                    st = search_step_raw(root["searchId"], [a])
                except RuntimeError:
                    scores[a].append((-1.0, False))  # sim: illegal here
                    continue
                if cfg.crn:
                    torch.manual_seed(det_seed)
                scores[a].append(self._rollout(
                    st, me, self._sim_trackers(me, guess), horizon,
                    stop_turn=(root_turn + depth - 1) if turn_mode
                    else None))
        except Exception:
            # search machinery failed on this state — the POLICY is the
            # fallback (fast, always available); never degrade to random
            self.forced_run = 0
            return [cands[0]]
        finally:
            search_end()
        self.forced_run = 0
        pol = cands[0]                   # policy argmax among candidates
        if not any(scores.values()):     # budget gave us nothing usable
            return [pol]
        mean = {a: (sum(v for v, _ in xs) / len(xs)) if xs else -2.0
                for a, xs in scores.items()}
        best = max(cands, key=lambda a: (mean[a], float(probs[a])))
        # min-evidence override rule: enough completed rollouts AND a
        # real margin over the policy's own choice — stiffer when the
        # winner's evidence is all value-leaves (no engine-true result)
        if best != pol:
            need = cfg.margin if any(t for _, t in scores[best]) \
                else (cfg.margin_v or cfg.margin)
            if (len(scores[best]) < cfg.min_ev or not scores[pol]
                    or mean[best] < mean[pol] + need):
                best = pol
        return [best]
