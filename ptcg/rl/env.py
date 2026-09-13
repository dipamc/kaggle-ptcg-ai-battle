"""PTCG PufferLib environment.

Modes:
- opponent="self" (default): BOTH seats learn. One obs row per game; the
  stream serves whichever seat must decide, obs always from the actor's
  perspective (GlobF.ACTOR_SEAT carries the seat). Rewards are written
  from the perspective of the seat that took the action; the negamax
  advantage sign-flips bootstrap across actor switches, so PPO credit
  is correct for both seats with one shared policy. No waiting/padding steps exist.
- opponent="random": learner-vs-random (clean single-agent MDP).
- opponent="league": per EPISODE each game draws its opponent — mirror
  self-play (mix_self), a scripted kaggle-format anchor from
  script_dirs (mix_script — plays its OWN decklist from its deck call,
  answers every one of its selects, exec-loaded the way
  kaggle-environments loads it), a frozen checkpoint sampled from
  pool_dir (mix_pool, CPU inference in-worker, LRU-cached), else
  random. Scripted and pool draws are PFSP-weighted: per-worker EMA of
  the learner's win rate per opponent id, sample weight
  max(floor, (1-ema)^2) — beaten opponents demote to rare forgetting
  checks but are never removed. The negamax advantage handles the
  mixture transparently: vs-opponent rows have a constant actor seat,
  so the sign-flip reduces to stock GAE.

Per RL step a row consumes exactly one policy decision. Between
decisions the env auto-resolves forced selects (either seat) and, in
random mode, the opponent's moves. Multi-selects decompose into
sequential picks + STOP (buffers.STOP). Rewards: +-1 per prize from the
actor's perspective, terminal top-up to +-12 (draws total 0).

Per-seat InfoTrackers feed the blind tracker zones; the OracleLedger
feeds the critic-only oracle block. They never mix.

Each PTCGEnv instance runs `num_envs` concurrent battles in-process;
pufferlib.vector.make(...) fans instances across worker processes.
"""
import glob
import json
import os
import random
import time

import gymnasium
import numpy as np
import pufferlib

from . import battle as battle_mod
from .buffers import OBS_SIZE, MAX_OPTIONS, STOP
from .encoder import encode, write_oracle
from .buffers import FROZEN_COL
from .oracle import OracleLedger
from ..tracker import InfoTracker

DECK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "decks", "pool")
CAUSES = ("prize", "bench", "deck", "other")


def load_deck_pool(deck_dir=DECK_DIR) -> list[list[int]]:
    decks = []
    for f in sorted(glob.glob(os.path.join(deck_dir, "*.csv"))):
        with open(f) as fh:
            ids = [int(line) for line in fh if line.strip()]
        if len(ids) == 60:
            decks.append(ids)
    if not decks:
        raise FileNotFoundError(f"no decks in {deck_dir}")
    return decks


def random_answer(sel: dict, rng: random.Random) -> list[int]:
    n = len(sel["option"])
    k = rng.randint(sel["minCount"], min(sel["maxCount"], n))
    return sorted(rng.sample(range(n), k))


def forced_answer(sel: dict):
    n = len(sel["option"])
    if sel["minCount"] == sel["maxCount"] == n:
        return list(range(n))
    if n == 1 and sel["minCount"] >= 1:
        return [0]
    return None


class _Game:
    __slots__ = ("handle", "obs", "learner_seat", "trackers", "oracle",
                 "frozen_idx",
                 "picked", "cum_p0", "pending_p0", "decisions",
                 "engine_steps", "forced_run", "rng", "deck_pair",
                 "deck_idx", "prize_rewards",
                 "opp_kind")  # None=mirror, "random", or a checkpoint path

    def __init__(self, rng, prize_rewards=True):
        self.rng = rng
        self.prize_rewards = prize_rewards
        self.handle = None

    def start(self, decks, script_deck=None, my_idx=None, opp_idx=None):
        """my_idx/opp_idx pin the learner/opponent seat's deck (pool
        index); None = uniform draw. script_deck still overrides the
        opponent seat's deck entirely."""
        if self.handle is not None:
            self.handle.finish()
        self.learner_seat = self.rng.randint(0, 1)   # random/league only
        iL = self.rng.randrange(len(decks)) if my_idx is None else my_idx
        iO = self.rng.randrange(len(decks)) if opp_idx is None else opp_idx
        i0, i1 = (iL, iO) if self.learner_seat == 0 else (iO, iL)
        d0, d1 = decks[i0], decks[i1]
        if script_deck is not None:
            # scripted opponent plays its own decklist; idx -1 keeps
            # external decks out of the deck tallies (the log's 'opp'
            # field still names the anchor)
            if self.learner_seat == 0:
                d1, i1 = script_deck, -1
            else:
                d0, i0 = script_deck, -1
        self.deck_idx = (i0, i1)   # stable ids: sorted-glob position
        self.deck_pair = (d0, d1)
        self.handle = battle_mod.BattleHandle(d0, d1)
        self.oracle = OracleLedger((d0, d1), self.handle)
        self.trackers = [InfoTracker(d0), InfoTracker(d1)]
        self.obs = self.handle.obs()
        self._ingest()
        self.picked = []
        self.cum_p0 = 0.0
        self.pending_p0 = 0.0
        self.decisions = 0
        self.engine_steps = 0
        # per-seat: "my selects auto-resolved since my last decision".
        # A shared scalar counted the OTHER seat's forced submits into
        # mirror rows but not vs-opponent rows — a mode fingerprint.
        self.forced_run = [0, 0]

    def _ingest(self):
        """Feed a freshly-arrived engine obs to the receiving seat's
        tracker and to the oracle ledger (exactly once per obs)."""
        if self.obs["current"] is not None:
            seat = self.obs["current"]["yourIndex"]
            self.trackers[seat].update(self.obs)
            self.oracle.on_seat_obs(self.obs)

    def prizes_p0(self):
        pl = self.obs["current"]["players"]
        return len(pl[0]["prize"]), len(pl[1]["prize"])

    def submit(self, answer):
        """One engine transition; accrues prize rewards in p0 currency
        (reward='win' mode accrues nothing — terminal +-1 only)."""
        p0, p1 = self.prizes_p0()
        self.obs = self.handle.select(answer)
        self.engine_steps += 1
        if self.prize_rewards:
            q0, q1 = self.prizes_p0()
            self.pending_p0 += (p0 - q0) - (p1 - q1)
        self.picked = []
        self._ingest()


class PTCGEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs=32, deck_dir=DECK_DIR, opponent="self",
                 pool_dir=None, mix_self=0.4, mix_pool=0.4,
                 script_dirs=None, mix_script=0.0, mix_frozen=0.0,
                 n_frozen=1,
                 meta_decks=None, opp_meta_frac=0.0, meta_frac=1.0,
                 script_weights=None,
                 log_interval=128, max_engine_steps=3000,
                 replay_dir=None, replay_every=0, replay_teams=None,
                 reward="prize", deck_log_dir=None, buf=None, seed=0):
        assert opponent in ("self", "random", "league")
        # "prize": +-1 per prize + terminal top-up to +-12 (draws 0).
        # "win": pure terminal +-1, no prize signal (draws 0).
        assert reward in ("prize", "win")
        self.win_r = 12.0 if reward == "prize" else 1.0
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1e6, high=1e6, shape=(OBS_SIZE,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(MAX_OPTIONS)
        self.num_agents = num_envs
        self.render_mode = None
        super().__init__(buf=buf)

        self.opponent = opponent
        self.decks = load_deck_pool(deck_dir)
        self.max_engine_steps = max_engine_steps
        self.log_interval = log_interval
        self.pool_dir = pool_dir
        self.mix_self, self.mix_pool = mix_self, mix_pool
        # scripted kaggle-format anchors (abspath: workers chdir into
        # the agent dir around every call)
        self.script_dirs = [os.path.abspath(d) for d in (script_dirs or [])]
        self.mix_script = mix_script if self.script_dirs else 0.0
        # frozen: same-architecture v4 opponent, acted on GPU by the
        # driver (RoutedPolicy) -- its rows leave the env like mirror
        # rows instead of being resolved in-worker on CPU
        self.mix_frozen = mix_frozen
        # size of the frozen LEAGUE; one is drawn uniformly per game
        self.n_frozen = max(1, int(n_frozen))
        assert (self.mix_self + self.mix_script + self.mix_pool
                + self.mix_frozen) <= 1.0 + 1e-9
        # v4 meta-seat mode: learner seat ALWAYS plays a pinned meta
        # deck (uniform); mirror opponent seat draws meta with
        # opp_meta_frac else full pool. meta_decks = filename substrings
        # resolved against the sorted pool glob.
        self._meta_idx = []
        if meta_decks:
            files = [os.path.basename(f) for f in
                     sorted(glob.glob(os.path.join(deck_dir, "*.csv")))]
            for pat in meta_decks:
                hits = [i for i, f in enumerate(files) if pat in f]
                assert len(hits) == 1, f"meta deck {pat!r}: {len(hits)} matches"
                self._meta_idx.append(hits[0])
        self.opp_meta_frac = opp_meta_frac
        # P(learner seat draws a pinned meta deck); the remainder
        # pilots a uniform full-pool deck (fork-experiment knob)
        self.meta_frac = meta_frac
        # fixed script sampling weights (disables PFSP for scripts)
        self.script_weights = script_weights
        if script_weights is not None:
            assert len(script_weights) == len(self.script_dirs)
        self._script_set = set(self.script_dirs)
        self._script_fns = {}   # dir -> exec-loaded agent fn
        # PFSP state: per-worker EMA of learner win rate per opponent id
        self._ema = {}
        self.ema_alpha, self.ema_floor = 0.02, 0.03
        self._pool_files = []
        self._pool_cache = {}   # path -> frozen policy (LRU, cap 4)
        self._scratch = np.zeros((1, OBS_SIZE), dtype=np.float32)
        self.games = [_Game(random.Random(seed * 100003 + i),
                            prize_rewards=(reward == "prize"))
                      for i in range(num_envs)]
        self.tick = 0
        self.engine_steps_total = 0  # finished episodes only; see bench
        self._stats = self._zero_stats()
        self.replay_dir = replay_dir
        self.replay_every = replay_every
        self.replay_teams = replay_teams or ["p0", "p1"]
        self._episode_counter = 0
        if replay_dir:
            os.makedirs(replay_dir, exist_ok=True)
        # per-episode deck outcome log (one file per worker process;
        # aggregate offline with ptcg.rl.deck_report)
        self._deck_log_path = None
        self._deck_buf = []
        if deck_log_dir:
            os.makedirs(deck_log_dir, exist_ok=True)
            self._deck_log_path = os.path.join(
                deck_log_dir, f"w{os.getpid()}.jsonl")

    @staticmethod
    def _zero_stats():
        return {"episodes": 0, "wins": 0, "first_wins": 0, "draws": 0,
                "ep_len": 0, "prize_margin": 0.0, "illegal": 0,
                "turns_first": 0, "turns_second": 0,
                "eps_mirror": 0, "eps_pool": 0, "eps_rand": 0,
                "eps_script": 0, "w_script": 0,
                "w_pool": 0, "w_rand": 0, "env_errors": 0,
                "ep_ret_abs": 0.0, "ep_ret_bad": 0,
                # winner-perspective SIGNED prize margin, decided games
                # only (negative = won by bench/deck-out while behind)
                "prize_margin_dec": 0.0, "eps_dec": 0,
                # win causes, kept UNMIXED: end_* = mirror games only;
                # wend_/lend_* = learner won/lost vs a real opponent
                **{f"{p}_{c}": 0 for p in ("end", "wend", "lend")
                   for c in CAUSES}}

    # -- opponent drawing / snapshot inference ---------------------------
    def _start_game(self, g: _Game):
        if self.opponent == "self":
            g.opp_kind = None
        elif self.opponent == "random":
            g.opp_kind = "random"
        else:  # league: per-episode draw, PFSP within script/pool bands
            r = g.rng.random()
            if r < self.mix_self:
                g.opp_kind = None
            elif r < self.mix_self + self.mix_script:
                g.opp_kind = (
                    g.rng.choices(self.script_dirs,
                                  weights=self.script_weights)[0]
                    if self.script_weights is not None
                    else self._pfsp_pick(g, self.script_dirs))
            elif r < self.mix_self + self.mix_script + self.mix_frozen:
                g.opp_kind = "frozen"
                g.frozen_idx = g.rng.randrange(self.n_frozen)
            elif r < (self.mix_self + self.mix_script + self.mix_frozen
                      + self.mix_pool) \
                    and self._refresh_pool():
                g.opp_kind = self._pfsp_pick(g, self._pool_files)
            else:
                g.opp_kind = "random"
        script_deck = None
        if g.opp_kind in self._script_set:
            try:
                script_deck = self._script_deck_call(g.opp_kind)
            except Exception:
                # one broken anchor must never kill a run: surface it
                # (env_errors) and play this game vs random instead
                import traceback
                traceback.print_exc()
                self._stats["env_errors"] += 1
                g.opp_kind = "random"
        my_idx = opp_idx = None
        if self._meta_idx:
            if g.rng.random() < self.meta_frac:
                my_idx = g.rng.choice(self._meta_idx)
            if g.opp_kind is None and g.rng.random() < self.opp_meta_frac:
                opp_idx = g.rng.choice(self._meta_idx)
        g.start(self.decks, script_deck=script_deck,
                my_idx=my_idx, opp_idx=opp_idx)

    @staticmethod
    def _opp_name(opp) -> str:
        return os.path.basename(opp).removesuffix(".pt")

    def _pfsp_pick(self, g: _Game, ids):
        """PFSP-style draw within one category: weight (1-ema)^2 with a
        never-remove floor — hard opponents dominate, beaten ones demote
        to rare forgetting checks. Unseen ids start at ema 0.5."""
        ws = [max(self.ema_floor,
                  (1.0 - self._ema.get(self._opp_name(x), 0.5)) ** 2)
              for x in ids]
        return g.rng.choices(ids, weights=ws)[0]

    # -- scripted kaggle-format anchors ----------------------------------
    def _script_call(self, d, obs):
        fn = self._script_fns.get(d)
        if fn is None:
            from .arena import load_kaggle_agent
            fn = self._script_fns[d] = load_kaggle_agent(d)
            # alakazam_rising references sys only on its no-engine-search
            # branch (never taken on kaggle, where search_begin_input is
            # always supplied) without importing it — seed it so its
            # authored fallback (disable search, play heuristic) works.
            # NOTE: anchors therefore play WITHOUT engine search here;
            # eval_watch's battery still faces them at full strength.
            ns = getattr(fn, "__globals__", None)
            if ns is not None:
                import sys as _sys
                ns.setdefault("sys", _sys)
        old = os.getcwd()
        os.chdir(d)      # their deck.csv reads are cwd-relative
        try:
            return fn(obs)
        finally:
            os.chdir(old)

    def _script_deck_call(self, d) -> list[int]:
        """Game-start deck call, kaggle semantics (also lets stateful
        agents reset between games)."""
        from .arena import DECK_CALL
        deck = [int(c) for c in self._script_call(d, dict(DECK_CALL))]
        assert len(deck) == 60, f"{d}: deck call returned {len(deck)} cards"
        return deck

    def _refresh_pool(self) -> bool:
        if self.pool_dir:
            self._pool_files = sorted(
                glob.glob(os.path.join(self.pool_dir, "*.pt")))
        return bool(self._pool_files)

    def _pool_policy(self, path):
        pol = self._pool_cache.pop(path, None)
        if pol is None:
            import torch
            from .model import PTCGTransformer
            torch.set_num_threads(1)  # many workers per host; don't oversubscribe
            pol = PTCGTransformer(self)
            sd = torch.load(path, map_location="cpu")
            pol.load_state_dict(
                {k.replace("module.", ""): v for k, v in sd.items()})
            pol.eval()
            pol.freeze_tables()   # static tables: huge batch-1 CPU saving
            while len(self._pool_cache) >= 4:
                self._pool_cache.pop(next(iter(self._pool_cache)))
        self._pool_cache[path] = pol  # re-insert = LRU touch
        return pol

    def _snapshot_answer(self, g: _Game, seat: int) -> list[int]:
        """Frozen-checkpoint opponent answers one select (multi-picks via
        the same sequential decomposition the learner uses). Oracle block
        is left zero — policy logits are proven independent of it."""
        import torch
        pol = self._pool_policy(g.opp_kind)
        sel = g.obs["select"]
        n = len(sel["option"])
        picked = []
        while True:
            stop_ok = (len(picked) >= sel["minCount"]
                       and (sel["maxCount"] > 1 or sel["minCount"] == 0))
            encode(self._scratch[0], g.obs, picked, stop_ok, 0,
                   tracker=g.trackers[seat])
            with torch.no_grad():
                logits = pol.forward_policy(torch.from_numpy(self._scratch))
            a = int(torch.distributions.Categorical(logits=logits[0]).sample())
            if a == STOP:
                return picked
            if a >= n or a in picked:  # mask makes this near-impossible
                a = next((j for j in range(n) if j not in picked), None)
                if a is None:
                    return picked
            if sel["maxCount"] == 1:
                return [a]
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                return picked

    # -- core row driver -------------------------------------------------
    def _advance(self, g: _Game, i: int):
        """Run engine to the next policy decision (encode row) or end."""
        while True:
            cur = g.obs["current"]
            if cur["result"] != -1:
                self._finish_episode(g, i, result=cur["result"])
                return
            if g.engine_steps >= self.max_engine_steps:
                self._finish_episode(g, i, result=None)
                return
            sel = g.obs["select"]
            seat = cur["yourIndex"]
            if (g.opp_kind is not None and g.opp_kind != "frozen"
                    and seat != g.learner_seat):
                if g.opp_kind in self._script_set:
                    # scripted anchors answer EVERY select, forced ones
                    # included — exact kaggle runner semantics
                    ans = [int(a) for a in self._script_call(g.opp_kind,
                                                             g.obs)]
                else:
                    fa = forced_answer(sel)
                    if fa is not None:
                        g.submit(fa)
                        g.forced_run[seat] += 1
                        continue
                    ans = random_answer(sel, g.rng) \
                        if g.opp_kind == "random" \
                        else self._snapshot_answer(g, seat)
                try:
                    g.submit(ans)
                except RuntimeError:
                    if ans:   # non-empty rejection: let the guard handle it
                        raise
                    # engine rejected an advertised-legal [] (Select error
                    # 4 on minCount=0) — retry with the first option
                    g.submit([0])
                continue
            fa = forced_answer(sel)
            if fa is not None:
                g.submit(fa)
                g.forced_run[seat] += 1
                continue
            self._encode_row(g, i)
            return

    def _encode_row(self, g: _Game, i: int):
        sel = g.obs["select"]
        seat = g.obs["current"]["yourIndex"]
        stop_allowed = (len(g.picked) >= sel["minCount"]
                        and (sel["maxCount"] > 1 or sel["minCount"] == 0))
        encode(self.observations[i], g.obs, g.picked, stop_allowed,
               g.forced_run[seat], tracker=g.trackers[seat])
        write_oracle(self.observations[i], *g.oracle.emit(g.obs))
        if g.opp_kind == "frozen" and seat != g.learner_seat:
            # idx+1 so 0 stays "learner row"; the driver routes this row
            # through league member frozen_idx and the PPO policy loss
            # skips it (encode() zeroed the slot)
            self.observations[i][FROZEN_COL] = float(g.frozen_idx + 1)
        g.forced_run[seat] = 0
        g.decisions += 1

    @staticmethod
    def _end_cause(cur, winner: int) -> str:
        """Classify a decided game by win condition, in rule-check order:
        winner took all prizes > loser had no Pokemon to promote >
        loser could not draw at turn start (deck out)."""
        loser = cur["players"][1 - winner]
        if not cur["players"][winner]["prize"]:
            return "prize"
        if not any(p for p in loser["active"]) \
                and not any(p for p in loser["bench"]):
            return "bench"
        if loser["deckCount"] == 0:
            return "deck"
        return "other"

    def _perspective(self, g: _Game, actor: int) -> float:
        # Negamax reward contract: rewards[t+1] must be in
        # the currency of the actor who ACTED at t. Frozen-league games
        # surface OPPONENT rows, so the true actor seat signs the reward
        # there, exactly like mirror games. Every other opponent kind
        # answers between rows (single-learner-seat game), so forcing the
        # learner's seat is what folds those moves into the env dynamics.
        if g.opp_kind is not None and g.opp_kind != "frozen":
            actor = g.learner_seat
        return 1.0 if actor == 0 else -1.0

    def _finish_episode(self, g: _Game, i: int, result):
        """result: winner seat 0/1, 2 for draw, None for truncation."""
        margin_p0 = g.cum_p0 + g.pending_p0     # accrued reward, pre-top-up
        if result is None:
            self.truncations[i] = True
        else:
            self.terminals[i] = True
            if result in (0, 1):
                bonus_w = self.win_r - (margin_p0 if result == 0
                                        else -margin_p0)
                g.pending_p0 += bonus_w if result == 0 else -bonus_w
            else:                                   # draw: totals zero out
                g.pending_p0 += -margin_p0
            # invariant: episode total in p0 currency is exactly +-win_r
            tot = g.cum_p0 + g.pending_p0
            self._stats["ep_ret_abs"] += abs(tot)
            if abs(abs(tot) - self.win_r) > 1e-3 and abs(tot) > 1e-3:
                self._stats["ep_ret_bad"] += 1
        st = self._stats
        st["episodes"] += 1
        cur = g.obs["current"]
        if g.opp_kind is None:
            st["eps_mirror"] += 1
        elif g.opp_kind == "random":
            st["eps_rand"] += 1
        elif g.opp_kind in self._script_set:
            st["eps_script"] += 1
        else:
            st["eps_pool"] += 1
        cause = None
        if result in (0, 1):
            cause = self._end_cause(cur, result)
            if g.opp_kind is None:
                st[f"end_{cause}"] += 1
            elif result == g.learner_seat:
                st[f"wend_{cause}"] += 1
            else:
                st[f"lend_{cause}"] += 1
            if g.opp_kind is not None:
                won = result == g.learner_seat
                st["wins"] += won
                st["w_rand" if g.opp_kind == "random" else
                   ("w_script" if g.opp_kind in self._script_set
                    else "w_pool")] += won
                if g.opp_kind != "random":  # PFSP EMA, decided games only
                    nm = self._opp_name(g.opp_kind)
                    e_ = self._ema.get(nm, 0.5)
                    self._ema[nm] = e_ + self.ema_alpha * (float(won) - e_)
            st["first_wins"] += result == cur["firstPlayer"]
        elif result == 2:
            st["draws"] += 1
        st["ep_len"] += g.decisions
        # from the board, not reward accounting (works in reward='win' too)
        q0, q1 = g.prizes_p0()
        st["prize_margin"] += abs(q1 - q0)
        if result in (0, 1):
            st["eps_dec"] += 1
            st["prize_margin_dec"] += (q1 - q0) if result == 0 else (q0 - q1)
        self._episode_counter += 1
        if self.replay_dir and self.replay_every \
                and self._episode_counter % self.replay_every == 0:
            self._dump_replay(g, result)
        t = cur["turn"]
        st["turns_first"] += (t + 1) // 2
        st["turns_second"] += t // 2
        self.engine_steps_total += g.engine_steps
        if self._deck_log_path:
            opp = "self" if g.opp_kind is None else (
                "random" if g.opp_kind == "random"
                else os.path.basename(g.opp_kind).removesuffix(".pt"))
            self._deck_buf.append(json.dumps(
                {"d0": g.deck_idx[0], "d1": g.deck_idx[1],
                 "r": -1 if result is None else result, "c": cause,
                 "fp": cur["firstPlayer"], "turns": t, "opp": opp,
                 "t": int(time.time())}))
            if len(self._deck_buf) >= 50:
                self._flush_deck_log()
        return  # caller writes the reward, then resets

    def _flush_deck_log(self):
        if self._deck_buf:
            with open(self._deck_log_path, "a") as f:
                f.write("\n".join(self._deck_buf) + "\n")
            self._deck_buf = []

    # -- pufferlib API ---------------------------------------------------
    def reset(self, seed=None):
        for i, g in enumerate(self.games):
            self._start_game(g)
            self._advance(g, i)
        return self.observations, []

    def step(self, actions):
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        for i, g in enumerate(self.games):
            try:
                self._step_game(g, i, int(actions[i]))
            except Exception:
                # one freak engine state must never kill a long run —
                # surface it (env_errors in wandb) and restart the game
                import traceback
                traceback.print_exc()
                self._stats["env_errors"] += 1
                self.truncations[i] = True
                self._start_game(g)
                self._advance(g, i)

        self.tick += 1
        info = []
        st = self._stats
        if self.tick % self.log_interval == 0 and st["episodes"] > 0:
            e = st["episodes"]
            turns = st["turns_first"] + st["turns_second"]
            out = {"episodes": e, "ep_len": st["ep_len"] / e,
                   "prize_margin": st["prize_margin"] / e,
                   "first_win_rate": st["first_wins"] / max(1, e - st["draws"]),
                   "turns_first": st["turns_first"] / e,
                   "turns_second": st["turns_second"] / e,
                   "dec_per_turn": st["ep_len"] / max(1, turns),
                   "illegal": st["illegal"],
                   "env_errors": st["env_errors"],
                   "ep_ret_abs": st["ep_ret_abs"] / e,
                   "ep_ret_bad": st["ep_ret_bad"]}
            for pre in ("end", "wend", "lend"):
                tot = sum(st[f"{pre}_{c}"] for c in CAUSES)
                if tot:
                    for c in CAUSES:
                        # 'other' is the classifier's safety bucket —
                        # always 0 in practice; its very appearance on a
                        # dashboard is the alarm, so emit only if nonzero
                        if c != "other" or st[f"{pre}_{c}"]:
                            out[f"{pre}_{c}"] = st[f"{pre}_{c}"] / tot
            if self.opponent == "random":
                out["win_rate"] = st["wins"] / e
            elif self.opponent == "league":
                out["mirror_frac"] = st["eps_mirror"] / e
                if st["eps_pool"]:
                    out["win_vs_pool"] = st["w_pool"] / st["eps_pool"]
                if st["eps_rand"]:
                    out["win_vs_rand"] = st["w_rand"] / st["eps_rand"]
                if st["eps_script"]:
                    out["win_vs_script"] = st["w_script"] / st["eps_script"]
                for d_ in self.script_dirs:  # PFSP telemetry per anchor
                    nm = self._opp_name(d_)
                    if nm in self._ema:
                        out[f"wr_anchor_{nm}"] = self._ema[nm]
            info.append(out)
            self._stats = self._zero_stats()
        return (self.observations, self.rewards, self.terminals,
                self.truncations, info)

    @staticmethod
    def _unpicked(g: _Game, n: int):
        for j in range(n):
            if j not in g.picked:
                return j
        return None

    def _step_game(self, g: _Game, i: int, a: int):
        sel = g.obs["select"]
        actor = g.obs["current"]["yourIndex"]
        n = len(sel["option"])
        if a == STOP:
            stop_ok = (len(g.picked) >= sel["minCount"]
                       and (sel["maxCount"] > 1 or sel["minCount"] == 0))
            if stop_ok:
                try:
                    g.submit(g.picked)
                    self._flush(g, i, actor)
                    return
                except RuntimeError:
                    # engine rejected an advertised-legal submit (seen:
                    # "Select error 4 for []" on minCount=0 selects, ~2 per
                    # 3h). State is unchanged after a rejected Select —
                    # fall through and pick something instead.
                    self._stats["illegal"] += 1
                    a = self._unpicked(g, n)
                    if a is None:
                        raise  # outer guard restarts the game
            else:
                self._stats["illegal"] += 1
                a = self._unpicked(g, n)
        elif a >= n or a in g.picked:
            self._stats["illegal"] += 1
            a = self._unpicked(g, n)
        if a is None:
            # engine offered fewer distinct picks than maxCount promised
            # (contract violation, ~1 per 5M steps): submit what we have
            g.submit(g.picked)
            self._flush(g, i, actor)
            return
        if sel["maxCount"] == 1:
            g.submit([a])
            self._flush(g, i, actor)
        else:
            g.picked.append(a)
            if len(g.picked) >= sel["maxCount"]:
                g.submit(g.picked)
                self._flush(g, i, actor)
            else:
                self._encode_row(g, i)  # same select, next pick

    def _flush(self, g: _Game, i: int, actor: int):
        """Advance to the next decision, then credit the accrued reward
        (perspective of the seat that just acted) to this step."""
        self._advance(g, i)
        self.rewards[i] += self._perspective(g, actor) * g.pending_p0
        g.cum_p0 += g.pending_p0
        g.pending_p0 = 0.0
        if self.terminals[i] or self.truncations[i]:
            self._start_game(g)
            self._advance(g, i)  # fresh episode never ends pre-decision

    def _dump_replay(self, g: _Game, result):
        """Dump the finished game as a Kaggle-episode-shaped JSON, the
        same shape a replay viewer consumes."""
        import json
        frames = g.handle.visualize()
        if not frames:
            return
        frames[0]["action"] = [list(g.deck_pair[0]), list(g.deck_pair[1])]
        teams = list(self.replay_teams)
        if g.opp_kind is not None:
            opp_name = "random" if g.opp_kind == "random" else \
                os.path.basename(g.opp_kind).removesuffix(".pt")
            teams = [teams[0], opp_name]
            if g.learner_seat == 1:
                teams.reverse()  # labels follow the actual seat assignment
        # unique across env instances AND time — a fresh env (new eval
        # pass) restarts the counter, and pid alone caused overwrites
        import time as _time
        ep_id = f"9{int(_time.time()) % 10**7:07d}{self._episode_counter % 1000:03d}"
        rewards = [0, 0]
        if result in (0, 1):
            rewards = [1, -1] if result == 0 else [-1, 1]
        wrapped = {
            "id": f"rl-{ep_id}",
            "name": "rl-battle",
            "info": {"EpisodeId": int(ep_id), "TeamNames": teams},
            "rewards": rewards,
            "statuses": ["DONE", "DONE"],
            "steps": [[{"visualize": frames}]],
        }
        path = os.path.join(self.replay_dir, f"{ep_id}.json")
        with open(path, "w") as f:
            json.dump(wrapped, f)

    def close(self):
        if self._deck_log_path:
            self._flush_deck_log()
        for g in self.games:
            if g.handle is not None:
                g.handle.finish()
