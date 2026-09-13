"""Dump a parity stream: real games driven by the python env with random
legal actions, capturing (a) every raw GetBattleData JSON string each seat
ingested, (b) every VisualizeData payload the oracle pulled, (c) every
encode() call's inputs, and (d) the exact float32 rows python produced.

The C replayer (native/ptcg/parity_replay.c) feeds the same strings through
the C tracker/oracle/encoder and compares rows bit-level.

Output:
  native/parity_stream.jsonl  — records:
      {"start": {"deck0": [...60], "deck1": [...60]}}
      {"obs": "<raw json>", "viz": "<raw json>" | null}
      {"enc": {"picked": [...], "stop": 0/1, "forced": n}}
  native/parity_rows.bin      — float32 rows, one per "enc" record, in order

Run from repo root: PYTHONPATH=data:. python native/tools/dump_stream.py [games]
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

from cg.sim import lib
import ptcg.rl.battle as battle_mod
from ptcg.rl import env as env_mod
from ptcg.rl.buffers import OBS_SIZE, OFFSETS

STREAM = open("native/parity_stream.jsonl", "w")
ROWS = open("native/parity_rows.bin", "wb")
N_ROWS = [0]
IN_START = [False]


class HookedHandle(battle_mod.BattleHandle):
    """BattleHandle that captures raw obs/viz strings as the env consumes them."""

    def obs(self):
        sd = lib.GetBattleData(self.ptr)
        raw = sd.json.decode()
        self._pending_obs = raw          # emitted lazily, with viz attached
        self._pending_viz = None
        return json.loads(raw)

    def visualize(self):
        raw = lib.VisualizeData(self.ptr).decode()
        self._pending_viz = raw
        return json.loads(raw)


def flush_obs(handle):
    raw = getattr(handle, "_pending_obs", None)
    if raw is None:
        return
    STREAM.write(json.dumps({"obs": raw, "viz": handle._pending_viz}) + "\n")
    handle._pending_obs = None
    handle._pending_viz = None


orig_ingest = env_mod._Game._ingest
orig_start = env_mod._Game.start
orig_encode_row = env_mod.PTCGEnv._encode_row


def ingest_hook(self):
    orig_ingest(self)
    if not IN_START[0]:              # start()'s first obs flushes after the marker
        flush_obs(self.handle)


def start_hook(self, decks, script_deck=None, my_idx=None, opp_idx=None):
    IN_START[0] = True
    try:
        orig_start(self, decks, script_deck=script_deck,
                   my_idx=my_idx, opp_idx=opp_idx)
    finally:
        IN_START[0] = False
    STREAM.write(json.dumps({"start": {"deck0": list(self.deck_pair[0]),
                                       "deck1": list(self.deck_pair[1])}}) + "\n")
    flush_obs(self.handle)


def encode_row_hook(self, g, i):
    sel = g.obs["select"]
    seat = g.obs["current"]["yourIndex"]
    stop_allowed = (len(g.picked) >= sel["minCount"]
                    and (sel["maxCount"] > 1 or sel["minCount"] == 0))
    forced = g.forced_run[seat]
    orig_encode_row(self, g, i)
    STREAM.write(json.dumps({"enc": {"picked": list(g.picked),
                                     "stop": int(stop_allowed),
                                     "forced": forced}}) + "\n")
    ROWS.write(np.ascontiguousarray(
        self.observations[i], dtype=np.float32).tobytes())
    N_ROWS[0] += 1


env_mod._Game._ingest = ingest_hook
env_mod._Game.start = start_hook
env_mod.PTCGEnv._encode_row = encode_row_hook
battle_mod.BattleHandle = HookedHandle


def main(n_games=8):
    env = env_mod.PTCGEnv(num_envs=1, opponent="league", mix_self=0.9,
                          mix_pool=0.0, mix_script=0.0, reward="win",
                          seed=1234)
    rng = random.Random(7)
    env.reset()
    games_done = 0
    steps = 0
    a0, b0 = OFFSETS["opt_mask"]
    while games_done < n_games:
        mask = env.observations[0][a0:b0]
        legal = np.nonzero(mask)[0]
        assert len(legal) > 0, "no legal actions"
        a = int(rng.choice(legal))
        _, _, terminals, truncations, _ = env.step(np.array([a]))
        steps += 1
        if terminals[0] or truncations[0]:
            games_done += 1
    env.close()
    STREAM.close()
    ROWS.close()
    print(f"dumped {games_done} games, {steps} steps, {N_ROWS[0]} rows "
          f"({N_ROWS[0] * OBS_SIZE * 4 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
