"""Dump a parity stream for CHOSEN decks, without pufferlib or the RL env.

dump_stream.py drives the full PTCGEnv, which pulls in pufferlib+gymnasium and
draws decks from the pool at random. That is the right capture for general
coverage (it reaches multi-select `picked` runs and forced-run counters), but
it cannot target an archetype — and a mechanic that only a handful of decks
can trigger ends up with a few rows of coverage in a 1300-row stream, which is
not enough to say the C port of the tracker agrees with the python one.

This drives BattleHandle directly, exactly like tools/validate_obs_v4.py, over
a deck glob you choose. Same stream format as dump_stream.py, so
native/ptcg/parity_replay.c consumes it unchanged:

    {"start": {"deck0": [...60], "deck1": [...60]}}
    {"obs": "<raw json>", "viz": "<raw json>" | null}
    {"enc": {"picked": [...], "stop": 0/1, "forced": n}}

Deliberately answers one option at a time (picked always empty, stop false):
those paths are covered by the general stream, and keeping the driver simple
is what makes it trustworthy as a gate.

Run from repo root:
  PYTHONPATH=data:. python3 native/tools/dump_stream_decks.py \\
      OUT_STREAM.jsonl OUT_ROWS.bin 'decks/pool/slowking*.csv' [n_games]
"""
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

from cg.sim import lib                                  # noqa: E402
import ptcg.rl.battle as battle_mod                     # noqa: E402
from ptcg.rl.buffers import OBS_SIZE                    # noqa: E402
from ptcg.rl.encoder import encode, write_oracle        # noqa: E402
from ptcg.rl.oracle import OracleLedger                 # noqa: E402
from ptcg.tracker import InfoTracker                    # noqa: E402


class HookedHandle(battle_mod.BattleHandle):
    """Captures the raw obs/viz strings as they are consumed, so the stream
    records exactly the bytes the C replayer must parse."""

    def obs(self):
        raw = lib.GetBattleData(self.ptr).json.decode()
        self.last_obs_raw = raw
        self.last_viz_raw = None
        return json.loads(raw)

    # BattleHandle.select() returns self.obs(), so the override above is what
    # captures the post-select raw string too — no select() hook needed.

    def visualize(self):
        raw = lib.VisualizeData(self.ptr).decode()
        self.last_viz_raw = raw
        return json.loads(raw)


def main(out_stream, out_rows, pattern, n_games):
    decks = []
    for f in sorted(glob.glob(pattern)):
        ids = [int(x) for x in open(f).read().split()]
        if len(ids) == 60:
            decks.append(ids)
    assert decks, f"no 60-card decks matched {pattern}"
    rng = random.Random(11)
    row = np.zeros(OBS_SIZE, dtype=np.float32)
    n_rows = n_obs = 0

    with open(out_stream, "w") as fs, open(out_rows, "wb") as fr:
        for _ in range(n_games):
            d0, d1 = rng.choice(decks), rng.choice(decks)
            h = HookedHandle(d0, d1)
            trackers = [InfoTracker(d0), InfoTracker(d1)]
            oracle = OracleLedger((d0, d1), h)
            fs.write(json.dumps({"start": {"deck0": list(d0),
                                           "deck1": list(d1)}}) + "\n")
            obs = h.obs()
            steps = 0
            while obs["current"]["result"] == -1 and steps < 3000:
                seat = obs["current"]["yourIndex"]
                raw = h.last_obs_raw
                trackers[seat].update(obs)
                oracle.on_seat_obs(obs)      # may pull viz; recorded below
                fs.write(json.dumps({"obs": raw,
                                     "viz": h.last_viz_raw}) + "\n")
                n_obs += 1

                encode(row, obs, [], False, 0, tracker=trackers[seat])
                write_oracle(row, *oracle.emit(obs))
                fs.write(json.dumps(
                    {"enc": {"picked": [], "stop": 0, "forced": 0}}) + "\n")
                fr.write(np.ascontiguousarray(row, dtype=np.float32).tobytes())
                n_rows += 1

                sel = obs["select"]
                n = len(sel["option"])
                lo, hi = sel["minCount"], min(sel["maxCount"], n)
                k = rng.randint(lo, max(lo, hi))
                obs = h.select(rng.sample(range(n), min(k, n)) or [0])
                steps += 1
            h.finish()

    print(f"{n_games} games / {n_obs} obs -> {n_rows} rows "
          f"({n_rows * OBS_SIZE * 4 / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 12)
