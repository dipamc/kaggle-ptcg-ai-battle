#!/usr/bin/env python3
"""Self-checkpoint deck eval: ONE checkpoint pilots BOTH seats, only the decks
differ. Isolates deck effect with the policy held exactly constant.

    PYTHONPATH=data:. python selfeval.py job.json [n_workers]

job: {ckpt, cells: [[my_deck, opp_deck], ...], games, device, heads, temp,
      seed, out}

Imports ONLY arena + search_agent on purpose: eval_watch drags in pufferlib
via a module-level `from .env import PTCGEnv` that this path never uses.
Seats alternate every game, so a first-player advantage cannot bias a cell.
"""
import json
import os
import random
import sys


def run_cells(cells, ckpt, device, heads, games, temp, seed):
    from ptcg.rl.arena import play as arena_play
    from ptcg.rl.search_agent import SearchAgent
    rng = random.Random(seed)
    agents = {}

    def get(deck):
        if deck not in agents:
            agents[deck] = SearchAgent(ckpt, deck, cfg=None, device=device,
                                       heads=heads, temp=temp)
        return agents[deck]

    out = []
    for my_deck, opp_deck in cells:
        me, opp = get(my_deck), get(opp_deck)
        wins = 0
        for i in range(games):
            # my_seat alternates; play() returns the winning seat index
            r, steps, dec, cause = arena_play(me, opp.agent, ".", i % 2, rng, [])
            wins += (r == i % 2)
        out.append({"my_deck": my_deck, "opp_deck": opp_deck,
                    "games": games, "wins": wins, "wr": wins / games})
        print(f"  {os.path.basename(my_deck)[:34]:34s} vs "
              f"{os.path.basename(opp_deck)[:34]:34s} {wins}/{games} "
              f"= {wins/games:.3f}", flush=True)
    return out


def main():
    job = json.load(open(sys.argv[1]))
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cells = [tuple(c) for c in job["cells"]]

    if nw <= 1:
        res = run_cells(cells, job["ckpt"], job["device"], job["heads"],
                        job["games"], job.get("temp", 0.0), job["seed"])
        json.dump(res, open(job["out"], "w"), indent=1)
        return

    # round-robin split; each worker gets a decorrelated seed, so results are
    # statistically equivalent to serial but not bit-identical
    parts = []
    for w in range(nw):
        sub = cells[w::nw]
        if not sub:
            continue
        p = dict(job, cells=sub, seed=job["seed"] + 1000 * w,
                 out=f"{job['out']}.part{w}")
        jp = f"{sys.argv[1]}.part{w}"
        json.dump(p, open(jp, "w"))
        parts.append((jp, p["out"]))
    pids = []
    for jp, _ in parts:
        pid = os.fork()
        if pid == 0:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.execvp(sys.executable, [sys.executable, sys.argv[0], jp])
        pids.append(pid)
    bad = 0
    for pid in pids:
        _, st = os.waitpid(pid, 0)
        bad += (st != 0)
    res = []
    for jp, op in parts:
        if os.path.exists(op):
            res.extend(json.load(open(op)))
            os.remove(op)
        os.remove(jp)
    if bad:
        print(f"WARNING: {bad}/{len(pids)} workers exited non-zero", flush=True)
    json.dump(res, open(job["out"], "w"), indent=1)
    print(f"wrote {job['out']}: {len(res)} cells")


if __name__ == "__main__":
    main()
