#!/usr/bin/env python3
"""selfeval with a batched-inference server: same job format, same game loop,
same decision semantics — only the model forward moves to one shared process.

    PYTHONPATH=data:. python selfeval_batched.py job.json [n_workers]

Why: per-obs GPU inference measured a wash (launch-overhead + spin-wait
bound; see decklab/config.json _notes). Here W engine workers run the
UNTOUCHED SearchAgent/arena_play loop, but `_logits` ships the encoded row
over shared memory to a server that stacks every pending request into ONE
forward. All agents share one checkpoint (selfeval invariant), so a single
resident model serves every game. Batches form naturally: whatever arrives
during the previous forward is the next batch — no wait window, no added
latency.

Worker crash => its cells are missing from the output, same contract as
selfeval.py (caller counts cells). Results are NOT bit-identical to
selfeval.py (batched matmul reduction order), but with temp=0 the per-game
trajectories only diverge on float-tie argmax flips — validate statistically.
"""
import json
import os
import select as _select
import struct
import sys
import time
from multiprocessing import Pipe, shared_memory

N_LOGIT_MAX = 4096      # response slot stride (floats, minus the value slot);
                        # BOTH sides must use this stride — the real logit
                        # width only says how much of the slot is meaningful


def worker(job, wid, req_shm_name, resp_shm_name, pipe, obs_size):
    """One engine worker: the selfeval cell loop with remote forwards."""
    import numpy as np
    # read logits width from the server before touching torch/model
    hdr = pipe.recv_bytes()
    n_logit = struct.unpack("i", hdr)[0]
    req = shared_memory.SharedMemory(name=req_shm_name)
    resp = shared_memory.SharedMemory(name=resp_shm_name)
    row_slot = np.ndarray((obs_size,), dtype=np.float32,
                          buffer=req.buf, offset=wid * obs_size * 4)
    slot = np.ndarray((N_LOGIT_MAX + 1,), dtype=np.float32,
                      buffer=resp.buf, offset=wid * (N_LOGIT_MAX + 1) * 4)
    out_slot = slot[:n_logit + 1]   # [0:n_logit] logits, [n_logit] value

    # torch is pre-imported by the parent (inherited via fork, COW pages —
    # no per-worker import cost); ptcg/cg imports stay HERE so each child
    # dlopens the engine itself and gets an independent C-side RNG state
    import torch
    import random
    from ptcg.rl.search_agent import SearchAgent
    from ptcg.rl.arena import play as arena_play

    class RemoteAgent(SearchAgent):
        def _logits(self, obs, picked, stop_ok, forced_run, tracker):
            from ptcg.rl.encoder import encode
            encode(self.row[0], obs, picked, stop_ok, forced_run,
                   tracker=tracker)
            row_slot[:] = self.row[0]
            pipe.send_bytes(b"r")
            if not pipe.poll(120):
                raise RuntimeError("inference server timeout")
            pipe.recv_bytes()
            self.evals += 1
            return torch.from_numpy(out_slot[:n_logit].copy()), float(out_slot[n_logit])

    rng = random.Random(job["seed"])
    agents = {}

    def get(deck):
        if deck not in agents:
            agents[deck] = RemoteAgent(job["ckpt"], deck, cfg=None,
                                       device="cpu", heads=job["heads"],
                                       temp=job.get("temp", 0.0))
        return agents[deck]

    out = []
    for my_deck, opp_deck, games in job["cells"]:
        me, opp = get(my_deck), get(opp_deck)
        wins = 0
        for i in range(games):
            r, steps, dec, cause = arena_play(me, opp.agent, ".", i % 2, rng, [])
            wins += (r == i % 2)
        out.append({"my_deck": my_deck, "opp_deck": opp_deck,
                    "games": games, "wins": wins})
        print(f"  [w{wid}] {os.path.basename(my_deck)[:30]:30s} vs "
              f"{os.path.basename(opp_deck)[:30]:30s} {wins}/{games}",
              flush=True)
    json.dump(out, open(job["out"], "w"))
    pipe.send_bytes(b"q")
    pipe.close()


def shard_cells(cells, games, nw):
    """Split each cell's games into EVEN-sized shards so games within a cell
    parallelize across workers (selfeval.py serializes them — the latency
    floor for small probe jobs). Even sizes keep seat alternation balanced
    inside every shard. Target ~2 shards per worker for load balance."""
    per = max(2, 2 * round(games * len(cells) / (2.0 * nw) / max(len(cells), 1)))
    shards = []
    for my, opp in cells:
        left = games
        while left > 0:
            g = min(per, left)
            if left - g == 1:       # never strand an odd single game
                g += 1
            shards.append((my, opp, g))
            left -= g
    return shards


def main():
    job = json.load(open(sys.argv[1]))
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cells = shard_cells([tuple(c) for c in job["cells"]], job["games"], nw)

    # Preload the heavy import so forked workers inherit it for free.
    # Thread caps MUST be set before these imports: OpenBLAS/OMP size their
    # pools at library load, every forked child inherits that pool config,
    # and N children x 16 threads thrashes the host (2x slowdown).
    # CUDA must NOT be touched before the forks — model .to(device) happens
    # strictly after every worker exists.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    import numpy  # noqa: F401
    import torch  # noqa: F401
    from ptcg.rl.buffers import OBS_SIZE

    parts, pipes = [], []
    kids = []
    req = shared_memory.SharedMemory(create=True, size=nw * OBS_SIZE * 4)
    resp = shared_memory.SharedMemory(create=True,
                                      size=nw * (N_LOGIT_MAX + 1) * 4)
    try:
        for w in range(nw):
            sub = cells[w::nw]
            if not sub:
                continue
            p = dict(job, cells=sub, seed=job["seed"] + 1000 * w,
                     out=f"{job['out']}.part{w}")
            server_end, worker_end = Pipe()
            pid = os.fork()
            if pid == 0:
                server_end.close()
                os.environ["OMP_NUM_THREADS"] = "1"
                try:
                    worker(p, w, req.name, resp.name, worker_end, OBS_SIZE)
                    os._exit(0)
                except Exception as e:
                    print(f"worker {w} died: {e}", flush=True)
                    os._exit(1)
            worker_end.close()
            kids.append(pid)
            pipes.append((w, server_end))
            parts.append(p["out"])

        # ---- inference server (this process) ----
        import numpy as np
        import torch
        from ptcg.rl.model import PTCGTransformer, arch_from_state_dict
        dev = torch.device(job.get("device", "cpu"))
        if dev.type == "cpu":
            # the pre-import caps above would strangle a CPU-device server;
            # torch's runtime API overrides them for this process only
            torch.set_num_threads(min(8, os.cpu_count() or 8))
        sd = torch.load(job["ckpt"], map_location="cpu")
        policy = PTCGTransformer(**arch_from_state_dict(sd, heads=job["heads"]))
        policy.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
        policy.eval().to(dev)
        policy.freeze_tables()
        with torch.no_grad():
            probe_out = policy.forward_eval(torch.zeros((1, OBS_SIZE)).to(dev))
        n_logit = int(probe_out[0].shape[-1])
        assert n_logit <= N_LOGIT_MAX, f"logits {n_logit} > shm slot"
        req_arr = np.ndarray((nw, OBS_SIZE), dtype=np.float32, buffer=req.buf)
        resp_arr = np.ndarray((nw, N_LOGIT_MAX + 1), dtype=np.float32,
                              buffer=resp.buf)
        for _, pe in pipes:
            pe.send_bytes(struct.pack("i", n_logit))

        live = dict(pipes)
        t0, forwards, served, bmax = time.time(), 0, 0, 0
        while live:
            ready, _, _ = _select.select([pe for pe in live.values()], [], [], 5.0)
            if not ready:
                continue
            batch = []
            for w in list(live):
                pe = live[w]
                got_req = False
                while pe.poll():
                    try:
                        msg = pe.recv_bytes()
                    except EOFError:
                        msg = b"q"
                    if msg == b"q":
                        del live[w]
                        got_req = False
                        break
                    got_req = True     # collapse any dupes; one row per worker
                if got_req:
                    batch.append(w)
            if not batch:
                continue
            x = torch.from_numpy(req_arr[batch].copy()).to(dev)
            with torch.no_grad():
                out = policy.forward_eval(x)
            logits = out[0].float().cpu().numpy()
            vals = out[1].float().cpu().numpy().reshape(-1)
            for bi, w in enumerate(batch):
                resp_arr[w, :n_logit] = logits[bi]
                resp_arr[w, n_logit] = vals[bi]
                live[w].send_bytes(b"d")
            forwards += 1
            served += len(batch)
            bmax = max(bmax, len(batch))

        bad = 0
        for pid in kids:
            _, st = os.waitpid(pid, 0)
            bad += (st != 0)
        agg = {}
        for op in parts:
            if os.path.exists(op):
                for r in json.load(open(op)):
                    k = (r["my_deck"], r["opp_deck"])
                    a = agg.setdefault(k, {"my_deck": k[0], "opp_deck": k[1],
                                           "games": 0, "wins": 0})
                    a["games"] += r["games"]
                    a["wins"] += r["wins"]
                os.remove(op)
        res = list(agg.values())
        for r in res:
            r["wr"] = r["wins"] / max(r["games"], 1)
        if bad:
            print(f"WARNING: {bad}/{len(kids)} workers exited non-zero", flush=True)
        json.dump(res, open(job["out"], "w"), indent=1)
        dt = time.time() - t0
        tot_games = sum(r["games"] for r in res)
        print(f"wrote {job['out']}: {len(res)} cells; {tot_games} games in "
              f"{dt:.1f}s = {tot_games/dt:.1f} games/s; {forwards} forwards, "
              f"mean batch {served/max(forwards,1):.1f}, max {bmax}", flush=True)
    finally:
        req.close(); req.unlink()
        resp.close(); resp.unlink()


if __name__ == "__main__":
    main()
