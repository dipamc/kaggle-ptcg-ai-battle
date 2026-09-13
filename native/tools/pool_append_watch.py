"""Publish mid-run deck-pool appends on a step schedule (docs/deck-pool.md).

The trainer never grows its own pool: it polls a request file for
"<version> <blob path>" and swaps at a rollout/train boundary once every rank
reads the same request. This is the writer half — the thing that decides WHEN.

  PYTHONPATH=. python3 native/tools/pool_append_watch.py \\
      --log runs/<RUN>.log --request runs/<RUN>.poolreq --world 2 \\
      --at 15000000=native/ptcg_tables_s2.bin \\
      --at 30000000=native/ptcg_tables_s3.bin \\
      --at 45000000=native/ptcg_tables_s4.bin

Three things worth knowing:

* **--at takes GLOBAL steps.** The trainer's log line prints rank 0's own
  `global_step`, and `total_timesteps` is divided by world size, so global
  progress is `printed_step * world`. Passing --world wrong is the easy way to
  fire every append at half or double the intended point.
* **Blobs must be a true append chain**, each built with `--append-to` against
  the previous listing. `pt_reload_decks` rejects a removal, any change to an
  existing row, or a changed coverage pool — a rejected blob is logged by the
  trainer and training continues on the old pool, so a bad chain fails quietly
  in the sense that the run keeps going with fewer decks than you think.
* Versions start at 2. The initial pool arrives via PTCG_TABLES, not a
  request, and the trainer treats version 0 as "no request".

Safe to restart: it reads the whole log from the top, so anything already past
its step is published immediately, and the trainer ignores a version it has
already applied (it remembers across resumes in <run_dir>/pool_version).
"""
import argparse
import os
import re
import sys
import time

STEP_RE = re.compile(r"^epoch (\d+) step (\d+) ")


def latest_step(log_path):
    """Rank 0's last reported global_step, or None if it has not logged yet."""
    if not os.path.exists(log_path):
        return None
    last = None
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                last = int(m.group(2))
    return last


def publish(request_path, version, blob):
    tmp = request_path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(f"{version} {blob}\n")
    os.replace(tmp, request_path)          # atomic: ranks never see a partial


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", required=True, help="trainer stdout log (rank 0)")
    p.add_argument("--request", required=True, help="--deck-pool-file path")
    p.add_argument("--world", type=int, default=1, help="number of ranks")
    p.add_argument("--at", action="append", default=[], metavar="STEPS=BLOB",
                   help="publish BLOB once global steps reach STEPS; repeatable")
    p.add_argument("--poll", type=float, default=30.0, help="seconds")
    a = p.parse_args()

    schedule = []
    for i, item in enumerate(a.at):
        steps, _, blob = item.partition("=")
        if not blob:
            raise SystemExit(f"--at needs STEPS=BLOB, got {item!r}")
        if not os.path.exists(blob):
            raise SystemExit(f"blob does not exist: {blob}")
        schedule.append((int(steps), i + 2, blob))    # versions start at 2
    schedule.sort()
    if not schedule:
        raise SystemExit("nothing scheduled")

    print(f"watching {a.log} (world={a.world}); schedule:", flush=True)
    for steps, ver, blob in schedule:
        print(f"  global step {steps:>12,} -> v{ver} {blob}", flush=True)

    done = 0
    while done < len(schedule):
        step = latest_step(a.log)
        if step is not None:
            world_step = step * a.world
            while done < len(schedule) and world_step >= schedule[done][0]:
                target, ver, blob = schedule[done]
                publish(a.request, ver, blob)
                print(f"[{time.strftime('%H:%M:%S')}] global step "
                      f"{world_step:,} >= {target:,} -> published v{ver} "
                      f"{blob}", flush=True)
                done += 1
        time.sleep(a.poll)

    print("all appends published; watcher exiting", flush=True)


if __name__ == "__main__":
    main()
