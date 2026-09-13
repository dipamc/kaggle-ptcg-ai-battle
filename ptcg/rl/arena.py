"""Arena: our (search) agent vs an external kaggle-style agent dir.

Hosts battles through cg.game (the singleton battle — one game at a
time per process — which also supplies search_begin_input so the
search agent works exactly as it would on Kaggle). The opponent dir
must contain a kaggle-format main.py + deck.csv; it is loaded the way
kaggle-environments loads it (exec, no __file__, last callable) with
cwd switched to its dir so relative deck.csv reads work.

    python -m ptcg.rl.arena --ckpt <model.pt> --deck decks/pool/... \
        --opponent opponents/meta18_B --games 20 --heads 4 \
        --search "actions=4,dets=2,horizon=20" --out results.jsonl
"""
import argparse
import json
import os
import random
import sys
import time


def load_kaggle_agent(agent_dir, isolate=("ptcg",)):
    """Exec-load a kaggle-format agent. `isolate` package names are
    hidden from sys.modules around the exec so a bundle that VENDORS
    them (e.g. an old-obs-VERSION checkpoint bundle carrying its own
    v3 ptcg) imports its own copy instead of the host process's —
    then unhidden, so the host keeps its versions. The bundle's
    module-level references keep working (they hold object refs)."""
    src = open(os.path.join(agent_dir, "main.py")).read()
    env = {}
    saved = {}
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in isolate):
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, agent_dir)
    old = os.getcwd()
    os.chdir(agent_dir)      # their deck.csv reads are cwd-relative
    try:
        exec(compile(src, "<string>", "exec"), env)
    finally:
        os.chdir(old)
        sys.path.remove(agent_dir)
        for name in list(sys.modules):
            if any(name == p or name.startswith(p + ".") for p in isolate):
                del sys.modules[name]
        sys.modules.update(saved)
    return [v for v in env.values() if callable(v)][-1]


DECK_CALL = {"select": None, "logs": [], "current": None,
             "search_begin_input": None}  # full kaggle deck-call shape


def play(my_agent, opp_fn, opp_dir, my_seat, rng, decision_times):
    from cg import game
    my_deck = my_agent.agent(dict(DECK_CALL))
    old = os.getcwd()
    os.chdir(opp_dir)
    try:
        opp_deck = opp_fn(dict(DECK_CALL))
    finally:
        os.chdir(old)
    decks = [my_deck, opp_deck] if my_seat == 0 else [opp_deck, my_deck]
    obs, sd = game.battle_start(decks[0], decks[1])
    assert obs is not None, f"battle_start failed: {sd.errorType}"
    steps = 0
    my_decisions = 0
    while obs["current"]["result"] == -1 and steps < 3000:
        seat = obs["current"]["yourIndex"]
        if seat == my_seat:
            t0 = time.time()
            ans = my_agent.agent(obs)
            decision_times.append(time.time() - t0)
            my_decisions += 1
        else:
            os.chdir(opp_dir)
            try:
                ans = opp_fn(obs)
            finally:
                os.chdir(old)
        try:
            obs = game.battle_select(ans)
        except Exception:
            if ans:
                # side that produced the bad answer forfeits
                game.battle_finish()
                return (1 - seat), steps, my_decisions, "illegal"
            obs = game.battle_select([0])
        steps += 1
    r = obs["current"]["result"]
    cur = obs["current"]
    cause = None
    if r in (0, 1):
        w, l = cur["players"][r], cur["players"][1 - r]
        cause = ("prize" if not w["prize"] else
                 "bench" if not any(p for p in l["active"]) and
                 not any(p for p in l["bench"]) else
                 "deck" if l["deckCount"] == 0 else "other")
    game.battle_finish()
    return r, steps, my_decisions, cause


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--search", default=None,
                    help="e.g. 'mode=turn,actions=6,dets=3,gate_gap=0.35,"
                         "min_ev=3,margin=0.15'; omit = policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    # Not inferrable from the weights -- qkv is d x d at any head count, so a
    # wrong value loads cleanly and computes different attention (see
    # arch_from_state_dict). Must match how the run was LAUNCHED: 4 for
    # d128 models, 8 for d256 ones.
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="real-play sampling temp; 0 = greedy argmax")
    ap.add_argument("--winline", type=int, default=0,
                    help="1 = engine-proven win-this-turn override")
    ap.add_argument("--wl-worlds", type=int, default=32,
                    help="verification worlds; cheap under --wl-certify 1 "
                         "(replays), expensive under 0 (full searches)")
    ap.add_argument("--wl-nodes", type=int, default=3000)
    ap.add_argument("--wl-certify", type=int, default=1,
                    help="1 = the executed line must replay to a win in "
                         "every world (v3); 0 = shipped v1/v2 rule")
    ap.add_argument("--wl-strict", type=int, default=0,
                    help="certify only: replay must also match world 0's "
                         "visible states (costs recall, buys no soundness)")
    ap.add_argument("--wl-cands", type=int, default=3,
                    help="candidate lines tried per decision")
    ap.add_argument("--benchguard", type=int, default=0,
                    help="1 = play a Basic instead of ENDing at 1-in-play")
    ap.add_argument("--ens", default=None,
                    help="extra checkpoints 'path:heads,path:heads' whose "
                         "softmax probs are averaged with the primary")
    ap.add_argument("--vrank", type=float, default=0.0,
                    help="V-leaf override threshold (0 = off)")
    ap.add_argument("--vrank-gate", type=float, default=0.35)
    ap.add_argument("--vrank-worlds", type=int, default=2)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    from .search_agent import SearchAgent, SearchCfg
    cfg = None
    if args.search:
        def _coerce(v):
            for t in (int, float):
                try:
                    return t(v)
                except ValueError:
                    pass
            return v
        cfg = SearchCfg(**{k: _coerce(v) for k, v in
                           (kv.split("=") for kv in args.search.split(","))})
    ens = None
    if args.ens:
        ens = [(p, int(h)) for p, h in
               (kv.rsplit(":", 1) for kv in args.ens.split(","))]
    me = SearchAgent(args.ckpt, args.deck, cfg=cfg, seed=args.seed,
                     device=args.device, heads=args.heads, temp=args.temp,
                     winline=args.winline, wl_worlds=args.wl_worlds,
                     wl_nodes=args.wl_nodes, wl_certify=args.wl_certify,
                     wl_strict=args.wl_strict, wl_cands=args.wl_cands,
                     benchguard=args.benchguard,
                     ens=ens, vrank=args.vrank, vrank_gate=args.vrank_gate,
                     vrank_worlds=args.vrank_worlds)
    opp = load_kaggle_agent(args.opponent)

    rng = random.Random(args.seed)
    wins = draws = 0
    times = []
    rows = []
    for i in range(args.games):
        seat = i % 2
        me.evals = 0
        me.wl_checked = me.wl_proved = me.wl_nodes_spent = 0
        me.wl_cands_tried = me.wl_vfail_win = me.wl_vfail_key = 0
        me.wl_cached = me.wl_replans = 0
        me.bg_seen = me.bg_fired = 0
        t0 = time.time()
        r, steps, dec, cause = play(me, opp, args.opponent, seat, rng, times)
        won = r == seat
        wins += won
        draws += r == 2
        rows.append({"game": i, "seat": seat, "result": int(r), "won": bool(won),
                     "cause": cause, "steps": steps, "my_decisions": dec,
                     "evals": me.evals, "wall_s": round(time.time() - t0, 1),
                     "wl_checked": me.wl_checked, "wl_proved": me.wl_proved,
                     "wl_nodes": me.wl_nodes_spent,
                     "wl_cands": me.wl_cands_tried,
                     "wl_vfail_win": me.wl_vfail_win,
                     "wl_vfail_key": me.wl_vfail_key,
                     "wl_cached": me.wl_cached, "wl_replans": me.wl_replans,
                     "bg_seen": me.bg_seen, "bg_fired": me.bg_fired})
        print(f"[{args.tag or 'arena'}] g{i} seat{seat} "
              f"{'W' if won else ('D' if r == 2 else 'L')} cause={cause} "
              f"dec={dec} evals={me.evals} wl={me.wl_checked}/{me.wl_proved} "
              f"cand={me.wl_cands_tried} "
              f"vfail={me.wl_vfail_win}/{me.wl_vfail_key} "
              f"bg={me.bg_seen}/{me.bg_fired} {rows[-1]['wall_s']}s", flush=True)
    import numpy as np
    t = np.array(times) if times else np.array([0.0])
    summary = {"tag": args.tag, "opponent": os.path.basename(args.opponent),
               "search": args.search, "games": args.games, "wins": wins,
               "draws": draws, "win_rate": round(wins / args.games, 3),
               "dec_ms_mean": round(float(t.mean() * 1000), 1),
               "dec_ms_max": round(float(t.max() * 1000), 1)}
    print(json.dumps(summary), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            for r_ in rows:
                f.write(json.dumps({**r_, **{"tag": args.tag,
                        "opponent": summary["opponent"],
                        "search": args.search}}) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")


if __name__ == "__main__":
    main()
