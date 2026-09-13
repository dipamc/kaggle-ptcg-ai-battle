"""Native CUDA model vs torch PTCGTransformer, same weights, same obs batch.

Run on the CUDA host (needs the built _C and a GPU), from repo root, STRICT fp32:

  PTCG_TF32=0 PTCG_TABLES=native/ptcg_tables.bin PYTHONPATH=data:.:native \\
      python native/tools/model_parity.py \\
      --rows native/parity_rows.bin --weights <run_dir>/init.bin

parity_rows.bin = fp32 obs rows from native/tools/dump_stream.py (any checkpoint's
rows work; the batch just has to be real obs so the mask/options are live).

Compares the fused decoder output (B, 65) = [64 raw option scores || oracle
value] against torch forward_eval:
  - option scores where opt_mask == 1 (torch masks invalid ones to -1e9
    in-model; native masks later at sample/PPO time, so raw invalid entries
    are expected to differ — they are excluded)
  - value (col 64 vs oracle_v)
  - KL(torch || native) over the masked softmax, the number the PPO loss
    actually feels. Early-training reference kl vs a torch run was ~0.15-0.2;
    parity here should be ~1e-6.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "native"))
sys.path.insert(0, os.path.join(os.getcwd(), "native", "tools"))

# Take the arch from native_weights rather than restating it: this file used
# to carry its own `D, HEADS, FFN, LAYERS = 128, 4, 256, 4`, so after a size
# change the gate silently compared a d256 blob against a d128 torch model and
# died in load_state_dict instead of reporting parity. A parity tool that can
# disagree with the thing it validates is worse than no parity tool.
from native_weights import (  # noqa: E402  (native/tools on path)
    blob_to_sd, D, HEADS, FFN, LAYERS)
OBS = 10944


def native_forward(rows, weights, agents, threads):
    os.environ.setdefault("PTCG_TABLES", "native/ptcg_tables.bin")
    os.environ["PTCG_INIT_WEIGHTS"] = weights
    from puffer_ptcg import _C
    n = rows.shape[0]
    args = {
        "env_name": "ptcg", "reset_state": True, "cudagraphs": -1,
        "profile": False, "rank": 0, "world_size": 1, "gpu_id": 0,
        "nccl_id": b"", "seed": 1,
        "train": {
            "horizon": 64, "total_timesteps": 1000000, "learning_rate": 1e-3,
            "min_lr_ratio": 0.0, "anneal_lr": 0, "beta1": 0.95,
            "beta2": 0.999, "eps": 1e-12, "minibatch_size": n,
            "replay_ratio": 1.0, "max_grad_norm": 1.5, "clip_coef": 0.2,
            "vf_clip_coef": 0.2, "vf_coef": 2.0, "ent_coef": 0.01,
            "min_ent_coef_ratio": 0.1, "anneal_ent_coef": 0, "gamma": 1.0,
            "gae_lambda": 0.95, "vtrace_rho_clip": 1.0, "vtrace_c_clip": 1.0,
            "prio_alpha": 0.8, "prio_beta0": 0.2, "gpus": 1,
        },
        "vec": {"total_agents": agents, "num_buffers": 2,
                "num_threads": threads},
        "env": {"mix_self": 0.98, "reward_win": 1, "max_engine_steps": 3000,
                "seed": 7},
        "policy": {"hidden_size": 65, "num_layers": 1},
    }
    pl = _C.create_pufferl(args)
    out = _C.debug_forward(pl, rows)
    _C.close(pl)
    # decoder tensor comes back segment-major, e.g. (segments, horizon*65);
    # rows are contiguous either way
    return np.asarray(out).reshape(n, -1)


def torch_forward(rows, weights, device):
    """Run the torch reference in a SUBPROCESS. Both the torch model init
    (ptcg.rl.cards -> cg.api) and our _C initialize the same loaded libcg;
    doing both in one process double-inits the engine and dies inside it
    ("buffer full. capacity:7"). Process isolation is the fix, not ordering."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rows_f = os.path.join(td, "rows.npy")
        out_f = os.path.join(td, "torch_out.npz")
        np.save(rows_f, rows)
        subprocess.check_call(
            [sys.executable, os.path.abspath(__file__), "--_torch-child",
             rows_f, weights, device, out_f],
            cwd=os.getcwd())
        z = np.load(out_f)
        return z["logits"], z["value"], z["mask"]


def _torch_child(rows_f, weights, device, out_f):
    import torch
    from ptcg.rl.model import PTCGTransformer, _take

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    rows = np.load(rows_f)
    m = PTCGTransformer(d=D, layers=LAYERS, heads=HEADS, ffn=FFN,
                        critic="oracle")
    sd = m.state_dict()
    sd.update(blob_to_sd(np.fromfile(weights, dtype=np.float32)))
    m.load_state_dict(sd)
    m = m.to(device).float().eval()
    obs = torch.from_numpy(rows).to(device)
    with torch.no_grad():
        logits, v = m.forward_eval(obs)
        mask = (_take(obs, "opt_mask") != 0)
    np.savez(out_f, logits=logits.cpu().numpy(),
             value=v.squeeze(-1).cpu().numpy(), mask=mask.cpu().numpy())


def masked_softmax(scores, mask, neg):
    z = np.where(mask, scores, neg).astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z) * mask
    return e / e.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="native/parity_rows.bin")
    p.add_argument("--weights", required=True,
                   help="flat fp32 blob (init.bin or a run checkpoint .bin)")
    p.add_argument("--n", type=int, default=512,
                   help="batch size (= native minibatch)")
    p.add_argument("--agents", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max |score diff| on valid options to pass")
    a = p.parse_args()

    rows = np.fromfile(a.rows, dtype=np.float32).reshape(-1, OBS)
    if rows.shape[0] < a.n:
        reps = -(-a.n // rows.shape[0])
        rows = np.tile(rows, (reps, 1))
    rows = np.ascontiguousarray(rows[:a.n])
    print(f"{a.n} obs rows from {a.rows}")

    if os.environ.get("PTCG_TF32", "1") != "0":
        print("WARN: PTCG_TF32 not set to 0 — native gemms will use TF32 "
              "while torch runs 'highest'; expect ~1e-3 not ~1e-5 agreement")

    t_logits, t_value, mask = torch_forward(rows, a.weights, a.device)
    n_out = native_forward(rows, a.weights, a.agents, a.threads)
    n_scores, n_value = n_out[:, :64], n_out[:, 64]

    valid = mask.astype(bool)
    d_scores = np.abs(t_logits - n_scores)[valid]
    d_value = np.abs(t_value - n_value)
    p_t = masked_softmax(t_logits, valid, -1e9)
    p_n = masked_softmax(n_scores, valid, -1e4)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p_t > 0, p_t * (np.log(p_t) - np.log(p_n)), 0.0)
    kl = np.nansum(kl, axis=1)

    print(f"valid options/row: mean {valid.sum(1).mean():.1f}")
    print(f"scores  (valid): max|d| {d_scores.max():.3e}  "
          f"mean|d| {d_scores.mean():.3e}")
    print(f"value:           max|d| {d_value.max():.3e}  "
          f"mean|d| {d_value.mean():.3e}")
    print(f"KL(torch||native): max {kl.max():.3e}  mean {kl.mean():.3e}")
    worst = np.argsort(kl)[-5:][::-1]
    for i in worst:
        print(f"  row {i}: kl {kl[i]:.3e}  "
              f"argmax t={p_t[i].argmax()} n={p_n[i].argmax()}  "
              f"value t={t_value[i]:+.4f} n={n_value[i]:+.4f}")
    agree = (p_t.argmax(1) == p_n.argmax(1)).mean()
    print(f"argmax agreement: {agree * 100:.2f}%")

    ok = d_scores.max() < a.tol and d_value.max() < a.tol
    print("PARITY OK" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_torch-child":
        _torch_child(*sys.argv[2:6])
    else:
        main()
