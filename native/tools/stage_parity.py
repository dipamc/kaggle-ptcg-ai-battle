"""Localize a native-vs-torch forward divergence stage by stage.

Native side dumps every named stage via PTCG_DUMP_DIR (see forward_core);
torch side captures the same tensors with module hooks in a SUBPROCESS
(both runtimes double-init libcg if they share a process). Stages compare
in dataflow order; the first one over tolerance is where the bug lives.

  PTCG_TF32=0 PTCG_TABLES=native/ptcg_tables.bin PYTHONPATH=data:.:native \\
      python native/tools/stage_parity.py --weights <blob.bin> [--n 512]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))
sys.path.insert(0, os.path.join(os.getcwd(), "native"))
sys.path.insert(0, os.path.join(os.getcwd(), "native", "tools"))

# Arch comes from native_weights so this bisect tool cannot drift out of sync
# with the model it is bisecting (see the note in model_parity.py).
from native_weights import D, HEADS, FFN, LAYERS  # noqa: E402
OBS, LTOK, LSEQ, NOPT = 10944, 160, 161, 64

# (name, per-row shape, seq-sliced?) in dataflow order. seq-sliced stages
# compare only the torch prefix (1+L positions).
STAGES = [
    ("att_tab", None, False), ("sk_tab", None, False),
    ("a_mean", None, False), ("a_max", None, False),
    ("s_mean", None, False), ("s_max", None, False),
    ("card_in", None, False), ("card_ln", None, False),
    ("card_h1", None, False), ("card_vec", None, False),
    ("tok_cat", (LTOK, 324), False), ("tokp", (LTOK, D), False),
    ("dec_in", (696,), False), ("g_out", (128,), False),
    ("d_out", (64,), False), ("bc_out", (D,), False),
    ("keep", (LSEQ,), True), ("x0", (LSEQ, D), True),
    ("x1", (LSEQ, D), True), ("x2", (LSEQ, D), True),
    ("x3", (LSEQ, D), True), ("x4", (LSEQ, D), True),
    ("h", (LSEQ, D), True),
    ("q_in", (NOPT, 408), False), ("q_vec", (NOPT, D), False),
    ("sc_in", (NOPT, 576), False), ("out", (65,), False),
]


def native_side(rows, weights, dump_dir):
    from model_parity import native_forward
    os.environ["PTCG_DUMP_DIR"] = dump_dir
    native_forward(rows, weights, 64, 2)


def _torch_child(rows_f, weights, out_dir):
    import torch
    from ptcg.rl.model import PTCGTransformer, _take
    from native_weights import blob_to_sd

    torch.set_float32_matmul_precision("highest")
    rows = np.load(rows_f)
    m = PTCGTransformer(d=D, layers=LAYERS, heads=HEADS, ffn=FFN,
                        critic="oracle")
    sd = m.state_dict()
    sd.update(blob_to_sd(np.fromfile(weights, dtype=np.float32)))
    m.load_state_dict(sd)
    m = m.float().eval()

    cap = {}

    def grab(name, t):
        cap[name] = t.detach().float().cpu().numpy()

    def out_hook(name):                    # hooks MUST return None or they
        def h(mod, i, o):                  # replace the module's output
            grab(name, o)
        return h

    def io_hook(name_i, name_o):
        def h(mod, i, o):
            grab(name_i, i[0])
            grab(name_o, o)
        return h

    def x0_pre_hook(mod, args):
        grab("x0", args[0])
        grab("keep", args[1].float())

    m.token_proj.register_forward_hook(io_hook("tok_cat", "tokp"))
    m.global_mlp.register_forward_hook(out_hook("g_out"))
    m.dec_mlp.register_forward_hook(io_hook("dec_in", "d_out"))
    m.bcast.register_forward_hook(out_hook("bc_out"))
    m.blocks[0].register_forward_pre_hook(x0_pre_hook)
    for bi, blk in enumerate(m.blocks):
        blk.register_forward_hook(out_hook(f"x{bi + 1}"))
    m.ln_f.register_forward_hook(out_hook("h"))
    m.q_mlp.register_forward_hook(io_hook("q_in", "q_vec"))
    m.score_mlp.register_forward_hook(io_hook("sc_in", "_sc_out"))

    obs = torch.from_numpy(rows)
    with torch.no_grad():
        logits_masked, v = m.forward_eval(obs)
        c = m._core(obs)  # second pass, deterministic — reuse hooks' capture
    cv, av = m._card_table()
    cap["card_vec"] = cv.detach().float().cpu().numpy()
    cap["att_tab"] = av.detach().float().cpu().numpy()
    # card-chain intermediates, recomputed exactly as _card_table does
    with torch.no_grad():
        a_g = av[m.card_attacks]
        a_mask = (m.card_attacks > 0).unsqueeze(-1)
        cap["a_mean"] = ((a_g * a_mask).sum(1) /
                         a_mask.sum(1).clamp(min=1)).float().numpy()
        cap["a_max"] = (a_g * a_mask).amax(1).float().numpy()
        sk_tab = m.skill_mlp(torch.cat(
            [m.skill_id_emb.weight, m.skill_text], dim=-1))
        cap["sk_tab"] = sk_tab.float().numpy()
        s_g = sk_tab[m.card_skills]
        s_mask = (m.card_skills > 0).unsqueeze(-1)
        cap["s_mean"] = ((s_g * s_mask).sum(1) /
                         s_mask.sum(1).clamp(min=1)).float().numpy()
        cap["s_max"] = (s_g * s_mask).amax(1).float().numpy()
        card_in = torch.cat(
            [m.card_id_emb.weight, m.card_static, m.card_text,
             torch.from_numpy(cap["a_mean"]), torch.from_numpy(cap["a_max"]),
             torch.from_numpy(cap["s_mean"]), torch.from_numpy(cap["s_max"])],
            dim=-1)
        cap["card_in"] = card_in.float().numpy()
        card_ln = m.card_mlp[0](card_in)
        cap["card_ln"] = card_ln.float().numpy()
        cap["card_h1"] = m.card_mlp[1](card_ln).float().numpy()
    # raw scores: score_mlp output pre-mask is what native emits; rebuild from
    # masked logits is lossy at -1e9, so recompute out from _core + value
    scores = c["logits"].detach().float().cpu().numpy()
    cap["out"] = np.concatenate(
        [scores, v.detach().float().cpu().numpy().reshape(-1, 1)], axis=1)
    cap["L"] = np.array([c["L"]])
    np.savez(os.path.join(out_dir, "torch_stages.npz"), **cap)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="native/parity_rows.bin")
    p.add_argument("--weights", required=True)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--tol", type=float, default=2e-3)
    a = p.parse_args()

    rows = np.fromfile(a.rows, dtype=np.float32).reshape(-1, OBS)
    rows = np.ascontiguousarray(rows[:a.n])

    with tempfile.TemporaryDirectory() as td:
        rows_f = os.path.join(td, "rows.npy")
        np.save(rows_f, rows)
        subprocess.check_call(
            [sys.executable, os.path.abspath(__file__), "--_torch-child",
             rows_f, a.weights, td], cwd=os.getcwd())
        t = np.load(os.path.join(td, "torch_stages.npz"))
        L = int(t["L"][0])
        print(f"torch prefix L = {L} (native runs full {LTOK})")

        dump = os.path.join(td, "native")
        os.makedirs(dump)
        native_side(rows, a.weights, dump)

        n_rows = rows.shape[0]
        first_bad = None
        for name, shape, sliced in STAGES:
            f = os.path.join(dump, f"{name}.bin")
            if not os.path.exists(f) or name not in t:
                print(f"{name:>9}: MISSING ({'native' if not os.path.exists(f) else 'torch'})")
                continue
            nat = np.fromfile(f, dtype=np.float32)
            tor = t[name]
            if shape is None:                      # tables
                nat = nat.reshape(tor.shape)
            else:
                nat = nat.reshape((n_rows,) + shape)
                if sliced:                         # torch ran a 1+L prefix
                    nat = nat[:, :1 + L]
                tor = tor.reshape(nat.shape)
            d = np.abs(nat - tor)
            md, loc = d.max(), np.unravel_index(d.argmax(), d.shape)
            ok = md < a.tol
            print(f"{name:>9}: max|d| {md:.3e} at {tuple(int(v) for v in loc)}"
                  f"  {'ok' if ok else '<-- FIRST DIVERGENCE' if first_bad is None else 'bad'}")
            if not ok and first_bad is None:
                first_bad = name
        print(f"\nfirst divergent stage: {first_bad or 'NONE — parity clean'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_torch-child":
        _torch_child(*sys.argv[2:5])
    else:
        main()
