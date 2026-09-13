"""torch PTCGTransformer <-> native flat fp32 blob (PT_PARAMS order).

The order and shapes here MUST mirror PT_PARAMS in native/model/ptcg_model.cu.
Uses:
  export init:  PYTHONPATH=data:. python native/tools/native_weights.py init OUT.bin [--seed 42]
  torch->blob:  ... native_weights.py export CKPT.pt OUT.bin
  blob->torch:  ... native_weights.py import BLOB.bin OUT.pt   (for eval_watch)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "data"))

# MUST MATCH native/model/ptcg_model.cu's PT_D / PT_H / PT_FFN / PT_LAYERS.
# This file is a SECOND COPY of the weight layout: the CUDA side writes the
# blob, this reads it back into a torch state dict, and eval_watch runs it on
# every checkpoint. A desync here does not crash — it produces a silently
# wrong .pt, which means wrong evals and wrong submission bundles. If you
# change the size in ptcg_model.cu, change it here in the same commit.
# A d128/h4 trunk is 128, 4, 256, 4.
D, HEADS, FFN, LAYERS, DC, DA = 256, 8, 512, 4, 128, 64
G = 128                     # global-summary MLP width; independent of D
NC, NA, NS = 1284, 1573, 448
# derived concat widths — mirror the torch nn.Linear input sizes
Q_IN = 16 + 2 * D + 2 * 64 + 8
SCORE_IN = 3 * D + G + 64
VAL_IN = D + G
ORC_IN = 134


def _block(prefix):
    return [
        (f"{prefix}.ln1.weight", (D,)), (f"{prefix}.ln1.bias", (D,)),
        (f"{prefix}.qkv.weight", (3 * D, D)), (f"{prefix}.qkv.bias", (3 * D,)),
        (f"{prefix}.proj.weight", (D, D)), (f"{prefix}.proj.bias", (D,)),
        (f"{prefix}.ln2.weight", (D,)), (f"{prefix}.ln2.bias", (D,)),
        (f"{prefix}.ffn.0.weight", (FFN, D)), (f"{prefix}.ffn.0.bias", (FFN,)),
        (f"{prefix}.ffn.2.weight", (D, FFN)), (f"{prefix}.ffn.2.bias", (D,)),
    ]


def param_order():
    """(torch_name, shape) in native PT_PARAMS order."""
    order = [
        ("card_id_emb.weight", (NC, 64)),
        ("attack_id_emb.weight", (NA, 48)),
        ("skill_id_emb.weight", (NS, 48)),
        ("attack_mlp.0.weight", (64, 127)), ("attack_mlp.0.bias", (64,)),
        ("skill_mlp.0.weight", (DC, 112)), ("skill_mlp.0.bias", (DC,)),
        ("card_mlp.0.weight", (634,)), ("card_mlp.0.bias", (634,)),
        ("card_mlp.1.weight", (256, 634)), ("card_mlp.1.bias", (256,)),
        ("card_mlp.3.weight", (DC, 256)), ("card_mlp.3.bias", (DC,)),
        ("zone_emb.weight", (16, 8)),
        ("owner_emb.weight", (4, 4)),
        ("token_proj.weight", (D, 324)), ("token_proj.bias", (D,)),
        ("readout", (D,)),
        ("global_mlp.0.weight", (G, 64)), ("global_mlp.0.bias", (G,)),
        ("global_mlp.2.weight", (G, G)), ("global_mlp.2.bias", (G,)),
        ("sel_type_emb.weight", (29, 8)),
        ("ctx_emb.weight", (65, 32)),
        ("dec_mlp.0.weight", (64, 696)), ("dec_mlp.0.bias", (64,)),
        ("dec_mlp.2.weight", (64, 64)), ("dec_mlp.2.bias", (64,)),
        ("bcast.weight", (D, G + 64)), ("bcast.bias", (D,)),
    ]
    for i in range(LAYERS):
        order += _block(f"blocks.{i}")
    order += [
        ("ln_f.weight", (D,)), ("ln_f.bias", (D,)),
        ("opt_type_emb.weight", (33, 16)),
        ("opt_card.weight", (64, DC)), ("opt_card.bias", (64,)),
        ("opt_attack.weight", (64, DA)), ("opt_attack.bias", (64,)),
        ("q_mlp.0.weight", (D, Q_IN)), ("q_mlp.0.bias", (D,)),
        ("q_mlp.2.weight", (D, D)), ("q_mlp.2.bias", (D,)),
        ("score_mlp.0.weight", (256, SCORE_IN)), ("score_mlp.0.bias", (256,)),
        ("score_mlp.2.weight", (1, 256)), ("score_mlp.2.bias", (1,)),
        ("value_mlp.0.weight", (256, VAL_IN)), ("value_mlp.0.bias", (256,)),
        ("value_mlp.2.weight", (1, 256)), ("value_mlp.2.bias", (1,)),
        ("oracle_proj.weight", (D, ORC_IN)), ("oracle_proj.bias", (D,)),
        ("critic_readout", (D,)),
    ]
    order += _block("critic_blocks.0")
    order += [
        ("oracle_value_mlp.0.weight", (256, VAL_IN)), ("oracle_value_mlp.0.bias", (256,)),
        ("oracle_value_mlp.2.weight", (1, 256)), ("oracle_value_mlp.2.bias", (1,)),
        ("aux_hand.weight", (NC, VAL_IN)), ("aux_hand.bias", (NC,)),
        ("aux_prize.weight", (1, D)), ("aux_prize.bias", (1,)),
    ]
    return order


def sd_to_blob(sd):
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    chunks = []
    for name, shape in param_order():
        t = sd[name].detach().float().cpu().numpy()
        assert tuple(t.shape) == shape, f"{name}: {t.shape} != {shape}"
        chunks.append(np.ascontiguousarray(t).ravel())
    return np.concatenate(chunks)


def blob_to_sd(blob):
    import torch
    sd, off = {}, 0
    for name, shape in param_order():
        n = int(np.prod(shape))
        sd[name] = torch.from_numpy(
            blob[off:off + n].reshape(shape).copy())
        off += n
    assert off == blob.size, f"blob has {blob.size} floats, consumed {off}"
    return sd


def fresh_model(seed):
    import torch
    from ptcg.rl.model import PTCGTransformer
    torch.manual_seed(seed)
    return PTCGTransformer(d=D, layers=LAYERS, heads=HEADS, ffn=FFN,
                           critic="oracle")


def main():
    cmd = sys.argv[1]
    if cmd == "init":
        out = sys.argv[2]
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
        m = fresh_model(seed)
        blob = sd_to_blob(m.state_dict())
        blob.astype(np.float32).tofile(out)
        print(f"init blob: {out} ({blob.size} params, seed {seed})")
    elif cmd == "export":
        import torch
        sd = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
        if not isinstance(sd, dict) or "card_id_emb.weight" not in {
                k.replace("module.", "") for k in sd}:
            sd = sd.get("state_dict", sd)
        blob = sd_to_blob(sd)
        blob.astype(np.float32).tofile(sys.argv[3])
        print(f"exported {sys.argv[2]} -> {sys.argv[3]} ({blob.size} params)")
    elif cmd == "import":
        import torch
        blob = np.fromfile(sys.argv[2], dtype=np.float32)
        sd = blob_to_sd(blob)
        # non-persistent buffers (static tables) rebuild on load; state_dict
        # from a live model includes them — merge from a fresh model so
        # eval_watch's load_state_dict sees every key
        m = fresh_model(0)
        full = m.state_dict()
        full.update(sd)
        torch.save(full, sys.argv[3])
        print(f"imported {sys.argv[2]} -> {sys.argv[3]}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
