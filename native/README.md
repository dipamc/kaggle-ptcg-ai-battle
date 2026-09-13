# native/ — the C env and the CUDA trainer

The Pokemon TCG environment in C, the policy/value model in CUDA, and a vendored, patched PufferLib
native trainer, all linked into one Python extension. There is no PyTorch in the training loop.

Full instructions — prerequisites, data, flags, multi-GPU, telemetry, checkpoints, tooling — are in
`docs/training.md`. The local changes to PufferLib are described in `PATCHES.md`.

## Layout

| path | what |
|---|---|
| `ptcg/` | the C environment: `obs.c`, `tracker.c`, `encoder.c`, `env.c`, `tables.c`, the pufferlib binding `binding.c`, and the header `ptcg_env.h`, plus the standalone tests `test_env.c`, `parity_replay.c`, `test_reload.c`, `test_deck_lock.c`, `test_deck_weights.c`, `test_stall.c` |
| `model/` | the CUDA model: `ptcg_model.cu` (sizes, weight table, tables and trunk), `ptcg_model2.cu` (drivers, aux losses, negamax advantage), `ptcg_model3.cu` (transformer blocks, forward), `ptcg_model4.cu` (backward, weight loading), `ptcg_kernels.cuh` |
| `src/` | PufferLib at commit `c5d3c63` with local patches: `pufferlib.cu`, `bindings.cu`, `muon.cu`, `kernels.cu`, `vecenv.h`, `ocean.cu`, `models.cu`, `tensor.h`, plus `VENDORED_COMMIT` and `PUFFERLIB_LICENSE` |
| `patches/` | `pufferlib-c5d3c63.patch`, the unified diff of everything in `src/` |
| `vendor/` | cJSON, used by the C env to parse engine payloads |
| `tools/` | launch and maintenance scripts: `native_weights.py`, `export_tables.py`, `supervisor_native.sh`, `native_eval_watch.sh`, `model_parity.py`, `stage_parity.py`, `dump_stream.py`, `dump_stream_decks.py`, `pool_append_watch.py`, `test_weight_request.py` |
| `train_native.py` | the training driver: argument parsing, per-rank processes, the epoch loop, telemetry, checkpointing |
| `build_native.sh`, `Makefile` | the CUDA build and the CPU-only test builds |
| `puffer_ptcg/`, `build/` | build outputs: the extension module and object files |

## Build

```bash
bash native/build_native.sh          # --debug for -O0 -g
```

Requires `nvcc`, `gcc`, `pybind11`, `numpy`, NCCL and `data/cg/libcg.so`. It produces
`native/puffer_ptcg/_C<ext>.so`. Set `PTCG_SM` to build for a GPU architecture other than the one in
the box, and `PY` to choose the interpreter whose headers are used. `NCCL_HOME` pins the NCCL to
build and link against; use it when the system NCCL was built for a newer CUDA than the driver
supports, which shows up at launch as `ncclCommInitRank failed ... unhandled cuda error` with
`NCCL WARN Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'` under
`NCCL_DEBUG=WARN`. The wheel torch installs is always a matching one:

```bash
NCCL_HOME=$(python -c "import nvidia.nccl;print(nvidia.nccl.__path__[0])") bash native/build_native.sh
ldd native/puffer_ptcg/_C*.so | grep nccl      # should point into the venv
```

## Test

The env tests build with the host compiler only — no GPU, no CUDA:

```bash
make -C native test          # invariants over 16 envs x 20000 steps
make -C native parity        # C encoder vs recorded python rows, bit-level
make -C native clean
```

`reload`, `decklock` and `deckweights` test the mid-run deck-pool contract and take tables blobs as
make variables — `make -C native reload V1=<blob> GROWN=<blob+appended> INSERTED=<reordered blob>`,
`make -C native decklock V1=… GROWN=…`, `make -C native deckweights V1=… [GROWN=…]`; how to build
those blobs is in `docs/deck-pool.md`. `make -C native build/test_stall` builds the activation-cap
test, which is run by hand. The model has its own gate on a GPU box:

```bash
PTCG_TF32=0 PTCG_TABLES=native/ptcg_tables.bin PYTHONPATH=data:.:native \
python3 native/tools/model_parity.py --rows native/parity_rows.bin --weights <blob.bin>
```

## Key facts

- The observation is 10944 fp32 values, with a 64-wide action mask that also lives inside the
  observation. Ids are exact in fp32.
- **fp32 only.** The build passes `-DPRECISION_FLOAT`; bf16 corrupts the id fields in the
  observation. TF32 matmuls are on by default and `PTCG_TF32=0` forces true fp32.
- One agent per env, one env per game; the self-play stream serves whichever seat is to act, and the
  advantage is negamax — the value sign flips when consecutive timesteps belong to different seats.
- Truncation (the engine-step limit, or an engine error) resets silently with no terminal flag, so
  the trainer bootstraps across it.
- Weights are a flat fp32 blob in the `PT_PARAMS` order declared in `model/ptcg_model.cu`;
  `tools/native_weights.py` converts to and from a torch `state_dict`. Initial weights must come
  from a torch-exported blob (`PTCG_INIT_WEIGHTS`) — there is no native random init.
- The grads arena mirrors the params arena exactly, including 16-byte alignment gaps, because Muon
  walks both with one shared offset.
