# Local patches to the vendored PufferLib trainer

`native/src/` is PufferLib's native CUDA trainer, vendored at commit `c5d3c63`
(`native/src/VENDORED_COMMIT`) and modified in place; in-tree the changes are marked `LOCAL PATCH`
or `PTCG:`. The full unified diff is `native/patches/pufferlib-c5d3c63.patch`, covering
`pufferlib.cu`, `bindings.cu`, `muon.cu`, `kernels.cu`, `vecenv.h` and `ocean.cu` (`models.cu` and
`tensor.h` are untouched).

To re-apply on a clean checkout:

```bash
cd PufferLib && git checkout c5d3c63 && patch -p1 < .../pufferlib-c5d3c63.patch
```

`native/build_native.sh` compiles only `src/bindings.cu`, which includes `pufferlib.cu` and through
it everything else, with `-DPTCG_NATIVE -DPRECISION_FLOAT` (plus `-DOBS_TENSOR_T=FloatTensor
-DENV_NAME=ptcg`). fp32 only: bf16 corrupts the id fields in the observation.

## Custom network hook and the ocean.cu replacement

`ocean.cu` is replaced wholesale: instead of the sample environments' encoders (and the cuDNN
dependency they bring), it includes `ptcg_model.cu` and defines `create_custom_encoder`,
`create_custom_decoder` and `create_custom_network`, installing the PTCG forward/backward, weight
registration and allocation vtables when the env name is `ptcg`. Upstream calls the first two from
`build_policy` but not the third, so `build_policy` gains a `create_custom_network(env_name,
&network)` call — without it the mingru network stays wired up and the custom trunk is never used.

## Negamax advantage under PTCG_NATIVE

Both seats of a self-play game write into one row, so a reward is in the currency of whichever seat
acted. Under `#ifdef PTCG_NATIVE`, `train_impl` calls `ptcg_puff_advantage` (in `native/model/`)
instead of `puff_advantage_cuda`: the same v-trace/GAE recursion with a sign that flips whenever
consecutive timesteps belong to different seats, read from a seat column in the observation.

## 16-byte aligned arenas and packed weight blobs

cuBLAS selects vectorized kernels by shape and architecture and hard-faults on unaligned operands,
so the params and grads arenas 16-byte-align every tensor (`kernels.cu` carries the rule as a
comment). Every consumer that assumed a packed arena mirrors that alignment: `muon_init` sizes the
momentum buffer by `total_bytes`, Muon's per-parameter walk rounds its offset up to 4 floats, and
`grad_puf`/`param_puf` span the padded arenas. On-disk blobs stay packed, so `pufferlib.cu` gains
`packed_blob_to_arena_host` / `arena_host_to_packed_blob`, and `save_weights`, `load_weights`,
`save_state`, `load_state` and the frozen-bank loader stage through a host image of the arena.

## KL / value-loss gate and the Muon 1-D path

`ppo_loss_reduce` takes a nullable `guard_mb` pointer and writes this minibatch's mean approx-KL and
mean value loss into a small guard buffer (`KlGuardIdx` in `pufferlib.cu`: `MB_KL`, `MB_VF`, `GATE`,
`SKIPPED`, `SKIPPED_VF`). In `train_impl` the two are allreduced with `ncclAvg` — a per-rank skip
decision would fork the ranks' weights, and the two slots are adjacent so one 2-float collective
covers both — then `kl_gate_kernel` writes `GATE` from the `kl_skip_threshold` / `vf_skip_threshold`
comparison, failing closed on non-finite values. It is all stream-ordered, so it captures into the
train graph.

In `muon.cu`, `muon_nesterov`, `muon_weight_update` and the new `muon_nesterov_ema` take that gate
pointer and return early when it is 0 — a branch, not a multiply, because `0 * nan` is `nan`. The
momentum launch also moves out of one arena-wide call into the per-parameter walk so 1-D parameters
can use EMA accumulation (`m = mu*m + (1-mu)*g`, update `g + m`) and a separate decoupled weight
decay `wd_1d`, while tensors of 2 or more dimensions keep upstream's summed momentum bit-for-bit.
`bindings.cu` exposes the three knobs with protective defaults; `puf_log` reports `kl_skipped` /
`vf_skipped`.

## LR warmup, anchored cosine, start_epoch

`HypersT` gains `lr_warmup_epochs`, `lr_anneal_from_epoch` and `start_epoch`. In `train_impl` the
`cudaMemcpy` to `lr_ptr` is hoisted out of the `anneal_lr` branch — on a flat-LR run that branch was
the only write, so a warmup gated behind it would silently do nothing. The cosine is evaluated
relative to `lr_anneal_from_epoch` (0 reproduces upstream), so enabling annealing mid-run spans
"here to the target" rather than the tail of a curve anchored before the run began; the warmup
multiplies afterwards so the two compose, and its ramp counts epochs relative to `start_epoch`,
which makes it meaningful on a restart.

## Gradient accumulation

`accum_minibatches` groups micro-batches into one optimizer step. `create_pufferl` allocates a
second grads arena (`grad_accum_puf`) with the same padded layout, `accum_grads_kernel` folds each
micro-batch's gradients in at `1/k`, and the guard, the Muon step and the accumulator memset run
only on a group's final micro-batch. Groups must tile the epoch exactly or the last partial group
would never step, which `train_impl` and `train_native.py` both check. A second cudagraph,
`train_accum_cudagraph`, covers the non-final variant, which has no optimizer step in it.

## League routing

Opt-in through `league_routing`. The env stamps a frozen-bank id (0 = learner) into one observation
column and exports which column through the weak symbol `env_league_tag_col`. In the rollout,
`league_extract_tag_kernel` moves the tag into its own rollout tensor and zeroes the column (every
checkpoint was trained with it at 0), the primary bank forwards every row, each frozen bank forwards
its `bank_layout` slice into per-(bank, buffer) scratch, and `league_merge_kernel` overwrites the
tagged rows' action and logprob; values are never merged. In the loss the tag rides along to the
minibatch, and the PPO kernel masks frozen decisions out of the policy gradient, entropy, KL and
clipfrac (renormalizing by the learner-decision count), pins their importance ratio to 1 and zeroes
their policy gradient explicitly, while the value loss keeps every decision.
`zero_frozen_advantages` is skipped here; with the flag off the kernel takes the original
all-element path unchanged.

## New Python bindings

`bindings.cu` adds `save_state` / `load_state` (weights + Muon momentum + step/epoch, `PTCGSTA1`
header, tmp + fsync + rename), `set_start` (stamp `global_step`/`epoch` for a weights-only restart),
`debug_forward` (run the train-path forward on a caller-supplied observation batch and return the
decoder output — the model parity gate), `reload_decks` and `set_deck_weights` /
`clear_deck_weights` (mid-run deck-pool and sampling-weight changes forwarded to the C env), and
`allreduce_i32` (integer min/max/sum across ranks, which is how the deck-pool protocol proves the
ranks agree). `puf_log` also returns the applied learning rate read from Muon's device pointer, so
telemetry cannot drift from the schedule.

## Decoder-row mirror for the event log

`vecenv.h` gains `dec` / `gpu_dec` / `dec_cols`: the decoder row (every action head's logits plus
the fused value column) copied back to the host each step and handed to the env as `env->dec`, so
the event log can record what the policy thought at each decision. It is opt-in at build time
(`MY_DECODER_EXPORT`, which the PTCG binding defines) and again at runtime (only under
`PTCG_EVENT_LOG`), so a normal run allocates nothing.

## DDP robustness

- Rollout capture uses `cudaStreamCaptureModeThreadLocal` instead of `Global`, with a verbose error
  and abort on failure: buffer threads capture concurrently, which global mode turns into a failure.
- `ncclCommInitRank` is error-checked and followed by an eager warmup allreduce, turning a broken
  transport into a clean error at startup (naming `NCCL_P2P_DISABLE=1` as the fix) and guaranteeing
  connections exist before capture, which NCCL requires for captured collectives.
- Stream syncs in the rollout loop, the train end-sync and the warmup device sync are checked and
  abort on error; an unchecked sync after a poisoned context leaves the run spinning on zeroed
  actions at absurd SPS.
- `ptcg_wait_relax` replaces pure spin-waits on buffer state: spin briefly, then sleep 50us per
  check. Idle spinners otherwise burn a core each and starve the env workers on a CPU-quota'd box,
  deadlocking a multi-rank run.
- `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` is set before the context exists, so idle
  syncs sleep instead of spinning (`PTCG_BLOCKING_SYNC=0` opts out), and `PTCG_DEBUG_ROLLOUT=1` adds
  a rollout progress heartbeat.

## Miscellaneous

- Env-log dict capacity raised from 64 to 128 entries; per-bank league keys would otherwise overflow
  it.
- Custom decoders are no longer type-punned to `DecoderWeights`. The PTCG decoder is an 8-byte stub,
  so reading `dw->continuous` there is heap garbage; the rollout and train paths gate on the
  env-level `is_continuous` flag instead.
- `close_impl` guards graph destruction on `train_captured` / `train_accum_captured` / a non-null
  rollout graph array and frees the accumulation arena, so a graphs-off run can shut down.
- The warmup iterations' loss accumulators and guard counters are reset afterwards, so the first
  logged epoch reflects real rollouts.
