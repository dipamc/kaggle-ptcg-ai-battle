# Observation and model

The agent sees one flat float32 vector per decision and answers with an index into a fixed 64-slot action
space. This document covers that vector, the bookkeeping that fills it, and the network that consumes it.

## The observation row

`ptcg/rl/buffers.py` is the single source of truth for the layout. It is frozen: checkpoints are bound to
it, and `VERSION` (currently 4) is bumped on any change. Everything is float32, including ids — integers
below 2^24 are exact in fp32 and the model casts them back with `.long()`. `OBS_SIZE` is **10944**, and
`OFFSETS` holds the `(start, end)` of each block:

| block | shape | floats | offset |
|---|---|---|---|
| `tok_int` | `MAX_TOKENS x TOK_INT` = 160 x 5 | 800 | 0 |
| `tok_float` | `MAX_TOKENS x TOK_F` = 160 x 56 | 8960 | 800 |
| `opt_int` | `MAX_OPTIONS x OPT_INT` = 64 x 6 | 384 | 9760 |
| `opt_float` | `MAX_OPTIONS x OPT_F` = 64 x 8 | 512 | 10144 |
| `opt_mask` | `MAX_OPTIONS` = 64 | 64 | 10656 |
| `global_f` | `GLOBAL_F` = 64 | 64 | 10720 |
| `dec_int` | `DEC_INT` = 8 | 8 | 10784 |
| `dec_float` | `DEC_F` = 16 | 16 | 10792 |
| `oracle` | `ORACLE_F` = 136 | 136 | 10808 |

Two absolute columns are exported because code outside the model reads them without unpacking the row:
`SEAT_COL` (10751, `GlobF.ACTOR_SEAT`) tells the advantage kernel which seat acted, and `FROZEN_COL` (10774,
`GlobF.FROZEN_ACTOR`) carries `bank_index + 1` when a frozen league opponent acted the row. Both are zeroed
before any forward pass. `ptcg/rl/encoder.py` fills a row with `encode(row, obs, picked, stop_allowed,
forced_run, tracker=...)`, which zeroes it first; `write_oracle(...)` then fills the oracle block.

## Tokens

A token is one entity. `tok_int` holds `[card_id, zone, owner, tool0_id, tool1_id]`, `tok_float` holds 56
scalars indexed by `TokF`. Tokens fill a contiguous prefix, and the model slices each batch to its longest
real prefix (typically 40-90 of 160) before the trunk runs. There is no positional encoding: the tokens are
a set.

| zone (`buffers.Zone`, 0 = padding) | source |
|---|---|
| `MY_ACTIVE`, `MY_BENCH`, `OPP_ACTIVE`, `OPP_BENCH` | one token per board Pokemon, both sides |
| `MY_HAND` | one token per hand card, individually, so PLAY option indices resolve 1:1 |
| `STADIUM` | the stadium in play, owner-tagged |
| `MY_DISCARD`, `OPP_DISCARD` | count-collapsed: one token per distinct card id, `TokF.COPIES` = n/4 |
| `LOOKING` | the cards a select is showing (`select["deck"]`, `current["looking"]`), count-collapsed, so deck-search options have pointer targets |
| `MY_UNSEEN`, `MY_PRIZE_KNOWN` | tracker: my deck-plus-prize multiset, and my exact prizes once determined — `MY_UNSEEN` is then the deck-only remainder |
| `OPP_HAND_KNOWN`, `OPP_REVEALED` | tracker: opponent hand cards whose identity leaked, and opponent cards known to be back in their deck |

Count-collapsing is lossless because identical copies are interchangeable, and the 4-copy rule keeps
distinct ids per zone small. Board-Pokemon tokens carry the richest features: HP fractions and totals,
damage, a 12-wide attached-energy histogram from `ENERGY0`, tool and evolution-stack counts, `APPEAR`, the
five status flags (active only), `IS_ACTIVE`, and two per-step flags — `REFERENCED` (some option this step
points here) and `PICKED`. From `ATK0` come four groups of three derived features, one per top-4 attack by
damage: affordable now, damage/300, cost deficit clamped to 3 (`_deficit` does the engine's energy
matching). `WEAK_HIT`/`RES_HIT` compare this Pokemon's energy type to the opposing active's weakness and
resistance, and `RETREAT_OK` says the attached energy covers retreat. The last floats are the
lingering-effect layer — `EFF_CANT_ATTACK`, `EFF_CANT_RETREAT`, `EFF_PREVENT_ALL`, `EFF_DMG_TAKEN_MOD` and
`EFF_DMG_DEALT_MOD` (signed, /100), `EFF_DELAYED_DMG`, `EFF_DELAYED_KO`, `EFF_ATTACK_LOCKED`,
`EVOLVED_THIS_TURN`, `HEALED_THIS_TURN` — where 1.0 means the effect binds this turn and 0.5 that it is
registered and binds later, or is soft.

## Options and the action space

The action space is `Discrete(MAX_OPTIONS)` = 64. Slots 0..62 index `select["option"]` directly; slot 63 is
`STOP`. `opt_int` is `[opt_type, ptr1+1, ptr2+1, card_id, attack_id, number]`; pointers are token indices
plus one with 0 meaning null, and the `+1` lines up with the READOUT token at trunk position 0, so pointer
`k` reads `h[:, k]` directly. PLAY points at the hand card, ATTACH and EVOLVE at source and target, ATTACK
at my active, and card/discard/ability options at whatever token `(playerIndex, area, index)` resolves to.
`opt_float` carries `NUMBER`, `COUNT`, `PICKED`, `IS_STOP`, `SPECIAL` and `ENERGY_IDX`.

**Multi-select decomposition.** A select with `maxCount > 1` is answered one pick at a time. Each pick is a
normal forward pass over the same select with already-chosen indices flagged in `OptF.PICKED` and their
tokens in `TokF.PICKED`; `STOP` becomes legal once `len(picked) >= minCount`, and on `minCount == 0`
single-selects where it means "pick none". Choosing `STOP` submits the accumulated answer, and
`DecF.PICKED`, `GlobF.PICKS_MIN_LEFT` and `GlobF.PICKS_MAX_LEFT` say where in the sequence the net is.

**Class dedup masking.** The engine does not deduplicate equivalent options: four copies of one card in hand
are four indices. The encoder computes an equivalence key per option and sets `opt_mask` on exactly one
representative per class, so the policy's categorical is natively over classes and log-probs, ratios and
entropy stay consistent. The key is conservative: board-entity targets never collapse (two same-id Pokemon
with different HP are not equivalent), while hand, deck, discard, looking and face-down prize copies of one
card id are exactly symmetric and do.

## Globals, decision features and the oracle block

`global_f` (64 floats, `GlobF`) is the state belonging to no entity: turn and parity, who went first and
whether that is decided, action count, the four once-per-turn flags, prize/deck/hand/bench counts, both
actives' status conditions, the multi-select counters, the actor seat, tracker summaries
(`OPP_HAND_UNKNOWN`, `MY_PRIZES_KNOWN`), six player-level lingering locks (item, supporter, evolve, per
side), player damage boosts, six cumulative opponent-behaviour counters (energy removed, tools removed,
forced switches, per side), `MY_KOD_LAST_TURN` / `OPP_KOD_LAST_TURN`, and four static one-ply lethal
estimates (`MY_ACTIVE_IN_DANGER`, `..._P1`, and the opponent equivalents) from `_threat`.

`dec_int` is the current question: `SELECT_TYPE`, `CONTEXT`, `EFFECT_CARD`, `CONTEXT_CARD`, `MY_LAST_ATTACK`
/ `OPP_LAST_ATTACK` (zeroed once that Pokemon is no longer active) and `MY_SUPPORTER` / `OPP_SUPPORTER`.
`dec_float` holds `MIN_COUNT`, `MAX_COUNT`, `N_OPTIONS`, `REMAIN_DMG`, `REMAIN_ENERGY`, `PICKED` and
`FORCED_RUN` (selects auto-resolved since the last real decision).

The **oracle block** is ground truth for the critic and nothing else: 24 slots of opponent hand as `(id,
count/4)`, 24 slots of their unseen cards as `(id, deck_count/4, prize_count/4)`, 8 slots of my exact prizes
as `(id, count/4)`. Unknown identities are written as `UNK_CARD` (1283, the top of the vocab slack) rather
than guessed. The policy head has no forward path from these floats; leak safety is structural.

## InfoTracker: what the agent may legitimately know

`ptcg/tracker.py` maintains, from the blind observation stream of one game, everything deducible without
cheating. Construct it with your 60-card deck, call `update(obs)` on every observation.

- `my_unseen()` is your 60-list minus everything visible in your own zones, including the limbo correction:
  a trainer that has left the hand and not reached the discard is in no zone, so the resolving
  `select["effect"]` card is subtracted when its serial is nowhere to be found.
- `my_prize_known` becomes exact the first time a select shows your whole deck (prizes = unseen pool minus
  shown deck, guarded by a count check), is then maintained from your own logs since prize takes carry a
  card id, and is invalidated if an unidentified card enters the prize zone.
- `opp_known_hand` and `opp_known_deck` map serial to card id. Serials are unique per card and logs carry
  them, so hand knowledge never goes stale: an effect revealing a card into their hand records it, playing
  it removes it; ordinary draws reveal nothing. `opp_visible()` and `opp_revealed()` give the
  archetype-matching multisets, and `clone()` makes a cheap copy for search rollouts.

## Lingering effects

The engine never exposes effects that persist past the action that created them; they surface only as
options silently missing from later selects. `ptcg/effects.py` reconstructs that state from the log stream
using `data/lingering_effects.json` — card-text-derived records with fields `source_kind`, `source_name`,
`effect_class`, `target`, `duration`, `trigger_timing`, `magnitude` and `condition`. `_build_tables()` joins
them against the engine's tables into `attack_fx` and `play_fx`; attack records fire on `ATTACK` logs and
item/supporter records on `PLAY` logs, while ability records are skipped because the log enum has no
ability-used event.

`EffectTracker` registers an instance with a bind window `[from, to]` derived from `duration` (`SAME_TURN`,
`MY_NEXT_TURN`, `OPP_NEXT_TURN`, `UNTIL_END_OF_*`, `WHILE_CONDITION`) against a replayed turn counter that
advances on `TURN_START`. Instances attach to a Pokemon's **serial**, not its board slot, and clear when it
leaves play, is switched, or evolves; player-level classes (`CANT_PLAY_ITEM`, `CANT_PLAY_SUPPORTER`,
`CANT_EVOLVE`, `CANT_PLAY_POKEMON_FROM_HAND`) go to `player_locks` instead. Every flag has a strength: 1.0
while binding, 0.5 while registered but not yet binding, halved again when the registration is *soft* —
coin-flip conditions logs cannot resolve, and registrations against a side holding an effect-prevention
shield (a board ability, or an attached card protecting its holder). `token_flags(serial, turn)` and
`player_flags(seat, turn)` are what the encoder reads; the same object carries the temporal counters
(`energy_removed`, `tools_removed`, `forced_switches`, `evolved_now`, `healed_now`, `last_ko_turn`,
`supporter_now` / `supporter_prev`) feeding the global and decision blocks.

## OracleLedger: ground truth for the critic

`ptcg/rl/oracle.py` is env-side and structurally separate from the tracker. It knows both 60-card decks
because the environment dealt them, pins the initial prize six from a single `VisualizeData` parse during
setup (freezing at turn 1, since per-step calls cost 26-38 ms), and refreshes each seat's exact hand
whenever that seat receives an observation. Decks are derived: 60 minus hand, minus visible zones, minus
prizes. Where it cannot be certain — hidden draws since a seat's last decision, or a prize count disagreeing
with the public one — it emits `UNK_CARD` rows rather than wrong truth. `emit(obs)` returns the four
counters `write_oracle` needs.

## The network

`ptcg/rl/model.py` defines `PTCGTransformer`, with defaults `d=256`, `layers=4`, `heads=8`, `ffn=512`,
`d_c=128`, `d_a=64`, `critic_layers=1`, `critic="oracle"`.

**Static encoders.** `ptcg/rl/cards.py` builds four numpy tables once per process from the engine's card and
attack data: `card_static` (58 features per card — type and energy one-hots, the
basic/stage/ex/megaEx/tera/aceSpec flags, hp, retreat cost, prize liability, weakness and resistance
one-hots, attack and skill counts), `card_attacks` (up to 4 attack ids), `card_skills` (up to 2 ids in a
skill vocabulary of distinct `(name, text)` pairs) and `att_static` (15 per attack). `load_text_tables()`
adds the frozen text-embedding channel from `data/embed_pca.npz`, row-aligned to those tables: 128 dims per
card, 64 per attack, 64 per skill. Each forward rebuilds the composed tables in `_card_table` — AttackVec
from `attack_id_emb(48) ‖ att_static ‖ att_text`, SkillVec from `skill_id_emb(48) ‖ skill_text`, and CardVec
from `card_id_emb(64) ‖ card_static ‖ card_text` plus mean- and max-pools of its attacks' and skills'
vectors, through a 634-wide MLP ending at `d_c`. That is cheap at ~1.3k rows and lets gradients reach the
tables; `freeze_tables()` precomputes them for frozen policies, where the rebuild would otherwise dominate
batch-1 CPU inference.

**Tokens, context and trunk.** Each token is `[CardVec(card_id) ‖ sum of attached-tool CardVecs ‖ tok_float
‖ zone_emb(8) ‖ owner_emb(4)]`, 324 wide, projected to `d`; a learned `readout` parameter is prepended as a
true CLS with no input features. `global_mlp` maps `global_f` to 128 and `dec_mlp` maps the decision block
(select-type and context embeddings, the four card and two attack vectors named by `dec_int`, and
`dec_float`, 696 wide) to 64; `bcast(g ‖ d_vec)` is added to *every* token input, so globals reach layer 0
without an attention hop. The trunk is `layers` pre-LN blocks with manual QKV plus
`scaled_dot_product_attention` under a key-padding mask, then `ln_f`; `Block` builds QKV by hand because
`nn.MultiheadAttention` falls back to a weight-materializing path under a padding mask and runs out of
memory at training minibatch sizes. Per option slot, `q = q_mlp([opt_type_emb(16) ‖ h(ptr1) ‖ h(ptr2) ‖
opt_card(CardVec) ‖ opt_attack(AttackVec) ‖ opt_float])`, then `score = score_mlp([q ‖ r ‖ g ‖ d_vec ‖
q*r])` with `r` the READOUT output, masked to `-1e9` wherever `opt_mask` is 0.

**Value and aux heads.** `value_mlp([r ‖ g])` is the blind critic: it sees exactly what the policy sees, and
it is the head that ships. `_oracle_value` mounts a training-only top above the trunk — the 56 oracle slots
become tokens through `oracle_proj([CardVec(id) ‖ 6 typed features])`, and `critic_blocks` run joint
attention over `[critic_readout ‖ trunk outputs ‖ oracle tokens]` before `oracle_value_mlp([x0 ‖ g])`. With
`critic="oracle"` (the default) the oracle value is the PPO baseline and the blind head trains on the same
returns; with `critic="blind"` the roles swap. Both heads are linear, not tanh. `aux_hand` predicts opponent
hand counts over the card vocabulary from `[r ‖ g]` through a softplus, and `aux_prize` predicts per
`MY_UNSEEN` token whether that card is prized; both targets come from the oracle block of the same row.
`forward_policy` returns logits only, `forward_eval` returns `(logits, value)` following `critic_mode`,
`forward_train` returns everything the trainer needs. At the default size the model is **4,465,812
parameters** — trunk 2.11M, critic top 0.53M, auxiliary hand head 0.49M.

`native/model/ptcg_model.cu` is a from-scratch fp32 CUDA implementation of the same network, used by the
native trainer. Its size knob is the compile-time block `PT_D` / `PT_H` / `PT_FFN` / `PT_LAYERS`, mirrored
by `D, HEADS, FFN, LAYERS` in `native/tools/native_weights.py`; read both before changing either, since
every other width is derived. `native/tools/model_parity.py` runs both on the same weights and the same real
observation rows, comparing option scores, value, and the KL between the masked softmaxes.

## Weight formats

Torch checkpoints are plain state dicts. The native trainer reads and writes a packed fp32 blob whose order
is `PT_PARAMS` in `ptcg_model.cu`, mirrored name-for-name by `param_order()` in
`native/tools/native_weights.py`:

```bash
PYTHONPATH=data:. python native/tools/native_weights.py init  OUT.bin --seed 42
PYTHONPATH=data:. python native/tools/native_weights.py export CKPT.pt OUT.bin
PYTHONPATH=data:. python native/tools/native_weights.py import BLOB.bin OUT.pt
```

`import` merges the blob over a fresh model's state dict so the non-persistent static-table buffers are
present for `load_state_dict`. `arch_from_state_dict(sd, heads=...)` recovers `d`, `ffn` and `layers` from
the weights, because a mismatch there raises on load. It **cannot** recover the head count: `qkv` is `d x d`
for any number of heads, so a wrong `--heads` loads cleanly and computes different attention. Every loader
takes the value explicitly, and it must match how the run was launched.

## Rewards and advantage

`build_args` in `native/train_native.py` sets `reward_win: 1`: a terminal +1 to the winner, -1 to the loser,
nothing else. (`reward_win: 0` switches the environment to per-prize rewards topped up to +/-12; the trainer
does not use it.) Defaults are `--gamma 1.0` and `--gae-lambda 0.95`. Self-play uses **one row per game**.
The observation is always from the perspective of whichever seat must decide, and `GlobF.ACTOR_SEAT` records
which physical seat that was. The advantage is therefore negamax: the CUDA kernel `k_negamax_adv` reads the
seat column at `t` and `t+1` and multiplies the bootstrap and the trace by `-1` whenever they differ, so one
shared policy gets correct credit for both seats. Against a fixed opponent the seat is constant and the rule
reduces to ordinary GAE. Truncation at `max_engine_steps` (3000) resets **silently**, with no terminal flag,
so the advantage bootstraps across the boundary; environment errors restart the same way.

## Text embeddings

`data/embed_pca.npz` already covers the card sheet, so nothing needs regenerating to train. To rebuild it:

```bash
PYTHONPATH=data python3 tools/build_embed_texts.py   # -> data/embed_texts.json
python3 tools/embed_card_texts.py                    # needs OPENAI_API_KEY
python3 tools/build_embed_pca.py                     # -> data/embed_pca.npz
```

The first step renders one text per card, attack and skill, aligned row-for-row with the model's tables;
skill order must match `cards.build_tables` exactly. The second embeds them, and the third whitens and
row-normalizes a PCA projection to 128 / 64 / 64 dims so the frozen text channel does not dwarf the learned
id embeddings.

## Validating the observation

`tools/validate_obs_v4.py` drives `BattleHandle` with random legal answers, feeds per-seat `InfoTracker`s
exactly as the environment does, encodes every decision and checks five properties: every row is finite; a
hard `EFF_CANT_ATTACK` or `EFF_CANT_RETREAT` implies the engine offers no corresponding option; a hard item
or supporter lock implies no PLAY option for a hand card of that type; whenever `tracker.my_prize_known` is
set it equals the `OracleLedger`'s exact prize multiset; and the effect tracker's turn counter follows the
engine's. The argument is a game count; run it after any change to the encoder, tracker, effects layer or
effect table.

```bash
PYTHONPATH=data:. python3 tools/validate_obs_v4.py 120
```
