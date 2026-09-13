// PTCG native env: layout constants (mirror of ptcg/rl/buffers.py, VERSION 4),
// parsed-obs structs, tracker/oracle state, and the Env struct the pufferlib
// static vec drives. Pure C99 + cJSON; no CUDA, no python.
#ifndef PTCG_ENV_H
#define PTCG_ENV_H

#include <stdint.h>
#include <stdbool.h>
#include "cJSON.h"

// ---------------------------------------------------------------- buffers.py
#define PT_MAX_TOKENS 160
#define PT_TOK_INT 5
#define PT_TOK_F 56
#define PT_MAX_OPTIONS 64
#define PT_STOP (PT_MAX_OPTIONS - 1)
#define PT_MAX_ANCHORS 16   // pinned/oversampled decks

// First deck id belonging to the COVERAGE pool. Training ids are [0, n_decks)
// and coverage ids are [PT_DECK_ALT_BASE, PT_DECK_ALT_BASE + n_decks_alt).
//
// The gap is deliberate. If coverage started at n_decks, appending ONE
// training deck mid-run would renumber every coverage deck -- the
// exact silent-renaming failure the append-only design exists to prevent, just
// moved to the other pool. A fixed base makes append-only true for both.
//
// It therefore also caps the training pool: n_decks must stay < this, and the
// loader rejects a blob that breaks it rather than letting the two id spaces
// collide. The deck matrix is capped at the same value, so a coverage id can
// never land in it (which is the intent: coverage games are excluded anyway).
#define PT_DECK_ALT_BASE 1024
#define PT_MAX_BANKS 16     // frozen league opponents (per-bank Log slots)
#define PT_OPT_INT 6
#define PT_OPT_F 8
#define PT_GLOBAL_F 64
#define PT_DEC_INT 8
#define PT_DEC_F 16
#define PT_ORACLE_HAND_SLOTS 24
#define PT_ORACLE_SPLIT_SLOTS 24
#define PT_ORACLE_PRIZE_SLOTS 8
#define PT_ORACLE_F (PT_ORACLE_HAND_SLOTS*2 + PT_ORACLE_SPLIT_SLOTS*3 + PT_ORACLE_PRIZE_SLOTS*2)

#define PT_N_CARDS (1267 + 17)
#define PT_N_ATTACKS (1556 + 17)
#define PT_N_SKILLS (421 + 27)
#define PT_N_ZONES 16
#define PT_N_OWNERS 4
#define PT_N_OPT_TYPES (17 + 16)
#define PT_OT_STOP 17
#define PT_N_SELECT_TYPES (13 + 16)
#define PT_N_CONTEXTS (49 + 16)
#define PT_UNK_CARD (PT_N_CARDS - 1)

// section offsets (order in buffers.py _build_offsets)
#define PT_OFF_TOK_INT   0
#define PT_OFF_TOK_F     (PT_OFF_TOK_INT + PT_MAX_TOKENS*PT_TOK_INT)
#define PT_OFF_OPT_INT   (PT_OFF_TOK_F + PT_MAX_TOKENS*PT_TOK_F)
#define PT_OFF_OPT_F     (PT_OFF_OPT_INT + PT_MAX_OPTIONS*PT_OPT_INT)
#define PT_OFF_OPT_MASK  (PT_OFF_OPT_F + PT_MAX_OPTIONS*PT_OPT_F)
#define PT_OFF_GLOBAL    (PT_OFF_OPT_MASK + PT_MAX_OPTIONS)
#define PT_OFF_DEC_INT   (PT_OFF_GLOBAL + PT_GLOBAL_F)
#define PT_OFF_DEC_F     (PT_OFF_DEC_INT + PT_DEC_INT)
#define PT_OFF_ORACLE    (PT_OFF_DEC_F + PT_DEC_F)
#define PT_OBS_SIZE      (PT_OFF_ORACLE + PT_ORACLE_F)

#define PT_SEAT_COL   (PT_OFF_GLOBAL + 31)   // GlobF.ACTOR_SEAT
#define PT_FROZEN_COL (PT_OFF_GLOBAL + 54)   // GlobF.FROZEN_ACTOR

// Zone ids
enum { ZONE_PAD=0, ZONE_MY_ACTIVE=1, ZONE_MY_BENCH=2, ZONE_OPP_ACTIVE=3,
       ZONE_OPP_BENCH=4, ZONE_MY_HAND=5, ZONE_STADIUM=6, ZONE_MY_DISCARD=7,
       ZONE_OPP_DISCARD=8, ZONE_MY_UNSEEN=9, ZONE_OPP_HAND_KNOWN=10,
       ZONE_OPP_REVEALED=11, ZONE_LOOKING=12, ZONE_MY_PRIZE_KNOWN=13 };
enum { OWN_NEUTRAL=0, OWN_MINE=1, OWN_OPP=2 };

// TokF indices
enum { TF_COPIES=0, TF_HP_FRAC=1, TF_HP=2, TF_MAX_HP=3, TF_DMG=4,
       TF_N_ENERGY=5, TF_ENERGY0=6, TF_N_TOOLS=18, TF_STACK=19, TF_APPEAR=20,
       TF_POISONED=21, TF_BURNED=22, TF_ASLEEP=23, TF_PARALYZED=24,
       TF_CONFUSED=25, TF_IS_ACTIVE=26, TF_REFERENCED=27, TF_PICKED=28,
       TF_ATK0=29, TF_WEAK_HIT=41, TF_RES_HIT=42, TF_RETREAT_OK=43,
       TF_EFF_CANT_ATTACK=44, TF_EFF_CANT_RETREAT=45, TF_EFF_PREVENT_ALL=46,
       TF_EFF_DMG_TAKEN_MOD=47, TF_EFF_DMG_DEALT_MOD=48, TF_EFF_DELAYED_DMG=49,
       TF_EFF_DELAYED_KO=50, TF_EFF_ATTACK_LOCKED=51, TF_EVOLVED_THIS_TURN=52,
       TF_HEALED_THIS_TURN=53 };

// OptF indices
enum { OF_NUMBER=0, OF_COUNT=1, OF_PICKED=2, OF_IS_STOP=3, OF_SPECIAL=4,
       OF_ENERGY_IDX=5 };

// GlobF indices
enum { GF_TURN=0, GF_PARITY=1, GF_I_AM_FIRST=2, GF_FIRST_DECIDED=3,
       GF_ACTIONS=4, GF_SUPPORTER=5, GF_STADIUM_PLAYED=6, GF_ENERGY_ATTACHED=7,
       GF_RETREATED=8, GF_MY_PRIZES=9, GF_OPP_PRIZES=10, GF_MY_DECK=11,
       GF_OPP_DECK=12, GF_MY_HAND=13, GF_OPP_HAND=14, GF_MY_BENCH=15,
       GF_OPP_BENCH=16, GF_BENCH_MAX=17, GF_STADIUM_PRESENT=18,
       GF_MY_STATUS0=19, GF_OPP_STATUS0=24, GF_PICKS_MIN_LEFT=29,
       GF_PICKS_MAX_LEFT=30, GF_ACTOR_SEAT=31, GF_OPP_HAND_UNKNOWN=32,
       GF_MY_PRIZES_KNOWN=33, GF_MY_LOCK_ITEM=34, GF_MY_LOCK_SUPPORTER=35,
       GF_MY_LOCK_EVOLVE=36, GF_OPP_LOCK_ITEM=37, GF_OPP_LOCK_SUPPORTER=38,
       GF_OPP_LOCK_EVOLVE=39, GF_MY_DMG_BOOST=40, GF_OPP_DMG_BOOST=41,
       GF_MY_ENERGY_REMOVED=42, GF_OPP_ENERGY_REMOVED=43, GF_MY_TOOLS_REMOVED=44,
       GF_OPP_TOOLS_REMOVED=45, GF_MY_FORCED_SWITCHES=46, GF_OPP_FORCED_SWITCHES=47,
       GF_MY_KOD_LAST_TURN=48, GF_OPP_KOD_LAST_TURN=49, GF_MY_ACTIVE_IN_DANGER=50,
       GF_MY_ACTIVE_IN_DANGER_P1=51, GF_OPP_ACTIVE_IN_DANGER=52,
       GF_OPP_ACTIVE_IN_DANGER_P1=53, GF_FROZEN_ACTOR=54 };

enum { DI_SELECT_TYPE=0, DI_CONTEXT=1, DI_EFFECT_CARD=2, DI_CONTEXT_CARD=3,
       DI_MY_LAST_ATTACK=4, DI_OPP_LAST_ATTACK=5, DI_MY_SUPPORTER=6,
       DI_OPP_SUPPORTER=7 };
enum { DF_MIN_COUNT=0, DF_MAX_COUNT=1, DF_N_OPTIONS=2, DF_REMAIN_DMG=3,
       DF_REMAIN_ENERGY=4, DF_PICKED=5, DF_FORCED_RUN=6 };

// AreaType (cg.api)
enum { AREA_DECK=1, AREA_HAND=2, AREA_DISCARD=3, AREA_ACTIVE=4, AREA_BENCH=5,
       AREA_PRIZE=6, AREA_STADIUM=7, AREA_ENERGY=8, AREA_TOOL=9,
       AREA_PRE_EVOLUTION=10, AREA_PLAYER=11, AREA_LOOKING=12 };
// OptionType (cg.api)
enum { OT_NUMBER=0, OT_YES=1, OT_NO=2, OT_CARD=3, OT_TOOL_CARD=4,
       OT_ENERGY_CARD=5, OT_ENERGY=6, OT_PLAY=7, OT_ATTACH=8, OT_EVOLVE=9,
       OT_ABILITY=10, OT_DISCARD=11, OT_RETREAT=12, OT_ATTACK=13, OT_END=14,
       OT_SKILL=15, OT_SPECIAL=16 };
// LogType (cg.api)
enum { LOG_SHUFFLE=0, LOG_HAS_BASIC=1, LOG_TURN_START=2, LOG_TURN_END=3,
       LOG_DRAW=4, LOG_DRAW_REVERSE=5, LOG_MOVE_CARD=6, LOG_MOVE_REVERSE=7,
       LOG_SWITCH=8, LOG_CHANGE=9, LOG_PLAY=10, LOG_ATTACH=11, LOG_EVOLVE=12,
       LOG_DEVOLVE=13, LOG_MOVE_ATTACHED=14, LOG_ATTACK=15, LOG_HP_CHANGE=16,
       LOG_POISONED=17, LOG_BURNED=18, LOG_ASLEEP=19, LOG_PARALYZED=20,
       LOG_CONFUSED=21, LOG_COIN=22, LOG_RESULT=23 };

// ----------------------------------------------------------- parsed obs view
// Thin structs over one GetBattleData JSON document. All optional ints use
// PT_NONE when the field is absent/null (python None).
#define PT_NONE (-1000000)
#define PT_MAX_LIST 64      // per-zone card list cap (hand/discard/bench...)
#define PT_MAX_DECKSEL 64   // select.deck / looking cap
#define PT_MAX_LOGS 256
#define PT_MAX_OPTS 128     // engine can offer >64; env encodes first 63

typedef struct {
    int id, serial, playerIndex;   // PT_NONE where unknown/facedown
} PCard;

typedef struct {
    int id, serial, hp, maxHp;
    bool appearThisTurn;
    int n_energies; int energies[16];
    int n_energyCards; PCard energyCards[16];
    int n_tools; PCard tools[4];
    int n_preEvolution; PCard preEvolution[4];
    bool present;                  // false = None (facedown active)
} PPokemon;

typedef struct {
    int n_active; PPokemon active[1];
    int n_bench; PPokemon bench[8];
    int benchMax, deckCount, handCount;
    int n_discard; PCard discard[PT_MAX_LIST];
    int n_prize; PCard prize[8];          // id PT_NONE = facedown
    int n_hand; PCard hand[PT_MAX_LIST];  // n_hand=-1 => None (opponent)
    bool poisoned, burned, asleep, paralyzed, confused;
} PPlayer;

typedef struct {
    int type, number, area, index, playerIndex, toolIndex, energyIndex,
        count, inPlayArea, inPlayIndex, attackId, cardId, serial,
        specialConditionType;   // PT_NONE for absent
} POption;

typedef struct {
    int type, playerIndex, cardId, serial, fromArea, toArea,
        serialActive, serialBench, serialBefore, serialAfter, serialTarget,
        attackId, value, head, result,   // head: 0/1, PT_NONE absent
        // isRecover distinguishes "became poisoned" from "cured of poison" on
        // the special-condition types; reason is RESULT's win cause. Both are
        // 100% present on their types and neither is derivable.
        isRecover, reason;
} PLog;

typedef struct {
    bool has_select;
    int type, context, minCount, maxCount, remainDamageCounter,
        remainEnergyCost;
    int n_options; POption option[PT_MAX_OPTS];
    int n_deck; PCard deck[PT_MAX_DECKSEL];   // -1 => None
    bool has_effect; PCard effect;
    bool has_contextCard; PCard contextCard;
} PSelect;

typedef struct {
    bool has_current;
    int turn, turnActionCount, yourIndex, firstPlayer, result;
    bool supporterPlayed, stadiumPlayed, energyAttached, retreated;
    int n_stadium; PCard stadium[1];
    int n_looking; PCard looking[PT_MAX_DECKSEL];  // -1 => None
    PPlayer players[2];
    PSelect select;
    int n_logs; PLog logs[PT_MAX_LOGS];
} PObs;

// parse one GetBattleData JSON string into *out (fully overwrites).
// Returns 0 on success.
int pt_obs_parse(const char* json, PObs* out);

// ------------------------------------------------------------- static tables
#define FXREC_I 8
typedef struct {
    int key, kind, flag_tokf, lock, sign, target, condition, duration;
    float magnitude;
} FxRec;
// FxRec.kind: 0 plain-max flag | 1 dmg_taken | 2 dmg_dealt | 3 delayed_dmg
//             | 4 delayed_ko | -1 lock
enum { FXT_SELF=0, FXT_DEFENDING=1, FXT_ALL_MY=2, FXT_ALL_OPP=3,
       FXT_OPP_PLAYER=4, FXT_OTHER=5 };
enum { FXC_NONE=0, FXC_COIN_HEADS=1, FXC_COIN_TAILS=2, FXC_OTHER=3 };
enum { FXD_SAME_TURN=0, FXD_MY_NEXT=1, FXD_OPP_NEXT=2, FXD_UNTIL_END_MY_NEXT=3,
       FXD_UNTIL_END_OPP_NEXT=4, FXD_WHILE_CONDITION=5, FXD_OTHER=6 };
enum { LOCK_ITEM=1, LOCK_SUPPORTER=2, LOCK_EVOLVE=3 };

typedef struct {
    // env-side
    float* atk_damage;            // (N_ATTACKS)
    int8_t* atk_cost;             // (N_ATTACKS, 12)
    int8_t* energy_type;          // (N_CARDS)
    int8_t* weakness;             // (N_CARDS)
    int8_t* resistance;           // (N_CARDS)
    int8_t* retreat;              // (N_CARDS)
    int32_t* top_attacks;         // (N_CARDS, 4)
    uint8_t* supporters;          // (N_CARDS) membership
    uint8_t* shields;
    uint8_t* pokemon_ids;
    FxRec* attack_fx; int n_attack_fx;   // sorted by key
    FxRec* play_fx; int n_play_fx;
    int32_t* attack_fx_head;      // (N_ATTACKS) first index into attack_fx or -1
    int32_t* attack_fx_cnt;       // (N_ATTACKS)
    int32_t* play_fx_head;        // (N_CARDS)
    int32_t* play_fx_cnt;
    int32_t* decks; int n_decks;  // (n_decks, 60) TRAINING pool
    // Coverage pool: drawn ONLY by the random opponent's seat. Its ids start
    // at a FIXED base (PT_DECK_ALT_BASE), not at n_decks, so that appending a
    // training deck mid-run cannot renumber it. Absent in older blobs, which
    // read back as n_decks_alt = 0.
    int32_t* decks_alt; int n_decks_alt;  // (n_decks_alt, 60)
    // Per-deck sampling weights over the TRAINING pool. NULL means uniform,
    // which is the unweighted draw and is kept as an exact fast path -- a run
    // that never sets weights executes byte-identical sampling code.
    //
    // deck_w is normalised to mean 1 on publish, so weights are RELATIVE
    // WITHIN the training pool: scaling the whole file up or down cannot
    // change how often the random opponent reaches into the coverage pool.
    // deck_cw is the cumulative sum over the DRAW SLOT space -- training slots
    // first, then coverage slots each at weight 1 -- so the one array serves
    // both draw bounds (training-only, and training+coverage).
    float* deck_w;                // (n_decks) or NULL
    double* deck_cw;              // (n_decks + n_decks_alt) or NULL
    // model-side (loaded for the GPU; kept host-side here)
    float* card_static;           // (N_CARDS, 58)
    int32_t* card_attacks;        // (N_CARDS, 4)
    int32_t* card_skills;         // (N_CARDS, 2)
    float* att_static;            // (N_ATTACKS, 15)
    float* card_text;             // (N_CARDS, 128)
    float* att_text;              // (N_ATTACKS, 64)
    float* skill_text;            // (N_SKILLS, 64)
} PtTables;

// Load blob (path from env var PTCG_TABLES or default native/ptcg_tables.bin).
// Idempotent, thread-safe for concurrent readers after first call.
const PtTables* pt_tables(void);

// Swap in a grown TRAINING deck pool from a new blob, mid-run. APPEND ONLY:
// the new blob's first n_decks rows must be bytewise identical to the current
// ones, and its coverage pool must be unchanged -- see the long comment on the
// definition for why the coverage pool is part of the contract.
//
// *** CALLERS MUST BE PARKED. *** This does a non-atomic {pointer, count}
// publish, which is safe only because every env thread sits at the
// static_vec_omp_step fork-join barrier while the trainer is in _C.train().
// If a reload ever moves off that boundary, {decks, n_decks} must become one
// atomically-swapped struct or a reader will index past the end.
//
// Returns the new n_decks on success, or -1 with a message on stderr and NO
// state mutated. Deliberately does not abort(): refusing the swap and training
// on for another hour beats killing an 8-rank run over a bad file.
int pt_reload_decks(const char* path);

// Publish per-deck sampling weights over the training pool. `n` must equal the
// live n_decks -- a mismatch means the caller resolved names against a
// different pool listing, which is exactly the bug this rejects.
//
// Weights are non-negative and need not sum to anything: they are normalised
// to mean 1 internally. Zero disables a deck (it becomes unreachable in the
// draw, while keeping its id, so earlier deck matrices stay summable).
//
// Unlike pt_reload_decks this takes the write lock itself and does NOT require
// parked callers -- it publishes {deck_w, deck_cw} under the lock that
// game_start already holds for reading. Cheap either way: rebuilding is one
// pass over n_decks.
//
// Returns the number of ACTIVE decks (weight > 0) on success, or -1 with a
// message on stderr and NO state mutated. Rejects: length mismatch, negative,
// NaN, infinite, or all-zero weights.
int pt_set_deck_weights(const float* w, int n);

// Back to the uniform draw. Idempotent.
void pt_clear_deck_weights(void);

// Reader side of the deck-table lock. Hold across the deck DRAW and the COPY
// together (deck_row hands out a pointer into the table). Uncontended in normal
// operation; see the long comment in tables.c.
void pt_decks_rdlock(void);
void pt_decks_unlock(void);
// Writer side, exposed for tests that need to prove readers actually block.
void pt_decks_wrlock(void);

// --------------------------------------------------------------- small maps
// serial -> value open-addressing map (serials are small positive ints)
#define PT_MAP_CAP 512          // power of two; per-game serial space is ~200
typedef struct { int32_t keys[PT_MAP_CAP]; int32_t vals[PT_MAP_CAP]; int n; } SMap;
void smap_clear(SMap* m);
void smap_put(SMap* m, int key, int val);   // upsert
int  smap_get(const SMap* m, int key);      // PT_NONE if missing
void smap_del(SMap* m, int key);

typedef uint16_t CCount[PT_N_CARDS];        // dense card-id multiset
void pt_ccount_sub_clamp(uint16_t* a, const uint16_t* b);

// ------------------------------------------------------------ effect tracker
#define PT_MAX_FX_INST 128
typedef struct {
    const FxRec* rec;
    int bind_from, bind_to;      // bind_to PT_NONE = open-ended
    bool soft;
    int serial;                  // owning serial for serial_fx, else PT_NONE
    int player;                  // for player_locks / dmg_boost
    bool live;
} FxInst;

typedef struct {
    int turn, first_player, turn_drift;
    int actives[2];                       // running active serial, PT_NONE none
    FxInst inst[PT_MAX_FX_INST]; int n_inst;   // serial_fx + locks + boosts
    int last_attack_id[2], last_attack_serial[2];
    int supporter_now[2], supporter_prev[2];
    int energy_removed[2], tools_removed[2], forced_switches[2];
    SMap evolved_now, healed_now;         // serial -> 1 (this turn)
    int last_ko_turn[2];
} EffState;

// ---------------------------------------------------------------- tracker
typedef struct {
    int32_t my_deck_list[60];
    SMap opp_known_hand;        // serial -> cardId
    SMap opp_known_deck;
    CCount my_prize_known; bool my_prize_known_set;
    EffState fx;
} Tracker;

// ---------------------------------------------------------------- oracle
typedef struct {
    CCount decks[2];
    CCount hand[2];             // last-exact snapshot per seat
    CCount prize[2];
    bool frozen;
} Oracle;

// ------------------------------------------------------- per-game event log
// Compact form of PLog for offline card-level analysis. 16 bytes; the engine
// emits a few hundred per game, so a full 3B run is tens of GB (see
// docs/training.md). Fields dropped from PLog are the ones no analysis has
// asked for yet (serialBefore/After/Active/Bench) - add them here rather than
// widening the record at read time.
#define PT_MAX_GAME_EVENTS 4096
typedef struct {
    uint8_t  type;        // PLog.type (PLAY=10, DRAW=4, MOVE_CARD=6, ...)
    uint8_t  seat;        // playerIndex
    uint8_t  turn;        // obs.turn when emitted, saturating at 255
    int8_t   head;        // coin flip: 0/1, -1 absent
                          //   17..21: isRecover (1 cured, 0 inflicted)
    int16_t  card_id;     // -1 absent.  23 RESULT: the result code
    uint16_t serial;      // card instance, so two copies are distinguishable
                          //   9 CHANGE / 14 MOVE_ATTACHED: serialBefore
    int16_t  attack_id;   // -1 absent
    int16_t  value;       // damage / count, clamped to int16
                          //   14 MOVE_ATTACHED: the OLD host serial
                          //   23 RESULT: reason (1 prizes, 2 deck-out,
                          //              3 no active Pokemon, 4 card effect)
    int8_t   from_area;
    int8_t   to_area;
    uint16_t serial_target;  // 9/14: serialAfter (the NEW host)
} PtEvent;

// Per-log view-invariant signature, kept for EVERY engine log including the
// ones ev_keep() drops -- the dropped types (SHUFFLE, TURN_START/END) are the
// anchors that keep the two seats' views aligned, so they cannot be discarded
// before the merge. See ev_capture() for why the merge exists at all.
#define PT_MAX_GAME_SIGS 4096
#define PT_EV_LOOKAHEAD  12

// Decoder row exported per decision: PT_MAX_OPTIONS logits then the value.
#define PT_DEC_COLS (PT_MAX_OPTIONS + 1)
// Per-game fp16 payload holding, for each policy decision, the value head and
// the logits of the options actually on offer (n_options of them, plus the
// STOP slot). Stored packed and variable-length rather than a fixed 64 wide:
// the mean decision offers 5.6 options, so the fixed form would be ~10x
// larger AND would not compress, because the masked slots hold un-masked
// network output rather than a constant.
#define PT_MAX_GAME_PAYLOAD 16384
typedef struct {
    uint64_t sig;        // hash of the fields both seats agree on
    int16_t  ev_idx;     // slot in PtGame.ev[], or -1 if ev_keep() dropped it
    uint8_t  info;       // how much this view knew; a better view overwrites
    uint8_t  pad;
} PtSig;

// ------------------------------------------------------------------- game
typedef struct {
    void* battle;               // engine battle ptr
    PObs obs;                   // last parsed obs
    int learner_seat;
    Tracker trackers[2];
    Oracle oracle;
    int picked[PT_MAX_OPTIONS]; int n_picked;
    double cum_p0, pending_p0;
    int decisions, engine_steps;
    int forced_run[2];
    // Solar Transfer stall cap (docs/training.md): activations of
    // card 652's ability per seat in the turn ab652_turn. Reset on turn
    // change and at game_start.
    int ab652_turn;
    int ab652_used[2];
    uint64_t rng;               // pcg32 state
    int deck_idx[2];
    int32_t deck_pair[2][60];
    int opp_kind;               // 0 mirror, 1 random, 2 league (frozen bank env->tag)
    // event log; n_ev_seen counts every PHYSICAL event the engine emitted this
    // game (post-merge, post-filter), n_ev only those that fit, so truncation
    // is measurable rather than silent
    int n_ev, n_ev_seen;
    PtEvent ev[PT_MAX_GAME_EVENTS];
    // merge state: the global signature stream and each seat's cursor into it
    int n_sig, ev_cursor[2];
    PtSig sig[PT_MAX_GAME_SIGS];
    // policy logits + value, one variable-length entry per PT_EV_DECISION
    // record, in the same order. n_dec_dropped counts decisions that did not
    // fit so a short payload is measurable rather than silently truncating.
    int n_payload, n_dec, n_dec_dropped;
    uint16_t payload[PT_MAX_GAME_PAYLOAD];
    // True prize contents, snapshotted the moment the oracle pins them down
    // (both seats still on 6). Oracle.prize[] is the prizes REMAINING -- it
    // decrements as they are taken -- so it must be captured then, not at
    // game end. Face-down prizes are otherwise unrecoverable from the log.
    uint8_t  ev_pz_n, ev_pz_seat[16], ev_pz_cnt[16];
    int16_t  ev_pz_cid[16];
} PtGame;

// ---------------------------------------------------------------- Log/Env
// Aggregated per-episode sums; n counts episodes (vecenv divides by n).
typedef struct {
    float n;
    float ep_len, prize_margin, prize_margin_dec, first_wins, draws, decided;
    float turns_first, turns_second;
    float eps_mirror, eps_rand, w_rand, wins;
    float illegal, env_errors, ep_ret_abs, ep_ret_bad;
    float end_prize, end_bench, end_deck, end_other;       // mirror games
    float wend_prize, wend_bench, wend_deck, wend_other;   // won vs random
    float lend_prize, lend_bench, lend_deck, lend_other;   // lost vs random
    // league (frozen-opponent) games; kept apart so win_vs_rand stays pure
    float eps_league, w_league, league_len;
    float lgw_prize, lgw_bench, lgw_deck, lgw_other;       // won vs league
    float lgl_prize, lgl_bench, lgl_deck, lgl_other;       // lost vs league
    float league_n_b[PT_MAX_BANKS], league_w_b[PT_MAX_BANKS];
    float engine_steps, episode_return;
    // Per-deck TRAINING win rate for the tracked decks (the --anchor-decks
    // list, in that order). Deck sampling is unaffected -- these indices only
    // label which decks to count. Decided games only (draws/truncations are
    // excluded from both counters, so the ratio is a true win rate).
    // NOTE: Log is aggregated by summing sizeof(Log)/sizeof(float) floats and
    // then dividing every field by the episode count, so both counters shrink
    // by the same factor and deck_wins/deck_games stays exact.
    float deck_games[PT_MAX_ANCHORS], deck_wins[PT_MAX_ANCHORS];
} Log;

typedef struct {
    // pufferlib slot pointers (1 agent per env)
    float* observations;        // -> vec buffer row (PT_OBS_SIZE floats)
    float* actions;             // 1 float
    float* rewards;
    float* terminals;
    unsigned char* action_mask; // 64 bytes
    // The decoder row that produced this step's action: PT_MAX_OPTIONS logits
    // then the value head, mirrored host-side by the vecenv worker. NULL when
    // the trainer is not exporting it (test_env, and any run with the event
    // log off) -- the env must tolerate that.
    const float* dec;
    Log log;                    // pending aggregate (reset on read)
    int num_agents;
    // config
    float mix_self;             // P(mirror); remainder -> random opponent
    int max_engine_steps;
    float win_r;                // terminal reward magnitude (reward=win: 1.0)
    int prize_rewards;          // 0 for reward=win
    // Anchor decks: the pinned bench decks the eval battery scores. Training
    // uniformly over all 203 pool decks means the decks we actually SHIP are
    // seen ~2.5% of the time each; these knobs oversample them. Indices are
    // positions in the sorted-glob deck table (same construction as
    // deck_report.deck_names(), so Python can compute them by name).
    // Probability is applied per SEAT: the learner's seat uses
    // p_anchor_learner, the opposing seat uses p_anchor_self_opp in self
    // games and p_anchor_league_opp against a frozen/league opponent.
    int n_anchors;
    int anchor_idx[PT_MAX_ANCHORS];
    float p_anchor_learner;
    float p_anchor_self_opp;
    float p_anchor_league_opp;
    // League row assignment (vecenv MY_USES_TAGS): tag 0 = normal self-play
    // row (mirror/random via mix_self); tag b in [1, PT_MAX_BANKS] = every
    // game on this row is played against frozen weight bank b. The trainer
    // stamps tags to match its bank_layout slices; the opponent seat's rows
    // then carry the tag in obs col PT_FROZEN_COL for routing/loss masking.
    int tag;
    int boundary_reached;       // required by MY_USES_TAGS; unused here
    // state
    PtGame game;
    uint64_t seed;
} Env;

void ptcg_env_init(Env* env, uint64_t seed);
void c_reset(Env* env);
void c_step(Env* env);
void c_close(Env* env);
void c_render(Env* env);

// encoder (mirrors ptcg/rl/encoder.py exactly)
void pt_encode(float* row, const PObs* obs, const int* picked, int n_picked,
               bool stop_allowed, int forced_run, const Tracker* tr);
void pt_write_oracle(float* row, const Oracle* orc, const PObs* obs);
void pt_write_mask(unsigned char* mask, const float* row);

// tracker/effects/oracle updates
void tracker_init(Tracker* t, const int32_t* deck60);
void tracker_update(Tracker* t, const PObs* obs);
void tracker_my_unseen(const Tracker* t, const PObs* obs, CCount out);
void tracker_opp_known_hand(const Tracker* t, CCount out);
void tracker_opp_known_deck(const Tracker* t, CCount out);
void oracle_init(Oracle* o, const int32_t* d0, const int32_t* d1);
// viz(ctx) returns a VisualizeData JSON string; called only while the
// prize six is still being pinned (parity harness injects captured payloads)
typedef const char* (*PtVizFn)(void* ctx);
void oracle_on_seat_obs(Oracle* o, const PObs* obs, PtVizFn viz, void* ctx);
void oracle_emit(const Oracle* o, const PObs* obs, CCount opp_hand,
                 CCount opp_deck, CCount opp_prize, CCount my_prize);
void effstate_init(EffState* e);
void fx_token_flags(const EffState* e, int serial, int t, float* f);
void fx_player_flags(const EffState* e, int seat, int t, float locks[4],
                     float* boost);
int fx_last_attack_id(const EffState* e, int seat, int active_serial);

// engine wrapper (thread-safe per battle)
void* pt_battle_start(const int32_t* d0, const int32_t* d1);
const char* pt_battle_data(void* battle);
int pt_battle_select(void* battle, const int* idx, int n);
void pt_battle_finish(void* battle);
const char* pt_battle_visualize(void* battle);
void pt_engine_init(void);

#endif // PTCG_ENV_H
