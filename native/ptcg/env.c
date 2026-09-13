// Game driver: port of ptcg/rl/env.py (_Game + PTCGEnv step machinery)
// for the pufferlib native backend. One env = one game = one agent row.
// Self-play serves both seats through the row. Untagged rows split
// mirror-vs-random at mix_self; rows tagged with a frozen bank id play
// LEAGUE games (opp_kind 2): the opponent seat's decisions surface as
// normal encoded rows (carrying the bank id in obs col PT_FROZEN_COL) so
// the GPU can act them with frozen weights — the trainer masks those
// decisions out of the policy loss.
//
// Divergences from the python env, all deliberate and none affecting
// training math:
//   - no replay dumps / deck jsonl logs (analysis tooling stays python)
//   - truncation (max_engine_steps, env_error restart) resets silently
//     with NO terminal flag — pufferl 3.0 never fed truncations to the
//     advantage kernel (pufferl.py:275 stores `d` only), so parity means
//     bootstrapping across the boundary exactly like the torch stack.
//   - Solar Transfer stall cap (PT_AB652_*, docs/training.md):
//     native-only guard, absent from the python env. Inert for any pool
//     without card 652, which includes every parity stream recorded so far.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "ptcg_env.h"

// ------------------------------------------------- deck-vs-deck result matrix
// Full pairwise "which deck beat whom" over the WHOLE 203-deck pool, for
// offline analysis (archetype/variant strength). It cannot live in Log: that
// is per-Env and aggregated by summing the struct, so an n_decks^2 matrix
// would cost ~330KB * 1024 envs per rank. Instead one process-global matrix,
// flushed to a file and reset on an interval; the supervisor ships the files
// to HF. Config arrives through the ENVIRONMENT, not the kwargs Dict, because
// that Dict carries doubles only and cannot pass a path.
//
//   PTCG_DECK_MATRIX        file prefix; unset = feature off (zero cost)
//   PTCG_DECK_MATRIX_EVERY  episodes per flush (default 20000)
//
// The vecenv steps envs from several threads, so increments take a mutex.
// At ~100 finished episodes/s the lock is uncontended in practice.
static pthread_mutex_t g_dm_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t *g_dm_games = NULL, *g_dm_wins = NULL;
static int g_dm_n = 0, g_dm_state = 0, g_dm_seq = 0;   // state: 0 unknown 1 on 2 off
static long g_dm_eps = 0, g_dm_every = 20000;
static const char *g_dm_prefix = NULL;

// Current training epoch, or 0 if unknown. The env has no notion of epochs,
// so the trainer drops it in "<prefix>.epoch" once per epoch (tmp+rename, and
// per-rank so there is no writer race). Deliberately NOT an env var: Python's
// os.environ assignment calls setenv(), which is not safe against concurrent
// getenv() on the env-stepping threads.
static uint32_t read_epoch_file(const char *prefix) {
    if (!prefix) return 0u;
    char epath[544];
    snprintf(epath, sizeof(epath), "%s.epoch", prefix);
    FILE *e = fopen(epath, "r");
    if (!e) return 0u;
    long v = 0;
    int got = fscanf(e, "%ld", &v);
    fclose(e);
    return (got == 1 && v >= 0) ? (uint32_t)v : 0u;
}

static uint32_t dm_read_epoch(void) { return read_epoch_file(g_dm_prefix); }

// Caller must hold g_dm_lock.
static void dm_flush_locked(void) {
    if (!g_dm_games || g_dm_eps <= 0) return;
    uint32_t epoch = dm_read_epoch();
    // Epoch goes in the NAME as well as the header so a directory listing is
    // already a time series without opening every file.
    char path[576];
    snprintf(path, sizeof(path), "%s_e%08u_%06d.bin", g_dm_prefix, epoch, g_dm_seq);
    char tmp[608];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return;                       // never fatal: telemetry only
    // header v2: magic, version, n_decks, episodes in THIS interval, seq, epoch
    // (v1 had no epoch field; tools/deck_matrix.py reads both)
    uint32_t hdr[6] = {0x4D445450u /* 'PTDM' */, 2u, (uint32_t)g_dm_n,
                       (uint32_t)g_dm_eps, (uint32_t)g_dm_seq, epoch};
    size_t cells = (size_t)g_dm_n * (size_t)g_dm_n;
    int ok = fwrite(hdr, sizeof(hdr), 1, f) == 1
             && fwrite(g_dm_games, sizeof(uint32_t), cells, f) == cells
             && fwrite(g_dm_wins, sizeof(uint32_t), cells, f) == cells;
    fclose(f);
    if (ok && rename(tmp, path) == 0) {   // tmp+rename so a reader never sees a partial file
        g_dm_seq++;
        memset(g_dm_games, 0, cells * sizeof(uint32_t));
        memset(g_dm_wins, 0, cells * sizeof(uint32_t));
        g_dm_eps = 0;
    } else {
        remove(tmp);
    }
}

// Record one decided game from the learner seat's perspective.
static void dm_record(int n_decks, int ld, int od, int won) {
    if (g_dm_state == 2) return;
    pthread_mutex_lock(&g_dm_lock);
    if (g_dm_state == 0) {                // first call: latch config
        g_dm_prefix = getenv("PTCG_DECK_MATRIX");
        const char *ev = getenv("PTCG_DECK_MATRIX_EVERY");
        if (ev && atol(ev) > 0) g_dm_every = atol(ev);
        // Pre-size to a CONSTANT width rather than the current pool size, so
        // matrices stay summable as the pool grows mid-run (docs/deck-pool.md).
        // Without this, g_dm_n latches at the pool size it first saw
        // and the ld/od guard below silently DISCARDS every game involving a
        // deck appended later -- no counter, no warning.
        // 1024^2 * 4B * 2 arrays = 8 MB, irrelevant next to the rollout buffers.
        // Readers cope with a names listing shorter than the width; trailing
        // rows are all-zero by construction (see deck_matrix.fit_to_names).
        if (g_dm_prefix && n_decks > 0) {
            int width = PT_DECK_ALT_BASE;
            const char *mx = getenv("PTCG_DECK_MATRIX_MAX");
            if (mx && atoi(mx) > 0) width = atoi(mx);
            if (width < n_decks) width = n_decks;   // never lose the live pool
            // Never at or above the coverage base: coverage ids start there,
            // and they must stay out of the matrix.
            if (width > PT_DECK_ALT_BASE) width = PT_DECK_ALT_BASE;
            size_t cells = (size_t)width * (size_t)width;
            g_dm_games = (uint32_t*)calloc(cells, sizeof(uint32_t));
            g_dm_wins = (uint32_t*)calloc(cells, sizeof(uint32_t));
            g_dm_n = width;
        }
        g_dm_state = (g_dm_games && g_dm_wins) ? 1 : 2;
        if (g_dm_state == 2) { free(g_dm_games); free(g_dm_wins);
                               g_dm_games = g_dm_wins = NULL; }
    }
    if (g_dm_state == 1 && ld >= 0 && ld < g_dm_n && od >= 0 && od < g_dm_n) {
        size_t k = (size_t)ld * (size_t)g_dm_n + (size_t)od;
        g_dm_games[k]++;
        if (won) g_dm_wins[k]++;
        if (++g_dm_eps >= g_dm_every) dm_flush_locked();
    }
    pthread_mutex_unlock(&g_dm_lock);
}

// ----------------------------------------------------------- game event log
// Every engine log event of every non-random game, for offline card-level
// credit assignment. Same shape as the deck matrix above: process-global,
// mutex-guarded, rotated on an episode interval, shipped to HF by the
// supervisor. Config via the environment for the same reason.
//
//   PTCG_EVENT_LOG        file prefix; unset = feature off (zero cost)
//   PTCG_EVENT_LOG_EVERY  episodes per file (default 20000)
//
// RANDOM-OPPONENT GAMES ARE EXCLUDED (opp_kind 1): that seat is not a policy,
// so its card choices carry no credit signal and would dilute the dataset.
// Draws and max_engine_steps truncations ARE written, with result 2 and -2 --
// dropping them is what let the stall exploit hide in the deck matrix.
#define PT_EV_ACTION   200        // ours, not an engine LogType
#define PT_EV_DECISION 201        // ditto: one per policy decision
#define PT_EV_PRIZE    202        // ditto: the true prize contents

static inline int16_t ev_clamp16(int v) {
    return (int16_t)(v > 32767 ? 32767 : (v < -32768 ? -32768 : v));
}

#define PT_EV_OURS(t) ((t) == PT_EV_ACTION || (t) == PT_EV_DECISION)

static pthread_mutex_t g_ev_lock = PTHREAD_MUTEX_INITIALIZER;
static FILE *g_ev_f = NULL;
static int g_ev_state = 0, g_ev_seq = 0;      // state: 0 unknown 1 on 2 off
static long g_ev_eps = 0, g_ev_every = 20000;
static const char *g_ev_prefix = NULL;
static char g_ev_tmp[608];

static void ev_open_locked(uint32_t epoch) {
    char path[576];
    snprintf(path, sizeof(path), "%s_e%08u_%06d.bin", g_ev_prefix, epoch, g_ev_seq);
    snprintf(g_ev_tmp, sizeof(g_ev_tmp), "%s.tmp", path);
    g_ev_f = fopen(g_ev_tmp, "wb");
    if (!g_ev_f) return;
    // header: magic 'PTEV', version, record size, epoch
    // v2: the episode header gained payload_bytes and dec_dropped, and the
    // episode gained a trailing fp16 payload of policy logits + value.
    uint32_t hdr[4] = {0x50544556u, 2u, (uint32_t)sizeof(PtEvent), epoch};
    fwrite(hdr, sizeof(hdr), 1, g_ev_f);
}

// Caller holds g_ev_lock. tmp+rename so a reader never sees a partial file.
static void ev_close_locked(uint32_t epoch) {
    if (!g_ev_f) return;
    fclose(g_ev_f);
    g_ev_f = NULL;
    char path[576];
    snprintf(path, sizeof(path), "%s_e%08u_%06d.bin", g_ev_prefix, epoch, g_ev_seq);
    if (rename(g_ev_tmp, path) == 0) g_ev_seq++;
    else remove(g_ev_tmp);
    g_ev_eps = 0;
}

static void ev_write(const PtGame* g, int result) {
    if (g_ev_state == 2) return;
    pthread_mutex_lock(&g_ev_lock);
    if (g_ev_state == 0) {
        g_ev_prefix = getenv("PTCG_EVENT_LOG");
        const char *e = getenv("PTCG_EVENT_LOG_EVERY");
        if (e && atol(e) > 0) g_ev_every = atol(e);
        g_ev_state = g_ev_prefix ? 1 : 2;
    }
    if (g_ev_state == 1) {
        uint32_t epoch = read_epoch_file(g_ev_prefix);
        if (!g_ev_f) ev_open_locked(epoch);
        if (g_ev_f) {
            // 16-byte episode header, then n_ev events
            uint32_t h0 = epoch;
            uint16_t dl = (uint16_t)g->deck_idx[g->learner_seat];
            uint16_t dop = (uint16_t)g->deck_idx[1 - g->learner_seat];
            uint8_t seat = (uint8_t)g->learner_seat;
            int8_t res = (int8_t)result;
            uint8_t ok = (uint8_t)g->opp_kind;
            // was a spare pad byte; firstPlayer is not derivable from the
            // event stream and costs nothing here
            uint8_t first = (uint8_t)(g->obs.firstPlayer < 0 ? 255
                                      : g->obs.firstPlayer);
            // ---- true prize contents (oracle) ----
            // Prizes are face-down to their OWN owner at every step, so the
            // event stream can only ever show them as anonymous deck->prize
            // reverse records; a card that is prized and never taken is not
            // recoverable from the log at all, and eliminating over "never
            // seen all game" does NOT work (that set is prizes PLUS whatever
            // was still in the deck at the end). Offline analysis then has to
            // fall back on the tracker's deduction, which only pins the
            // multiset down once a deck search reveals the deck -- measured at
            // 84% of turn-start draws on one deck, with the rest unusable.
            //
            // The env already knows the truth: Oracle.prize[seat] is exact
            // from turn 1 (tracker.c, oracle_parse_prizes off the visualize
            // frames). Emitting it costs ~12 records and removes the deduction
            // step entirely. This is ORACLE information -- it was never
            // available to the policy and must not be fed back into one.
            PtEvent pz[16];
            int n_pz = 0;
            for (int k = 0; k < (int)g->ev_pz_n && n_pz < 16; k++) {
                PtEvent* e = &pz[n_pz++];
                memset(e, 0, sizeof(*e));
                e->type = PT_EV_PRIZE;
                e->seat = g->ev_pz_seat[k];
                e->card_id = g->ev_pz_cid[k];
                e->value = (int16_t)g->ev_pz_cnt[k];   // copies in the prizes
                e->attack_id = -1;
                e->from_area = -1;
                e->to_area = 6;                        // PRIZE
            }
            uint16_t nev = (uint16_t)(g->n_ev + n_pz);
            uint16_t nseen = (uint16_t)(g->n_ev_seen > 65535 ? 65535 : g->n_ev_seen);
            uint32_t pbytes = (uint32_t)g->n_payload * 2u;
            uint16_t ndrop = (uint16_t)(g->n_dec_dropped > 65535 ? 65535
                                        : g->n_dec_dropped);
            uint16_t pad0 = 0;
            fwrite(&h0, 4, 1, g_ev_f);
            fwrite(&dl, 2, 1, g_ev_f);   fwrite(&dop, 2, 1, g_ev_f);
            fwrite(&seat, 1, 1, g_ev_f); fwrite(&res, 1, 1, g_ev_f);
            fwrite(&ok, 1, 1, g_ev_f);   fwrite(&first, 1, 1, g_ev_f);
            fwrite(&nev, 2, 1, g_ev_f);  fwrite(&nseen, 2, 1, g_ev_f);
            fwrite(&pbytes, 4, 1, g_ev_f);
            fwrite(&ndrop, 2, 1, g_ev_f); fwrite(&pad0, 2, 1, g_ev_f);
            if (n_pz > 0) fwrite(pz, sizeof(PtEvent), (size_t)n_pz, g_ev_f);
            // sig[] carries global order for engine events; ACTION records sit
            // outside it and are spliced back in at their anchor.
            int ai = 0;
            for (int i = 0; i <= g->n_sig; i++) {
                while (ai < g->n_ev) {
                    const PtEvent* a = &g->ev[ai];
                    if (!PT_EV_OURS(a->type)) { ai++; continue; }
                    if ((int)a->serial > i) break;
                    PtEvent out = *a;
                    out.serial = 0;                  // strip the anchor
                    fwrite(&out, sizeof(PtEvent), 1, g_ev_f);
                    ai++;
                }
                if (i < g->n_sig && g->sig[i].ev_idx >= 0)
                    fwrite(&g->ev[g->sig[i].ev_idx], sizeof(PtEvent), 1, g_ev_f);
            }
            // fp16 policy logits + value, one variable-length entry per
            // DECISION record, same order. Length comes from that record's
            // n_options, so the two are read in lockstep.
            if (g->n_payload > 0)
                fwrite(g->payload, 2, (size_t)g->n_payload, g_ev_f);
            if (++g_ev_eps >= g_ev_every) ev_close_locked(epoch);
        }
    }
    pthread_mutex_unlock(&g_ev_lock);
}

// Append the engine events produced by the transition just parsed into g->obs.
// Random-opponent games are skipped here so they cost nothing.
//
// Dropped outright: SHUFFLE(0), TURN_START(2), TURN_END(3). No card, and every
// event carries `turn` so the boundaries are implied. They still get a slot in
// g->sig[] -- they are the anchors the two seats' views align on.
//
// DRAW_REVERSE(5) and MOVE_CARD_REVERSE(7) are NOT dropped here. They
// are the observer's view of an event the owner also reports; the merge in
// ev_capture() upgrades them in place to the owner's DRAW(4)/MOVE_CARD(6) when
// that view arrives. What survives as a 5 or a 7 is an event NO seat ever saw
// the identity of -- overwhelmingly deck->prize, which is face-down to both
// players and is the only way the prize zone ever gets filled. Dropping those
// unconditionally makes prize and hand counts unreconstructable; the residual
// is 12/game, exactly 6 prizes x 2 players. See docs/training.md.
static inline int ev_keep(int type) {
    switch (type) {
        case 0: case 2: case 3: return 0;
        default: return 1;
    }
}

// --------------------------------------------------------------- merge
// g->obs.logs is the ACTING seat's window: every event since THAT seat last
// acted. The seats' cursors sit at different places, so consecutive windows
// overlap and each physical event is presented once per seat's view. Writing
// both inflates the file ~1.5x and, worse, doubles every per-card count --
// measured PLAY ratio exactly 2.00 over 344 episodes.
//
// ev_sig() is what the two views agree on. DRAW/DRAW_REVERSE collapse to one
// signature (the observer sees no cardId) and so do MOVE_CARD/_REVERSE; for
// everything else both seats report the same fields.
static inline uint64_t ev_sig(const PLog* L) {
    int st = (L->type == 5) ? 4 : (L->type == 7) ? 6 : L->type;
    uint64_t h = 1469598103934665603ULL;
#define EV_MIX(x) do { h ^= (uint64_t)(int64_t)(x); h *= 1099511628211ULL; } while (0)
    EV_MIX(st); EV_MIX(L->playerIndex);
    if (st == 6) { EV_MIX(L->fromArea); EV_MIX(L->toArea); }
    else if (st != 4) {
        EV_MIX(L->serial);       EV_MIX(L->serialTarget);
        EV_MIX(L->serialActive); EV_MIX(L->serialBench);
        EV_MIX(L->serialBefore); EV_MIX(L->serialAfter);
        EV_MIX(L->value);        EV_MIX(L->attackId);   EV_MIX(L->head);
    }
#undef EV_MIX
    return h;
}

// How much this view knew. A view that names more of the cards wins.
static inline int ev_info(const PLog* L) {
    int n = 0;
    if (L->cardId       != PT_NONE) n++;
    if (L->serial       != PT_NONE) n++;
    if (L->serialTarget != PT_NONE) n++;
    if (L->serialActive != PT_NONE) n++;
    if (L->serialBench  != PT_NONE) n++;
    if (L->serialBefore != PT_NONE) n++;
    if (L->serialAfter  != PT_NONE) n++;
    return n;
}

static void ev_fill(PtEvent* e, const PLog* L, int turn, int acting_seat) {
    e->type = (uint8_t)L->type;
    // top bit of seat = this event happened on the seat that is NOT to
    // act, i.e. it was public. Restores the visibility distinction the
    // dropped _REVERSE types carried, at zero extra bytes.
    int pi = L->playerIndex;
    e->seat = (uint8_t)(pi < 0 ? 255 : (pi | (pi != acting_seat ? 0x80 : 0)));
    e->turn = (uint8_t)(turn > 255 ? 255 : (turn < 0 ? 0 : turn));
    e->head = (int8_t)(L->head == PT_NONE ? -1 : L->head);
    e->card_id = ev_clamp16(L->cardId == PT_NONE ? -1 : L->cardId);
    e->serial = (uint16_t)(L->serial == PT_NONE ? 0 : L->serial);
    e->attack_id = ev_clamp16(L->attackId == PT_NONE ? -1 : L->attackId);
    e->value = ev_clamp16(L->value == PT_NONE ? 0 : L->value);
    e->from_area = (int8_t)(L->fromArea == PT_NONE ? -1 : L->fromArea);
    e->to_area = (int8_t)(L->toArea == PT_NONE ? -1 : L->toArea);
    e->serial_target = (uint16_t)(L->serialTarget == PT_NONE ? 0 : L->serialTarget);
    // SWITCH names its two Pokemon in serialActive/serialBench; CHANGE and
    // MOVE_ATTACHED name theirs in serialBefore/serialAfter. Reading
    // serialActive/serialBench for CHANGE (as this did) left the record empty,
    // and MOVE_ATTACHED had no route for its host serials at all -- you knew a
    // card moved but not from where to where.
    if (L->type == 8) {
        if (L->serialActive != PT_NONE) e->serial = (uint16_t)L->serialActive;
        if (L->serialBench  != PT_NONE) e->serial_target = (uint16_t)L->serialBench;
    } else if (L->type == 9) {
        if (L->serialBefore != PT_NONE) e->serial = (uint16_t)L->serialBefore;
        if (L->serialAfter  != PT_NONE) e->serial_target = (uint16_t)L->serialAfter;
    } else if (L->type == 14) {
        if (L->serialAfter  != PT_NONE) e->serial_target = (uint16_t)L->serialAfter;
        if (L->serialBefore != PT_NONE) e->value = ev_clamp16(L->serialBefore);
    } else if (L->type >= 17 && L->type <= 21) {
        e->head = (int8_t)(L->isRecover == PT_NONE ? -1 : L->isRecover);
    } else if (L->type == 23) {
        e->card_id = ev_clamp16(L->result == PT_NONE ? -1 : L->result);
        e->value = ev_clamp16(L->reason == PT_NONE ? 0 : L->reason);
    }
}

// Allocate (or reuse) the ev[] slot behind sig entry `si` and fill it.
static void ev_realize(PtGame* g, int si, const PLog* L, int turn, int seat) {
    if (!ev_keep(L->type)) return;
    int slot = g->sig[si].ev_idx;
    if (slot < 0) {
        g->n_ev_seen++;
        if (g->n_ev >= PT_MAX_GAME_EVENTS) return;   // measurable, not silent
        slot = g->n_ev++;
        g->sig[si].ev_idx = (int16_t)slot;
    }
    ev_fill(&g->ev[slot], L, turn, seat);
}

// The engine takes no seed (pt_battle_start(d0,d1)), so a logged game cannot
// be replayed deterministically and the option set at each decision is NOT
// recoverable after the fact. Record the choice itself: type 200, card_id =
// submitted option index, value = how many options were on offer, head = the
// position within a multi-pick. n_options==1 means the env answered a forced
// select, not a policy decision.
// ACTION records are ours, not the engine's, so they must NOT take a slot in
// sig[]: a seat only ever aligns against engine logs, and an ACTION sitting
// between them blocks the other seat's insert (it lands before the ACTION,
// which silently pushes the first seat's cursor past an event it had not
// consumed -- that single interaction was worth ~0.8x of duplication). They
// carry an anchor instead: "this decision happened before sig entry N", which
// ev_write() uses to interleave them back into the stream. The anchor lives in
// `serial`, unused for ACTION, and is zeroed on the way out.
static void ev_action(PtGame* g, const int* ans, int n) {
    if (g_ev_state == 2 || g->opp_kind == 1) return;
    int nopt = g->obs.select.has_select ? g->obs.select.n_options : 0;
    for (int k = 0; k < n; k++) {
        g->n_ev_seen++;
        if (g->n_ev >= PT_MAX_GAME_EVENTS) continue;
        PtEvent* e = &g->ev[g->n_ev++];
        memset(e, 0, sizeof(*e));
        e->type = PT_EV_ACTION;
        e->seat = (uint8_t)(g->obs.yourIndex < 0 ? 255 : g->obs.yourIndex);
        e->turn = (uint8_t)(g->obs.turn > 255 ? 255 : (g->obs.turn < 0 ? 0 : g->obs.turn));
        e->head = (int8_t)(k > 127 ? 127 : k);
        e->card_id = ev_clamp16(ans[k]);
        e->attack_id = -1;
        e->value = ev_clamp16(nopt);
        e->serial = (uint16_t)g->n_sig;      // anchor, stripped by ev_write
    }
}

// Append at the end of the global stream.
static void ev_push(PtGame* g, const PLog* L, uint64_t s, int turn, int seat) {
    if (g->n_sig >= PT_MAX_GAME_SIGS) return;
    int si = g->n_sig++;
    g->sig[si].sig = s;
    g->sig[si].ev_idx = -1;
    g->sig[si].info = (uint8_t)ev_info(L);
    ev_realize(g, si, L, turn, seat);
}

// This seat saw an event that belongs BEFORE entries already in the stream
// (its own deck search, which the opponent is sometimes not told about at
// all). Only sig[] is order-bearing -- ev[] is an append-only pool indexed
// through sig[].ev_idx -- so keeping global order costs one memmove of 8-byte
// records and no event rewriting.
static void ev_insert(PtGame* g, int gi, const PLog* L, uint64_t s,
                      int turn, int seat) {
    if (g->n_sig >= PT_MAX_GAME_SIGS) return;
    memmove(&g->sig[gi + 1], &g->sig[gi],
            (size_t)(g->n_sig - gi) * sizeof(PtSig));
    g->n_sig++;
    g->sig[gi].sig = s;
    g->sig[gi].ev_idx = -1;
    g->sig[gi].info = (uint8_t)ev_info(L);
    for (int p = 0; p < 2; p++)
        if (g->ev_cursor[p] > gi) g->ev_cursor[p]++;
    ev_realize(g, gi, L, turn, seat);
}

// ---------------------------------------------------- policy logits + value
// IEEE half, round-to-nearest-even. Logits sit in [-30, 30] and the value head
// in [-1, 1], so half has ample range and ~3 decimal digits -- far more than
// the analysis needs, at half the bytes of fp32.
static uint16_t ev_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127;
    uint32_t man = x & 0x7FFFFFu;
    if (exp == 128) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));  // inf/nan
    if (exp > 15) return (uint16_t)(sign | 0x7C00u);                          // overflow
    if (exp < -24) return (uint16_t)sign;                                     // underflow
    if (exp < -14) {                                                          // subnormal
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(-14 - exp) + 13;
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1u) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (uint32_t)((exp + 15) << 10) | (man >> 13);
    if ((man & 0x1FFFu) > 0x1000u || ((man & 0x1FFFu) == 0x1000u && (h & 1u)))
        h++;
    return (uint16_t)(sign | h);
}

// One record per POLICY decision, emitted before the action is applied so the
// option set is still the one the model scored. card_id is the raw sampled
// index (PT_STOP included), value the number of options offered, head the
// position within an accumulating multi-pick. The payload entry that follows
// is: value head, then the logits of options 0..n-1, then the STOP logit.
static void ev_decision(PtGame* g, const float* dec, int a) {
    if (g_ev_state == 2 || g->opp_kind == 1) return;
    const PSelect* sel = &g->obs.select;
    int nopt = sel->has_select ? sel->n_options : 0;
    int n = nopt < PT_MAX_OPTIONS - 1 ? nopt : PT_MAX_OPTIONS - 1;

    // NOT counted in n_ev_seen: that counts ENGINE events, and is how a
    // 4096-cap truncation is detected. Decision drops have their own counter.
    // No decoder row at all (test_env, or a driver that does not export one):
    // emit nothing. A DECISION record with no payload behind it would break
    // the k-th-record-is-the-k-th-entry mapping, which is the only thing tying
    // the two together -- so record and payload are emitted as a pair or not
    // at all, and the count of what did not fit is carried in the header.
    if (!dec) { g->n_dec_dropped++; return; }
    int need = n + 2;                        // value + n logits + STOP
    if (g->n_ev >= PT_MAX_GAME_EVENTS
        || g->n_payload + need > PT_MAX_GAME_PAYLOAD) {
        g->n_dec_dropped++;
        return;
    }
    PtEvent* e = &g->ev[g->n_ev++];
    memset(e, 0, sizeof(*e));
    e->type = PT_EV_DECISION;
    e->seat = (uint8_t)(g->obs.yourIndex < 0 ? 255 : g->obs.yourIndex);
    e->turn = (uint8_t)(g->obs.turn > 255 ? 255 : (g->obs.turn < 0 ? 0 : g->obs.turn));
    e->head = (int8_t)(g->n_picked > 127 ? 127 : g->n_picked);
    e->card_id = ev_clamp16(a);
    e->attack_id = -1;
    e->value = ev_clamp16(nopt);
    e->serial = (uint16_t)g->n_sig;          // anchor, stripped by ev_write
    g->n_dec++;
    g->payload[g->n_payload++] = ev_half(dec[PT_MAX_OPTIONS]);   // value head
    for (int j = 0; j < n; j++)
        g->payload[g->n_payload++] = ev_half(dec[j]);
    g->payload[g->n_payload++] = ev_half(dec[PT_STOP]);
}

static void ev_capture(PtGame* g) {
    if (g_ev_state == 2 || g->opp_kind == 1) return;
    int turn = g->obs.turn;
    int seat = g->obs.yourIndex;
    if (seat < 0 || seat > 1) seat = 0;
    // latched once: getenv() on the stepping threads is neither cheap nor safe
    // against a concurrent setenv()
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("PTCG_EVENT_DEBUG") ? 1 : 0;
    if (dbg) {
        fprintf(stderr, "WINDOW seat=%d turn=%d n_logs=%d n_sig=%d cur=[%d,%d]\n",
                seat, turn, g->obs.n_logs, g->n_sig,
                g->ev_cursor[0], g->ev_cursor[1]);
        for (int i = 0; i < g->obs.n_logs; i++) {
            const PLog* L = &g->obs.logs[i];
            fprintf(stderr, "   t=%-3d pi=%-2d cid=%-5d ser=%-4d %d->%d val=%d\n",
                    L->type, L->playerIndex, L->cardId, L->serial,
                    L->fromArea, L->toArea, L->value);
        }
    }
    int gi = g->ev_cursor[seat];
    if (gi > g->n_sig) gi = g->n_sig;

    for (int wi = 0; wi < g->obs.n_logs; wi++) {
        const PLog* L = &g->obs.logs[wi];
        uint64_t s = ev_sig(L);

        if (gi < g->n_sig && g->sig[gi].sig == s) {
            // Same physical event, already in the stream. Keep the better
            // view; the first-seen position stays, which is what puts a move
            // at when it happened rather than when its owner next acted.
            if (ev_info(L) > (int)g->sig[gi].info) {
                g->sig[gi].info = (uint8_t)ev_info(L);
                ev_realize(g, gi, L, turn, seat);
            }
            gi++;
            continue;
        }
        if (gi >= g->n_sig) { ev_push(g, L, s, turn, seat); gi = g->n_sig; continue; }

        // Out of step. Either the stream holds events this seat is not told
        // about (skip them), or this seat is reporting something new (splice
        // it in). A plain suffix match desynchronises permanently the first
        // time either happens, which is why this is a two-pointer diff.
        int gskip = -1, wskip = -1;
        for (int k = 1; k < PT_EV_LOOKAHEAD; k++) {
            if (wskip < 0 && gi + k < g->n_sig && g->sig[gi + k].sig == s)
                wskip = k;
            if (gskip < 0 && wi + k < g->obs.n_logs
                && ev_sig(&g->obs.logs[wi + k]) == g->sig[gi].sig)
                gskip = k;
            if (wskip >= 0 && gskip >= 0) break;
        }
        if (wskip >= 0 && (gskip < 0 || wskip <= gskip)) {
            gi += wskip;
            wi--;                    // re-test this log against the new gi
            continue;
        }
        ev_insert(g, gi, L, s, turn, seat);
        gi++;
    }
    g->ev_cursor[seat] = gi;
}

// ------------------------------------------------------------------ rng
static inline uint32_t pcg32(uint64_t* s) {
    uint64_t old = *s;
    *s = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}

static inline uint32_t rng_below(uint64_t* s, uint32_t n) {   // [0, n)
    return n ? (uint32_t)(((uint64_t)pcg32(s) * n) >> 32) : 0;
}

static inline double rng_double(uint64_t* s) {
    return pcg32(s) * (1.0 / 4294967296.0);
}

// ------------------------------------------------ Solar Transfer stall cap
// Mega Venusaur ex (card 652), "Solar Transfer": "as often as you like
// during your turn" + a reversible effect + no opponent priority = a legal
// infinite loop (docs/training.md). Cap ACTIVATIONS at 10 per seat per
// turn: over-budget activation options are cleared from the policy mask
// (encode_row) and from the random opponent's draw pool (random_answer).
// Deliberately scoped to the one card — no live-pool deck carries any
// "as often as you like" ability, so this is inert for every current deck.
// The sub-selects an activation opens (move source/target) are other option
// types and are never counted or masked, so an in-flight resolution cannot
// be starved. forced_answer stays unfiltered: forced means no alternative.
#define PT_AB652_CARD 652
#define PT_AB652_CAP  10

// Is this option an activation of card 652's ability? Mirrors the encoder's
// resolve_card fallback for the board slot the option points at (the engine
// sometimes omits cardId on ability options).
static int ab652_option(const PObs* obs, const POption* o) {
    if (o->type != OT_ABILITY && o->type != OT_SKILL) return 0;
    int cid = o->cardId == PT_NONE ? 0 : o->cardId;
    if (!cid) {
        int p = o->playerIndex == PT_NONE ? obs->yourIndex : o->playerIndex;
        int idx = o->index == PT_NONE ? -1 : o->index;
        if (p < 0 || p > 1 || idx < 0) return 0;
        const PPlayer* pl = &obs->players[p];
        if (o->area == AREA_ACTIVE)
            cid = (idx < pl->n_active && pl->active[idx].present)
                      ? pl->active[idx].id : 0;
        else if (o->area == AREA_BENCH && idx < pl->n_bench)
            cid = pl->bench[idx].id;
    }
    return cid == PT_AB652_CARD;
}

static void ab652_sync(PtGame* g) {
    if (g->obs.turn != g->ab652_turn) {
        g->ab652_turn = g->obs.turn;
        g->ab652_used[0] = g->ab652_used[1] = 0;
    }
}

// -------------------------------------------------------------- game core
static const char* live_viz(void* ctx) { return pt_battle_visualize(ctx); }

static void ingest(PtGame* g) {
    if (!g->obs.has_current) return;
    int seat = g->obs.yourIndex;
    tracker_update(&g->trackers[seat], &g->obs);
    oracle_on_seat_obs(&g->oracle, &g->obs, live_viz, g->battle);
    // Snapshot the prizes the first time the oracle has them: it holds the
    // prizes REMAINING and decrements them as they are taken, so by ev_write
    // the winner's are gone.
    if (g_ev_state != 2 && g->ev_pz_n == 0 && g->oracle.frozen) {
        for (int s = 0; s < 2 && g->ev_pz_n < 16; s++) {
            const uint16_t* pc = g->oracle.prize[s];
            for (int cid = 0; cid < PT_N_CARDS && g->ev_pz_n < 16; cid++) {
                if (!pc[cid]) continue;
                int k = g->ev_pz_n++;
                g->ev_pz_seat[k] = (uint8_t)s;
                g->ev_pz_cid[k] = ev_clamp16(cid);
                g->ev_pz_cnt[k] = (uint8_t)(pc[cid] > 255 ? 255 : pc[cid]);
            }
        }
    }
}

static void prizes_p0(const PtGame* g, int* p0, int* p1) {
    *p0 = g->obs.players[0].n_prize;
    *p1 = g->obs.players[1].n_prize;
}

// one engine transition; accrues prize rewards in p0 currency.
// Returns the engine Select error (0 = ok; obs unchanged on error).
static int game_submit(Env* env, PtGame* g, const int* ans, int n) {
    int p0 = 0, p1 = 0;
    // Count Solar Transfer activations against the PRE-submit obs (the turn
    // the action belongs to), apply only after the engine accepts — a
    // rejected select mutates nothing and must not count.
    ab652_sync(g);
    int ab_inc = 0, ab_seat = g->obs.yourIndex;
    for (int k = 0; k < n; k++)
        if (ans[k] >= 0 && ans[k] < g->obs.select.n_options
            && ab652_option(&g->obs, &g->obs.select.option[ans[k]]))
            ab_inc++;
    ev_action(g, ans, n);        // pre-submit: obs still holds the option set
    if (env->prize_rewards) prizes_p0(g, &p0, &p1);
    int err = pt_battle_select(g->battle, ans, n);
    if (err != 0) return err;
    const char* json = pt_battle_data(g->battle);
    if (pt_obs_parse(json, &g->obs) != 0) return -100;
    if (ab_inc > 0 && (ab_seat == 0 || ab_seat == 1))
        g->ab652_used[ab_seat] += ab_inc;
    ev_capture(g);
    g->engine_steps++;
    if (env->prize_rewards) {
        int q0, q1;
        prizes_p0(g, &q0, &q1);
        g->pending_p0 += (double)((p0 - q0) - (p1 - q1));
    }
    g->n_picked = 0;
    ingest(g);
    return 0;
}

// Draw one seat's deck: with probability p_anchor pick uniformly among the
// anchor decks, otherwise uniformly over the whole pool. p_anchor 0 (or no
// anchors configured) reproduces the plain uniform draw exactly.
// Deck ids: [0, n_decks) is the training pool, and the coverage pool lives at
// [PT_DECK_ALT_BASE, PT_DECK_ALT_BASE + n_decks_alt) -- a FIXED base, not
// n_decks, so appending a training deck mid-run cannot renumber coverage.
// Every seat draws from the training pool; the RANDOM opponent's seat draws
// from both, so the learner meets lists the training pool does not contain.
// Coverage ids never reach the deck matrix: they are above its width, and
// random-opponent games are excluded from it anyway (see finish_episode).
static const int32_t* deck_row(const PtTables* T, int idx) {
    if (T->n_decks_alt > 0 && idx >= PT_DECK_ALT_BASE)
        return T->decks_alt + (size_t)(idx - PT_DECK_ALT_BASE) * 60;
    return T->decks + (size_t)idx * 60;
}

// The two pools are contiguous to draw from but not to address: slot s in
// [0, n_decks + n_decks_alt) maps to a training id, or to a coverage id
// above the base. Keeping the draw dense means the sampling distribution is
// unchanged by the renumbering.
static inline int pool_slot_to_id(const PtTables* T, int slot) {
    return slot < T->n_decks ? slot
                             : PT_DECK_ALT_BASE + (slot - T->n_decks);
}

// Weighted draw over the slot space [0, n_pool), by binary search on the
// cumulative weights. Called once per EPISODE, not per step, so ~10
// comparisons at n=1024 is free and the readable form wins over an alias
// table.
//
// A zero-weight deck is UNREACHABLE, not merely unlikely: its cumulative entry
// equals its predecessor's, so `cw[mid] > u` can never first become true at
// that index. This relies on rng_double() being strictly < 1, which makes
// u < total strictly, so some index always qualifies.
static inline int weighted_slot(PtGame* g, const double* cw, int n_pool) {
    double u = rng_double(&g->rng) * cw[n_pool - 1];
    int lo = 0, hi = n_pool - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cw[mid] > u) hi = mid; else lo = mid + 1;
    }
    return lo;
}

// `n_pool` bounds the draw: n_decks for a seat that must play a training
// deck, n_decks + n_decks_alt for the random opponent, which may also draw
// coverage decks. Anchors stay available either way -- they are training
// ids, valid under both bounds.
static int draw_deck(PtGame* g, const PtTables* T, const Env* env,
                     float p_anchor, int n_pool) {
    if (env->n_anchors > 0 && p_anchor > 0.0f
        && rng_double(&g->rng) < (double)p_anchor) {
        int a = (int)rng_below(&g->rng, (uint32_t)env->n_anchors);
        int idx = env->anchor_idx[a];
        if (idx >= 0 && idx < T->n_decks) return idx;   // ignore a bad index
    }
    // n_pool counts SLOTS across both pools; map the dense slot to the sparse
    // id space (training below the base, coverage above it).
    //
    // deck_cw NULL is the unweighted uniform draw, kept as an exact fast path:
    // a run that never sets weights is bit-for-bit unaffected by it. The
    // cumulative table spans training slots then coverage slots, so the same
    // array serves both values of n_pool.
    if (T->deck_cw && n_pool > 0)
        return pool_slot_to_id(T, weighted_slot(g, T->deck_cw, n_pool));
    return pool_slot_to_id(T, (int)rng_below(&g->rng, (uint32_t)n_pool));
}

static void game_start(Env* env) {
    PtGame* g = &env->game;
    const PtTables* T = pt_tables();
    if (g->battle) { pt_battle_finish(g->battle); g->battle = NULL; }
    // Tagged rows always play their frozen bank; untagged rows draw
    // mirror below mix_self, else random opponent.
    if (env->tag > 0) g->opp_kind = 2;
    else g->opp_kind = rng_double(&g->rng) < env->mix_self ? 0 : 1;
    g->learner_seat = (int)rng_below(&g->rng, 2);
    // Per-seat deck draw. The opposing seat oversamples anchors at a different
    // rate depending on who is playing it: another copy of the learner (self)
    // vs a frozen league checkpoint.
    float p_opp = (g->opp_kind == 2) ? env->p_anchor_league_opp
                                     : env->p_anchor_self_opp;
    // ONE critical section over the draw AND the copy. Splitting them would
    // look correct and still be wrong: draw_deck reads n_decks for its bound,
    // deck_row returns a pointer INTO the table, and the memcpy dereferences
    // it -- a swap landing between lookup and copy reads freed memory.
    // Uncontended in normal operation (the swap fires at the rollout/train
    // boundary with every env thread parked), and taken once per EPISODE, not
    // per step, so the cost is nil. It is here because episodes do not align
    // with rollout phases: game_start fires at arbitrary moments on arbitrary
    // env threads, so there is no episode boundary to synchronise on instead.
    pt_decks_rdlock();
    int n_opp_pool = (g->opp_kind == 1) ? T->n_decks + T->n_decks_alt
                                        : T->n_decks;
    int iL = draw_deck(g, T, env, env->p_anchor_learner, T->n_decks);
    int iO = draw_deck(g, T, env, p_opp, n_opp_pool);
    int i0 = g->learner_seat == 0 ? iL : iO;
    int i1 = g->learner_seat == 0 ? iO : iL;
    g->deck_idx[0] = i0; g->deck_idx[1] = i1;
    memcpy(g->deck_pair[0], deck_row(T, i0), 60 * sizeof(int32_t));
    memcpy(g->deck_pair[1], deck_row(T, i1), 60 * sizeof(int32_t));
    pt_decks_unlock();
    g->battle = pt_battle_start(g->deck_pair[0], g->deck_pair[1]);
    if (!g->battle) {
        fprintf(stderr, "ptcg env: BattleStart failed (decks %d, %d)\n", i0, i1);
        env->log.env_errors += 1;
        // fall through with obs zeroed; c_step's guard restarts
        memset(&g->obs, 0, sizeof(g->obs));
        return;
    }
    oracle_init(&g->oracle, g->deck_pair[0], g->deck_pair[1]);
    tracker_init(&g->trackers[0], g->deck_pair[0]);
    tracker_init(&g->trackers[1], g->deck_pair[1]);
    const char* json = pt_battle_data(g->battle);
    pt_obs_parse(json, &g->obs);
    // BEFORE capture: setup events belong to this game. The merge cursors reset
    // with them or the next game aligns against the previous game's stream.
    g->n_ev = g->n_ev_seen = 0;
    g->n_sig = 0;
    g->ev_cursor[0] = g->ev_cursor[1] = 0;
    g->n_payload = g->n_dec = g->n_dec_dropped = 0;
    g->ev_pz_n = 0;
    ev_capture(g);
    ingest(g);
    g->n_picked = 0;
    g->cum_p0 = 0.0;
    g->pending_p0 = 0.0;
    g->decisions = 0;
    g->engine_steps = 0;
    g->forced_run[0] = g->forced_run[1] = 0;
    // Explicit reset: a new game can open on the same turn NUMBER the last
    // one ended on, which ab652_sync alone would read as "same turn".
    g->ab652_turn = -1;
    g->ab652_used[0] = g->ab652_used[1] = 0;
}

// forced_answer: NULL if not forced; else fills ans, returns count >= 0
static int forced_answer(const PSelect* sel, int* ans) {
    int n = sel->n_options;
    if (sel->minCount == sel->maxCount && sel->minCount == n) {
        for (int i = 0; i < n; i++) ans[i] = i;
        return n;
    }
    if (n == 1 && sel->minCount >= 1) { ans[0] = 0; return 1; }
    return -1;
}

static int random_answer(PtGame* g, const PSelect* sel, int* ans) {
    int n = sel->n_options;
    // Solar Transfer over budget: drop its activation options from the draw
    // pool (the random-opponent mirror of the encode_row mask filter). When
    // nothing is dropped the pool is the identity and the rng call sequence
    // below is unchanged, so recorded games replay bit-exact. Fails open
    // if the filtered pool could not satisfy minCount.
    int pool[PT_MAX_OPTS], np = 0;
    ab652_sync(g);
    int seat = g->obs.yourIndex;
    int over = (seat == 0 || seat == 1) && g->ab652_used[seat] >= PT_AB652_CAP;
    for (int i = 0; i < n; i++)
        if (!(over && ab652_option(&g->obs, &sel->option[i]))) pool[np++] = i;
    if (np < 1 || np < sel->minCount) {
        np = n;
        for (int i = 0; i < n; i++) pool[i] = i;
    }
    int hi = sel->maxCount < np ? sel->maxCount : np;
    int lo = sel->minCount < hi ? sel->minCount : hi;
    int k = lo + (hi > lo ? (int)rng_below(&g->rng, (uint32_t)(hi - lo + 1)) : 0);
    for (int i = 0; i < k; i++) {          // partial Fisher-Yates
        int j = i + (int)rng_below(&g->rng, (uint32_t)(np - i));
        int t = pool[i]; pool[i] = pool[j]; pool[j] = t;
        ans[i] = pool[i];
    }
    // python returns sorted(sample)
    for (int i = 1; i < k; i++) {
        int v = ans[i], j = i - 1;
        while (j >= 0 && ans[j] > v) { ans[j + 1] = ans[j]; j--; }
        ans[j + 1] = v;
    }
    return k;
}

// Negamax reward contract: the reward observed at
// row t+1 must be in the currency of the actor who ACTED at t. Mirror and
// league games surface both seats as rows, so the true actor seat signs the
// reward. Random games never surface opponent rows — every row is the
// learner's, so their rewards stay in learner currency (forcing the actor
// there is what keeps opponent moves folded into the env dynamics).
static double perspective(const PtGame* g, int actor) {
    if (g->opp_kind == 1) actor = g->learner_seat;
    return actor == 0 ? 1.0 : -1.0;
}

// _end_cause: 0 prize, 1 bench, 2 deck, 3 other
static int end_cause(const PObs* obs, int winner) {
    const PPlayer* loser = &obs->players[1 - winner];
    if (obs->players[winner].n_prize == 0) return 0;
    bool any_active = loser->n_active > 0 && loser->active[0].present;
    if (!any_active && loser->n_bench == 0) return 1;
    if (loser->deckCount == 0) return 2;
    return 3;
}

// result: 0/1 winner seat, 2 draw, -2 truncation. Sets g_done flags.
static void finish_episode(Env* env, PtGame* g, int result,
                           bool* done_terminal, bool* done_trunc) {
    Log* st = &env->log;
    double margin_p0 = g->cum_p0 + g->pending_p0;
    if (result == -2) {
        *done_trunc = true;                 // NO terminal flag (see header)
    } else {
        *done_terminal = true;
        env->terminals[0] = 1.0f;
        if (result == 0 || result == 1) {
            double bonus_w = env->win_r - (result == 0 ? margin_p0 : -margin_p0);
            g->pending_p0 += result == 0 ? bonus_w : -bonus_w;
        } else {                            // draw: totals zero out
            g->pending_p0 += -margin_p0;
        }
        double tot = g->cum_p0 + g->pending_p0;
        double at = tot < 0 ? -tot : tot;
        st->ep_ret_abs += (float)at;
        if ((at > env->win_r + 1e-3 || at < env->win_r - 1e-3) && at > 1e-3)
            st->ep_ret_bad += 1;
    }
    st->n += 1;
    st->episode_return += (float)(g->cum_p0 + g->pending_p0);
    if (g->opp_kind == 0) st->eps_mirror += 1;
    else if (g->opp_kind == 2) {
        st->eps_league += 1;
        int b = env->tag - 1;
        if (b >= 0 && b < PT_MAX_BANKS) st->league_n_b[b] += 1;
    } else st->eps_rand += 1;
    if (result == 0 || result == 1) {
        bool won = result == g->learner_seat;
        int cause = end_cause(&g->obs, result);
        float* cause_row = NULL;
        if (g->opp_kind == 0) cause_row = &st->end_prize;
        else if (g->opp_kind == 2) cause_row = won ? &st->lgw_prize
                                                   : &st->lgl_prize;
        else cause_row = won ? &st->wend_prize : &st->lend_prize;
        cause_row[cause] += 1;              // prize/bench/deck/other adjacency
        if (g->opp_kind == 1) {
            st->wins += won;
            st->w_rand += won;
        } else if (g->opp_kind == 2) {
            st->w_league += won;
            int b = env->tag - 1;
            if (b >= 0 && b < PT_MAX_BANKS) st->league_w_b[b] += won;
        }
        st->first_wins += result == g->obs.firstPlayer;
        st->decided += 1;
        int q0 = g->obs.players[0].n_prize, q1 = g->obs.players[1].n_prize;
        st->prize_margin_dec += (float)(result == 0 ? q1 - q0 : q0 - q1);
        // Per-deck training win rate, for the decks named in --anchor-decks.
        // Credit the LEARNER seat's deck: in mirror/league games the learner
        // seat is drawn uniformly, so this is an unbiased sample of "how does
        // this deck do in the games we actually train on". Sampling is NOT
        // changed by this -- anchor_idx here is only a label.
        int ld = g->deck_idx[g->learner_seat];
        for (int i = 0; i < env->n_anchors; i++) {
            if (env->anchor_idx[i] == ld) {
                st->deck_games[i] += 1;
                if (result == g->learner_seat) st->deck_wins[i] += 1;
                break;
            }
        }
        // Full pairwise matrix over the TRAINING pool (offline analysis).
        // No-op unless PTCG_DECK_MATRIX is set.
        // Games against the RANDOM opponent are EXCLUDED: that seat is not a
        // policy, so the result measures the random agent rather than the
        // matchup. It is also the only seat that can hold a coverage deck,
        // which is why no coverage id ever reaches the matrix and the matrix
        // stays n_decks square.
        if (g->opp_kind != 1)
            dm_record(pt_tables()->n_decks, ld,
                      g->deck_idx[1 - g->learner_seat],
                      result == g->learner_seat);
    } else if (result == 2) {
        st->draws += 1;
    }
    // Event log takes EVERY outcome, decided or not (see the header note).
    if (g->opp_kind != 1) ev_write(g, result);
    {
        int q0 = g->obs.players[0].n_prize, q1 = g->obs.players[1].n_prize;
        int d = q1 - q0;
        st->prize_margin += (float)(d < 0 ? -d : d);
    }
    st->ep_len += g->decisions;
    if (g->opp_kind == 2) st->league_len += g->decisions;
    int t = g->obs.turn;
    st->turns_first += (t + 1) / 2;
    st->turns_second += t / 2;
    st->engine_steps += g->engine_steps;
}

static void encode_row(Env* env, PtGame* g) {
    const PSelect* sel = &g->obs.select;
    int seat = g->obs.yourIndex;
    bool stop_allowed = g->n_picked >= sel->minCount
                        && (sel->maxCount > 1 || sel->minCount == 0);
    pt_encode(env->observations, &g->obs, g->picked, g->n_picked, stop_allowed,
              g->forced_run[seat], &g->trackers[seat]);
    pt_write_oracle(env->observations, &g->oracle, &g->obs);
    // Solar Transfer over budget: clear its activation options from the ROW
    // mask before pt_write_mask, so the byte mask and the model's mask input
    // agree. Never empties the mask — the selects that offer an ability
    // always offer END (or STOP), and the keep>0 guard fails open on any
    // select where that ever stopped holding.
    ab652_sync(g);
    if ((seat == 0 || seat == 1) && g->ab652_used[seat] >= PT_AB652_CAP) {
        float* mrow = env->observations + PT_OFF_OPT_MASK;
        int n = sel->n_options < PT_MAX_OPTIONS - 1 ? sel->n_options
                                                    : PT_MAX_OPTIONS - 1;
        int drop[PT_MAX_OPTIONS], nd = 0, keep = 0;
        for (int j = 0; j < PT_MAX_OPTIONS; j++) {
            if (mrow[j] == 0.0f) continue;
            if (j < n && ab652_option(&g->obs, &sel->option[j])) drop[nd++] = j;
            else keep++;
        }
        if (nd > 0 && keep > 0)
            for (int i = 0; i < nd; i++) mrow[drop[i]] = 0.0f;
    }
    pt_write_mask(env->action_mask, env->observations);
    // League opponent decision: stamp the bank id (idx+1 convention, 0 =
    // learner) so the trainer can route it to frozen weights and mask it
    // out of the policy loss. pt_encode memsets the row, so learner rows
    // are guaranteed 0 here. The trainer strips this column before any
    // network forward — every checkpoint trained with it at 0.
    if (g->opp_kind == 2 && seat != g->learner_seat)
        env->observations[PT_FROZEN_COL] = (float)env->tag;
    g->forced_run[seat] = 0;
    g->decisions++;
}

// run engine to the next policy decision (encode row) or episode end.
// Returns 0 ok; -1 env error (caller restarts the game).
static int advance(Env* env, PtGame* g, bool* done_terminal, bool* done_trunc) {
    int ans[PT_MAX_OPTS];
    while (1) {
        if (!g->obs.has_current) return -1;
        if (g->obs.result != -1) {
            finish_episode(env, g, g->obs.result, done_terminal, done_trunc);
            return 0;
        }
        if (g->engine_steps >= env->max_engine_steps) {
            finish_episode(env, g, -2, done_terminal, done_trunc);
            return 0;
        }
        const PSelect* sel = &g->obs.select;
        if (!sel->has_select) return -1;
        int seat = g->obs.yourIndex;
        // Random opponents only: answer without surfacing a row. League
        // (opp_kind 2) opponent decisions fall through to encode_row so the
        // frozen network can act on them.
        if (g->opp_kind == 1 && seat != g->learner_seat) {
            int k = forced_answer(sel, ans);
            if (k >= 0) {
                if (game_submit(env, g, ans, k) != 0) return -1;
                g->forced_run[seat]++;
                continue;
            }
            k = random_answer(g, sel, ans);
            int err = game_submit(env, g, ans, k);
            if (err != 0) {
                if (k != 0) return -1;
                // engine rejected an advertised-legal [] — retry option 0
                ans[0] = 0;
                if (game_submit(env, g, ans, 1) != 0) return -1;
            }
            continue;
        }
        int k = forced_answer(sel, ans);
        if (k >= 0) {
            if (game_submit(env, g, ans, k) != 0) return -1;
            g->forced_run[seat]++;
            continue;
        }
        encode_row(env, g);
        return 0;
    }
}

static void restart_after_error(Env* env) {
    env->log.env_errors += 1;
    bool dt = false, dr = false;
    game_start(env);
    if (env->game.battle && advance(env, &env->game, &dt, &dr) == 0) return;
    // pathological: keep trying fresh games (bounded)
    for (int tries = 0; tries < 8; tries++) {
        dt = dr = false;
        game_start(env);
        if (env->game.battle && advance(env, &env->game, &dt, &dr) == 0) return;
        env->log.env_errors += 1;
    }
    fprintf(stderr, "ptcg env: unrecoverable env error\n");
}

// advance to next decision, credit accrued reward to this step (actor's
// perspective), reset on episode end. Mirrors PTCGEnv._flush.
static void flush_step(Env* env, PtGame* g, int actor) {
    bool done_terminal = false, done_trunc = false;
    if (advance(env, g, &done_terminal, &done_trunc) != 0) {
        restart_after_error(env);
        return;
    }
    env->rewards[0] += (float)(perspective(g, actor) * g->pending_p0);
    g->cum_p0 += g->pending_p0;
    g->pending_p0 = 0.0;
    if (done_terminal || done_trunc) {
        bool dt = false, dr = false;
        game_start(env);
        if (!env->game.battle
            || advance(env, &env->game, &dt, &dr) != 0) {
            restart_after_error(env);
        }
        // fresh episode never ends pre-decision
    }
}

static int unpicked(const PtGame* g, int n) {
    for (int j = 0; j < n; j++) {
        bool in = false;
        for (int i = 0; i < g->n_picked; i++)
            if (g->picked[i] == j) { in = true; break; }
        if (!in) return j;
    }
    return -1;
}

// one policy action (mirrors PTCGEnv._step_game)
static void step_game(Env* env, PtGame* g, int a) {
    const PSelect* sel = &g->obs.select;
    int actor = g->obs.yourIndex;
    int n = sel->n_options;
    if (a == PT_STOP) {
        bool stop_ok = g->n_picked >= sel->minCount
                       && (sel->maxCount > 1 || sel->minCount == 0);
        if (stop_ok) {
            int err = game_submit(env, g, g->picked, g->n_picked);
            if (err == 0) { flush_step(env, g, actor); return; }
            // engine rejected an advertised-legal submit — pick instead
            env->log.illegal += 1;
            a = unpicked(g, n);
            if (a < 0) { restart_after_error(env); return; }
        } else {
            env->log.illegal += 1;
            a = unpicked(g, n);
        }
    } else if (a >= n || a < 0) {
        env->log.illegal += 1;
        a = unpicked(g, n);
    } else {
        for (int i = 0; i < g->n_picked; i++)
            if (g->picked[i] == a) {
                env->log.illegal += 1;
                a = unpicked(g, n);
                break;
            }
    }
    if (a < 0) {
        // engine offered fewer distinct picks than maxCount promised
        if (game_submit(env, g, g->picked, g->n_picked) != 0) {
            restart_after_error(env);
            return;
        }
        flush_step(env, g, actor);
        return;
    }
    if (sel->maxCount == 1) {
        int one[1] = {a};
        if (game_submit(env, g, one, 1) != 0) { restart_after_error(env); return; }
        flush_step(env, g, actor);
    } else {
        g->picked[g->n_picked++] = a;
        if (g->n_picked >= sel->maxCount) {
            if (game_submit(env, g, g->picked, g->n_picked) != 0) {
                restart_after_error(env);
                return;
            }
            flush_step(env, g, actor);
        } else {
            encode_row(env, g);          // same select, next pick
        }
    }
}

// ------------------------------------------------------------ env lifecycle
void ptcg_env_init(Env* env, uint64_t seed) {
    memset(&env->game, 0, sizeof(env->game));
    env->game.rng = seed * 100003ULL + 0x9E3779B97F4A7C15ULL;
    env->seed = seed;
    memset(&env->log, 0, sizeof(env->log));
}

void c_reset(Env* env) {
    pt_engine_init();
    bool dt = false, dr = false;
    game_start(env);
    if (!env->game.battle || advance(env, &env->game, &dt, &dr) != 0)
        restart_after_error(env);
}

void c_step(Env* env) {
    int a = (int)env->actions[0];
    // Before step_game: obs still holds the option set the decoder scored.
    ev_decision(&env->game, env->dec, a);
    step_game(env, &env->game, a);
}

void c_close(Env* env) {
    if (env->game.battle) { pt_battle_finish(env->game.battle); env->game.battle = NULL; }
}

void c_render(Env* env) { (void)env; }
