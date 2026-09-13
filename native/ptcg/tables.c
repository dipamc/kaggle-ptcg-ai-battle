// Static-tables blob loader + serial maps + engine wrapper.
//
// pthread_rwlock_* is a POSIX extension, and glibc hides it under -std=c11
// (which defines __STRICT_ANSI__). build_native.sh uses -std=gnu11 so the
// CUDA/training build never saw this, but the Makefile's CPU targets --
// test, parity, reload, decklock -- would not compile on Linux at all.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <math.h>
#include "ptcg_env.h"

// Defined with the deck-weight code below; pt_reload_decks needs it to keep a
// weighted distribution consistent when the pool grows under it.
static int rebuild_deck_cw(void);

// ------------------------------------------------------------------ blob
typedef struct { char name[24]; int dtype, ndim; int64_t shape[4]; void* data; } Section;

static PtTables g_tables;
static Section g_sections[32];
static int g_nsections;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

// Guards the {decks, n_decks} pair against a mid-run append.
//
// The swap is triggered at the rollout/train boundary, where the env threads
// are parked at the fork-join barrier -- so in normal operation this lock is
// uncontended and costs nothing. It is here so that safety does not DEPEND on
// that parking: episodes do not align with rollout phases (a rollout collects
// a fixed horizon and stops mid-episode), so game_start fires at arbitrary
// moments on arbitrary env threads, and there is no global episode boundary to
// synchronise on. If anything ever overlaps rollout with training, this is
// what turns a silent use-after-free into a brief wait.
//
// Readers must hold it across the deck DRAW and the COPY together: deck_row()
// returns a pointer INTO the table, so releasing between lookup and memcpy
// would still let the array be freed underneath.
static pthread_rwlock_t g_decks_lock = PTHREAD_RWLOCK_INITIALIZER;

void pt_decks_rdlock(void) { pthread_rwlock_rdlock(&g_decks_lock); }
void pt_decks_unlock(void) { pthread_rwlock_unlock(&g_decks_lock); }
void pt_decks_wrlock(void) { pthread_rwlock_wrlock(&g_decks_lock); }

static void* sec(const char* name, int64_t expect_elems, int elem_size) {
    for (int i = 0; i < g_nsections; i++) {
        if (strcmp(g_sections[i].name, name) == 0) {
            int64_t n = 1;
            for (int d = 0; d < g_sections[i].ndim; d++) n *= g_sections[i].shape[d];
            if (expect_elems >= 0 && n != expect_elems) {
                fprintf(stderr, "ptcg tables: %s has %lld elems, expected %lld\n",
                        name, (long long)n, (long long)expect_elems);
                abort();
            }
            (void)elem_size;
            return g_sections[i].data;
        }
    }
    fprintf(stderr, "ptcg tables: missing section %s\n", name);
    abort();
}

static int64_t sec_dim0(const char* name) {
    for (int i = 0; i < g_nsections; i++)
        if (strcmp(g_sections[i].name, name) == 0) return g_sections[i].shape[0];
    return 0;
}

static void load_blob(void) {
    const char* path = getenv("PTCG_TABLES");
    if (!path) path = "native/ptcg_tables.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ptcg tables: cannot open %s (set PTCG_TABLES)\n", path); abort(); }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PTCGTAB1", 8)) {
        fprintf(stderr, "ptcg tables: bad magic in %s\n", path); abort();
    }
    int32_t hdr[15];
    if (fread(hdr, 4, 15, f) != 15) abort();
    // layout constants must match compile-time mirror of buffers.py
    assert(hdr[0] == PT_OBS_SIZE && "OBS_SIZE mismatch vs python buffers.py");
    assert(hdr[1] == PT_MAX_TOKENS && hdr[2] == PT_MAX_OPTIONS);
    assert(hdr[3] == PT_TOK_INT && hdr[4] == PT_TOK_F);
    assert(hdr[5] == PT_OPT_INT && hdr[6] == PT_OPT_F);
    assert(hdr[7] == PT_GLOBAL_F && hdr[8] == PT_DEC_INT && hdr[9] == PT_DEC_F);
    assert(hdr[10] == PT_ORACLE_F);
    assert(hdr[11] == PT_N_CARDS && hdr[12] == PT_N_ATTACKS && hdr[13] == PT_N_SKILLS);
    g_nsections = hdr[14];
    assert(g_nsections <= 32);
    static const int esize[4] = {4, 4, 1, 1};
    for (int i = 0; i < g_nsections; i++) {
        Section* s = &g_sections[i];
        if (fread(s->name, 1, 24, f) != 24) abort();
        int32_t meta[2];
        if (fread(meta, 4, 2, f) != 2) abort();
        s->dtype = meta[0]; s->ndim = meta[1];
        if (fread(s->shape, 8, 4, f) != 4) abort();
        int64_t n = 1;
        for (int d = 0; d < s->ndim; d++) n *= s->shape[d];
        int64_t bytes = n * esize[s->dtype];
        s->data = malloc(bytes > 0 ? bytes : 1);
        if (bytes && fread(s->data, 1, bytes, f) != (size_t)bytes) abort();
    }
    fclose(f);

    PtTables* t = &g_tables;
    t->atk_damage = sec("atk_damage", PT_N_ATTACKS, 4);
    t->atk_cost = sec("atk_cost", PT_N_ATTACKS * 12, 1);
    t->energy_type = sec("energy_type", PT_N_CARDS, 1);
    t->weakness = sec("weakness", PT_N_CARDS, 1);
    t->resistance = sec("resistance", PT_N_CARDS, 1);
    t->retreat = sec("retreat", PT_N_CARDS, 1);
    t->top_attacks = sec("top_attacks", PT_N_CARDS * 4, 4);
    t->supporters = sec("supporters", PT_N_CARDS, 1);
    t->shields = sec("shields", PT_N_CARDS, 1);
    t->pokemon_ids = sec("pokemon_ids", PT_N_CARDS, 1);
    t->n_attack_fx = (int)sec_dim0("attack_fx");
    t->n_play_fx = (int)sec_dim0("play_fx");
    int32_t* afx = sec("attack_fx", (int64_t)t->n_attack_fx * FXREC_I, 4);
    float* amag = sec("attack_fx_mag", t->n_attack_fx, 4);
    int32_t* pfx = sec("play_fx", (int64_t)t->n_play_fx * FXREC_I, 4);
    float* pmag = sec("play_fx_mag", t->n_play_fx, 4);
    t->attack_fx = calloc(t->n_attack_fx ? t->n_attack_fx : 1, sizeof(FxRec));
    t->play_fx = calloc(t->n_play_fx ? t->n_play_fx : 1, sizeof(FxRec));
    for (int i = 0; i < t->n_attack_fx; i++) {
        int32_t* r = afx + i * FXREC_I;
        t->attack_fx[i] = (FxRec){r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], amag[i]};
    }
    for (int i = 0; i < t->n_play_fx; i++) {
        int32_t* r = pfx + i * FXREC_I;
        t->play_fx[i] = (FxRec){r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], pmag[i]};
    }
    // key -> [head, cnt] index (records sorted by key in the exporter)
    t->attack_fx_head = malloc(PT_N_ATTACKS * 4);
    t->attack_fx_cnt = calloc(PT_N_ATTACKS, 4);
    t->play_fx_head = malloc(PT_N_CARDS * 4);
    t->play_fx_cnt = calloc(PT_N_CARDS, 4);
    for (int i = 0; i < PT_N_ATTACKS; i++) t->attack_fx_head[i] = -1;
    for (int i = 0; i < PT_N_CARDS; i++) t->play_fx_head[i] = -1;
    for (int i = 0; i < t->n_attack_fx; i++) {
        int k = t->attack_fx[i].key;
        if (k >= 0 && k < PT_N_ATTACKS) {
            if (t->attack_fx_head[k] < 0) t->attack_fx_head[k] = i;
            t->attack_fx_cnt[k]++;
        }
    }
    for (int i = 0; i < t->n_play_fx; i++) {
        int k = t->play_fx[i].key;
        if (k >= 0 && k < PT_N_CARDS) {
            if (t->play_fx_head[k] < 0) t->play_fx_head[k] = i;
            t->play_fx_cnt[k]++;
        }
    }
    t->n_decks = (int)sec_dim0("decks");
    // Training ids must stay below the coverage base or the two id spaces
    // collide and deck_row() starts handing out coverage rows for training ids.
    if (t->n_decks > PT_DECK_ALT_BASE) {
        fprintf(stderr, "ptcg tables: %d training decks exceeds PT_DECK_ALT_BASE "
                "(%d); raise the base and renumber the coverage pool\n",
                t->n_decks, PT_DECK_ALT_BASE);
        abort();
    }
    t->decks = sec("decks", (int64_t)t->n_decks * 60, 4);
    // optional: sec_dim0 returns 0 for a missing section, so a blob built
    // without --alt-pool simply has no coverage decks and draw_deck falls
    // back to the training pool for every seat
    t->n_decks_alt = (int)sec_dim0("decks_alt");
    t->decks_alt = t->n_decks_alt
        ? sec("decks_alt", (int64_t)t->n_decks_alt * 60, 4) : NULL;
    t->card_static = sec("card_static", (int64_t)PT_N_CARDS * 58, 4);
    t->card_attacks = sec("card_attacks", (int64_t)PT_N_CARDS * 4, 4);
    t->card_skills = sec("card_skills", (int64_t)PT_N_CARDS * 2, 4);
    t->att_static = sec("att_static", (int64_t)PT_N_ATTACKS * 15, 4);
    t->card_text = sec("card_text", (int64_t)PT_N_CARDS * 128, 4);
    t->att_text = sec("att_text", (int64_t)PT_N_ATTACKS * 64, 4);
    t->skill_text = sec("skill_text", (int64_t)PT_N_SKILLS * 64, 4);
}

const PtTables* pt_tables(void) {
    pthread_once(&g_once, load_blob);
    return &g_tables;
}

// ------------------------------------------------------- mid-run deck append
// Read ONE named i32 section out of a blob into a fresh buffer. Returns the
// row count (shape[0]) and stores the buffer, or -1 if the section is absent.
// Deliberately does not touch g_sections/g_tables: a reload that is going to
// be rejected must not have mutated anything by the time we find out.
static int64_t read_i32_section(FILE* f, const char* want,
                                int nsections, int32_t** out) {
    *out = NULL;
    int64_t found = -1;
    static const int esize[4] = {4, 4, 1, 1};
    for (int i = 0; i < nsections; i++) {
        char name[25];
        int32_t meta[2];
        int64_t shape[4];
        if (fread(name, 1, 24, f) != 24) return -1;
        name[24] = '\0';                      // blob names need not be terminated
        if (fread(meta, 4, 2, f) != 2) return -1;
        if (fread(shape, 8, 4, f) != 4) return -1;
        // guard dtype and ndim BEFORE they index esize[]/shape[]
        if (meta[0] < 0 || meta[0] > 3) return -1;
        if (meta[1] < 0 || meta[1] > 4) return -1;
        int64_t n = 1;
        for (int d = 0; d < meta[1]; d++) {
            if (shape[d] < 0) return -1;
            n *= shape[d];
        }
        int64_t bytes = n * esize[meta[0]];
        if (strcmp(name, want) == 0 && found < 0) {
            // exporter's _DT: 0=f32, 1=i32, 2=i8, 3=u8 (export_tables.py:227)
            if (meta[0] != 1) return -1;              // must be i32
            int32_t* buf = (int32_t*)malloc(bytes > 0 ? (size_t)bytes : 1);
            if (!buf) return -1;
            if (bytes && fread(buf, 1, (size_t)bytes, f) != (size_t)bytes) {
                free(buf); return -1;
            }
            *out = buf;
            found = shape[0];
        } else if (fseek(f, (long)bytes, SEEK_CUR) != 0) {
            return -1;
        }
    }
    return found;
}

#define RELOAD_FAIL(...) do { \
    fprintf(stderr, "pt_reload_decks: " __VA_ARGS__); \
    free(new_decks); free(new_alt); if (f) fclose(f); return -1; \
} while (0)

int pt_reload_decks(const char* path) {
    const PtTables* cur = pt_tables();       // force first load if needed
    int32_t* new_decks = NULL;
    int32_t* new_alt = NULL;
    FILE* f = NULL;

    if (!path) RELOAD_FAIL("null path\n");
    f = fopen(path, "rb");
    if (!f) RELOAD_FAIL("cannot open %s\n", path);

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "PTCGTAB1", 8))
        RELOAD_FAIL("bad magic in %s\n", path);
    int32_t hdr[15];
    if (fread(hdr, 4, 15, f) != 15) RELOAD_FAIL("short header in %s\n", path);
    // Same layout gate as load_blob, but as a rejection rather than an assert:
    // a stale blob is an operator error, not a reason to kill the run.
    if (hdr[0] != PT_OBS_SIZE || hdr[1] != PT_MAX_TOKENS || hdr[2] != PT_MAX_OPTIONS
        || hdr[3] != PT_TOK_INT || hdr[4] != PT_TOK_F
        || hdr[5] != PT_OPT_INT || hdr[6] != PT_OPT_F
        || hdr[7] != PT_GLOBAL_F || hdr[8] != PT_DEC_INT || hdr[9] != PT_DEC_F
        || hdr[10] != PT_ORACLE_F || hdr[11] != PT_N_CARDS
        || hdr[12] != PT_N_ATTACKS || hdr[13] != PT_N_SKILLS)
        RELOAD_FAIL("%s layout disagrees with this binary (obs %d vs %d)\n",
                    path, hdr[0], PT_OBS_SIZE);
    int nsec = hdr[14];
    if (nsec <= 0 || nsec > 32) RELOAD_FAIL("bad section count %d\n", nsec);

    long after_hdr = ftell(f);
    int64_t new_n = read_i32_section(f, "decks", nsec, &new_decks);
    if (new_n < 0 || !new_decks) RELOAD_FAIL("no usable 'decks' section in %s\n", path);
    if (fseek(f, after_hdr, SEEK_SET) != 0) RELOAD_FAIL("seek failed\n");
    int64_t new_an = read_i32_section(f, "decks_alt", nsec, &new_alt);
    if (new_an < 0) { new_an = 0; free(new_alt); new_alt = NULL; }

    // Everything above is file I/O into LOCAL buffers, deliberately outside
    // the lock: no reason to stall the env threads for a 1.7 MB read. From
    // here on we compare against and mutate the live table, so take the lock.
    // The hold is a ~92 KB memcmp plus a few stores -- microseconds.
#define RELOAD_FAIL_UNLOCK(...) do { \
    pthread_rwlock_unlock(&g_decks_lock); \
    RELOAD_FAIL(__VA_ARGS__); \
} while (0)
    pthread_rwlock_wrlock(&g_decks_lock);

    // --- append-only contract -------------------------------------------
    if (new_n < cur->n_decks)
        RELOAD_FAIL_UNLOCK("%s has %lld training decks, current pool has %d -- "
                    "that is a REMOVAL, which must be a stop/resume\n",
                    path, (long long)new_n, cur->n_decks);
    if (new_n > PT_DECK_ALT_BASE)
        RELOAD_FAIL_UNLOCK("%s has %lld training decks, which reaches the "
                    "coverage id base (%d) -- the id spaces would collide\n",
                    path, (long long)new_n, PT_DECK_ALT_BASE);
    for (int i = 0; i < cur->n_decks; i++) {
        const int32_t* a = cur->decks + (size_t)i * 60;
        const int32_t* b = new_decks + (size_t)i * 60;
        if (memcmp(a, b, 60 * sizeof(int32_t)) != 0)
            RELOAD_FAIL_UNLOCK("%s changes existing deck id %d -- a reorder "
                        "wearing an append's clothes; every id after it would "
                        "silently rename\n", path, i);
    }
    // The coverage pool must not move either. deck_row() addresses it as
    // (idx - n_decks), so growing the TRAINING pool already shifts every
    // coverage id by k. That is tolerable only because coverage ids are
    // excluded from the deck matrix and are redrawn every episode -- but if
    // the coverage CONTENTS also changed under us, a persisted coverage id
    // would resolve to a different list with nothing to detect it.
    if (new_an != cur->n_decks_alt)
        RELOAD_FAIL_UNLOCK("%s has %lld coverage decks, current has %d -- the "
                    "coverage pool must be unchanged across an append\n",
                    path, (long long)new_an, cur->n_decks_alt);
    if (new_an > 0 && memcmp(cur->decks_alt, new_alt,
                             (size_t)new_an * 60 * sizeof(int32_t)) != 0)
        RELOAD_FAIL_UNLOCK("%s changes the coverage pool contents\n", path);

    // --- publish (callers are parked; see the header comment) ------------
    int old_n = g_tables.n_decks;
    int32_t* old_decks = g_tables.decks;
    g_tables.decks = new_decks;
    g_tables.n_decks = (int)new_n;
    // keep g_sections consistent so a later sec("decks") cannot hand out the
    // buffer we are about to free
    for (int i = 0; i < g_nsections; i++)
        if (strcmp(g_sections[i].name, "decks") == 0) {
            g_sections[i].data = new_decks;
            g_sections[i].shape[0] = new_n;
        }
    // A weighted pool must survive an append. Decks appended mid-run start at
    // the mean (1.0, since deck_w is normalised to mean 1), so growing the
    // pool cannot silently re-shape the distribution over the decks that were
    // already there. If the grow fails we drop to uniform rather than leave
    // deck_w shorter than n_decks, which would read past the end on the very
    // next draw.
    if (g_tables.deck_w) {
        float* gw = (float*)realloc(g_tables.deck_w,
                                    (size_t)new_n * sizeof(float));
        if (gw) {
            for (int64_t i = old_n; i < new_n; i++) gw[i] = 1.0f;
            g_tables.deck_w = gw;
        } else {
            free(g_tables.deck_w);
            g_tables.deck_w = NULL;
            fprintf(stderr, "ptcg deck weights: out of memory growing to %lld "
                    "decks; falling back to UNIFORM sampling\n",
                    (long long)new_n);
        }
        if (rebuild_deck_cw() != 0) {
            free(g_tables.deck_w);
            g_tables.deck_w = NULL;
            rebuild_deck_cw();
            fprintf(stderr, "ptcg deck weights: could not rebuild after the "
                    "append; falling back to UNIFORM sampling\n");
        }
    }
    // Safe to free under the write lock: no reader can still hold a pointer
    // into the old array -- parking the env threads alone does not guarantee
    // that.
    if (old_decks) free(old_decks);
    pthread_rwlock_unlock(&g_decks_lock);
    free(new_alt);                 // verified identical; keep the live one
    fclose(f);
    return (int)new_n;
}

#undef RELOAD_FAIL_UNLOCK
#undef RELOAD_FAIL

// ------------------------------------------------- per-deck sampling weights
// Rebuild the cumulative table from g_tables.deck_w. THE CALLER HOLDS THE
// WRITE LOCK. Normalising to mean 1 over the training pool is what keeps the
// coverage share fixed: the random opponent draws from training+coverage
// slots, so if the training weights were used raw, halving every weight in the
// file would silently double how often it reached for a coverage deck.
//
// Returns 0, or -1 (leaving deck_cw NULL) if the weights cannot produce a
// valid distribution.
static int rebuild_deck_cw(void) {
    PtTables* t = &g_tables;
    free(t->deck_cw);
    t->deck_cw = NULL;
    if (!t->deck_w) return 0;                 // uniform fast path
    int n = t->n_decks, na = t->n_decks_alt;
    if (n <= 0) return -1;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += t->deck_w[i];
    if (!(sum > 0.0)) return -1;
    double scale = (double)n / sum;           // -> mean 1 over the training pool
    double* cw = (double*)malloc((size_t)(n + na) * sizeof(double));
    if (!cw) return -1;
    double acc = 0.0;
    for (int i = 0; i < n; i++) { acc += (double)t->deck_w[i] * scale; cw[i] = acc; }
    for (int i = 0; i < na; i++) { acc += 1.0; cw[n + i] = acc; }
    t->deck_cw = cw;
    return 0;
}

int pt_set_deck_weights(const float* w, int n) {
    pt_tables();                              // force first load if needed
    if (!w || n <= 0) {
        fprintf(stderr, "ptcg deck weights: null or empty vector\n");
        return -1;
    }
    // Validate BEFORE touching live state: a rejected request must leave the
    // run sampling exactly as it was.
    double sum = 0.0;
    int active = 0;
    for (int i = 0; i < n; i++) {
        // NaN fails the >= test, so this catches NaN and negatives together;
        // isfinite catches +inf, which would otherwise make every other deck
        // unreachable rather than erroring.
        if (!(w[i] >= 0.0f) || !isfinite(w[i])) {
            fprintf(stderr, "ptcg deck weights: entry %d is %g -- weights must "
                    "be finite and >= 0\n", i, (double)w[i]);
            return -1;
        }
        sum += w[i];
        if (w[i] > 0.0f) active++;
    }
    if (!(sum > 0.0)) {
        fprintf(stderr, "ptcg deck weights: all %d weights are zero -- that "
                "would leave nothing to sample\n", n);
        return -1;
    }

    pthread_rwlock_wrlock(&g_decks_lock);
    if (n != g_tables.n_decks) {
        int live = g_tables.n_decks;
        pthread_rwlock_unlock(&g_decks_lock);
        fprintf(stderr, "ptcg deck weights: got %d weights for a %d-deck pool "
                "-- the caller resolved names against a different listing, so "
                "every id past the first difference would be wrong\n", n, live);
        return -1;
    }
    float* nw = (float*)malloc((size_t)n * sizeof(float));
    if (!nw) {
        pthread_rwlock_unlock(&g_decks_lock);
        fprintf(stderr, "ptcg deck weights: out of memory\n");
        return -1;
    }
    memcpy(nw, w, (size_t)n * sizeof(float));
    float* old_w = g_tables.deck_w;
    double* old_cw = g_tables.deck_cw;
    g_tables.deck_w = nw;
    g_tables.deck_cw = NULL;                  // rebuild_deck_cw frees/replaces
    if (rebuild_deck_cw() != 0) {
        free(nw);
        g_tables.deck_w = old_w;              // restore; nothing observable moved
        g_tables.deck_cw = old_cw;
        pthread_rwlock_unlock(&g_decks_lock);
        fprintf(stderr, "ptcg deck weights: could not build the cumulative "
                "table; staying on the previous distribution\n");
        return -1;
    }
    free(old_w);
    free(old_cw);
    pthread_rwlock_unlock(&g_decks_lock);
    return active;
}

void pt_clear_deck_weights(void) {
    pt_tables();
    pthread_rwlock_wrlock(&g_decks_lock);
    free(g_tables.deck_w);
    free(g_tables.deck_cw);
    g_tables.deck_w = NULL;
    g_tables.deck_cw = NULL;
    pthread_rwlock_unlock(&g_decks_lock);
}

// ------------------------------------------------------------------ smap
void smap_clear(SMap* m) {
    memset(m->keys, 0xff, sizeof(m->keys));  // -1 = empty
    m->n = 0;
}

static inline int smap_slot(const SMap* m, int key) {
    unsigned h = (unsigned)key * 2654435761u;
    int i = (int)(h & (PT_MAP_CAP - 1));
    while (m->keys[i] != -1 && m->keys[i] != key) i = (i + 1) & (PT_MAP_CAP - 1);
    return i;
}

void smap_put(SMap* m, int key, int val) {
    if (key < 0) return;
    int i = smap_slot(m, key);
    if (m->keys[i] == -1) { m->keys[i] = key; m->n++; }
    m->vals[i] = val;
    assert(m->n < PT_MAP_CAP - 8 && "smap overflow");
}

int smap_get(const SMap* m, int key) {
    if (key < 0) return PT_NONE;
    int i = smap_slot(m, key);
    return m->keys[i] == key ? m->vals[i] : PT_NONE;
}

// deletion via tombstone-free rehash of the probe cluster
void smap_del(SMap* m, int key) {
    if (key < 0) return;
    int i = smap_slot(m, key);
    if (m->keys[i] != key) return;
    m->keys[i] = -1; m->n--;
    int j = (i + 1) & (PT_MAP_CAP - 1);
    while (m->keys[j] != -1) {
        int k = m->keys[j], v = m->vals[j];
        m->keys[j] = -1; m->n--;
        smap_put(m, k, v);
        j = (j + 1) & (PT_MAP_CAP - 1);
    }
}

// ------------------------------------------------------------ engine wrapper
typedef struct { void* battlePtr; int errorPlayer; int errorType; } CgStartData;
typedef struct { const char* json; unsigned char* data; int count; int selectPlayer; } CgSerialData;
extern void GameInitialize(void);
extern CgStartData BattleStart(int* cards);
extern void BattleFinish(void* ptr);
extern CgSerialData GetBattleData(void* ptr);
extern int Select(void* ptr, int* arg, int n);
extern const char* VisualizeData(void* ptr);

static pthread_once_t g_engine_once = PTHREAD_ONCE_INIT;
static void engine_init_once(void) { GameInitialize(); }
void pt_engine_init(void) { pthread_once(&g_engine_once, engine_init_once); }

void* pt_battle_start(const int32_t* d0, const int32_t* d1) {
    int cards[120];
    for (int i = 0; i < 60; i++) { cards[i] = d0[i]; cards[60 + i] = d1[i]; }
    CgStartData sd = BattleStart(cards);
    return sd.battlePtr;
}

const char* pt_battle_data(void* battle) { return GetBattleData(battle).json; }
int pt_battle_select(void* battle, const int* idx, int n) { return Select(battle, (int*)idx, n); }
void pt_battle_finish(void* battle) { if (battle) BattleFinish(battle); }
const char* pt_battle_visualize(void* battle) { return VisualizeData(battle); }

// model-side accessor for the same blob (GPU static tables)
const void* pt_tables_handle(void) { return pt_tables(); }

// typed accessors for the CUDA side (avoids struct-layout mirroring)
const float* pt_tab_card_static(void)  { return pt_tables()->card_static; }
const int*   pt_tab_card_attacks(void) { return pt_tables()->card_attacks; }
const int*   pt_tab_card_skills(void)  { return pt_tables()->card_skills; }
const float* pt_tab_att_static(void)   { return pt_tables()->att_static; }
const float* pt_tab_card_text(void)    { return pt_tables()->card_text; }
const float* pt_tab_att_text(void)     { return pt_tables()->att_text; }
const float* pt_tab_skill_text(void)   { return pt_tables()->skill_text; }
