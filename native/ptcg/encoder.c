// encode() + write_oracle() — exact port of ptcg/rl/encoder.py.
// Token order, class-dedup keys, normalizations and truthiness quirks
// (python `x or 0`) are all preserved; the parity harness compares rows
// bit-level against the python encoder on identical obs streams.
#include <string.h>
#include <stdio.h>
#include "ptcg_env.h"

static const int CARD_ZONES[] = {AREA_DECK, AREA_HAND, AREA_DISCARD, AREA_PRIZE, AREA_LOOKING};

static bool in_card_zones(int area) {
    for (int i = 0; i < 5; i++) if (CARD_ZONES[i] == area) return true;
    return false;
}

// python `x or 0` with PT_NONE as None
static inline int or0(int v) { return v == PT_NONE ? 0 : v; }

// ---------------------------------------------------------------- ptr map
// (playerIndex, area, index) -> token idx. area<=12, index<=63, player 0/1.
typedef struct { int16_t map[2][13][PT_MAX_LIST]; } PtrMap;

static void ptr_clear(PtrMap* p) { memset(p->map, 0xff, sizeof(p->map)); }

static void ptr_put(PtrMap* p, int pidx, int area, int idx, int tok) {
    if (pidx < 0 || pidx > 1 || area < 0 || area > 12 || idx < 0 || idx >= PT_MAX_LIST) return;
    p->map[pidx][area][idx] = (int16_t)tok;
}

// python dict-miss semantics: a key containing None never hits
static int ptr_get(const PtrMap* p, int pidx, int area, int idx) {
    if (pidx == PT_NONE || area == PT_NONE || idx == PT_NONE) return -1;
    if (pidx < 0 || pidx > 1 || area < 0 || area > 12 || idx < 0 || idx >= PT_MAX_LIST) return -1;
    return p->map[pidx][area][idx];
}

// ------------------------------------------------------------ derived feats
static void energy_counts(const int* energies, int n, int att[12]) {
    memset(att, 0, 12 * sizeof(int));
    for (int i = 0; i < n; i++)
        if (energies[i] >= 0 && energies[i] < 12) att[energies[i]]++;
}

// missing energy for attack aid given attached counts (encoder.py _deficit)
static int deficit(const PtTables* T, const int att[12], int total, int aid) {
    const int8_t* cost = T->atk_cost + (size_t)aid * 12;
    int typed_missing = 0, used = 0;
    for (int t = 1; t < 12; t++) {
        int need = cost[t], have = att[t];
        used += need < have ? need : have;
        typed_missing += need > have ? need - have : 0;
    }
    int tr_need = (cost[5] > att[5] ? cost[5] - att[5] : 0)
                + (cost[7] > att[7] ? cost[7] - att[7] : 0);
    int tr_used = att[11] < tr_need ? att[11] : tr_need;
    typed_missing -= tr_used;
    used += tr_used;
    int wild_used = att[10] < typed_missing ? att[10] : typed_missing;
    typed_missing -= wild_used;
    used += wild_used;
    int colorless_missing = cost[0] > (total - used) ? cost[0] - (total - used) : 0;
    return typed_missing + colorless_missing;
}

static void attack_feats(const PtTables* T, float* f, int card_id,
                         const int* energies, int n_en, int opp_active_id) {
    int att[12];
    energy_counts(energies, n_en, att);
    int total = n_en;
    for (int k = 0; k < 4; k++) {
        int aid = T->top_attacks[card_id * 4 + k];
        if (aid == 0) continue;
        int d = deficit(T, att, total, aid);
        f[TF_ATK0 + 3 * k] = d == 0 ? 1.0f : 0.0f;
        f[TF_ATK0 + 3 * k + 1] = T->atk_damage[aid] / 300.0f;
        f[TF_ATK0 + 3 * k + 2] = (d < 3 ? d : 3) / 3.0f;
    }
    int my_type = T->energy_type[card_id];
    if (opp_active_id) {
        f[TF_WEAK_HIT] = T->weakness[opp_active_id] == my_type ? 1.0f : 0.0f;
        f[TF_RES_HIT] = T->resistance[opp_active_id] == my_type ? 1.0f : 0.0f;
    }
    f[TF_RETREAT_OK] = total >= T->retreat[card_id] ? 1.0f : 0.0f;
}

// one-ply lethal estimate (encoder.py _threat)
static void threat(const PtTables* T, int att_id, const int* att_en, int n_att_en,
                   int def_id, int def_hp, float extra_dmg,
                   float* now, float* p1) {
    int att[12];
    energy_counts(att_en, n_att_en, att);
    int total = n_att_en;
    int my_type = T->energy_type[att_id];
    *now = 0.0f; *p1 = 0.0f;
    for (int k = 0; k < 4; k++) {
        int aid = T->top_attacks[att_id * 4 + k];
        if (aid == 0) continue;
        float dmg = T->atk_damage[aid];
        if (dmg <= 0) continue;
        if (T->weakness[def_id] == my_type) dmg *= 2;
        if (T->resistance[def_id] == my_type) dmg -= 30;
        dmg += extra_dmg;
        if (dmg >= def_hp) {
            int d = deficit(T, att, total, aid);
            if (d == 0) *now = 1.0f;
            if (d <= 1) *p1 = 1.0f;
        }
    }
}

// -------------------------------------------------------------- dedup keys
typedef struct { int f[10]; } OptKey;

static bool key_eq(const OptKey* a, const OptKey* b) {
    return memcmp(a->f, b->f, sizeof(a->f)) == 0;
}

// resolve_card(pidx, area, idx): card id at location; 0 unknown/facedown/oob
static int resolve_card(const PObs* obs, int seat, int pidx, int area, int idx) {
    if (pidx == PT_NONE) pidx = seat;
    if (pidx < 0 || pidx > 1) return 0;
    const PPlayer* pl = &obs->players[pidx];
    if (idx == PT_NONE) idx = -1;
    switch (area) {
    case AREA_HAND:
        if (pl->n_hand <= 0 || idx < 0 || idx >= pl->n_hand) return 0;
        return or0(pl->hand[idx].id);
    case AREA_DISCARD:
        if (idx < 0 || idx >= pl->n_discard) return 0;
        return or0(pl->discard[idx].id);
    case AREA_PRIZE:
        if (idx < 0 || idx >= pl->n_prize) return 0;
        return or0(pl->prize[idx].id) < 0 ? 0 : or0(pl->prize[idx].id);
    case AREA_DECK:
        if (obs->select.n_deck <= 0 || idx < 0 || idx >= obs->select.n_deck) return 0;
        return or0(obs->select.deck[idx].id) < 0 ? 0 : or0(obs->select.deck[idx].id);
    case AREA_LOOKING:
        if (idx < 0 || idx >= obs->n_looking) return 0;
        return or0(obs->looking[idx].id) < 0 ? 0 : or0(obs->looking[idx].id);
    case AREA_ACTIVE:
        if (idx < 0 || idx >= pl->n_active || !pl->active[idx].present) return 0;
        return pl->active[idx].id;
    case AREA_BENCH:
        if (idx < 0 || idx >= pl->n_bench) return 0;
        return pl->bench[idx].id;
    case AREA_STADIUM:
        if (idx < 0 || idx >= obs->n_stadium) return 0;
        return or0(obs->stadium[idx].id);
    default:
        return 0;
    }
}

// ------------------------------------------------------------------ encode
void pt_encode(float* row, const PObs* obs, const int* picked, int n_picked,
               bool stop_allowed, int forced_run, const Tracker* tr) {
    const PtTables* T = pt_tables();
    memset(row, 0, PT_OBS_SIZE * sizeof(float));
    const PSelect* sel = &obs->select;
    int seat = obs->yourIndex;
    const PPlayer* me = &obs->players[seat];
    const PPlayer* opp = &obs->players[1 - seat];
    const EffState* fx = tr ? &tr->fx : NULL;
    int turn_now = obs->turn;

    int opp_active_id = 0, opp_active_serial = PT_NONE;
    const PPokemon* opp_active_pk = NULL;
    if (opp->n_active && opp->active[0].present) {
        opp_active_id = opp->active[0].id;
        opp_active_serial = opp->active[0].serial;
        opp_active_pk = &opp->active[0];
    }
    int my_active_id = 0, my_active_serial = PT_NONE;
    const PPokemon* my_active_pk = NULL;
    if (me->n_active && me->active[0].present) {
        my_active_id = me->active[0].id;
        my_active_serial = me->active[0].serial;
        my_active_pk = &me->active[0];
    }

    float* tok_int = row + PT_OFF_TOK_INT;      // (160, 5)
    float* tok_f = row + PT_OFF_TOK_F;          // (160, 56)
    int nt = 0;
    PtrMap ptr;
    ptr_clear(&ptr);

    #define TI(t, k) tok_int[(t) * PT_TOK_INT + (k)]
    #define TFV(t) (tok_f + (t) * PT_TOK_F)

    // add(card_id, zone, owner) -> token idx or -1
    #define ADD_TOKEN(cid, zone, owner, out_t) do { \
        if (nt >= PT_MAX_TOKENS) { out_t = -1; } \
        else { out_t = nt; TI(nt, 0) = (float)(cid); TI(nt, 1) = (float)(zone); \
               TI(nt, 2) = (float)(owner); nt++; } } while (0)

    // board Pokemon (order: me active, me bench, opp active, opp bench)
    for (int side = 0; side < 2; side++) {
        const PPlayer* pl = side == 0 ? me : opp;
        int pidx = side == 0 ? seat : 1 - seat;
        int owner = side == 0 ? OWN_MINE : OWN_OPP;
        int az = side == 0 ? ZONE_MY_ACTIVE : ZONE_OPP_ACTIVE;
        int bz = side == 0 ? ZONE_MY_BENCH : ZONE_OPP_BENCH;
        int vs_id = side == 0 ? opp_active_id : my_active_id;
        for (int part = 0; part < 2; part++) {
            int count = part == 0 ? pl->n_active : pl->n_bench;
            for (int i = 0; i < count; i++) {
                const PPokemon* p = part == 0 ? &pl->active[i] : &pl->bench[i];
                if (part == 0 && !p->present) continue;
                int t;
                ADD_TOKEN(p->id, part == 0 ? az : bz, owner, t);
                if (t < 0) continue;
                ptr_put(&ptr, pidx, part == 0 ? AREA_ACTIVE : AREA_BENCH, i, t);
                float* f = TFV(t);
                int max_hp = p->maxHp ? p->maxHp : 1;
                f[TF_HP_FRAC] = (float)p->hp / max_hp;
                f[TF_HP] = p->hp / 300.0f;
                f[TF_MAX_HP] = max_hp / 300.0f;
                f[TF_DMG] = (max_hp - p->hp) / 10.0f / 30.0f;
                f[TF_N_ENERGY] = p->n_energies / 5.0f;
                for (int e = 0; e < p->n_energies; e++)
                    if (p->energies[e] >= 0 && p->energies[e] < 12)
                        f[TF_ENERGY0 + p->energies[e]] += 1 / 3.0f;
                f[TF_N_TOOLS] = p->n_tools / 2.0f;
                f[TF_STACK] = p->n_preEvolution / 3.0f;
                f[TF_APPEAR] = p->appearThisTurn ? 1.0f : 0.0f;
                if (part == 0) {                          // active: status
                    f[TF_POISONED] = pl->poisoned;
                    f[TF_BURNED] = pl->burned;
                    f[TF_ASLEEP] = pl->asleep;
                    f[TF_PARALYZED] = pl->paralyzed;
                    f[TF_CONFUSED] = pl->confused;
                    f[TF_IS_ACTIVE] = 1.0f;
                }
                for (int k = 0; k < 2 && k < p->n_tools; k++)
                    TI(t, 3 + k) = (float)or0(p->tools[k].id);
                if (fx) {
                    fx_token_flags(fx, p->serial, turn_now, f);
                    if (p->serial != PT_NONE
                        && smap_get(&fx->evolved_now, p->serial) != PT_NONE)
                        f[TF_EVOLVED_THIS_TURN] = 1.0f;
                    if (p->serial != PT_NONE
                        && smap_get(&fx->healed_now, p->serial) != PT_NONE)
                        f[TF_HEALED_THIS_TURN] = 1.0f;
                }
                if (p->id >= 0 && p->id < PT_N_CARDS)
                    attack_feats(T, f, p->id, p->energies, p->n_energies, vs_id);
            }
        }
    }

    // my hand (individual tokens)
    if (me->n_hand > 0)
        for (int i = 0; i < me->n_hand; i++) {
            int t;
            ADD_TOKEN(or0(me->hand[i].id), ZONE_MY_HAND, OWN_MINE, t);
            if (t >= 0) ptr_put(&ptr, seat, AREA_HAND, i, t);
        }

    // stadium
    for (int i = 0; i < obs->n_stadium; i++) {
        int t;
        ADD_TOKEN(or0(obs->stadium[i].id), ZONE_STADIUM,
                  obs->stadium[i].playerIndex == seat ? OWN_MINE : OWN_OPP, t);
        (void)t;
    }

    // count-collapsed zones: one token per distinct id, in list order;
    // ptr entries for EVERY index point at the id's representative token
    int rep_ids[PT_MAX_LIST], rep_tok[PT_MAX_LIST];
    #define ADD_COLLAPSED(cards, ncards, zone, owner, pidx, area) do { \
        int n_rep = 0; \
        for (int i = 0; i < (ncards); i++) { \
            int cid_raw = (cards)[i].id; \
            int cid = cid_raw == PT_NONE ? 0 : cid_raw; \
            int t = -1; \
            for (int r = 0; r < n_rep; r++) \
                if (rep_ids[r] == cid) { t = rep_tok[r]; break; } \
            if (t >= 0) { \
                TFV(t)[TF_COPIES] += 1 / 4.0f; \
            } else { \
                ADD_TOKEN(cid, zone, owner, t); \
                if (t < 0) continue; \
                if (n_rep < PT_MAX_LIST) { rep_ids[n_rep] = cid; rep_tok[n_rep] = t; n_rep++; } \
                TFV(t)[TF_COPIES] = 1 / 4.0f; \
            } \
            if ((area) >= 0) ptr_put(&ptr, pidx, area, i, t); \
        } } while (0)

    ADD_COLLAPSED(me->discard, me->n_discard, ZONE_MY_DISCARD, OWN_MINE, seat, AREA_DISCARD);
    ADD_COLLAPSED(opp->discard, opp->n_discard, ZONE_OPP_DISCARD, OWN_OPP, 1 - seat, AREA_DISCARD);
    if (sel->has_select && sel->n_deck > 0)
        ADD_COLLAPSED(sel->deck, sel->n_deck, ZONE_LOOKING, OWN_MINE, seat, AREA_DECK);
    if (obs->n_looking > 0)
        ADD_COLLAPSED(obs->looking, obs->n_looking, ZONE_LOOKING, OWN_NEUTRAL, seat, AREA_LOOKING);

    // tracker zones: ascending-id counters
    #define ADD_COUNTER(counter, zone, owner) do { \
        for (int cid = 0; cid < PT_N_CARDS; cid++) { \
            if (!(counter)[cid]) continue; \
            int t; \
            ADD_TOKEN(cid, zone, owner, t); \
            if (t >= 0) TFV(t)[TF_COPIES] = (counter)[cid] / 4.0f; \
        } } while (0)

    int n_known_hand = 0;
    bool prizes_known = false;
    if (tr) {
        CCount unseen;
        tracker_my_unseen(tr, obs, unseen);
        if (tr->my_prize_known_set) {
            ADD_COUNTER(tr->my_prize_known, ZONE_MY_PRIZE_KNOWN, OWN_MINE);
            pt_ccount_sub_clamp(unseen, tr->my_prize_known);
        }
        ADD_COUNTER(unseen, ZONE_MY_UNSEEN, OWN_MINE);
        CCount known_hand;
        tracker_opp_known_hand(tr, known_hand);
        for (int cid = 0; cid < PT_N_CARDS; cid++) n_known_hand += known_hand[cid];
        ADD_COUNTER(known_hand, ZONE_OPP_HAND_KNOWN, OWN_OPP);
        CCount known_deck;
        tracker_opp_known_deck(tr, known_deck);
        ADD_COUNTER(known_deck, ZONE_OPP_REVEALED, OWN_OPP);
        prizes_known = tr->my_prize_known_set;
    }

    // ---- options ----
    float* opt_int = row + PT_OFF_OPT_INT;      // (64, 6)
    float* opt_f = row + PT_OFF_OPT_F;          // (64, 8)
    float* mask = row + PT_OFF_OPT_MASK;
    #define OI(j, k) opt_int[(j) * PT_OPT_INT + (k)]
    #define OFV(j, k) opt_f[(j) * PT_OPT_F + (k)]

    bool picked_set[PT_MAX_OPTIONS];
    memset(picked_set, 0, sizeof(picked_set));
    for (int i = 0; i < n_picked; i++)
        if (picked[i] >= 0 && picked[i] < PT_MAX_OPTIONS) picked_set[picked[i]] = true;

    OptKey keys[PT_MAX_OPTIONS];
    int n_keys = 0;
    int n = sel->n_options < PT_MAX_OPTIONS - 1 ? sel->n_options : PT_MAX_OPTIONS - 1;
    for (int j = 0; j < n; j++) {
        const POption* o = &sel->option[j];
        int ot = or0(o->type);
        int area = o->area, idx = o->index, pidx = o->playerIndex;
        OI(j, 0) = (float)ot;

        int card_id = or0(o->cardId);
        int p1 = -1, p2 = -1;
        if (ot == OT_PLAY) {
            card_id = resolve_card(obs, seat, seat, AREA_HAND, idx);
            p1 = ptr_get(&ptr, seat, AREA_HAND, or0(idx));
        } else if (ot == OT_CARD || ot == OT_DISCARD || ot == OT_ABILITY) {
            if (!card_id)
                card_id = resolve_card(obs, seat, pidx, area, idx);
            p1 = ptr_get(&ptr, pidx == PT_NONE ? seat : pidx, or0(area), or0(idx));
        } else if (ot == OT_ATTACH || ot == OT_EVOLVE) {
            card_id = resolve_card(obs, seat, seat, area, idx);
            p1 = ptr_get(&ptr, seat, or0(area), or0(idx));
            p2 = ptr_get(&ptr, seat, or0(o->inPlayArea), or0(o->inPlayIndex));
        } else if (ot == OT_TOOL_CARD || ot == OT_ENERGY_CARD || ot == OT_ENERGY) {
            p1 = ptr_get(&ptr, pidx == PT_NONE ? seat : pidx, or0(area), or0(idx));
        } else if (ot == OT_ATTACK) {
            p1 = ptr_get(&ptr, seat, AREA_ACTIVE, 0);
        }
        OI(j, 1) = p1 < 0 ? 0.0f : (float)(p1 + 1);
        OI(j, 2) = p2 < 0 ? 0.0f : (float)(p2 + 1);
        OI(j, 3) = (float)card_id;
        OI(j, 4) = (float)or0(o->attackId);
        OI(j, 5) = (float)or0(o->number);

        OFV(j, OF_NUMBER) = or0(o->number) / 30.0f;
        OFV(j, OF_COUNT) = or0(o->count) / 4.0f;
        OFV(j, OF_SPECIAL) = or0(o->specialConditionType) / 5.0f;
        OFV(j, OF_ENERGY_IDX) = or0(o->energyIndex) / 5.0f;
        if (picked_set[j]) {
            OFV(j, OF_PICKED) = 1.0f;
            if (p1 >= 0) TFV(p1)[TF_PICKED] = 1.0f;
            continue;   // picked indices are never legal again
        }
        if (p1 >= 0) TFV(p1)[TF_REFERENCED] = 1.0f;

        // equivalence class key (python tuple -> 10-int struct; PT_NONE = None)
        bool board_target = (area == AREA_ACTIVE || area == AREA_BENCH)
                            || o->inPlayArea != PT_NONE;
        OptKey k;
        memset(&k, 0, sizeof(k));
        if (ot == OT_SKILL) {
            k.f[0] = 1; k.f[1] = ot; k.f[2] = card_id; k.f[3] = o->serial;
        } else if (board_target || ot == OT_TOOL_CARD || ot == OT_ENERGY_CARD
                   || ot == OT_ENERGY) {
            int rc = (ot == OT_TOOL_CARD || ot == OT_ENERGY_CARD || ot == OT_ENERGY)
                         ? resolve_card(obs, seat, pidx, area, idx) : PT_NONE;
            k.f[0] = 2; k.f[1] = ot; k.f[2] = area; k.f[3] = idx; k.f[4] = pidx;
            k.f[5] = o->inPlayArea; k.f[6] = o->inPlayIndex; k.f[7] = card_id;
            k.f[8] = o->count; k.f[9] = rc;
        } else if ((area != PT_NONE && in_card_zones(area)) || ot == OT_PLAY) {
            k.f[0] = 3; k.f[1] = ot; k.f[2] = area; k.f[3] = pidx; k.f[4] = card_id;
        } else {
            k.f[0] = 4; k.f[1] = ot; k.f[2] = o->number; k.f[3] = o->attackId;
            k.f[4] = o->specialConditionType;
        }
        bool dup = false;
        for (int r = 0; r < n_keys; r++)
            if (key_eq(&keys[r], &k)) { dup = true; break; }
        if (!dup) {
            keys[n_keys++] = k;
            mask[j] = 1.0f;
        }
    }

    if (stop_allowed) {
        mask[PT_STOP] = 1.0f;
        OI(PT_STOP, 0) = (float)PT_OT_STOP;
        OFV(PT_STOP, OF_IS_STOP) = 1.0f;
    }

    // ---- global ----
    float* g = row + PT_OFF_GLOBAL;
    int first = obs->firstPlayer;
    g[GF_TURN] = obs->turn / 50.0f;
    g[GF_PARITY] = (float)(obs->turn % 2);
    g[GF_I_AM_FIRST] = first == seat ? 1.0f : 0.0f;
    g[GF_FIRST_DECIDED] = first != -1 ? 1.0f : 0.0f;
    g[GF_ACTIONS] = obs->turnActionCount / 20.0f;
    g[GF_SUPPORTER] = obs->supporterPlayed;
    g[GF_STADIUM_PLAYED] = obs->stadiumPlayed;
    g[GF_ENERGY_ATTACHED] = obs->energyAttached;
    g[GF_RETREATED] = obs->retreated;
    g[GF_MY_PRIZES] = me->n_prize / 6.0f;
    g[GF_OPP_PRIZES] = opp->n_prize / 6.0f;
    g[GF_MY_DECK] = me->deckCount / 60.0f;
    g[GF_OPP_DECK] = opp->deckCount / 60.0f;
    g[GF_MY_HAND] = me->handCount / 20.0f;
    g[GF_OPP_HAND] = opp->handCount / 20.0f;
    g[GF_MY_BENCH] = me->n_bench / 5.0f;
    g[GF_OPP_BENCH] = opp->n_bench / 5.0f;
    g[GF_BENCH_MAX] = me->benchMax / 8.0f;
    g[GF_STADIUM_PRESENT] = obs->n_stadium > 0 ? 1.0f : 0.0f;
    g[GF_MY_STATUS0 + 0] = me->poisoned;  g[GF_MY_STATUS0 + 1] = me->burned;
    g[GF_MY_STATUS0 + 2] = me->asleep;    g[GF_MY_STATUS0 + 3] = me->paralyzed;
    g[GF_MY_STATUS0 + 4] = me->confused;
    g[GF_OPP_STATUS0 + 0] = opp->poisoned; g[GF_OPP_STATUS0 + 1] = opp->burned;
    g[GF_OPP_STATUS0 + 2] = opp->asleep;   g[GF_OPP_STATUS0 + 3] = opp->paralyzed;
    g[GF_OPP_STATUS0 + 4] = opp->confused;
    int min_left = sel->minCount - n_picked;
    int max_left = sel->maxCount - n_picked;
    g[GF_PICKS_MIN_LEFT] = (min_left > 0 ? min_left : 0) / 5.0f;
    g[GF_PICKS_MAX_LEFT] = (max_left > 0 ? max_left : 0) / 10.0f;
    g[GF_ACTOR_SEAT] = (float)seat;
    int unk = opp->handCount - n_known_hand;
    g[GF_OPP_HAND_UNKNOWN] = (unk > 0 ? unk : 0) / 20.0f;
    g[GF_MY_PRIZES_KNOWN] = prizes_known ? 1.0f : 0.0f;
    if (fx) {
        float my_locks[4], op_locks[4], my_boost, op_boost;
        fx_player_flags(fx, seat, turn_now, my_locks, &my_boost);
        fx_player_flags(fx, 1 - seat, turn_now, op_locks, &op_boost);
        g[GF_MY_LOCK_ITEM] = my_locks[LOCK_ITEM];
        g[GF_MY_LOCK_SUPPORTER] = my_locks[LOCK_SUPPORTER];
        g[GF_MY_LOCK_EVOLVE] = my_locks[LOCK_EVOLVE];
        g[GF_OPP_LOCK_ITEM] = op_locks[LOCK_ITEM];
        g[GF_OPP_LOCK_SUPPORTER] = op_locks[LOCK_SUPPORTER];
        g[GF_OPP_LOCK_EVOLVE] = op_locks[LOCK_EVOLVE];
        g[GF_MY_DMG_BOOST] = my_boost;
        g[GF_OPP_DMG_BOOST] = op_boost;
        g[GF_MY_ENERGY_REMOVED] = fx->energy_removed[seat] / 10.0f;
        g[GF_OPP_ENERGY_REMOVED] = fx->energy_removed[1 - seat] / 10.0f;
        g[GF_MY_TOOLS_REMOVED] = fx->tools_removed[seat] / 5.0f;
        g[GF_OPP_TOOLS_REMOVED] = fx->tools_removed[1 - seat] / 5.0f;
        g[GF_MY_FORCED_SWITCHES] = fx->forced_switches[seat] / 5.0f;
        g[GF_OPP_FORCED_SWITCHES] = fx->forced_switches[1 - seat] / 5.0f;
        g[GF_MY_KOD_LAST_TURN] = fx->last_ko_turn[seat] >= turn_now - 1 ? 1.0f : 0.0f;
        g[GF_OPP_KOD_LAST_TURN] = fx->last_ko_turn[1 - seat] >= turn_now - 1 ? 1.0f : 0.0f;
    }
    if (my_active_pk && opp_active_pk) {
        // effect mods feed the threat estimate (python token_flags dicts)
        float myf[PT_TOK_F], opf[PT_TOK_F];
        memset(myf, 0, sizeof(myf));
        memset(opf, 0, sizeof(opf));
        if (fx) {
            fx_token_flags(fx, my_active_serial, turn_now, myf);
            fx_token_flags(fx, opp_active_serial, turn_now, opf);
        }
        threat(T, opp_active_id, opp_active_pk->energies, opp_active_pk->n_energies,
               my_active_id, my_active_pk->hp,
               (opf[TF_EFF_DMG_DEALT_MOD] + myf[TF_EFF_DMG_TAKEN_MOD]) * 100.0f,
               &g[GF_MY_ACTIVE_IN_DANGER], &g[GF_MY_ACTIVE_IN_DANGER_P1]);
        threat(T, my_active_id, my_active_pk->energies, my_active_pk->n_energies,
               opp_active_id, opp_active_pk->hp,
               (myf[TF_EFF_DMG_DEALT_MOD] + opf[TF_EFF_DMG_TAKEN_MOD]) * 100.0f,
               &g[GF_OPP_ACTIVE_IN_DANGER], &g[GF_OPP_ACTIVE_IN_DANGER_P1]);
    }

    // ---- decision ----
    float* di = row + PT_OFF_DEC_INT;
    float* df = row + PT_OFF_DEC_F;
    di[DI_SELECT_TYPE] = (float)or0(sel->type);
    di[DI_CONTEXT] = (float)or0(sel->context);
    di[DI_EFFECT_CARD] = sel->has_effect ? (float)or0(sel->effect.id) : 0.0f;
    di[DI_CONTEXT_CARD] = sel->has_contextCard ? (float)or0(sel->contextCard.id) : 0.0f;
    if (fx) {
        di[DI_MY_LAST_ATTACK] = (float)fx_last_attack_id(fx, seat, my_active_serial);
        di[DI_OPP_LAST_ATTACK] = (float)fx_last_attack_id(fx, 1 - seat, opp_active_serial);
        di[DI_MY_SUPPORTER] = (float)fx->supporter_now[seat];
        di[DI_OPP_SUPPORTER] = (float)fx->supporter_prev[1 - seat];
    }
    df[DF_MIN_COUNT] = sel->minCount / 5.0f;
    df[DF_MAX_COUNT] = sel->maxCount / 10.0f;
    df[DF_N_OPTIONS] = sel->n_options / 30.0f;
    df[DF_REMAIN_DMG] = sel->remainDamageCounter / 10.0f;
    df[DF_REMAIN_ENERGY] = sel->remainEnergyCost / 5.0f;
    df[DF_PICKED] = n_picked / 5.0f;
    df[DF_FORCED_RUN] = forced_run / 5.0f;
}

// write the oracle block AFTER pt_encode (which zeroed the row)
void pt_write_oracle(float* row, const Oracle* orc, const PObs* obs) {
    CCount opp_hand, opp_deck, opp_prize, my_prize;
    oracle_emit(orc, obs, opp_hand, opp_deck, opp_prize, my_prize);
    float* o = row + PT_OFF_ORACLE;
    int i = 0, slots = 0;
    for (int cid = 0; cid < PT_N_CARDS && slots < PT_ORACLE_HAND_SLOTS; cid++) {
        if (!opp_hand[cid]) continue;
        o[i] = (float)cid;
        o[i + 1] = opp_hand[cid] / 4.0f;
        i += 2; slots++;
    }
    i = PT_ORACLE_HAND_SLOTS * 2;
    slots = 0;
    for (int cid = 0; cid < PT_N_CARDS && slots < PT_ORACLE_SPLIT_SLOTS; cid++) {
        if (!opp_deck[cid] && !opp_prize[cid]) continue;
        o[i] = (float)cid;
        o[i + 1] = opp_deck[cid] / 4.0f;
        o[i + 2] = opp_prize[cid] / 4.0f;
        i += 3; slots++;
    }
    i = PT_ORACLE_HAND_SLOTS * 2 + PT_ORACLE_SPLIT_SLOTS * 3;
    slots = 0;
    for (int cid = 0; cid < PT_N_CARDS && slots < PT_ORACLE_PRIZE_SLOTS; cid++) {
        if (!my_prize[cid]) continue;
        o[i] = (float)cid;
        o[i + 1] = my_prize[cid] / 4.0f;
        i += 2; slots++;
    }
}

void pt_write_mask(unsigned char* mask, const float* row) {
    const float* m = row + PT_OFF_OPT_MASK;
    for (int j = 0; j < PT_MAX_OPTIONS; j++) mask[j] = m[j] != 0.0f;
}
