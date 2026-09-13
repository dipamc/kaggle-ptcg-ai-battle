// Parity replayer: feed the python-captured obs stream through the C
// tracker/oracle/encoder and compare rows BIT-level against the python rows.
//
// Usage: parity_replay <stream.jsonl> <rows.bin>   (paths repo-relative)
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptcg_env.h"

static const char* section_of(int col) {
    if (col < PT_OFF_TOK_F) return "tok_int";
    if (col < PT_OFF_OPT_INT) return "tok_float";
    if (col < PT_OFF_OPT_F) return "opt_int";
    if (col < PT_OFF_OPT_MASK) return "opt_float";
    if (col < PT_OFF_GLOBAL) return "opt_mask";
    if (col < PT_OFF_DEC_INT) return "global_f";
    if (col < PT_OFF_DEC_F) return "dec_int";
    if (col < PT_OFF_ORACLE) return "dec_float";
    return "oracle";
}

static const char* g_viz;   // injected VisualizeData payload for this ingest
static const char* viz_provider(void* ctx) { (void)ctx; return g_viz; }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s stream.jsonl rows.bin\n", argv[0]); return 2; }
    FILE* fs = fopen(argv[1], "r");
    FILE* fr = fopen(argv[2], "rb");
    if (!fs || !fr) { perror("open"); return 2; }
    pt_tables();

    static Tracker trackers[2];
    static Oracle oracle;
    static PObs obs;
    static float row[PT_OBS_SIZE], ref[PT_OBS_SIZE];
    static int picked[PT_MAX_OPTIONS];
    bool game_live = false;

    size_t cap = 1 << 20;
    char* line = malloc(cap);
    long n_rows = 0, n_bad_rows = 0, n_obs = 0, n_games = 0;
    long first_bad_row = -1;

    ssize_t len;
    while ((len = getline(&line, &cap, fs)) > 0) {
        cJSON* rec = cJSON_Parse(line);
        if (!rec) { fprintf(stderr, "bad stream line\n"); return 2; }
        cJSON* start = cJSON_GetObjectItemCaseSensitive(rec, "start");
        cJSON* obsj = cJSON_GetObjectItemCaseSensitive(rec, "obs");
        cJSON* enc = cJSON_GetObjectItemCaseSensitive(rec, "enc");
        if (start) {
            int32_t d0[60], d1[60];
            cJSON* a0 = cJSON_GetObjectItemCaseSensitive(start, "deck0");
            cJSON* a1 = cJSON_GetObjectItemCaseSensitive(start, "deck1");
            for (int i = 0; i < 60; i++) {
                d0[i] = (int32_t)cJSON_GetArrayItem(a0, i)->valuedouble;
                d1[i] = (int32_t)cJSON_GetArrayItem(a1, i)->valuedouble;
            }
            tracker_init(&trackers[0], d0);
            tracker_init(&trackers[1], d1);
            oracle_init(&oracle, d0, d1);
            game_live = true;
            n_games++;
        } else if (obsj) {
            if (!game_live) { fprintf(stderr, "obs before start\n"); return 2; }
            if (pt_obs_parse(obsj->valuestring, &obs) != 0) {
                fprintf(stderr, "obs parse failed at record %ld\n", n_obs);
                return 2;
            }
            n_obs++;
            if (obs.has_current) {
                int seat = obs.yourIndex;
                tracker_update(&trackers[seat], &obs);
                cJSON* viz = cJSON_GetObjectItemCaseSensitive(rec, "viz");
                g_viz = (viz && cJSON_IsString(viz)) ? viz->valuestring : NULL;
                oracle_on_seat_obs(&oracle, &obs, viz_provider, NULL);
                g_viz = NULL;
            }
        } else if (enc) {
            cJSON* pj = cJSON_GetObjectItemCaseSensitive(enc, "picked");
            int n_picked = cJSON_GetArraySize(pj);
            for (int i = 0; i < n_picked && i < PT_MAX_OPTIONS; i++)
                picked[i] = (int)cJSON_GetArrayItem(pj, i)->valuedouble;
            bool stop = cJSON_GetObjectItemCaseSensitive(enc, "stop")->valuedouble != 0;
            int forced = (int)cJSON_GetObjectItemCaseSensitive(enc, "forced")->valuedouble;
            int seat = obs.yourIndex;
            pt_encode(row, &obs, picked, n_picked, stop, forced, &trackers[seat]);
            pt_write_oracle(row, &oracle, &obs);
            if (fread(ref, sizeof(float), PT_OBS_SIZE, fr) != PT_OBS_SIZE) {
                fprintf(stderr, "rows.bin exhausted at row %ld\n", n_rows);
                return 2;
            }
            if (memcmp(row, ref, sizeof(row)) != 0) {
                n_bad_rows++;
                if (first_bad_row < 0) {
                    first_bad_row = n_rows;
                    int shown = 0;
                    for (int c = 0; c < PT_OBS_SIZE && shown < 12; c++) {
                        if (row[c] != ref[c]) {
                            printf("row %ld col %d [%s+%d]: C=%.6f py=%.6f\n",
                                   n_rows, c, section_of(c),
                                   c - (c < PT_OFF_TOK_F ? PT_OFF_TOK_INT :
                                        c < PT_OFF_OPT_INT ? PT_OFF_TOK_F :
                                        c < PT_OFF_OPT_F ? PT_OFF_OPT_INT :
                                        c < PT_OFF_OPT_MASK ? PT_OFF_OPT_F :
                                        c < PT_OFF_GLOBAL ? PT_OFF_OPT_MASK :
                                        c < PT_OFF_DEC_INT ? PT_OFF_GLOBAL :
                                        c < PT_OFF_DEC_F ? PT_OFF_DEC_INT :
                                        c < PT_OFF_ORACLE ? PT_OFF_DEC_F :
                                        PT_OFF_ORACLE),
                                   row[c], ref[c]);
                            shown++;
                        }
                    }
                }
            }
            n_rows++;
        }
        cJSON_Delete(rec);
    }
    // rows.bin must be fully consumed
    long extra = 0;
    float dummy;
    while (fread(&dummy, sizeof(float), 1, fr) == 1) extra++;
    printf("games=%ld obs=%ld rows=%ld mismatched_rows=%ld extra_ref_floats=%ld\n",
           n_games, n_obs, n_rows, n_bad_rows, extra);
    printf(n_bad_rows == 0 && extra == 0 && n_rows > 0 ? "PARITY OK\n" : "PARITY FAIL\n");
    return n_bad_rows != 0 || extra != 0 || n_rows == 0;
}
