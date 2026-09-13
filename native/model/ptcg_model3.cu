// Part 3: transformer blocks, drivers, encoder/decoder/network vtables.

struct PtcgActFull;   // fwd decl

// one pre-LN block forward. x_in/x_out (N,L,D) distinct buffers.
// scratch: a->s_ln (N,L,D), a->s_qkv (N,L,3D), s_q/s_k/s_v/s_ctx (N,H,L,Dh),
// s_ffn (N,L,FFN)
struct BlkScratch {
    float *ln, *qkv, *q, *k, *v, *ctx, *ffn, *dffn, *dq, *dk, *dv, *dp;
};

static void blk_fwd(BlkW* bw, BlkAct* ba, BlkScratch* sc, const float* x_in,
        float* x_out, const float* keep, int N, int L, cudaStream_t s) {
    long NL = (long)N * L;
    float scale = 1.0f / sqrtf((float)PT_DH);
    pt_ln_fwd<<<(int)NL, 128, 0, s>>>(sc->ln, ba->ln1_mean.data, ba->ln1_rstd.data,
        x_in, bw->ln1w, bw->ln1b, (int)NL, PT_D);
    pt_linear(sc->ln, bw->qkvw, bw->qkvb, sc->qkv, (int)NL, PT_D, 3 * PT_D, s);
    pt_qkv_split<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->q, sc->k, sc->v, sc->qkv,
        N, L, PT_H, PT_DH);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T, L, L, PT_DH,
        sc->q, (long)L * PT_DH, sc->k, (long)L * PT_DH,
        ba->probs.data, (long)L * L, N * PT_H, s);
    pt_attn_softmax_fwd<<<(int)((long)N * PT_H * L), 128, 0, s>>>(
        ba->probs.data, keep, N, PT_H, L, scale);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N, L, PT_DH, L,
        ba->probs.data, (long)L * L, sc->v, (long)L * PT_DH,
        sc->ctx, (long)L * PT_DH, N * PT_H, s);
    pt_heads_merge<<<ptg(NL * PT_D), PTB, 0, s>>>(ba->merged.data, sc->ctx,
        N, L, PT_H, PT_DH);
    pt_linear(ba->merged.data, bw->projw, bw->projb, ba->x_mid.data,
        (int)NL, PT_D, PT_D, s);
    pt_add<<<ptg(NL * PT_D), PTB, 0, s>>>(ba->x_mid.data, x_in, NL * PT_D);
    pt_ln_fwd<<<(int)NL, 128, 0, s>>>(sc->ln, ba->ln2_mean.data, ba->ln2_rstd.data,
        ba->x_mid.data, bw->ln2w, bw->ln2b, (int)NL, PT_D);
    pt_linear(sc->ln, bw->f1w, bw->f1b, ba->ffn1.data, (int)NL, PT_D, PT_FFN, s);
    pt_gelu_fwd<<<ptg(NL * PT_FFN), PTB, 0, s>>>(sc->ffn, ba->ffn1.data, NL * PT_FFN);
    pt_linear(sc->ffn, bw->f2w, bw->f2b, x_out, (int)NL, PT_FFN, PT_D, s);
    pt_add<<<ptg(NL * PT_D), PTB, 0, s>>>(x_out, ba->x_mid.data, NL * PT_D);
}

// block backward. dxA holds d(x_out) on entry and d(x_in) on exit.
// dxB is same-size scratch.
static void blk_bwd(BlkW* bw, BlkG* bg, BlkAct* ba, BlkScratch* sc,
        const float* x_in, float* dxA, float* dxB, int N, int L, cudaStream_t s) {
    long NL = (long)N * L;
    float scale = 1.0f / sqrtf((float)PT_DH);
    // ffn chain: recompute ln2_out + gelu(ffn1)
    k_ln_apply<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->ln, ba->x_mid.data,
        ba->ln2_mean.data, ba->ln2_rstd.data, bw->ln2w, bw->ln2b, NL, PT_D);
    pt_gelu_fwd<<<ptg(NL * PT_FFN), PTB, 0, s>>>(sc->ffn, ba->ffn1.data, NL * PT_FFN);
    pt_linear_bwd(sc->ffn, bw->f2w, dxA, sc->dffn, bg->f2w, bg->f2b,
        (int)NL, PT_FFN, PT_D, s);
    pt_gelu_bwd<<<ptg(NL * PT_FFN), PTB, 0, s>>>(sc->dffn, sc->dffn,
        ba->ffn1.data, NL * PT_FFN);
    pt_linear_bwd(sc->ln, bw->f1w, sc->dffn, dxB, bg->f1w, bg->f1b,
        (int)NL, PT_D, PT_FFN, s);
    pt_ln_bwd<<<(int)NL, 128, 0, s>>>(dxB, bg->ln2w, bg->ln2b, dxB,
        ba->x_mid.data, bw->ln2w, ba->ln2_mean.data, ba->ln2_rstd.data,
        (int)NL, PT_D);
    pt_add<<<ptg(NL * PT_D), PTB, 0, s>>>(dxA, dxB, NL * PT_D);  // dxA = d_x_mid
    // attn chain
    pt_linear_bwd(ba->merged.data, bw->projw, dxA, sc->ln /*d_merged*/,
        bg->projw, bg->projb, (int)NL, PT_D, PT_D, s);
    pt_heads_split<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->ctx /*d_ctx*/, sc->ln,
        N, L, PT_H, PT_DH);
    // recompute ln1_out, qkv, q/k/v
    k_ln_apply<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->ln, x_in,
        ba->ln1_mean.data, ba->ln1_rstd.data, bw->ln1w, bw->ln1b, NL, PT_D);
    pt_linear(sc->ln, bw->qkvw, bw->qkvb, sc->qkv, (int)NL, PT_D, 3 * PT_D, s);
    pt_qkv_split<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->q, sc->k, sc->v, sc->qkv,
        N, L, PT_H, PT_DH);
    // dV = P^T dctx ; dP = dctx V^T ; softmax bwd ; dK = dS^T Q ; dQ = dS K
    pt_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N, L, PT_DH, L,
        ba->probs.data, (long)L * L, sc->ctx, (long)L * PT_DH,
        sc->dv, (long)L * PT_DH, N * PT_H, s);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T, L, L, PT_DH,
        sc->ctx, (long)L * PT_DH, sc->v, (long)L * PT_DH,
        sc->dp, (long)L * L, N * PT_H, s);
    pt_attn_softmax_bwd<<<(int)((long)N * PT_H * L), 128, 0, s>>>(
        sc->dp, ba->probs.data, N, PT_H, L, scale);
    pt_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N, L, PT_DH, L,
        sc->dp, (long)L * L, sc->q, (long)L * PT_DH,
        sc->dk, (long)L * PT_DH, N * PT_H, s);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N, L, PT_DH, L,
        sc->dp, (long)L * L, sc->k, (long)L * PT_DH,
        sc->dq, (long)L * PT_DH, N * PT_H, s);
    pt_qkv_merge<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->qkv, sc->dq, sc->dk, sc->dv,
        N, L, PT_H, PT_DH);
    pt_linear_bwd(sc->ln, bw->qkvw, sc->qkv, dxB, bg->qkvw, bg->qkvb,
        (int)NL, PT_D, 3 * PT_D, s);
    pt_ln_bwd<<<(int)NL, 128, 0, s>>>(dxB, bg->ln1w, bg->ln1b, dxB,
        x_in, bw->ln1w, ba->ln1_mean.data, ba->ln1_rstd.data, (int)NL, PT_D);
    pt_add<<<ptg(NL * PT_D), PTB, 0, s>>>(dxA, dxB, NL * PT_D);  // dxA = d_x_in
}

// full activation struct registration (shared by rollout/train, train adds bwd)
struct PtcgActX {
    PtcgAct a;
    BlkScratch sc;
    PrecisionTensor s_ln, s_qkv, s_q, s_k, s_v, s_ctx, s_ffn, s_dffn,
                    s_dq, s_dk, s_dv, s_dp;
    PrecisionTensor tokp;      // (N,160,D) token_proj output
    PrecisionTensor d_tokp, d_tokcat, d_bc, d_bcin, d_decin, d_gin;
    PrecisionTensor g1a;       // global pre-gelu (N,128)
    PrecisionTensor d_otok_in, d_q, d_qin, d_scin, d_out64, d_ocard,
                    d_oatt, d_val, d_vin, d_v1;
    PrecisionTensor d_handlin, d_prizelogit;
    PrecisionTensor ov_out;    // oracle value (N,1)
};

static void ptcg_reg(PtcgActX* ax, PtcgWeights* w, Allocator* acts,
        Allocator* grads, int N, int train) {
    PtcgAct* a = &ax->a;
    a->N = N;
    a->train = train;
    if (train) {   // grads registered in canonical order == params order
        for (int i = 0; i < PT_NPARAM; i++) {
            w->g[i].shape[0] = PT_PARAMS[i].r;
            w->g[i].shape[1] = PT_PARAMS[i].c > 0 ? PT_PARAMS[i].c : 0;
            alloc_register(grads, &w->g[i]);
        }
    }
    table_reg(&a->tab, acts, grads, train);
    pt_reg(acts, &a->tok_cat, (long)N * PT_LTOK, PT_TOK_IN);
    pt_reg(acts, &ax->tokp, (long)N * PT_LTOK, PT_D);
    pt_reg(acts, &ax->g1a, N, 128);
    pt_reg(acts, &a->g_out, N, 128);
    pt_reg(acts, &a->dec_in, N, PT_DEC_IN);
    pt_reg(acts, &a->d1, N, 64);
    pt_reg(acts, &a->d_out, N, 64);
    pt_reg(acts, &a->bc_in, N, PT_BCAST_IN);
    pt_reg(acts, &a->bc_out, N, PT_D);
    // rollout has no backward: one shared trunk BlkAct + ping-pong x buffers
    int n_x = train ? PT_LAYERS + 1 : 2;
    int n_blk = train ? PT_LAYERS : 1;
    for (int i = 0; i < n_x; i++)
        pt_reg(acts, &a->x[i], (long)N * PT_LSEQ, PT_D);
    for (int i = 0; i < n_blk; i++)
        blk_reg(&a->blk[i], acts, N, PT_LSEQ, train);
    pt_reg(acts, &a->keep, N, PT_LSEQ);
    pt_reg(acts, &a->h, (long)N * PT_LSEQ, PT_D);
    pt_reg(acts, &a->lnf_mean, (long)N * PT_LSEQ, 0);
    pt_reg(acts, &a->lnf_rstd, (long)N * PT_LSEQ, 0);
    pt_reg(acts, &a->q_in, (long)N * PT_NOPT, PT_Q_IN);
    pt_reg(acts, &a->q1a, (long)N * PT_NOPT, PT_D);
    pt_reg(acts, &a->q_vec, (long)N * PT_NOPT, PT_D);
    pt_reg(acts, &a->sc_in, (long)N * PT_NOPT, PT_SCORE_IN);
    pt_reg(acts, &a->sc1, (long)N * PT_NOPT, 256);
    pt_reg(acts, &a->scores, (long)N * PT_NOPT, 0);
    pt_reg(acts, &a->ocard_in, (long)N * PT_NOPT, PT_DC);
    pt_reg(acts, &a->oatt_in, (long)N * PT_NOPT, PT_DA);
    pt_reg(acts, &a->otok_in, (long)N * PT_NORC, PT_ORC_IN);
    pt_reg(acts, &a->otok, (long)N * PT_NORC, PT_D);
    pt_reg(acts, &a->xc[0], (long)N * PT_LCRIT, PT_D);
    pt_reg(acts, &a->xc[1], (long)N * PT_LCRIT, PT_D);
    blk_reg(&a->cblk, acts, N, PT_LCRIT, train);
    pt_reg(acts, &a->ckeep, N, PT_LCRIT);
    pt_reg(acts, &a->ov_in, N, PT_VAL_IN);
    pt_reg(acts, &a->ov1, N, 256);
    pt_reg(acts, &ax->ov_out, N, 1);
    pt_reg(acts, &a->out, N, PT_NOPT + 1);
    // scratch (max seq = LCRIT for ln/qkv/heads buffers)
    long NLC = (long)N * PT_LCRIT;
    pt_reg(acts, &ax->s_ln, NLC, PT_D);
    pt_reg(acts, &ax->s_qkv, NLC, 3 * PT_D);
    pt_reg(acts, &ax->s_q, NLC, PT_D);
    pt_reg(acts, &ax->s_k, NLC, PT_D);
    pt_reg(acts, &ax->s_v, NLC, PT_D);
    pt_reg(acts, &ax->s_ctx, NLC, PT_D);
    pt_reg(acts, &ax->s_ffn, NLC, PT_FFN);
    if (train) {
        pt_reg(acts, &a->bv_in, N, PT_VAL_IN);
        pt_reg(acts, &a->bv1, N, 256);
        pt_reg(acts, &a->blind_v, N, 1);
        pt_reg(acts, &a->hand_lin, N, PT_NC);
        pt_reg(acts, &a->prize_logit, N, PT_LTOK);
        pt_reg(acts, &a->hand_tgt, N, PT_NC);
        pt_reg(acts, &a->prized, N, PT_NC);
        pt_reg(acts, &a->prize_cnt, 1, 0);
        pt_reg(acts, &ax->s_dffn, NLC, PT_FFN);
        pt_reg(acts, &ax->s_dq, NLC, PT_D);
        pt_reg(acts, &ax->s_dk, NLC, PT_D);
        pt_reg(acts, &ax->s_dv, NLC, PT_D);
        pt_reg(acts, &ax->s_dp, (long)N * PT_H * PT_LCRIT, PT_LCRIT);
        pt_reg(acts, &a->dx_a, NLC, PT_D);
        pt_reg(acts, &a->dx_b, NLC, PT_D);
        pt_reg(acts, &a->d_g, N, 128);
        pt_reg(acts, &a->d_dvec, N, 64);
        pt_reg(acts, &a->d_rg, N, PT_VAL_IN);
        pt_reg(acts, &ax->d_tokp, (long)N * PT_LTOK, PT_D);
        pt_reg(acts, &ax->d_tokcat, (long)N * PT_LTOK, PT_TOK_IN);
        pt_reg(acts, &ax->d_bc, N, PT_D);
        pt_reg(acts, &ax->d_bcin, N, PT_BCAST_IN);
        pt_reg(acts, &ax->d_decin, N, PT_DEC_IN);
        pt_reg(acts, &ax->d_gin, N, 128);
        pt_reg(acts, &ax->d_otok_in, (long)N * PT_NORC, PT_ORC_IN);
        pt_reg(acts, &ax->d_q, (long)N * PT_NOPT, PT_D);
        pt_reg(acts, &ax->d_qin, (long)N * PT_NOPT, PT_Q_IN);
        pt_reg(acts, &ax->d_scin, (long)N * PT_NOPT, PT_SCORE_IN);
        pt_reg(acts, &ax->d_out64, (long)N * PT_NOPT, 256);
        pt_reg(acts, &ax->d_ocard, (long)N * PT_NOPT, PT_DC);
        pt_reg(acts, &ax->d_oatt, (long)N * PT_NOPT, PT_DA);
        pt_reg(acts, &ax->d_val, N, 1);
        pt_reg(acts, &ax->d_vin, N, PT_VAL_IN);
        pt_reg(acts, &ax->d_v1, N, 256);
        pt_reg(acts, &ax->d_handlin, N, PT_NC);
        pt_reg(acts, &ax->d_prizelogit, N, PT_LTOK);
    }
}

static void scratch_bind(PtcgActX* ax) {
    ax->sc.ln = ax->s_ln.data;   ax->sc.qkv = ax->s_qkv.data;
    ax->sc.q = ax->s_q.data;     ax->sc.k = ax->s_k.data;
    ax->sc.v = ax->s_v.data;     ax->sc.ctx = ax->s_ctx.data;
    ax->sc.ffn = ax->s_ffn.data; ax->sc.dffn = ax->s_dffn.data;
    ax->sc.dq = ax->s_dq.data;   ax->sc.dk = ax->s_dk.data;
    ax->sc.dv = ax->s_dv.data;   ax->sc.dp = ax->s_dp.data;
}

static void k_prize_head_launch(PtcgAct* a, PtcgWeights* w, int N, cudaStream_t s);

static void pt_dump_stage(const char* dir, const char* name,
        const float* dev, long n) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(h, sizeof(float), n, f); fclose(f); }
    free(h);
}

// PTCG_SYNC_CHECK=1: sync + error-check after each forward stage
static int pt_sync_check_on(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PTCG_SYNC_CHECK"); v = e && e[0] == '1'; }
    return v;
}
#define PT_CK(stage) do { if (pt_sync_check_on()) { \
    cudaStreamSynchronize(s); cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { fprintf(stderr, "PTCG stage %s: %s (N=%d)\n", \
        stage, cudaGetErrorString(_e), N); abort(); } } } while (0)

// ----------------------------------------------------------------- forward
static void ptcg_forward_core(PtcgWeights* w, PtcgActX* ax, const float* obs,
        int N, int train, cudaStream_t s) {
    PtcgAct* a = &ax->a;
    a->obs_src = obs;
    scratch_bind(ax);
    pt_static_upload(s);
    tables_fwd(w, &a->tab, s);
    PT_CK("tables");
    const float* cv = a->tab.card_vec.data;
    const float* av = a->tab.att_tab.data;

    k_tok_cat<<<ptg((long)N * PT_LTOK * PT_TOK_IN), PTB, 0, s>>>(
        a->tok_cat.data, obs, cv, W("zone_emb"), W("owner_emb"), N);
    pt_linear(a->tok_cat.data, W("token_proj_w"), W("token_proj_b"),
        ax->tokp.data, N * PT_LTOK, PT_TOK_IN, PT_D, s);

    // global: g1a = lin1(glob); g_out = lin2(gelu(g1a)) — gelu into s_ln staging
    pt_slice_block<<<ptg((long)N * 64), PTB, 0, s>>>(ax->s_ln.data, obs, N,
        64, PT_OBS, PO_GLOB);
    pt_linear(ax->s_ln.data, W("global1_w"), W("global1_b"), ax->g1a.data,
        N, 64, 128, s);
    pt_gelu_fwd<<<ptg((long)N * 128), PTB, 0, s>>>(ax->s_qkv.data, ax->g1a.data, (long)N * 128);
    pt_linear(ax->s_qkv.data, W("global2_w"), W("global2_b"), a->g_out.data,
        N, 128, 128, s);

    // decision: d1 = lin1(dec_in); d_out = lin2(gelu(d1))
    k_dec_in<<<ptg((long)N * PT_DEC_IN), PTB, 0, s>>>(a->dec_in.data, obs,
        cv, av, W("sel_type_emb"), W("ctx_emb"), N);
    pt_linear(a->dec_in.data, W("dec1_w"), W("dec1_b"), a->d1.data, N,
        PT_DEC_IN, 64, s);
    pt_gelu_fwd<<<ptg((long)N * 64), PTB, 0, s>>>(ax->s_qkv.data, a->d1.data, (long)N * 64);
    pt_linear(ax->s_qkv.data, W("dec2_w"), W("dec2_b"), a->d_out.data, N, 64, 64, s);

    // bcast: bc_in = [g_out || d_out]; bc_out = bcast(bc_in)
    pt_copy_block<<<ptg((long)N * 128), PTB, 0, s>>>(a->bc_in.data,
        a->g_out.data, N, 128, PT_BCAST_IN, 0);
    pt_copy_block<<<ptg((long)N * 64), PTB, 0, s>>>(a->bc_in.data,
        a->d_out.data, N, 64, PT_BCAST_IN, 128);
    pt_linear(a->bc_in.data, W("bcast_w"), W("bcast_b"), a->bc_out.data,
        N, PT_BCAST_IN, PT_D, s);

    k_x0<<<ptg((long)N * PT_LSEQ * PT_D), PTB, 0, s>>>(a->x[0].data,
        a->keep.data, ax->tokp.data, a->bc_out.data, W("readout"), obs, N);
    PT_CK("x0");

    BlkW bw;
    static const char* bpre[4] = {"b0_", "b1_", "b2_", "b3_"};
    const float* x_last = nullptr;
    for (int i = 0; i < PT_LAYERS; i++) {
        blkw(w, bpre[i], &bw, nullptr);
        BlkAct* ba = train ? &a->blk[i] : &a->blk[0];
        const float* xi = train ? a->x[i].data : a->x[i % 2].data;
        float* xo = train ? a->x[i + 1].data : a->x[(i + 1) % 2].data;
        blk_fwd(&bw, ba, &ax->sc, xi, xo, a->keep.data, N, PT_LSEQ, s);
        PT_CK("trunk_blk");
        x_last = xo;
    }
    pt_ln_fwd<<<N * PT_LSEQ, 128, 0, s>>>(a->h.data, a->lnf_mean.data,
        a->lnf_rstd.data, x_last, W("lnf_w"), W("lnf_b"),
        N * PT_LSEQ, PT_D);

    // option scorer
    k_opt_gather<<<ptg((long)N * PT_NOPT * PT_DC), PTB, 0, s>>>(
        a->ocard_in.data, a->oatt_in.data, obs, cv, av, N);
    // opt_card/opt_attack linears into staging then q_in gather
    pt_linear(a->ocard_in.data, W("opt_card_w"), W("opt_card_b"),
        ax->s_q.data, N * PT_NOPT, PT_DC, 64, s);
    pt_linear(a->oatt_in.data, W("opt_attack_w"), W("opt_attack_b"),
        ax->s_k.data, N * PT_NOPT, PT_DA, 64, s);
    k_q_in<<<ptg((long)N * PT_NOPT * PT_Q_IN), PTB, 0, s>>>(a->q_in.data, obs,
        a->h.data, W("opt_type_emb"), ax->s_q.data, ax->s_k.data, N);
    pt_linear(a->q_in.data, W("q1_w"), W("q1_b"), a->q1a.data,
        N * PT_NOPT, PT_Q_IN, PT_D, s);
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * PT_D), PTB, 0, s>>>(ax->s_v.data,
        a->q1a.data, (long)N * PT_NOPT * PT_D);
    pt_linear(ax->s_v.data, W("q2_w"), W("q2_b"), a->q_vec.data,
        N * PT_NOPT, PT_D, PT_D, s);
    PT_CK("options_pre");
    k_sc_in<<<ptg((long)N * PT_NOPT * PT_SCORE_IN), PTB, 0, s>>>(a->sc_in.data,
        a->q_vec.data, a->h.data, a->g_out.data, a->d_out.data, N);
    pt_linear(a->sc_in.data, W("score1_w"), W("score1_b"), a->sc1.data,
        N * PT_NOPT, PT_SCORE_IN, 256, s);
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * 256), PTB, 0, s>>>(ax->s_ctx.data,
        a->sc1.data, (long)N * PT_NOPT * 256);
    pt_linear(ax->s_ctx.data, W("score2_w"), W("score2_b"), a->scores.data,
        N * PT_NOPT, 256, 1, s);

    // oracle critic top
    k_otok_in<<<ptg((long)N * PT_NORC * PT_ORC_IN), PTB, 0, s>>>(
        a->otok_in.data, a->ckeep.data, obs, cv, a->keep.data, N);
    pt_linear(a->otok_in.data, W("oracle_proj_w"), W("oracle_proj_b"),
        a->otok.data, N * PT_NORC, PT_ORC_IN, PT_D, s);
    k_xc<<<ptg((long)N * PT_LCRIT * PT_D), PTB, 0, s>>>(a->xc[0].data,
        a->h.data, a->otok.data, W("critic_readout"), N);
    PT_CK("otok");
    blkw(w, "c0_", &bw, nullptr);
    blk_fwd(&bw, &a->cblk, &ax->sc, a->xc[0].data, a->xc[1].data,
        a->ckeep.data, N, PT_LCRIT, s);
    PT_CK("critic_blk");
    k_val_in<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->ov_in.data,
        a->xc[1].data, PT_LCRIT, a->g_out.data, N);
    pt_linear(a->ov_in.data, W("oracle_v1_w"), W("oracle_v1_b"), a->ov1.data,
        N, PT_VAL_IN, 256, s);
    pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->ov1.data, (long)N * 256);
    pt_linear(ax->s_k.data, W("oracle_v2_w"), W("oracle_v2_b"), ax->ov_out.data,
        N, 256, 1, s);

    k_fuse_out<<<ptg((long)N * (PT_NOPT + 1)), PTB, 0, s>>>(a->out.data,
        a->scores.data, ax->ov_out.data, N);
    PT_CK("fuse");

    if (train) {
        // blind V + aux heads (targets consumed in backward)
        k_val_in<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->bv_in.data,
            a->h.data, PT_LSEQ, a->g_out.data, N);
        pt_linear(a->bv_in.data, W("value1_w"), W("value1_b"), a->bv1.data,
            N, PT_VAL_IN, 256, s);
        pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->bv1.data, (long)N * 256);
        pt_linear(ax->s_k.data, W("value2_w"), W("value2_b"), a->blind_v.data,
            N, 256, 1, s);
        pt_linear(a->bv_in.data, W("aux_hand_w"), W("aux_hand_b"),
            a->hand_lin.data, N, PT_VAL_IN, PT_NC, s);
        k_prize_head_launch(a, w, N, s);
    }

    // Stage-parity instrumentation: PTCG_DUMP_DIR=<dir> dumps every named
    // stage as raw f32 after the forward. native/tools/stage_parity.py reads these
    // against torch-hook captures to localize a divergence. Debug-only path.
    const char* dump_dir = getenv("PTCG_DUMP_DIR");
    if (dump_dir) {
        cudaStreamSynchronize(s);
        pt_dump_stage(dump_dir, "card_vec", cv, (long)PT_NC * PT_DC);
        pt_dump_stage(dump_dir, "att_tab", av, (long)PT_NA * PT_DA);
        pt_dump_stage(dump_dir, "sk_tab", a->tab.sk_tab.data, (long)PT_NS * PT_DC);
        pt_dump_stage(dump_dir, "a_mean", a->tab.a_mean.data, (long)PT_NC * 64);
        pt_dump_stage(dump_dir, "a_max", a->tab.a_max.data, (long)PT_NC * 64);
        pt_dump_stage(dump_dir, "s_mean", a->tab.s_mean.data, (long)PT_NC * PT_DC);
        pt_dump_stage(dump_dir, "s_max", a->tab.s_max.data, (long)PT_NC * PT_DC);
        pt_dump_stage(dump_dir, "card_in", a->tab.card_in.data, (long)PT_NC * PT_CARD_IN);
        pt_dump_stage(dump_dir, "card_ln", a->tab.card_ln.data, (long)PT_NC * PT_CARD_IN);
        pt_dump_stage(dump_dir, "card_h1", a->tab.card_h1.data, (long)PT_NC * 256);
        pt_dump_stage(dump_dir, "tok_cat", a->tok_cat.data, (long)N * PT_LTOK * PT_TOK_IN);
        pt_dump_stage(dump_dir, "tokp", ax->tokp.data, (long)N * PT_LTOK * PT_D);
        pt_dump_stage(dump_dir, "dec_in", a->dec_in.data, (long)N * PT_DEC_IN);
        pt_dump_stage(dump_dir, "g_out", a->g_out.data, (long)N * 128);
        pt_dump_stage(dump_dir, "d_out", a->d_out.data, (long)N * 64);
        pt_dump_stage(dump_dir, "bc_out", a->bc_out.data, (long)N * PT_D);
        pt_dump_stage(dump_dir, "keep", a->keep.data, (long)N * PT_LSEQ);
        pt_dump_stage(dump_dir, "x0", a->x[0].data, (long)N * PT_LSEQ * PT_D);
        if (train) {
            for (int i = 1; i <= PT_LAYERS; i++) {
                char nm[8];
                snprintf(nm, sizeof nm, "x%d", i);
                pt_dump_stage(dump_dir, nm, a->x[i].data, (long)N * PT_LSEQ * PT_D);
            }
        }
        pt_dump_stage(dump_dir, "h", a->h.data, (long)N * PT_LSEQ * PT_D);
        pt_dump_stage(dump_dir, "q_in", a->q_in.data, (long)N * PT_NOPT * PT_Q_IN);
        pt_dump_stage(dump_dir, "q_vec", a->q_vec.data, (long)N * PT_NOPT * PT_D);
        pt_dump_stage(dump_dir, "sc_in", a->sc_in.data, (long)N * PT_NOPT * PT_SCORE_IN);
        pt_dump_stage(dump_dir, "out", a->out.data, (long)N * (PT_NOPT + 1));
    }
}

// prize head needs a strided pass over h[:,1:,:] -> (N, LTOK)
__global__ void k_prize_head(float* logit, const float* h, const float* w,
        const float* b, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK) return;
    long n = i / PT_LTOK;
    int l = (int)(i % PT_LTOK);
    const float* hv = h + ((long)n * PT_LSEQ + 1 + l) * PT_D;
    float s = b[0];
    for (int d = 0; d < PT_D; d++) s += hv[d] * w[d];
    logit[i] = s;
}

__global__ void k_prize_head_bwd(float* d_h, float* d_w, float* d_b,
        const float* d_logit, const float* h, const float* w, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK) return;
    long n = i / PT_LTOK;
    int l = (int)(i % PT_LTOK);
    float g = d_logit[i];
    if (g == 0.f) return;
    const float* hv = h + ((long)n * PT_LSEQ + 1 + l) * PT_D;
    float* dh = d_h + ((long)n * PT_LSEQ + 1 + l) * PT_D;
    for (int d = 0; d < PT_D; d++) {
        atomicAdd(&dh[d], g * w[d]);
        atomicAdd(&d_w[d], g * hv[d]);
    }
    atomicAdd(d_b, g);
}

#include "ptcg_model4.cu"
