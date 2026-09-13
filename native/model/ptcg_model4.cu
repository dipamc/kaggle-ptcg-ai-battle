// Part 4: backward driver, weight loading, encoder/decoder/network vtables.

static void k_prize_head_launch(PtcgAct* a, PtcgWeights* w, int N, cudaStream_t s) {
    k_prize_head<<<ptg((long)N * PT_LTOK), PTB, 0, s>>>(a->prize_logit.data,
        a->h.data, W("aux_prize_w"), W("aux_prize_b"), N);
}

// ---------------------------------------------------------------- backward
// grad (N, 65) = [d_scores || d_value] from the passthrough decoder.
static void ptcg_backward_core(PtcgWeights* w, PtcgActX* ax, const float* grad,
        int N, cudaStream_t s) {
    PtcgAct* a = &ax->a;
    const float* obs = a->obs_src;
    scratch_bind(ax);
    // zero the whole param-grad span (g[] are contiguous in the grads arena)
    {
        char* lo = (char*)w->g[0].data;
        char* hi = (char*)w->g[PT_NPARAM - 1].data
                 + numel(w->g[PT_NPARAM - 1].shape) * sizeof(float);
        cudaMemsetAsync(lo, 0, hi - lo, s);
    }
    long NL = (long)N * PT_LSEQ;
    pt_zero<<<ptg(NL * PT_D), PTB, 0, s>>>(a->dx_a.data, NL * PT_D);   // d_h
    pt_zero<<<ptg((long)N * 128), PTB, 0, s>>>(a->d_g.data, (long)N * 128);
    pt_zero<<<ptg((long)N * 64), PTB, 0, s>>>(a->d_dvec.data, (long)N * 64);
    pt_zero<<<ptg((long)PT_NC * PT_DC), PTB, 0, s>>>(a->tab.d_card_vec.data, (long)PT_NC * PT_DC);
    pt_zero<<<ptg((long)PT_NA * 64), PTB, 0, s>>>(a->tab.d_att_tab.data, (long)PT_NA * 64);
    pt_zero<<<ptg((long)PT_NS * PT_DC), PTB, 0, s>>>(a->tab.d_sk_tab.data, (long)PT_NS * PT_DC);
    // NOTE d_h lives in dx_a (N,LSEQ,D). d_rg = grad into [r||g] pairs.

    // ---- aux losses (targets from obs + bound returns) ----
    pt_zero<<<ptg((long)N * PT_NC), PTB, 0, s>>>(a->hand_tgt.data, (long)N * PT_NC);
    pt_zero<<<ptg((long)N * PT_NC), PTB, 0, s>>>(a->prized.data, (long)N * PT_NC);
    k_aux_targets<<<ptg((long)N * 32), PTB, 0, s>>>(a->hand_tgt.data,
        a->prized.data, obs, N);
    // blind V: d = coef*(v - ret)/N -> value_mlp bwd -> d_rg
    k_blind_grad<<<ptg(N), PTB, 0, s>>>(ax->d_val.data, a->blind_v.data,
        g_mb_returns, N);
    pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->bv1.data, (long)N * 256);
    pt_linear_bwd(ax->s_k.data, W("value2_w"), ax->d_val.data, ax->d_v1.data,
        G("value2_w"), G("value2_b"), N, 256, 1, s);
    pt_gelu_bwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->d_v1.data, ax->d_v1.data,
        a->bv1.data, (long)N * 256);
    pt_linear_bwd(a->bv_in.data, W("value1_w"), ax->d_v1.data, a->d_rg.data,
        G("value1_w"), G("value1_b"), N, PT_VAL_IN, 256, s);
    // hand aux (shares bv_in input): accumulate into d_rg
    k_hand_grad<<<ptg((long)N * PT_NC), PTB, 0, s>>>(ax->d_handlin.data,
        a->hand_lin.data, a->hand_tgt.data, (long)N * PT_NC);
    pt_linear_bwd(a->bv_in.data, W("aux_hand_w"), ax->d_handlin.data,
        ax->d_vin.data, G("aux_hand_w"), G("aux_hand_b"), N, PT_VAL_IN, PT_NC, s);
    pt_add<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->d_rg.data,
        ax->d_vin.data, (long)N * PT_VAL_IN);
    // d_rg -> d_h[:,0] + d_g
    k_val_in_bwd<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->dx_a.data, PT_LSEQ,
        a->d_g.data, a->d_rg.data, N);
    // prize aux -> d_h token positions
    pt_zero<<<1, 1, 0, s>>>(a->prize_cnt.data, 1);
    k_prize_count<<<64, PTB, 0, s>>>(a->prize_cnt.data, obs, N);
    k_prize_grad<<<ptg((long)N * PT_LTOK), PTB, 0, s>>>(ax->d_prizelogit.data,
        a->prize_logit.data, a->prized.data, obs, a->prize_cnt.data, N);
    k_prize_head_bwd<<<ptg((long)N * PT_LTOK), PTB, 0, s>>>(a->dx_a.data,
        G("aux_prize_w"), G("aux_prize_b"), ax->d_prizelogit.data, a->h.data,
        W("aux_prize_w"), N);

    // ---- oracle value head (PPO value grad = grad[:,64]) ----
    pt_slice_block<<<ptg(N), PTB, 0, s>>>(ax->d_val.data, grad, N, 1,
        PT_NOPT + 1, PT_NOPT);
    pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->ov1.data, (long)N * 256);
    pt_linear_bwd(ax->s_k.data, W("oracle_v2_w"), ax->d_val.data, ax->d_v1.data,
        G("oracle_v2_w"), G("oracle_v2_b"), N, 256, 1, s);
    pt_gelu_bwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->d_v1.data, ax->d_v1.data,
        a->ov1.data, (long)N * 256);
    pt_linear_bwd(a->ov_in.data, W("oracle_v1_w"), ax->d_v1.data,
        a->d_rg.data, G("oracle_v1_w"), G("oracle_v1_b"), N, PT_VAL_IN, 256, s);
    // d_rg -> d_xc_out[:,0] + d_g ; critic block bwd -> d_h/d_otok/creadout
    long NLC = (long)N * PT_LCRIT;
    pt_zero<<<ptg(NLC * PT_D), PTB, 0, s>>>(a->dx_b.data, NLC * PT_D);
    k_val_in_bwd<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->dx_b.data, PT_LCRIT,
        a->d_g.data, a->d_rg.data, N);
    {
        BlkW bw; BlkG bg;
        blkw(w, "c0_", &bw, &bg);
        blk_bwd(&bw, &bg, &a->cblk, &ax->sc, a->xc[0].data, a->dx_b.data,
            ax->s_ln.data /*scratch as dxB2*/, N, PT_LCRIT, s);
    }
    pt_zero<<<ptg((long)N * PT_NORC * PT_D), PTB, 0, s>>>(a->otok.data,
        (long)N * PT_NORC * PT_D);   // reuse otok as d_otok
    k_xc_bwd<<<ptg(NLC * PT_D), PTB, 0, s>>>(G("critic_readout"), a->dx_a.data,
        a->otok.data, a->dx_b.data, N);
    pt_linear_bwd(a->otok_in.data, W("oracle_proj_w"), a->otok.data,
        ax->d_otok_in.data, G("oracle_proj_w"), G("oracle_proj_b"),
        N * PT_NORC, PT_ORC_IN, PT_D, s);
    k_otok_bwd<<<ptg((long)N * PT_NORC * PT_DC), PTB, 0, s>>>(
        a->tab.d_card_vec.data, ax->d_otok_in.data, obs, N);

    // ---- option scorer (d_scores = grad[:, :64]) ----
    pt_slice_block<<<ptg((long)N * PT_NOPT), PTB, 0, s>>>(ax->d_val.data /*reuse*/,
        grad, N, PT_NOPT, PT_NOPT + 1, 0);
    // NOTE d_val is (N,1); need (N,64) buffer — use d_q's first N*64 floats? no:
    // use ax->d_ocard (N*NOPT, DC) start as staging (N,64) — sized fine.
    pt_slice_block<<<ptg((long)N * PT_NOPT), PTB, 0, s>>>(ax->d_ocard.data,
        grad, N, PT_NOPT, PT_NOPT + 1, 0);
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * 256), PTB, 0, s>>>(ax->s_ctx.data,
        a->sc1.data, (long)N * PT_NOPT * 256);
    pt_linear_bwd(ax->s_ctx.data, W("score2_w"), ax->d_ocard.data,
        ax->d_out64.data, G("score2_w"), G("score2_b"), N * PT_NOPT, 256, 1, s);
    pt_gelu_bwd<<<ptg((long)N * PT_NOPT * 256), PTB, 0, s>>>(ax->d_out64.data,
        ax->d_out64.data, a->sc1.data, (long)N * PT_NOPT * 256);
    pt_linear_bwd(a->sc_in.data, W("score1_w"), ax->d_out64.data,
        ax->d_scin.data, G("score1_w"), G("score1_b"),
        N * PT_NOPT, PT_SCORE_IN, 256, s);
    k_sc_in_bwd<<<ptg((long)N * PT_NOPT * PT_D), PTB, 0, s>>>(ax->d_q.data,
        a->dx_a.data, a->d_g.data, a->d_dvec.data, ax->d_scin.data,
        a->q_vec.data, a->h.data, N);
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * PT_D), PTB, 0, s>>>(ax->s_v.data,
        a->q1a.data, (long)N * PT_NOPT * PT_D);
    pt_linear_bwd(ax->s_v.data, W("q2_w"), ax->d_q.data, ax->d_q.data,
        G("q2_w"), G("q2_b"), N * PT_NOPT, PT_D, PT_D, s);
    pt_gelu_bwd<<<ptg((long)N * PT_NOPT * PT_D), PTB, 0, s>>>(ax->d_q.data,
        ax->d_q.data, a->q1a.data, (long)N * PT_NOPT * PT_D);
    pt_linear_bwd(a->q_in.data, W("q1_w"), ax->d_q.data, ax->d_qin.data,
        G("q1_w"), G("q1_b"), N * PT_NOPT, PT_Q_IN, PT_D, s);
    k_q_in_bwd<<<ptg((long)N * PT_NOPT * PT_Q_IN), PTB, 0, s>>>(
        G("opt_type_emb"), a->dx_a.data, ax->d_ocard.data, ax->d_oatt.data,
        ax->d_qin.data, obs, N);
    pt_linear_bwd(a->ocard_in.data, W("opt_card_w"), ax->d_ocard.data,
        ax->d_ocard.data, G("opt_card_w"), G("opt_card_b"),
        N * PT_NOPT, PT_DC, 64, s);
    pt_linear_bwd(a->oatt_in.data, W("opt_attack_w"), ax->d_oatt.data,
        ax->d_oatt.data, G("opt_attack_w"), G("opt_attack_b"),
        N * PT_NOPT, PT_DA, 64, s);
    k_opt_gather_bwd<<<ptg((long)N * PT_NOPT * PT_DC), PTB, 0, s>>>(
        a->tab.d_card_vec.data, a->tab.d_att_tab.data, ax->d_ocard.data,
        ax->d_oatt.data, obs, N);

    // ---- ln_f + trunk ----
    pt_ln_bwd<<<(int)NL, 128, 0, s>>>(a->dx_b.data, G("lnf_w"), G("lnf_b"),
        a->dx_a.data, a->x[PT_LAYERS].data, W("lnf_w"), a->lnf_mean.data,
        a->lnf_rstd.data, (int)NL, PT_D);
    // dx_b now holds d_x4 in (N,LSEQ,D) region; move into dx_a for blk loop
    pt_copy<<<ptg(NL * PT_D), PTB, 0, s>>>(a->dx_a.data, a->dx_b.data, NL * PT_D);
    {
        BlkW bw; BlkG bg;
        static const char* bpre[4] = {"b0_", "b1_", "b2_", "b3_"};
        for (int i = PT_LAYERS - 1; i >= 0; i--) {
            blkw(w, bpre[i], &bw, &bg);
            blk_bwd(&bw, &bg, &a->blk[i], &ax->sc, a->x[i].data,
                a->dx_a.data, a->dx_b.data, N, PT_LSEQ, s);
        }
    }
    // ---- x0 split: readout / token_proj / bcast ----
    pt_zero<<<ptg((long)N * PT_D), PTB, 0, s>>>(ax->d_bc.data, (long)N * PT_D);
    k_x0_bwd<<<ptg(NL * PT_D), PTB, 0, s>>>(G("readout"), ax->d_tokp.data,
        ax->d_bc.data, a->dx_a.data, N);
    pt_linear_bwd(a->tok_cat.data, W("token_proj_w"), ax->d_tokp.data,
        ax->d_tokcat.data, G("token_proj_w"), G("token_proj_b"),
        N * PT_LTOK, PT_TOK_IN, PT_D, s);
    k_tok_cat_bwd<<<ptg((long)N * PT_LTOK * PT_TOK_IN), PTB, 0, s>>>(
        a->tab.d_card_vec.data, G("zone_emb"), G("owner_emb"),
        ax->d_tokcat.data, obs, N);
    pt_linear_bwd(a->bc_in.data, W("bcast_w"), ax->d_bc.data, ax->d_bcin.data,
        G("bcast_w"), G("bcast_b"), N, PT_BCAST_IN, PT_D, s);
    pt_slice_block_add<<<ptg((long)N * 128), PTB, 0, s>>>(a->d_g.data,
        ax->d_bcin.data, N, 128, PT_BCAST_IN, 0);
    pt_slice_block_add<<<ptg((long)N * 64), PTB, 0, s>>>(a->d_dvec.data,
        ax->d_bcin.data, N, 64, PT_BCAST_IN, 128);
    // ---- decision mlp ----
    pt_gelu_fwd<<<ptg((long)N * 64), PTB, 0, s>>>(ax->s_qkv.data, a->d1.data, (long)N * 64);
    pt_linear_bwd(ax->s_qkv.data, W("dec2_w"), a->d_dvec.data, a->d_dvec.data,
        G("dec2_w"), G("dec2_b"), N, 64, 64, s);
    pt_gelu_bwd<<<ptg((long)N * 64), PTB, 0, s>>>(a->d_dvec.data, a->d_dvec.data,
        a->d1.data, (long)N * 64);
    pt_linear_bwd(a->dec_in.data, W("dec1_w"), a->d_dvec.data, ax->d_decin.data,
        G("dec1_w"), G("dec1_b"), N, PT_DEC_IN, 64, s);
    k_dec_in_bwd<<<ptg((long)N * PT_DEC_IN), PTB, 0, s>>>(a->tab.d_card_vec.data,
        a->tab.d_att_tab.data, G("sel_type_emb"), G("ctx_emb"),
        ax->d_decin.data, obs, N);
    // ---- global mlp ----
    pt_gelu_fwd<<<ptg((long)N * 128), PTB, 0, s>>>(ax->s_qkv.data, ax->g1a.data, (long)N * 128);
    pt_linear_bwd(ax->s_qkv.data, W("global2_w"), a->d_g.data, a->d_g.data,
        G("global2_w"), G("global2_b"), N, 128, 128, s);
    pt_gelu_bwd<<<ptg((long)N * 128), PTB, 0, s>>>(a->d_g.data, a->d_g.data,
        ax->g1a.data, (long)N * 128);
    pt_slice_block<<<ptg((long)N * 64), PTB, 0, s>>>(ax->s_ln.data, obs, N,
        64, PT_OBS, PO_GLOB);
    pt_linear_bwd(ax->s_ln.data, W("global1_w"), a->d_g.data, nullptr,
        G("global1_w"), G("global1_b"), N, 64, 128, s);
    // ---- static tables chain ----
    tables_bwd(w, &a->tab, s);
}

// ------------------------------------------------------------ weight load
// Blob on disk is PACKED (torch-converter order, sum-of-numel floats); the
// params arena 16B-aligns each tensor. Copy per-param through the REGISTERED
// device pointers — no alignment-rule duplication here.
extern "C" void ptcg_load_init_weights_regs(PrecisionTensor* p, int n_params,
        long n_total, cudaStream_t s) {
    const char* path = getenv("PTCG_INIT_WEIGHTS");
    if (!path) {
        fprintf(stderr, "ptcg_model: PTCG_INIT_WEIGHTS not set — refusing "
            "random init (torch-matching init required)\n");
        abort();
    }
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ptcg_model: cannot open %s\n", path); abort(); }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (bytes != n_total * (long)sizeof(float)) {
        fprintf(stderr, "ptcg_model: %s has %ld bytes, expected %ld\n",
            path, bytes, n_total * (long)sizeof(float));
        abort();
    }
    float* host = (float*)malloc(bytes);
    if (fread(host, 1, bytes, f) != (size_t)bytes) abort();
    fclose(f);
    long off = 0;
    for (int i = 0; i < n_params; i++) {
        long ne = numel(p[i].shape);
        cudaMemcpyAsync(p[i].data, host + off, ne * sizeof(float),
                        cudaMemcpyHostToDevice, s);
        off += ne;
    }
    cudaStreamSynchronize(s);
    free(host);
    if (off != n_total) {
        fprintf(stderr, "ptcg_model: consumed %ld of %ld blob floats\n",
            off, n_total);
        abort();
    }
    printf("ptcg_model: loaded init weights from %s (%ld params)\n",
        path, n_total);
}

// =============================================================== vtables
struct PtcgEncCtx { PtcgWeights w; };   // per-Policy weights holder

static void* ptcg_enc_create_weights(void* self) {
    (void)self;
    return calloc(1, sizeof(PtcgWeights));
}

static void ptcg_enc_reg_params(void* wv, Allocator* alloc) {
    PtcgWeights* w = (PtcgWeights*)wv;
    for (int i = 0; i < PT_NPARAM; i++) {
        w->p[i].shape[0] = PT_PARAMS[i].r;
        w->p[i].shape[1] = PT_PARAMS[i].c > 0 ? PT_PARAMS[i].c : 0;
        alloc_register(alloc, &w->p[i]);
    }
}

static void ptcg_enc_init_weights(void* wv, ulong* seed, cudaStream_t stream) {
    PtcgWeights* w = (PtcgWeights*)wv;
    (void)seed;
    long total = 0;
    for (int i = 0; i < PT_NPARAM; i++) total += numel(w->p[i].shape);
    ptcg_load_init_weights_regs(w->p, PT_NPARAM, total, stream);
}

static void ptcg_enc_reg_train(void* wv, void* actv, Allocator* acts,
        Allocator* grads, int B_TT) {
    ptcg_reg((PtcgActX*)actv, (PtcgWeights*)wv, acts, grads, B_TT, 1);
}

static void ptcg_enc_reg_rollout(void* wv, void* actv, Allocator* alloc, int B) {
    ptcg_reg((PtcgActX*)actv, (PtcgWeights*)wv, alloc, nullptr, B, 0);
}

static PrecisionTensor ptcg_enc_forward(void* wv, void* actv,
        PrecisionTensor input, cudaStream_t stream) {
    PtcgActX* ax = (PtcgActX*)actv;
    int N = (int)input.shape[0];
    ptcg_forward_core((PtcgWeights*)wv, ax, input.data, N, ax->a.train, stream);
    return ax->a.out;
}

static void ptcg_enc_backward(void* wv, void* actv, PrecisionTensor grad,
        cudaStream_t stream) {
    PtcgActX* ax = (PtcgActX*)actv;
    ptcg_backward_core((PtcgWeights*)wv, ax, grad.data, ax->a.N, stream);
}

static void ptcg_enc_free_weights(void* wv) { free(wv); }
static void ptcg_enc_free_activations(void* actv) { free(actv); }

// passthrough decoder: forward = identity; backward assembles
// [grad_logits || grad_value] and returns it.
struct PtcgDecActs { PrecisionTensor grad_out; int B_TT; };

static PrecisionTensor ptcg_dec_forward(void* w, void* actv,
        PrecisionTensor input, cudaStream_t stream) {
    (void)w; (void)actv; (void)stream;
    return input;
}

static PrecisionTensor ptcg_dec_backward(void* w, void* actv,
        FloatTensor grad_logits, FloatTensor grad_logstd, FloatTensor grad_value,
        cudaStream_t stream) {
    (void)w; (void)grad_logstd;
    PtcgDecActs* a = (PtcgDecActs*)actv;
    int B_TT = a->B_TT;
    int od1 = PT_NOPT + 1;
    assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
        a->grad_out.data, grad_logits.data, grad_value.data, B_TT, PT_NOPT, od1);
    return a->grad_out;
}

static void ptcg_dec_init_weights(void* w, ulong* seed, cudaStream_t s) {
    (void)w; (void)seed; (void)s;
}
static void ptcg_dec_reg_params(void* w, Allocator* a) { (void)w; (void)a; }
static void ptcg_dec_reg_train(void* w, void* actv, Allocator* acts,
        Allocator* grads, int B_TT) {
    (void)w; (void)grads;
    PtcgDecActs* a = (PtcgDecActs*)actv;
    a->B_TT = B_TT;
    a->grad_out = (PrecisionTensor){.shape = {B_TT, PT_NOPT + 1}};
    alloc_register(acts, &a->grad_out);
}
static void ptcg_dec_reg_rollout(void* w, void* actv, Allocator* a, int B) {
    (void)w; (void)actv; (void)a; (void)B;
}
static void* ptcg_dec_create_weights(void* self) { (void)self; return calloc(1, 8); }
static void ptcg_dec_free_weights(void* w) { free(w); }
static void ptcg_dec_free_activations(void* a) { free(a); }

// identity network (our net is stateless; MinGRU bypassed)
static PrecisionTensor ptcg_net_forward(void* w, PrecisionTensor x,
        PrecisionTensor state, void* actv, cudaStream_t s) {
    (void)w; (void)state; (void)actv; (void)s;
    return x;
}
static PrecisionTensor ptcg_net_forward_train(void* w, PrecisionTensor x,
        PrecisionTensor state, void* actv, cudaStream_t s) {
    (void)w; (void)state; (void)actv; (void)s;
    return x;
}
static PrecisionTensor ptcg_net_backward(void* w, PrecisionTensor grad,
        void* actv, cudaStream_t s) {
    (void)w; (void)actv; (void)s;
    return grad;
}
static void ptcg_net_init_weights(void* w, ulong* seed, cudaStream_t s) {
    (void)w; (void)seed; (void)s;
}
static void ptcg_net_reg_params(void* w, Allocator* a) { (void)w; (void)a; }
static void ptcg_net_reg_train(void* w, void* actv, Allocator* acts,
        Allocator* grads, int B_TT) { (void)w; (void)actv; (void)acts; (void)grads; (void)B_TT; }
static void ptcg_net_reg_rollout(void* w, void* actv, Allocator* a, int B) {
    (void)w; (void)actv; (void)a; (void)B;
}
static void* ptcg_net_create_weights(void* self) { (void)self; return calloc(1, 8); }
static void ptcg_net_free_weights(void* w) { free(w); }
static void ptcg_net_free_activations(void* a) { free(a); }
