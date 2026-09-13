// Part 2: forward/backward drivers, aux losses, vtables, negamax advantage.

// re-apply a saved layernorm (exact recompute from input + saved stats)
__global__ void k_ln_apply(float* y, const float* x, const float* mean,
        const float* rstd, const float* w, const float* b, long N, int D) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= N * D) return;
    long n = i / D;
    int d = (int)(i % D);
    y[i] = (x[i] - mean[n]) * rstd[n] * w[d] + b[d];
}

// pool backward: scatter d_mean/d_max into d_tab through gather indices.
// d_mean/d_max are strided views into d_card_in (stride, offset).
__global__ void k_pool_bwd(float* d_tab, const float* dsrc, long stride,
        int off_mean, int off_max, const float* amax, const int* idx,
        int NC, int K, int E) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)NC * E) return;
    int n = (int)(i / E), e = (int)(i % E);
    int cnt = 0;
    for (int k = 0; k < K; k++) if (idx[n * K + k] > 0) cnt++;
    float dmean = dsrc[n * stride + off_mean + e];
    float dmax = dsrc[n * stride + off_max + e];
    float dm = dmean / (float)(cnt > 0 ? cnt : 1);
    int am = (int)amax[i];
    for (int k = 0; k < K; k++) {
        int id = idx[n * K + k];
        if (id <= 0) continue;                  // masked: a*mask kills grad
        float g = dm + (k == am ? dmax : 0.f);
        if (g != 0.f) atomicAdd(&d_tab[(long)id * E + e], g);
    }
}

// scatter grads of token-cat back: card ids, tool ids, zone/owner embeddings
__global__ void k_tok_cat_bwd(float* d_cv, float* d_zemb, float* d_oemb,
        const float* d_cat, const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK * PT_TOK_IN) return;
    int c = (int)(i % PT_TOK_IN);
    long nl = i / PT_TOK_IN;
    int l = (int)(nl % PT_LTOK), n = (int)(nl / PT_LTOK);
    float g = d_cat[i];
    if (g == 0.f) return;
    const float* ti = obs + (long)n * PT_OBS + PO_TOKI + l * 5;
    if (c < 128) {
        int cid = (int)ti[0];
        cid = cid < 0 ? 0 : (cid >= PT_NC ? PT_NC - 1 : cid);
        atomicAdd(&d_cv[(long)cid * PT_DC + c], g);
    } else if (c < 256) {
        int e = c - 128;
        int t0 = (int)ti[3], t1 = (int)ti[4];
        if (t0 > 0) atomicAdd(&d_cv[(long)(t0 >= PT_NC ? PT_NC-1 : t0) * PT_DC + e], g);
        if (t1 > 0) atomicAdd(&d_cv[(long)(t1 >= PT_NC ? PT_NC-1 : t1) * PT_DC + e], g);
    } else if (c < 312) {
        // tok_f: obs input, no grad
    } else if (c < 320) {
        int z = (int)ti[1]; z = z < 0 ? 0 : (z > 15 ? 15 : z);
        atomicAdd(&d_zemb[z * 8 + (c - 312)], g);
    } else {
        int o = (int)ti[2]; o = o < 0 ? 0 : (o > 3 ? 3 : o);
        atomicAdd(&d_oemb[o * 4 + (c - 320)], g);
    }
}

__global__ void k_dec_in_bwd(float* d_cv, float* d_av, float* d_semb,
        float* d_cemb, const float* d_in, const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_DEC_IN) return;
    int n = (int)(i / PT_DEC_IN), c = (int)(i % PT_DEC_IN);
    float g = d_in[i];
    if (g == 0.f) return;
    const float* di = obs + (long)n * PT_OBS + PO_DECI;
    if (c < 8) { int s = (int)di[0]; s = s < 0 ? 0 : (s > 28 ? 28 : s); atomicAdd(&d_semb[s * 8 + c], g); }
    else if (c < 40) { int s = (int)di[1]; s = s < 0 ? 0 : (s > 64 ? 64 : s); atomicAdd(&d_cemb[s * 32 + (c - 8)], g); }
    else if (c < 168) atomicAdd(&d_cv[(long)CLC((int)di[2]) * PT_DC + (c - 40)], g);
    else if (c < 296) atomicAdd(&d_cv[(long)CLC((int)di[3]) * PT_DC + (c - 168)], g);
    else if (c < 360) atomicAdd(&d_av[(long)CLA((int)di[4]) * PT_DA + (c - 296)], g);
    else if (c < 424) atomicAdd(&d_av[(long)CLA((int)di[5]) * PT_DA + (c - 360)], g);
    else if (c < 552) atomicAdd(&d_cv[(long)CLC((int)di[6]) * PT_DC + (c - 424)], g);
    else if (c < 680) atomicAdd(&d_cv[(long)CLC((int)di[7]) * PT_DC + (c - 552)], g);
}

// q_in backward: type emb, h-pointers, ocard/oatt slices (linear inputs), optf none
__global__ void k_q_in_bwd(float* d_temb, float* d_h, float* d_ocard,
        float* d_oatt, const float* d_qin, const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT * PT_Q_IN) return;
    int c = (int)(i % PT_Q_IN);
    long nj = i / PT_Q_IN;
    int j = (int)(nj % PT_NOPT), n = (int)(nj / PT_NOPT);
    float g = d_qin[i];
    const float* oi = obs + (long)n * PT_OBS + PO_OPTI + j * 6;
    if (c < 16) {
        if (g != 0.f) {
            int t = (int)oi[0]; t = t < 0 ? 0 : (t > 32 ? 32 : t);
            atomicAdd(&d_temb[t * 16 + c], g);
        }
    } else if (c < PT_QO_H2) {
        int p = (int)oi[1];
        if (g != 0.f && p > 0 && p <= PT_LTOK)
            atomicAdd(&d_h[((long)n * PT_LSEQ + p) * PT_D + (c - PT_QO_H1)], g);
    } else if (c < PT_QO_OC) {
        int p = (int)oi[2];
        if (g != 0.f && p > 0 && p <= PT_LTOK)
            atomicAdd(&d_h[((long)n * PT_LSEQ + p) * PT_D + (c - PT_QO_H2)], g);
    } else if (c < PT_QO_OA) {
        d_ocard[nj * 64 + (c - PT_QO_OC)] = g;
    } else if (c < PT_QO_OF) {
        d_oatt[nj * 64 + (c - PT_QO_OA)] = g;
    }
}

// sc_in backward: d_q (+prod), d_r/d_g/d_dvec accumulated over options
__global__ void k_sc_in_bwd(float* d_q, float* d_h, float* d_g, float* d_dvec,
        const float* d_scin, const float* q, const float* h, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT * PT_D) return;   // one thread per (n, j, d<PT_D)
    int d = (int)(i % PT_D);
    long nj = i / PT_D;
    int n = (int)(nj / PT_NOPT);
    const float* dsc = d_scin + nj * PT_SCORE_IN;
    const float* r = h + (long)n * PT_LSEQ * PT_D;
    float dq = dsc[d] + dsc[PT_SO_QR + d] * r[d];
    d_q[nj * PT_D + d] = dq;
    float dr = dsc[PT_SO_R + d] + dsc[PT_SO_QR + d] * q[nj * PT_D + d];
    atomicAdd(&d_h[(long)n * PT_LSEQ * PT_D + d], dr);      // r = h[:,0]
    // d_g is PT_G wide, NOT PT_D. The thread index runs to PT_D, so this
    // needs a bound — at d128 the two happened to be equal and the guard was
    // invisible; at any PT_D > PT_G it would write past the end of d_g.
    if (d < PT_G) atomicAdd(&d_g[(long)n * PT_G + d], dsc[PT_SO_G + d]);
    if (d < 64) atomicAdd(&d_dvec[(long)n * 64 + d], dsc[PT_SO_DV + d]);
}

__global__ void k_opt_gather_bwd(float* d_cv, float* d_av, const float* d_ocard,
        const float* d_oatt, const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT * PT_DC) return;
    int c = (int)(i % PT_DC);
    long nj = i / PT_DC;
    int j = (int)(nj % PT_NOPT), n = (int)(nj / PT_NOPT);
    const float* oi = obs + (long)n * PT_OBS + PO_OPTI + j * 6;
    float g = d_ocard[i];
    if (g != 0.f) {
        int cid = (int)oi[3];
        cid = cid < 0 ? 0 : (cid >= PT_NC ? PT_NC - 1 : cid);
        atomicAdd(&d_cv[(long)cid * PT_DC + c], g);
    }
    if (c < PT_DA) {
        float ga = d_oatt[nj * PT_DA + c];
        if (ga != 0.f) {
            int aid = (int)oi[4];
            aid = aid < 0 ? 0 : (aid >= PT_NA ? PT_NA - 1 : aid);
            atomicAdd(&d_av[(long)aid * PT_DA + c], ga);
        }
    }
}

__global__ void k_otok_bwd(float* d_cv, const float* d_otok_in,
        const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NORC * PT_DC) return;    // card_vec part only
    int c = (int)(i % PT_DC);
    long nk = i / PT_DC;
    int k = (int)(nk % PT_NORC), n = (int)(nk / PT_NORC);
    float g = d_otok_in[nk * PT_ORC_IN + c];
    if (g == 0.f) return;
    const float* o = obs + (long)n * PT_OBS + PO_ORC;
    int id = k < 24 ? (int)o[k * 2]
           : k < 48 ? (int)o[48 + (k - 24) * 3]
                    : (int)o[120 + (k - 48) * 2];
    id = id < 0 ? 0 : (id >= PT_NC ? PT_NC - 1 : id);
    atomicAdd(&d_cv[(long)id * PT_DC + c], g);
}

// split critic-input grad back to creadout / h / otok
__global__ void k_xc_bwd(float* d_cread, float* d_h, float* d_otok,
        const float* d_xc, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LCRIT * PT_D) return;
    int d = (int)(i % PT_D);
    long nl = i / PT_D;
    int l = (int)(nl % PT_LCRIT), n = (int)(nl / PT_LCRIT);
    float g = d_xc[i];
    if (l == 0) { if (g != 0.f) atomicAdd(&d_cread[d], g); }
    else if (l <= PT_LSEQ) d_h[((long)n * PT_LSEQ + (l - 1)) * PT_D + d] += g;
    else d_otok[((long)n * PT_NORC + (l - 1 - PT_LSEQ)) * PT_D + d] = g;
}

// value-input grad: seq pos0 + g
__global__ void k_val_in_bwd(float* d_seq, int LSEQ_, float* d_g,
        const float* d_in, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_VAL_IN) return;
    int n = (int)(i / PT_VAL_IN), c = (int)(i % PT_VAL_IN);
    float g = d_in[i];
    if (c < PT_D) atomicAdd(&d_seq[(long)n * LSEQ_ * PT_D + c], g);
    else atomicAdd(&d_g[(long)n * PT_G + (c - PT_D)], g);
}

// x0 backward: readout sum, token-proj grads, bcast row-sum
__global__ void k_x0_bwd(float* d_readout, float* d_tokp, float* d_bc,
        const float* d_x0, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LSEQ * PT_D) return;
    int d = (int)(i % PT_D);
    long nl = i / PT_D;
    int l = (int)(nl % PT_LSEQ), n = (int)(nl / PT_LSEQ);
    float g = d_x0[i];
    if (l == 0) {
        if (g != 0.f) atomicAdd(&d_readout[d], g);
    } else {
        d_tokp[((long)n * PT_LTOK + (l - 1)) * PT_D + d] = g;
        atomicAdd(&d_bc[(long)n * PT_D + d], g);
    }
}

// ---- aux targets + grads
__global__ void k_aux_targets(float* hand_tgt, float* prized, const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * 32) return;
    int n = (int)(i / 32), k = (int)(i % 32);
    const float* o = obs + (long)n * PT_OBS + PO_ORC;
    if (k < 24) {                                    // opp-hand counts
        int id = (int)o[k * 2];
        float cnt = o[k * 2 + 1] * 4.0f;
        if (id > 0 && id < PT_NC && id != PT_UNK && cnt != 0.f)
            atomicAdd(&hand_tgt[(long)n * PT_NC + id], cnt);
    } else if (k < 32) {                             // my-prize indicator
        int s = k - 24;
        int id = (int)o[120 + s * 2];
        float cnt = o[120 + s * 2 + 1];
        if (id > 0 && id < PT_NC && cnt > 0.f)
            prized[(long)n * PT_NC + id] = 1.f;      // UNK included (python)
    }
}

__global__ void k_hand_grad(float* d_lin, const float* lin, const float* tgt, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = lin[i];
    float pred = softplus_fwd(x);
    float g = AUX_HAND_COEF * 2.f * (pred - tgt[i]) / (float)n;
    d_lin[i] = softplus_bwd(g, x);
}

__global__ void k_prize_count(float* cnt, const float* obs, int N) {
    __shared__ float sh[PTB];
    float s = 0.f;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
         i < (long)N * PT_LTOK; i += (long)gridDim.x * blockDim.x) {
        long n = i / PT_LTOK;
        int l = (int)(i % PT_LTOK);
        float z = obs[n * PT_OBS + PO_TOKI + l * 5 + 1];
        if ((int)z == PT_ZONE_MY_UNSEEN) s += 1.f;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(cnt, sh[0]);
}

__global__ void k_prize_grad(float* d_logit, const float* logit,
        const float* prized, const float* obs, const float* cnt, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK) return;
    long n = i / PT_LTOK;
    int l = (int)(i % PT_LTOK);
    const float* ti = obs + n * PT_OBS + PO_TOKI + l * 5;
    if ((int)ti[1] != PT_ZONE_MY_UNSEEN) { d_logit[i] = 0.f; return; }
    int cid = (int)ti[0];
    cid = cid < 0 ? 0 : (cid >= PT_NC ? PT_NC - 1 : cid);
    float tgt = prized[n * PT_NC + cid];
    float p = 1.f / (1.f + expf(-logit[i]));
    float c = cnt[0] > 0.f ? cnt[0] : 1.f;
    d_logit[i] = AUX_PRIZE_COEF * (p - tgt) / c;
}

__global__ void k_blind_grad(float* d_v, const float* v, const float* ret, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < N) d_v[i] = AUX_VF_COEF * (v[i] - ret[i]) / (float)N;
}

// negamax vtrace/GAE (port of the python advantage path); seats from obs SEAT col
#define PT_SEAT_COL_M (PO_GLOB + 31)
__global__ void k_negamax_adv(const float* values, const float* rewards,
        const float* dones, const float* importance, const float* obs,
        float* adv, float gamma, float lambda, float rho_clip, float c_clip,
        int B, int T) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    long o = (long)row * T;
    const float* ob = obs + (long)row * T * PT_OBS;
    float lastlam = 0.f;
    for (int t = T - 2; t >= 0; t--) {
        float seat_t = ob[(long)t * PT_OBS + PT_SEAT_COL_M];
        float seat_n = ob[(long)(t + 1) * PT_OBS + PT_SEAT_COL_M];
        float sign = seat_t == seat_n ? 1.f : -1.f;
        float s = sign * (1.f - dones[o + t + 1]);
        float imp = importance[o + t];
        float rho = imp < rho_clip ? imp : rho_clip;
        float cc = imp < c_clip ? imp : c_clip;
        float delta = rho * (rewards[o + t + 1] + gamma * values[o + t + 1] * s
                             - values[o + t]);
        lastlam = delta + gamma * lambda * cc * s * lastlam;
        adv[o + t] = lastlam;
    }
}

extern "C" void ptcg_puff_advantage(PrecisionTensor* values, PrecisionTensor* rewards,
        PrecisionTensor* dones, PrecisionTensor* importance, PrecisionTensor* obs,
        PrecisionTensor* advantages, float gamma, float lambda, float rho_clip,
        float c_clip, cudaStream_t stream) {
    int B = (int)values->shape[0], T = (int)values->shape[1];
    k_negamax_adv<<<ptg(B), PTB, 0, stream>>>(values->data, rewards->data,
        dones->data, importance->data, obs->data, advantages->data,
        gamma, lambda, rho_clip, c_clip, B, T);
}

extern "C" void ptcg_bind_train_graph(void* train_buf_ptr) {
    // TrainGraph is defined later in pufferlib.cu; it is a plain struct of
    // PrecisionTensors — member 6 is mb_returns (state,obs,actions,logprobs,
    // advantages,values,returns,...). Keep in sync with register_train_buffers.
    PrecisionTensor* t = (PrecisionTensor*)train_buf_ptr;
    g_mb_returns = t[6].data;
    g_mb_returns_n = numel(t[6].shape);
}

// =========================================================== driver pieces
static void pt_reg(Allocator* a, PrecisionTensor* t, long r, long c) {
    if (c > 0) *t = (PrecisionTensor){.shape = {r, c}};
    else *t = (PrecisionTensor){.shape = {r}};
    alloc_register(a, t);
}

static void table_reg(TabAct* t, Allocator* acts, Allocator* grads, int train) {
    pt_reg(acts, &t->att_in, PT_NA, PT_ATT_IN);
    pt_reg(acts, &t->att_lin, PT_NA, 64);
    pt_reg(acts, &t->att_tab, PT_NA, 64);
    pt_reg(acts, &t->sk_in, PT_NS, PT_SK_IN);
    pt_reg(acts, &t->sk_lin, PT_NS, PT_DC);
    pt_reg(acts, &t->sk_tab, PT_NS, PT_DC);
    pt_reg(acts, &t->a_mean, PT_NC, 64); pt_reg(acts, &t->a_max, PT_NC, 64);
    pt_reg(acts, &t->a_amax, PT_NC, 64);
    pt_reg(acts, &t->s_mean, PT_NC, PT_DC); pt_reg(acts, &t->s_max, PT_NC, PT_DC);
    pt_reg(acts, &t->s_amax, PT_NC, PT_DC);
    pt_reg(acts, &t->card_in, PT_NC, PT_CARD_IN);
    pt_reg(acts, &t->ln_mean, PT_NC, 0); pt_reg(acts, &t->ln_rstd, PT_NC, 0);
    pt_reg(acts, &t->card_ln, PT_NC, PT_CARD_IN);
    pt_reg(acts, &t->card_h1, PT_NC, 256);
    pt_reg(acts, &t->card_g1, PT_NC, 256);
    pt_reg(acts, &t->card_vec, PT_NC, PT_DC);
    if (train) {
        pt_reg(acts, &t->d_card_vec, PT_NC, PT_DC);
        pt_reg(acts, &t->d_card_h, PT_NC, 256);
        pt_reg(acts, &t->d_card_g, PT_NC, 256);
        pt_reg(acts, &t->d_card_ln, PT_NC, PT_CARD_IN);
        pt_reg(acts, &t->d_card_in, PT_NC, PT_CARD_IN);
        pt_reg(acts, &t->d_att_tab, PT_NA, 64);
        pt_reg(acts, &t->d_att_lin, PT_NA, 64);
        pt_reg(acts, &t->d_sk_tab, PT_NS, PT_DC);
        pt_reg(acts, &t->d_sk_lin, PT_NS, PT_DC);
    }
    (void)grads;
}

static void blk_reg(BlkAct* b, Allocator* acts, int N, int L, int train) {
    pt_reg(acts, &b->probs, (long)N * PT_H * L, L);
    pt_reg(acts, &b->merged, (long)N * L, PT_D);
    pt_reg(acts, &b->x_mid, (long)N * L, PT_D);
    pt_reg(acts, &b->ffn1, (long)N * L, PT_FFN);
    pt_reg(acts, &b->ln1_mean, (long)N * L, 0);
    pt_reg(acts, &b->ln1_rstd, (long)N * L, 0);
    pt_reg(acts, &b->ln2_mean, (long)N * L, 0);
    pt_reg(acts, &b->ln2_rstd, (long)N * L, 0);
    (void)train;
}

// static-table device copies of the exporter blob (uploaded once)
struct PtStatic {
    float *card_static, *att_static, *card_text, *att_text, *skill_text;
    int *card_attacks, *card_skills;
    int ready;
};
static PtStatic g_st = {};

extern "C" {
const float* pt_tab_card_static(void);
const int*   pt_tab_card_attacks(void);
const int*   pt_tab_card_skills(void);
const float* pt_tab_att_static(void);
const float* pt_tab_card_text(void);
const float* pt_tab_att_text(void);
const float* pt_tab_skill_text(void);
}

static void pt_static_upload(cudaStream_t stream) {
    if (g_st.ready) return;
    #define UP(dst, src, n) do { \
        cudaError_t _e1 = cudaMalloc((void**)&dst, (size_t)(n) * 4); \
        cudaError_t _e2 = cudaMemcpy((void*)dst, (const void*)(src), (size_t)(n) * 4, cudaMemcpyHostToDevice); \
        if (_e1 != cudaSuccess || _e2 != cudaSuccess) { \
            fprintf(stderr, "pt_static_upload %s: %s / %s\n", #dst, \
                cudaGetErrorString(_e1), cudaGetErrorString(_e2)); abort(); } \
    } while (0)
    UP(g_st.card_static, pt_tab_card_static(), (long)PT_NC * 58);
    UP(g_st.att_static, pt_tab_att_static(), (long)PT_NA * 15);
    UP(g_st.card_text, pt_tab_card_text(), (long)PT_NC * 128);
    UP(g_st.att_text, pt_tab_att_text(), (long)PT_NA * 64);
    UP(g_st.skill_text, pt_tab_skill_text(), (long)PT_NS * 64);
    UP(g_st.card_attacks, pt_tab_card_attacks(), (long)PT_NC * 4);
    UP(g_st.card_skills, pt_tab_card_skills(), (long)PT_NC * 2);
    #undef UP
    (void)stream;
    g_st.ready = 1;
}

// ----- forward: card/attack/skill tables (autograd part of the graph)
static void tables_fwd(PtcgWeights* w, TabAct* t, cudaStream_t s) {
    k_att_in<<<ptg((long)PT_NA * PT_ATT_IN), PTB, 0, s>>>(t->att_in.data,
        W("attack_id_emb"), g_st.att_static, g_st.att_text);
    pt_linear(t->att_in.data, W("attack_mlp_w"), W("attack_mlp_b"),
        t->att_lin.data, PT_NA, PT_ATT_IN, 64, s);
    pt_gelu_fwd<<<ptg((long)PT_NA * 64), PTB, 0, s>>>(t->att_tab.data, t->att_lin.data, (long)PT_NA * 64);
    k_sk_in<<<ptg((long)PT_NS * PT_SK_IN), PTB, 0, s>>>(t->sk_in.data,
        W("skill_id_emb"), g_st.skill_text);
    pt_linear(t->sk_in.data, W("skill_mlp_w"), W("skill_mlp_b"),
        t->sk_lin.data, PT_NS, PT_SK_IN, PT_DC, s);
    pt_gelu_fwd<<<ptg((long)PT_NS * PT_DC), PTB, 0, s>>>(t->sk_tab.data, t->sk_lin.data, (long)PT_NS * PT_DC);
    k_pool<<<ptg((long)PT_NC * 64), PTB, 0, s>>>(t->a_mean.data, t->a_max.data,
        t->a_amax.data, t->att_tab.data, g_st.card_attacks, PT_NC, 4, 64);
    k_pool<<<ptg((long)PT_NC * PT_DC), PTB, 0, s>>>(t->s_mean.data, t->s_max.data,
        t->s_amax.data, t->sk_tab.data, g_st.card_skills, PT_NC, 2, PT_DC);
    k_card_in<<<ptg((long)PT_NC * PT_CARD_IN), PTB, 0, s>>>(t->card_in.data,
        W("card_id_emb"), g_st.card_static, g_st.card_text,
        t->a_mean.data, t->a_max.data, t->s_mean.data, t->s_max.data);
    pt_ln_fwd<<<PT_NC, 256, 0, s>>>(t->card_ln.data, t->ln_mean.data,
        t->ln_rstd.data, t->card_in.data, W("card_ln_w"), W("card_ln_b"),
        PT_NC, PT_CARD_IN);
    pt_linear(t->card_ln.data, W("card_mlp1_w"), W("card_mlp1_b"),
        t->card_h1.data, PT_NC, PT_CARD_IN, 256, s);
    pt_gelu_fwd<<<ptg((long)PT_NC * 256), PTB, 0, s>>>(t->card_g1.data, t->card_h1.data, (long)PT_NC * 256);
    pt_linear(t->card_g1.data, W("card_mlp2_w"), W("card_mlp2_b"),
        t->card_vec.data, PT_NC, 256, PT_DC, s);
}

static void tables_bwd(PtcgWeights* w, TabAct* t, cudaStream_t s) {
    // d_card_vec accumulated by callers; run the chain down to embeddings
    pt_linear_bwd(t->card_g1.data, W("card_mlp2_w"), t->d_card_vec.data,
        t->d_card_g.data, G("card_mlp2_w"), G("card_mlp2_b"), PT_NC, 256, PT_DC, s);
    pt_gelu_bwd<<<ptg((long)PT_NC * 256), PTB, 0, s>>>(t->d_card_h.data,
        t->d_card_g.data, t->card_h1.data, (long)PT_NC * 256);
    pt_linear_bwd(t->card_ln.data, W("card_mlp1_w"), t->d_card_h.data,
        t->d_card_ln.data, G("card_mlp1_w"), G("card_mlp1_b"), PT_NC, PT_CARD_IN, 256, s);
    pt_ln_bwd<<<PT_NC, 256, 0, s>>>(t->d_card_in.data, G("card_ln_w"),
        G("card_ln_b"), t->d_card_ln.data, t->card_in.data, W("card_ln_w"),
        t->ln_mean.data, t->ln_rstd.data, PT_NC, PT_CARD_IN);
    // split d_card_in: id emb / (skip static, text) / pools
    pt_slice_block_add<<<ptg((long)PT_NC * 64), PTB, 0, s>>>(G("card_id_emb"),
        t->d_card_in.data, PT_NC, 64, PT_CARD_IN, 0);
    // pool grads -> att/sk tables (strided reads out of d_card_in)
    k_pool_bwd<<<ptg((long)PT_NC * 64), PTB, 0, s>>>(t->d_att_tab.data,
        t->d_card_in.data, PT_CARD_IN, 250, 314, t->a_amax.data,
        g_st.card_attacks, PT_NC, 4, 64);
    k_pool_bwd<<<ptg((long)PT_NC * PT_DC), PTB, 0, s>>>(t->d_sk_tab.data,
        t->d_card_in.data, PT_CARD_IN, 378, 506, t->s_amax.data,
        g_st.card_skills, PT_NC, 2, PT_DC);
    pt_gelu_bwd<<<ptg((long)PT_NA * 64), PTB, 0, s>>>(t->d_att_lin.data,
        t->d_att_tab.data, t->att_lin.data, (long)PT_NA * 64);
    pt_linear_bwd(t->att_in.data, W("attack_mlp_w"), t->d_att_lin.data,
        t->att_in.data /*reuse as d*/, G("attack_mlp_w"), G("attack_mlp_b"),
        PT_NA, PT_ATT_IN, 64, s);
    pt_slice_block_add<<<ptg((long)PT_NA * 48), PTB, 0, s>>>(G("attack_id_emb"),
        t->att_in.data, PT_NA, 48, PT_ATT_IN, 0);
    pt_gelu_bwd<<<ptg((long)PT_NS * PT_DC), PTB, 0, s>>>(t->d_sk_lin.data,
        t->d_sk_tab.data, t->sk_lin.data, (long)PT_NS * PT_DC);
    pt_linear_bwd(t->sk_in.data, W("skill_mlp_w"), t->d_sk_lin.data,
        t->sk_in.data /*reuse as d*/, G("skill_mlp_w"), G("skill_mlp_b"),
        PT_NS, PT_SK_IN, PT_DC, s);
    pt_slice_block_add<<<ptg((long)PT_NS * 48), PTB, 0, s>>>(G("skill_id_emb"),
        t->sk_in.data, PT_NS, 48, PT_SK_IN, 0);
}

#include "ptcg_model3.cu"
