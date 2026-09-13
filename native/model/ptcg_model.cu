// PTCG transformer as a pufferlib-4.0 native custom encoder (fp32).
// Mirrors ptcg/rl/model.py exactly: static card/attack/skill tables rebuilt
// per forward, token projection + global/decision broadcast, pre-LN trunk
// with READOUT, pointer option scorer, oracle critic top (PPO value),
// blind-V + belief aux heads with their losses applied in backward.
//
// Weight layout is the canonical order in PT_PARAMS below; the torch
// exporter (native/tools/native_weights.py) mirrors it name-for-name.
#ifndef PTCG_MODEL_CU
#define PTCG_MODEL_CU

#include "ptcg_kernels.cuh"

// Trunk dims. THESE FOUR ARE THE WHOLE SIZE KNOB — every width below that
// depends on them is DERIVED, so switching size is editing this block and
// rebuilding. Must match the torch reference in ptcg/rl/model.py, whose
// defaults are d=256 layers=4 heads=8 ffn=512; the parity gate
// (native/tools/model_parity.py) is what proves they agree.
// A d128/h4 trunk is `PT_D 128 / PT_H 4 / PT_FFN 256`.
// NOTE PT_DH is 32 at BOTH sizes (128/4 == 256/8), which is why the
// attention path needs no changes between them.
#define PT_D 256
#define PT_H 8
#define PT_DH (PT_D / PT_H)
#define PT_FFN 512
// Card/attack vector widths and the global-summary MLP are INDEPENDENT of the
// trunk width — torch fixes them at d_c=128, d_a=64 and
// global_mlp = Linear(GLOBAL_F,128) -> Linear(128,128) whatever d is. Do not
// tie these to PT_D.
#define PT_DC 128
#define PT_DA 64
#define PT_G 128
#define PT_LAYERS 4
#define PT_LTOK 160
#define PT_LSEQ (PT_LTOK + 1)
#define PT_NOPT 64
#define PT_NORC 56
#define PT_LCRIT (1 + PT_LSEQ + PT_NORC)
#define PT_NC 1284
#define PT_NA 1573
#define PT_NS 448
#define PT_ATT_IN 127
#define PT_SK_IN 112
// Concat widths. CARD_IN/TOK_IN/DEC_IN/ORC_IN are built from PT_DC/PT_DA and
// obs fields only, so they do NOT move with the trunk width. The other four
// DO, and are derived here rather than written as literals — that is what
// makes a size change a one-block edit. Each mirrors a torch nn.Linear:
//   bcast      Linear(128 + 64, d)
//   q_mlp      Linear(16 + 2d + 2*64 + 8, d)
//   score_mlp  Linear(3d + 128 + 64, 256)
//   value_mlp  Linear(d + 128, 256)
// At d128 these evaluate to 192 / 408 / 576 / 256.
#define PT_CARD_IN 634
#define PT_TOK_IN 324
#define PT_DEC_IN 696
#define PT_ORC_IN 134
#define PT_BCAST_IN (PT_G + 64)
#define PT_Q_IN (16 + 2 * PT_D + 2 * 64 + 8)
#define PT_SCORE_IN (3 * PT_D + PT_G + 64)
#define PT_VAL_IN (PT_D + PT_G)
// offsets into the q_in concat: [type16 | h(p1) | h(p2) | ocard64 | oatt64 | optf8]
#define PT_QO_H1 16
#define PT_QO_H2 (PT_QO_H1 + PT_D)
#define PT_QO_OC (PT_QO_H2 + PT_D)
#define PT_QO_OA (PT_QO_OC + 64)
#define PT_QO_OF (PT_QO_OA + 64)
// offsets into the score concat: [q | r | g | dvec | q*r]
#define PT_SO_R PT_D
#define PT_SO_G (2 * PT_D)
#define PT_SO_DV (PT_SO_G + PT_G)
#define PT_SO_QR (PT_SO_DV + 64)
#define PT_OBS 10944
// obs section offsets (mirror ptcg_env.h; kept local to avoid C/CUDA include mix)
#define PO_TOKI 0
#define PO_TOKF (PT_LTOK * 5)
#define PO_OPTI (PO_TOKF + PT_LTOK * 56)
#define PO_OPTF (PO_OPTI + PT_NOPT * 6)
#define PO_MASK (PO_OPTF + PT_NOPT * 8)
#define PO_GLOB (PO_MASK + PT_NOPT)
#define PO_DECI (PO_GLOB + 64)
#define PO_DECF (PO_DECI + 8)
#define PO_ORC (PO_DECF + 16)
#define PT_ZONE_MY_UNSEEN 9
#define PT_UNK (PT_NC - 1)
#define AUX_VF_COEF 1.0f
#define AUX_HAND_COEF 0.1f
#define AUX_PRIZE_COEF 0.1f

// ------------------------------------------------------------ weight table
struct PtParamDef { const char* name; long r, c; };  // c==0 -> 1D
#define BLOCK_PARAMS(p) \
    {p "ln1_w",PT_D,0},{p "ln1_b",PT_D,0},{p "qkv_w",3*PT_D,PT_D},{p "qkv_b",3*PT_D,0}, \
    {p "proj_w",PT_D,PT_D},{p "proj_b",PT_D,0},{p "ln2_w",PT_D,0},{p "ln2_b",PT_D,0}, \
    {p "ffn1_w",PT_FFN,PT_D},{p "ffn1_b",PT_FFN,0},{p "ffn2_w",PT_D,PT_FFN},{p "ffn2_b",PT_D,0}
static const PtParamDef PT_PARAMS[] = {
    {"card_id_emb",PT_NC,64},{"attack_id_emb",PT_NA,48},{"skill_id_emb",PT_NS,48},
    {"attack_mlp_w",64,PT_ATT_IN},{"attack_mlp_b",64,0},
    {"skill_mlp_w",PT_DC,PT_SK_IN},{"skill_mlp_b",PT_DC,0},
    {"card_ln_w",PT_CARD_IN,0},{"card_ln_b",PT_CARD_IN,0},
    {"card_mlp1_w",256,PT_CARD_IN},{"card_mlp1_b",256,0},
    {"card_mlp2_w",PT_DC,256},{"card_mlp2_b",PT_DC,0},
    {"zone_emb",16,8},{"owner_emb",4,4},
    {"token_proj_w",PT_D,PT_TOK_IN},{"token_proj_b",PT_D,0},
    {"readout",PT_D,0},
    {"global1_w",PT_G,64},{"global1_b",PT_G,0},{"global2_w",PT_G,PT_G},{"global2_b",PT_G,0},
    {"sel_type_emb",29,8},{"ctx_emb",65,32},
    {"dec1_w",64,PT_DEC_IN},{"dec1_b",64,0},{"dec2_w",64,64},{"dec2_b",64,0},
    {"bcast_w",PT_D,PT_BCAST_IN},{"bcast_b",PT_D,0},
    BLOCK_PARAMS("b0_"),BLOCK_PARAMS("b1_"),BLOCK_PARAMS("b2_"),BLOCK_PARAMS("b3_"),
    {"lnf_w",PT_D,0},{"lnf_b",PT_D,0},
    {"opt_type_emb",33,16},
    {"opt_card_w",64,PT_DC},{"opt_card_b",64,0},
    {"opt_attack_w",64,PT_DA},{"opt_attack_b",64,0},
    {"q1_w",PT_D,PT_Q_IN},{"q1_b",PT_D,0},{"q2_w",PT_D,PT_D},{"q2_b",PT_D,0},
    {"score1_w",256,PT_SCORE_IN},{"score1_b",256,0},{"score2_w",1,256},{"score2_b",1,0},
    {"value1_w",256,PT_VAL_IN},{"value1_b",256,0},{"value2_w",1,256},{"value2_b",1,0},
    {"oracle_proj_w",PT_D,PT_ORC_IN},{"oracle_proj_b",PT_D,0},
    {"critic_readout",PT_D,0},
    BLOCK_PARAMS("c0_"),
    {"oracle_v1_w",256,PT_VAL_IN},{"oracle_v1_b",256,0},{"oracle_v2_w",1,256},{"oracle_v2_b",1,0},
    // aux_hand reads the SAME [r||g] concat as the value head, so its input is
    // PT_VAL_IN — not the 256 MLP-hidden width it happened to equal at d128.
    // torch: aux_hand = nn.Linear(d + 128, N_CARDS)
    {"aux_hand_w",PT_NC,PT_VAL_IN},{"aux_hand_b",PT_NC,0},
    {"aux_prize_w",1,PT_D},{"aux_prize_b",1,0},
};
#define PT_NPARAM (int)(sizeof(PT_PARAMS)/sizeof(PT_PARAMS[0]))

struct PtcgWeights {
    PrecisionTensor p[PT_NPARAM];       // params (registered in params alloc)
    PrecisionTensor g[PT_NPARAM];       // same-shape grads (grads alloc)
    int idx_by_hash[1];                 // unused; lookup by scan
};

static int pt_widx(const char* name) {
    for (int i = 0; i < PT_NPARAM; i++)
        if (strcmp(PT_PARAMS[i].name, name) == 0) return i;
    fprintf(stderr, "ptcg_model: unknown param %s\n", name);
    abort();
}
#define W(nm) (w->p[pt_widx(nm)].data)
#define G(nm) (w->g[pt_widx(nm)].data)

// per trunk/critic block param pointers
struct BlkW { float *ln1w,*ln1b,*qkvw,*qkvb,*projw,*projb,*ln2w,*ln2b,*f1w,*f1b,*f2w,*f2b; };
struct BlkG { float *ln1w,*ln1b,*qkvw,*qkvb,*projw,*projb,*ln2w,*ln2b,*f1w,*f1b,*f2w,*f2b; };
static void blkw(PtcgWeights* w, const char* pre, BlkW* o, BlkG* go) {
    char nm[32];
    #define GET(field, suffix) \
        snprintf(nm, 32, "%s%s", pre, suffix); \
        o->field = w->p[pt_widx(nm)].data; \
        if (go) go->field = w->g[pt_widx(nm)].data;
    GET(ln1w,"ln1_w") GET(ln1b,"ln1_b") GET(qkvw,"qkv_w") GET(qkvb,"qkv_b")
    GET(projw,"proj_w") GET(projb,"proj_b") GET(ln2w,"ln2_w") GET(ln2b,"ln2_b")
    GET(f1w,"ffn1_w") GET(f1b,"ffn1_b") GET(f2w,"ffn2_w") GET(f2b,"ffn2_b")
    #undef GET
}

// ------------------------------------------------------------- activations
// Table-build buffers (batch independent, needed fwd+bwd)
struct TabAct {
    PrecisionTensor att_in, att_lin, att_tab;         // (NA,127) (NA,64) (NA,64)
    PrecisionTensor sk_in, sk_lin, sk_tab;            // (NS,112) (NS,128) (NS,128)
    PrecisionTensor a_mean, a_max, a_amax;            // (NC,64)x3 (amax=idx as float)
    PrecisionTensor s_mean, s_max, s_amax;            // (NC,128)x3
    PrecisionTensor card_in, ln_mean, ln_rstd;        // (NC,634) (NC) (NC)
    PrecisionTensor card_ln, card_h1, card_g1, card_vec;  // (NC,634)(NC,256)(NC,256)(NC,128)
    // backward scratch
    PrecisionTensor d_card_vec, d_card_h, d_card_g, d_card_ln, d_card_in;
    PrecisionTensor d_att_tab, d_att_lin, d_sk_tab, d_sk_lin;
};

struct BlkAct {
    PrecisionTensor probs;        // (N,H,L,L) saved
    PrecisionTensor merged;       // (N,L,D) attn heads merged (proj input)
    PrecisionTensor x_mid;        // (N,L,D) after attn residual
    PrecisionTensor ffn1;         // (N,L,FFN) pre-gelu
    PrecisionTensor ln1_mean, ln1_rstd, ln2_mean, ln2_rstd;   // (N*L)
};

struct PtcgAct {
    int N, train;
    PrecisionTensor saved_obs;    // (N, OBS) train only (aux targets + scatters)
    const float* obs_src;         // rollout: points at input obs (no copy)
    TabAct tab;
    PrecisionTensor tok_cat;      // (N,160,324)
    PrecisionTensor g1, g_out;    // (N,128) global mlp: pre-gelu, out
    PrecisionTensor dec_in, d1, d_out;   // (N,696) (N,64 pre-gelu) (N,64)
    PrecisionTensor bc_in, bc_out;       // (N,192) (N,128)
    PrecisionTensor x[PT_LAYERS + 1];    // (N,LSEQ,D) trunk states
    BlkAct blk[PT_LAYERS];
    PrecisionTensor keep;         // (N,LSEQ) key mask
    PrecisionTensor h;            // (N,LSEQ,D) post ln_f
    PrecisionTensor lnf_mean, lnf_rstd;
    // options
    PrecisionTensor q_in, q1a, q_vec;    // (N,64,408) (N,64,128 pre-gelu) (N,64,128)
    PrecisionTensor sc_in, sc1;          // (N,64,576) (N,64,256 pre-gelu)
    PrecisionTensor scores;              // (N,64) staged raw scores. NOT
    // staging scratch: the critic block runs between score2 and k_fuse_out,
    // and its attention scratch aliases s_q — staging scores there gets
    // clobbered (found via model-parity: value exact, scores garbage).
    PrecisionTensor ocard_in, oatt_in;   // (N,64,128) (N,64,64) gathered
    // oracle critic
    PrecisionTensor otok_in, otok;       // (N,56,134) (N,56,128)
    PrecisionTensor xc[2];               // critic in/out (N,LCRIT,D)
    BlkAct cblk;
    PrecisionTensor ckeep;               // (N,LCRIT)
    PrecisionTensor ov_in, ov1;          // (N,256) cat[x2_0,g] ; pre-gelu
    // values / aux (train)
    PrecisionTensor bv_in, bv1, blind_v; // (N,256) (N,256) (N,1)
    PrecisionTensor hand_lin;            // (N,NC) pre-softplus
    PrecisionTensor prize_logit;         // (N,LTOK)
    PrecisionTensor hand_tgt, prized;    // (N,NC) (N,NC)
    PrecisionTensor out;                 // (N, NOPT+1) fused scores||value
    // backward scratch (shared)
    PrecisionTensor dx_a, dx_b;          // (N,LCRIT,D) generic seq grads (max seq)
    PrecisionTensor s_ln, s_qkv, s_q, s_k, s_v, s_ctx;  // recompute + grads
    PrecisionTensor s_ffn;               // (N,LSEQ,FFN)
    PrecisionTensor d_g, d_dvec, d_rg;   // (N,PT_G) (N,64) (N,PT_VAL_IN)
    PrecisionTensor d_out64;             // (N,64,*) scratch for option bwd
    PrecisionTensor d_qin, d_scin;
    PrecisionTensor prize_cnt;           // (1) masked-token count
};

// train-graph binding (mb_returns for the blind-V aux loss)
static float* g_mb_returns = nullptr;
static long g_mb_returns_n = 0;
extern "C" void ptcg_bind_train_graph(void* train_buf_ptr);

// ============================================================ fwd kernels
__global__ void k_att_in(float* out, const float* id_emb, const float* stat,
        const float* text) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)PT_NA * PT_ATT_IN) return;
    int n = (int)(i / PT_ATT_IN), c = (int)(i % PT_ATT_IN);
    float v;
    if (c < 48) v = id_emb[n * 48 + c];
    else if (c < 63) v = stat[n * 15 + (c - 48)];
    else v = text[n * 64 + (c - 63)];
    out[i] = v;
}

__global__ void k_sk_in(float* out, const float* id_emb, const float* text) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)PT_NS * PT_SK_IN) return;
    int n = (int)(i / PT_SK_IN), c = (int)(i % PT_SK_IN);
    out[i] = c < 48 ? id_emb[n * 48 + c] : text[n * 64 + (c - 48)];
}

// masked mean/max pools over gathered attack/skill vecs
__global__ void k_pool(float* mean, float* mx, float* amax, const float* tab,
        const int* idx, int NC, int K, int E) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)NC * E) return;
    int n = (int)(i / E), e = (int)(i % E);
    // torch: (a*mask).sum(1)/mask.sum().clamp(1) and (a*mask).amax(1) —
    // masked slots ARE 0-valued participants of the max
    float s = 0.f, m2 = -1e30f;
    int cnt = 0, am2 = 0;
    for (int k = 0; k < K; k++) {
        int id = idx[n * K + k];
        float v = id > 0 ? tab[(long)id * E + e] : 0.f;
        if (id > 0) { cnt++; s += v; }
        if (v > m2) { m2 = v; am2 = k; }
    }
    mean[i] = s / (float)(cnt > 0 ? cnt : 1);
    mx[i] = m2;
    amax[i] = (float)am2;
}

__global__ void k_card_in(float* out, const float* id_emb, const float* stat,
        const float* text, const float* am, const float* ax,
        const float* sm, const float* sx) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)PT_NC * PT_CARD_IN) return;
    int n = (int)(i / PT_CARD_IN), c = (int)(i % PT_CARD_IN);
    float v;
    if (c < 64) v = id_emb[n * 64 + c];
    else if (c < 122) v = stat[n * 58 + (c - 64)];
    else if (c < 250) v = text[n * 128 + (c - 122)];
    else if (c < 314) v = am[n * 64 + (c - 250)];
    else if (c < 378) v = ax[n * 64 + (c - 314)];
    else if (c < 506) v = sm[n * 128 + (c - 378)];
    else v = sx[n * 128 + (c - 506)];
    out[i] = v;
}

// token concat: [card_vec(cid), toolvec, tok_f(56), zone(8), owner(4)] = 324
__global__ void k_tok_cat(float* out, const float* obs, const float* cv,
        const float* zemb, const float* oemb, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK * PT_TOK_IN) return;
    int c = (int)(i % PT_TOK_IN);
    long nl = i / PT_TOK_IN;
    int l = (int)(nl % PT_LTOK), n = (int)(nl / PT_LTOK);
    const float* row = obs + (long)n * PT_OBS;
    const float* ti = row + PO_TOKI + l * 5;
    float v;
    if (c < 128) {
        int cid = (int)ti[0];
        cid = cid < 0 ? 0 : (cid >= PT_NC ? PT_NC - 1 : cid);
        v = cv[(long)cid * PT_DC + c];
    } else if (c < 256) {
        int e = c - 128;
        int t0 = (int)ti[3], t1 = (int)ti[4];
        float a = t0 > 0 ? cv[(long)(t0 >= PT_NC ? PT_NC-1 : t0) * PT_DC + e] : 0.f;
        float b = t1 > 0 ? cv[(long)(t1 >= PT_NC ? PT_NC-1 : t1) * PT_DC + e] : 0.f;
        v = a + b;
    } else if (c < 312) {
        v = row[PO_TOKF + l * 56 + (c - 256)];
    } else if (c < 320) {
        int z = (int)ti[1];
        z = z < 0 ? 0 : (z > 15 ? 15 : z);
        v = zemb[z * 8 + (c - 312)];
    } else {
        int o = (int)ti[2];
        o = o < 0 ? 0 : (o > 3 ? 3 : o);
        v = oemb[o * 4 + (c - 320)];
    }
    out[i] = v;
}

// dec concat (696): [sel8, ctx32, cv_eff, cv_ctx, av_my, av_opp, cv_msup, cv_osup, decf16]
__global__ void k_dec_in(float* out, const float* obs, const float* cv,
        const float* av, const float* semb, const float* cemb, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_DEC_IN) return;
    int n = (int)(i / PT_DEC_IN), c = (int)(i % PT_DEC_IN);
    const float* row = obs + (long)n * PT_OBS;
    const float* di = row + PO_DECI;
    #define CLC(x) ((x) < 0 ? 0 : ((x) >= PT_NC ? PT_NC - 1 : (x)))
    #define CLA(x) ((x) < 0 ? 0 : ((x) >= PT_NA ? PT_NA - 1 : (x)))
    float v;
    if (c < 8) { int s = (int)di[0]; s = s < 0 ? 0 : (s > 28 ? 28 : s); v = semb[s * 8 + c]; }
    else if (c < 40) { int s = (int)di[1]; s = s < 0 ? 0 : (s > 64 ? 64 : s); v = cemb[s * 32 + (c - 8)]; }
    else if (c < 168) v = cv[(long)CLC((int)di[2]) * PT_DC + (c - 40)];
    else if (c < 296) v = cv[(long)CLC((int)di[3]) * PT_DC + (c - 168)];
    else if (c < 360) v = av[(long)CLA((int)di[4]) * PT_DA + (c - 296)];
    else if (c < 424) v = av[(long)CLA((int)di[5]) * PT_DA + (c - 360)];
    else if (c < 552) v = cv[(long)CLC((int)di[6]) * PT_DC + (c - 424)];
    else if (c < 680) v = cv[(long)CLC((int)di[7]) * PT_DC + (c - 552)];
    else v = row[PO_DECF + (c - 680)];
    out[i] = v;
}

// assemble trunk input x0 (N,LSEQ,D): pos0 = readout; else token_proj+bcast.
// keep (N,LSEQ): pos0=1 else zone>0
__global__ void k_x0(float* x0, float* keep, const float* tokp,
        const float* bc, const float* readout, const float* obs, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LSEQ * PT_D) return;
    int d = (int)(i % PT_D);
    long nl = i / PT_D;
    int l = (int)(nl % PT_LSEQ), n = (int)(nl / PT_LSEQ);
    float v;
    if (l == 0) v = readout[d];
    else v = tokp[((long)n * PT_LTOK + (l - 1)) * PT_D + d]
             + bc[(long)n * PT_D + d];
    x0[i] = v;
    if (d == 0) {
        float kp = 1.f;
        if (l > 0) {
            float z = obs[(long)n * PT_OBS + PO_TOKI + (l - 1) * 5 + 1];
            kp = z > 0.f ? 1.f : 0.f;
        }
        keep[(long)n * PT_LSEQ + l] = kp;
    }
}

// option scorer input gathers
__global__ void k_opt_gather(float* ocard, float* oatt, const float* obs,
        const float* cv, const float* av, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT * PT_DC) return;
    int c = (int)(i % PT_DC);
    long nj = i / PT_DC;
    int j = (int)(nj % PT_NOPT), n = (int)(nj / PT_NOPT);
    const float* oi = obs + (long)n * PT_OBS + PO_OPTI + j * 6;
    int cid = (int)oi[3];
    cid = cid < 0 ? 0 : (cid >= PT_NC ? PT_NC - 1 : cid);
    ocard[i] = cv[(long)cid * PT_DC + c];
    if (c < PT_DA) {
        int aid = (int)oi[4];
        aid = aid < 0 ? 0 : (aid >= PT_NA ? PT_NA - 1 : aid);
        oatt[((long)n * PT_NOPT + j) * PT_DA + c] = av[(long)aid * PT_DA + c];
    }
}

// q_in (408): [type16, p1(128), p2(128), ocard64, oatt64, optf8]
__global__ void k_q_in(float* out, const float* obs, const float* h,
        const float* temb, const float* oc, const float* oa, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT * PT_Q_IN) return;
    int c = (int)(i % PT_Q_IN);
    long nj = i / PT_Q_IN;
    int j = (int)(nj % PT_NOPT), n = (int)(nj / PT_NOPT);
    const float* row = obs + (long)n * PT_OBS;
    const float* oi = row + PO_OPTI + j * 6;
    float v;
    if (c < 16) {
        int t = (int)oi[0];
        t = t < 0 ? 0 : (t > 32 ? 32 : t);
        v = temb[t * 16 + c];
    } else if (c < PT_QO_H2) {
        int p = (int)oi[1];   // 0 = null -> zeros; else h position p
        v = (p <= 0 || p > PT_LTOK) ? 0.f
            : h[((long)n * PT_LSEQ + p) * PT_D + (c - PT_QO_H1)];
    } else if (c < PT_QO_OC) {
        int p = (int)oi[2];
        v = (p <= 0 || p > PT_LTOK) ? 0.f
            : h[((long)n * PT_LSEQ + p) * PT_D + (c - PT_QO_H2)];
    } else if (c < PT_QO_OA) {
        v = oc[nj * 64 + (c - PT_QO_OC)];
    } else if (c < PT_QO_OF) {
        v = oa[nj * 64 + (c - PT_QO_OA)];
    } else {
        v = row[PO_OPTF + j * 8 + (c - PT_QO_OF)];
    }
    out[i] = v;
}

// sc_in (576): [q, r, g, d, q*r]
__global__ void k_sc_in(float* out, const float* q, const float* h,
        const float* g, const float* dv, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT * PT_SCORE_IN) return;
    int c = (int)(i % PT_SCORE_IN);
    long nj = i / PT_SCORE_IN;
    int j = (int)(nj % PT_NOPT), n = (int)(nj / PT_NOPT);
    (void)j;
    const float* r = h + (long)n * PT_LSEQ * PT_D;   // pos 0
    float v;
    if (c < PT_SO_R) v = q[nj * PT_D + c];
    else if (c < PT_SO_G) v = r[c - PT_SO_R];
    else if (c < PT_SO_DV) v = g[(long)n * PT_G + (c - PT_SO_G)];
    else if (c < PT_SO_QR) v = dv[(long)n * 64 + (c - PT_SO_DV)];
    else v = q[nj * PT_D + (c - PT_SO_QR)] * r[c - PT_SO_QR];
    out[i] = v;
}

// oracle token concat (134): [card_vec(id), feats6]; also critic keep mask
__global__ void k_otok_in(float* out, float* ckeep, const float* obs,
        const float* cv, const float* keep, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NORC * PT_ORC_IN) return;
    int c = (int)(i % PT_ORC_IN);
    long nk = i / PT_ORC_IN;
    int k = (int)(nk % PT_NORC), n = (int)(nk / PT_NORC);
    const float* o = obs + (long)n * PT_OBS + PO_ORC;
    int id;
    float f3 = 0, f4 = 0, f5 = 0;
    int kind;
    if (k < 24) { kind = 0; id = (int)o[k * 2]; f3 = o[k * 2 + 1]; }
    else if (k < 48) {
        kind = 1;
        int s = k - 24;
        id = (int)o[48 + s * 3];
        f4 = o[48 + s * 3 + 1];
        f5 = o[48 + s * 3 + 2];
    } else {
        kind = 2;
        int s = k - 48;
        id = (int)o[120 + s * 2];
        f3 = o[120 + s * 2 + 1];
    }
    int idc = id < 0 ? 0 : (id >= PT_NC ? PT_NC - 1 : id);
    float v;
    if (c < 128) v = cv[(long)idc * PT_DC + c];
    else if (c == 128) v = kind == 0 ? 1.f : 0.f;
    else if (c == 129) v = kind == 1 ? 1.f : 0.f;
    else if (c == 130) v = kind == 2 ? 1.f : 0.f;
    else if (c == 131) v = f3;
    else if (c == 132) v = f4;
    else v = f5;
    out[i] = v;
    if (c == 0) {
        // critic keep: [1(creadout), keep(LSEQ), ids>0(NORC)]
        float* ck = ckeep + (long)n * PT_LCRIT;
        if (k == 0) {
            ck[0] = 1.f;
            for (int l = 0; l < PT_LSEQ; l++) ck[1 + l] = keep[(long)n * PT_LSEQ + l];
        }
        ck[1 + PT_LSEQ + k] = id > 0 ? 1.f : 0.f;
    }
}

// critic input: [critic_readout, h(LSEQ), otok(NORC)]
__global__ void k_xc(float* xc, const float* h, const float* otok,
        const float* creadout, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LCRIT * PT_D) return;
    int d = (int)(i % PT_D);
    long nl = i / PT_D;
    int l = (int)(nl % PT_LCRIT), n = (int)(nl / PT_LCRIT);
    float v;
    if (l == 0) v = creadout[d];
    else if (l <= PT_LSEQ) v = h[((long)n * PT_LSEQ + (l - 1)) * PT_D + d];
    else v = otok[((long)n * PT_NORC + (l - 1 - PT_LSEQ)) * PT_D + d];
    xc[i] = v;
}

// value-head input: [seq0_state, g]
__global__ void k_val_in(float* out, const float* seq, int LSEQ_, const float* g, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_VAL_IN) return;
    int n = (int)(i / PT_VAL_IN), c = (int)(i % PT_VAL_IN);
    out[i] = c < PT_D ? seq[(long)n * LSEQ_ * PT_D + c]
                      : g[(long)n * PT_G + (c - PT_D)];
}

// fused out: [scores(64) || value]
__global__ void k_fuse_out(float* out, const float* sc, const float* val, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * (PT_NOPT + 1)) return;
    int n = (int)(i / (PT_NOPT + 1)), c = (int)(i % (PT_NOPT + 1));
    out[i] = c < PT_NOPT ? sc[(long)n * PT_NOPT + c] : val[n];
}

#include "ptcg_model2.cu"
#endif // PTCG_MODEL_CU
