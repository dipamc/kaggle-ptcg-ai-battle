// Primitive CUDA ops for the native PTCG model (fp32 build only).
// Every op has a hand-written backward. GEMMs run TF32 by default
// (PTCG_TF32=0 disables) to match torch.set_float32_matmul_precision('high')
// in the production pufferl 3.0 stack.
#ifndef PTCG_KERNELS_CUH
#define PTCG_KERNELS_CUH

#include <cublas_v2.h>
#include <math.h>

#ifdef PRECISION_FLOAT
// precision_t == float in this build; the model requires it.
#else
#error "ptcg native model requires the --float build (obs carry integer ids)"
#endif

#define PTB 256
static inline int ptg(long n) { return (int)((n + PTB - 1) / PTB); }

// ------------------------------------------------------------------ gemm
static cublasComputeType_t ptcg_compute_type(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("PTCG_TF32");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

static cublasHandle_t ptcg_cublas(void) {
    static thread_local cublasHandle_t h = nullptr;
    if (!h) {
        cublasCreate(&h);
        void* ws = nullptr;
        cudaMalloc(&ws, 32u << 20);
        cublasSetWorkspace(h, ws, 32u << 20);
    }
    return h;
}

// Row-major C(M,N) = op_a(A) @ op_b(B) [+ beta*C]
static inline void pt_gemm(cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, const float* A, const float* B, float* C,
        cudaStream_t stream, float alpha = 1.f, float beta = 0.f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasHandle_t h = ptcg_cublas();
    cublasSetStream(h, stream);
    cublasGemmEx(h, op_b, op_a, N, M, K, &alpha,
        B, CUDA_R_32F, ldb, A, CUDA_R_32F, lda, &beta,
        C, CUDA_R_32F, N, ptcg_compute_type(), CUBLAS_GEMM_DEFAULT);
}

// Batched row-major: C[b](M,N) = op_a(A[b]) @ op_b(B[b])
static inline void pt_gemm_batched(cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, const float* A, long strideA,
        const float* B, long strideB, float* C, long strideC, int batch,
        cudaStream_t stream, float alpha = 1.f, float beta = 0.f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasHandle_t h = ptcg_cublas();
    cublasSetStream(h, stream);
    cublasGemmStridedBatchedEx(h, op_b, op_a, N, M, K, &alpha,
        B, CUDA_R_32F, ldb, strideB, A, CUDA_R_32F, lda, strideA, &beta,
        C, CUDA_R_32F, N, strideC, batch, ptcg_compute_type(),
        CUBLAS_GEMM_DEFAULT);
}

// Linear: Y(N,O) = X(N,I) @ W(O,I)^T + b(O)
__global__ void pt_bias_add(float* y, const float* b, long n, int o) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % o];
}

static inline void pt_linear(const float* x, const float* w, const float* b,
        float* y, int N, int I, int O, cudaStream_t s) {
    pt_gemm(CUBLAS_OP_N, CUBLAS_OP_T, N, O, I, x, w, y, s);
    if (b) pt_bias_add<<<ptg((long)N * O), PTB, 0, s>>>(y, b, (long)N * O, O);
}

// Linear backward: dX = dY @ W ; dW += dY^T @ X ; db += colsum(dY)
// Row-sliced colsum: each block owns (row-slice, column-tile); threads sum
// their column over the slice (coalesced) and atomicAdd the partial into db.
// A one-thread-per-column loop would leave 128 threads covering 165K rows at
// token_proj bwd. Atomic merge order makes db fp-reorder-nondeterministic,
// same as the embedding scatters elsewhere in this backward.
#define PT_BGRAD_ROWS 256
__global__ void pt_bias_grad(float* db, const float* dy, long N, int O) {
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    if (o >= O) return;
    long n0 = (long)blockIdx.x * PT_BGRAD_ROWS;
    long n1 = n0 + PT_BGRAD_ROWS < N ? n0 + PT_BGRAD_ROWS : N;
    float s = 0.f;
    for (long n = n0; n < n1; n++) s += dy[n * O + o];
    atomicAdd(&db[o], s);
}

static inline void pt_linear_bwd(const float* x, const float* w, const float* dy,
        float* dx, float* dw, float* db, int N, int I, int O, cudaStream_t s,
        float beta_dw = 1.f) {
    if (dw) pt_gemm(CUBLAS_OP_T, CUBLAS_OP_N, O, I, N, dy, x, dw, s, 1.f, beta_dw);
    if (db) {
        dim3 g((unsigned)(((long)N + PT_BGRAD_ROWS - 1) / PT_BGRAD_ROWS),
               (unsigned)((O + PTB - 1) / PTB));
        pt_bias_grad<<<g, PTB, 0, s>>>(db, dy, (long)N, O);
    }
    if (dx) pt_gemm(CUBLAS_OP_N, CUBLAS_OP_N, N, I, O, dy, w, dx, s);
}

// ------------------------------------------------------------------ gelu
__global__ void pt_gelu_fwd(float* y, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    }
}

// dY -> dX given saved input x
__global__ void pt_gelu_bwd(float* dx, const float* dy, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float cdf = 0.5f * (1.f + erff(v * 0.70710678118654752f));
        float pdf = 0.3989422804014327f * expf(-0.5f * v * v);
        dx[i] = dy[i] * (cdf + v * pdf);
    }
}

// ------------------------------------------------------------- layernorm
// One block per row; D <= 1024. Saves mean and rstd. eps = 1e-5 (torch).
__global__ void pt_ln_fwd(float* y, float* mean, float* rstd, const float* x,
        const float* w, const float* b, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (long)row * D;
    float* yr = y + (long)row * D;
    __shared__ float sh[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) sh[0] = s / D;
    }
    __syncthreads();
    float mu = sh[0];
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = xr[i] - mu;
        v += d * d;
    }
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) sh[1] = rsqrtf(v / D + 1e-5f);
    }
    __syncthreads();
    float rs = sh[1];
    if (threadIdx.x == 0) { mean[row] = mu; rstd[row] = rs; }
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = (xr[i] - mu) * rs * w[i] + b[i];
}

// dX (written), dW/dB (atomicAdd across rows)
__global__ void pt_ln_bwd(float* dx, float* dw, float* db, const float* dy,
        const float* x, const float* w, const float* mean, const float* rstd,
        int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (long)row * D;
    const float* dyr = dy + (long)row * D;
    float* dxr = dx + (long)row * D;
    float mu = mean[row], rs = rstd[row];
    __shared__ float sh[64];
    float s1 = 0.f, s2 = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xh = (xr[i] - mu) * rs;
        float g = dyr[i] * w[i];
        s1 += g;
        s2 += g * xh;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, o);
        s2 += __shfl_down_sync(0xffffffff, s2, o);
    }
    if ((threadIdx.x & 31) == 0) {
        sh[threadIdx.x >> 5] = s1;
        sh[32 + (threadIdx.x >> 5)] = s2;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        int nw = (blockDim.x + 31) / 32;
        s1 = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
        s2 = threadIdx.x < nw ? sh[32 + threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) {
            s1 += __shfl_down_sync(0xffffffff, s1, o);
            s2 += __shfl_down_sync(0xffffffff, s2, o);
        }
        if (threadIdx.x == 0) { sh[0] = s1 / D; sh[32] = s2 / D; }
    }
    __syncthreads();
    float m1 = sh[0], m2 = sh[32];
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xh = (xr[i] - mu) * rs;
        float g = dyr[i] * w[i];
        dxr[i] = (g - m1 - xh * m2) * rs;
        atomicAdd(&dw[i], dyr[i] * xh);
        atomicAdd(&db[i], dyr[i]);
    }
}

// ------------------------------------------------- masked softmax (attn)
// scores (B,H,L,L): out-of-mask KEYS get -inf before softmax. Saves probs
// in place. keep (B, L) float 0/1 over key positions.
__global__ void pt_attn_softmax_fwd(float* sc, const float* keep,
        int B, int H, int L, float scale) {
    long row = blockIdx.x;                      // one block per (b, h, q)
    if (row >= (long)B * H * L) return;
    int b = (int)(row / ((long)H * L));
    float* r = sc + row * L;
    const float* k = keep + (long)b * L;
    __shared__ float sh[32];
    float mx = -1e30f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        float v = k[i] != 0.f ? r[i] * scale : -1e30f;
        r[i] = v;
        mx = fmaxf(mx, v);
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        mx = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : -1e30f;
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
        if (threadIdx.x == 0) sh[0] = mx;
    }
    __syncthreads();
    mx = sh[0];
    float sum = 0.f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
        float e = expf(r[i] - mx);
        r[i] = e;
        sum += e;
    }
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffff, sum, o);
        if (threadIdx.x == 0) sh[0] = sum;
    }
    __syncthreads();
    float inv = 1.f / sh[0];
    for (int i = threadIdx.x; i < L; i += blockDim.x) r[i] *= inv;
}

// softmax bwd in place: dP -> dS = P*(dP - sum(dP*P)); then unscale
__global__ void pt_attn_softmax_bwd(float* dp, const float* p,
        int B, int H, int L, float scale) {
    long row = blockIdx.x;
    if (row >= (long)B * H * L) return;
    float* dpr = dp + row * L;
    const float* pr = p + row * L;
    __shared__ float sh[32];
    float dot = 0.f;
    for (int i = threadIdx.x; i < L; i += blockDim.x) dot += dpr[i] * pr[i];
    for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = dot;
    __syncthreads();
    if (threadIdx.x < 32) {
        dot = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
        if (threadIdx.x == 0) sh[0] = dot;
    }
    __syncthreads();
    dot = sh[0];
    for (int i = threadIdx.x; i < L; i += blockDim.x)
        dpr[i] = pr[i] * (dpr[i] - dot) * scale;
}

// (B,L,3D) qkv rows -> Q,K,V (B,H,L,Dh) each
__global__ void pt_qkv_split(float* q, float* k, float* v, const float* qkv,
        int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    int D = H * Dh;
    const float* src = qkv + ((long)b * L + l) * 3 * D;
    long dst = (((long)b * H + h) * L + l) * Dh + d;
    q[dst] = src[h * Dh + d];
    k[dst] = src[D + h * Dh + d];
    v[dst] = src[2 * D + h * Dh + d];
}

// inverse: dQ,dK,dV (B,H,L,Dh) -> d_qkv (B,L,3D)
__global__ void pt_qkv_merge(float* dqkv, const float* dq, const float* dk,
        const float* dv, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    int D = H * Dh;
    float* dst = dqkv + ((long)b * L + l) * 3 * D;
    long src = (((long)b * H + h) * L + l) * Dh + d;
    dst[h * Dh + d] = dq[src];
    dst[D + h * Dh + d] = dk[src];
    dst[2 * D + h * Dh + d] = dv[src];
}

// (B,H,L,Dh) -> (B,L,D) concat heads
__global__ void pt_heads_merge(float* out, const float* x, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    out[((long)b * L + l) * H * Dh + h * Dh + d] = x[i];
}

__global__ void pt_heads_split(float* out, const float* x, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    out[i] = x[((long)b * L + l) * H * Dh + h * Dh + d];
}

// ---------------------------------------------------------- misc small ops
__global__ void pt_add(float* y, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void pt_zero(float* y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = 0.f;
}

__global__ void pt_copy(float* y, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

__global__ void pt_scale(float* y, float a, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] *= a;
}

// concat along last dim: copy src (N, D_src) into dst (N, D_dst) at col off
__global__ void pt_copy_block(float* dst, const float* src, int N,
        int d_src, int d_dst, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * d_src) return;
    int n = (int)(i / d_src), c = (int)(i % d_src);
    dst[(long)n * d_dst + off + c] = src[i];
}

// inverse: slice dst (N, d_src) out of src (N, d_dst) at col off
__global__ void pt_slice_block(float* dst, const float* src, int N,
        int d_src, int d_dst, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * d_src) return;
    int n = (int)(i / d_src), c = (int)(i % d_src);
    dst[i] = src[(long)n * d_dst + off + c];
}

// like pt_slice_block but ADDS into dst (grad accumulation)
__global__ void pt_slice_block_add(float* dst, const float* src, int N,
        int d_src, int d_dst, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * d_src) return;
    int n = (int)(i / d_src), c = (int)(i % d_src);
    dst[i] += src[(long)n * d_dst + off + c];
}

// Embedding gather: ids read from a strided float column of the obs.
// out (N, E) = table[clamp(int(obs[n*stride + col]), 0, V-1)]
__global__ void pt_gather_obs(float* out, const float* table, const float* obs,
        long stride, long col, long col_stride, int N, int V, int E) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * E) return;
    int n = (int)(i / E), e = (int)(i % E);
    int id = (int)obs[n * stride + col + (n % 1) * 0 + col_stride * 0];
    (void)col_stride;
    id = id < 0 ? 0 : (id >= V ? V - 1 : id);
    out[i] = table[(long)id * E + e];
}

// scatter-add backward for a gather with per-row ids
__global__ void pt_scatter_obs(float* dtable, const float* dout, const float* obs,
        long stride, long col, int N, int V, int E) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * E) return;
    int n = (int)(i / E), e = (int)(i % E);
    int id = (int)obs[n * stride + col];
    id = id < 0 ? 0 : (id >= V ? V - 1 : id);
    atomicAdd(&dtable[(long)id * E + e], dout[i]);
}

#endif // PTCG_KERNELS_CUH
