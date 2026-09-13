// bindings.cpp - Python bindings for pufferlib (torch-free)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unistd.h>
#include "pufferlib.cu"

#define _PUFFER_STRINGIFY(x) #x
#define PUFFER_STRINGIFY(x) _PUFFER_STRINGIFY(x)

namespace py = pybind11;

// Wrapper functions for Python bindings
pybind11::dict puf_log(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;

    // Summary
    int gpus = pufferl.hypers.world_size;
    long global_step = pufferl.global_step;
    long epoch = pufferl.epoch;
    double now = wall_clock();
    double dt = now - pufferl.last_log_time;
    long sps = dt > 0 ? (long)((global_step - pufferl.last_log_step) / dt) : 0;
    pufferl.last_log_time = now;
    pufferl.last_log_step = global_step;

    result["SPS"] = sps * gpus;
    result["agent_steps"] = global_step * gpus;
    result["uptime"] = now - pufferl.start_time;
    result["epoch"] = epoch;
    // The APPLIED learning rate, read back from the device pointer muon uses.
    // Mirroring the schedule independently on the python side for wandb would
    // mean making every schedule change twice or the telemetry lies
    // (docs/training.md).
    float lr_host = 0.0f;
    if (pufferl.muon.lr_ptr != nullptr) {
        cudaMemcpy(&lr_host, pufferl.muon.lr_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    }
    result["lr"] = lr_host;

    // Environment stats
    pybind11::dict env_dict;
    Dict* env_out = log_environments_impl(pufferl);
    for (int i = 0; i < env_out->size; i++) {
        env_dict[env_out->items[i].key] = env_out->items[i].value;
    }
    result["env"] = env_dict;

    // Losses
    pybind11::dict losses_dict;
    float losses_host[NUM_LOSSES];
    cudaMemcpy(losses_host, pufferl.losses_puf.data, sizeof(losses_host), cudaMemcpyDeviceToHost);
    // KL guard telemetry: minibatches vetoed since the last log. Only the
    // SKIPPED slot is reset — the GATE slot must keep its last value (muon
    // reads it every step) and MB_KL is overwritten every minibatch anyway.
    float guard_host[KL_GUARD_SLOTS];
    cudaMemcpy(guard_host, pufferl.kl_guard_puf.data, sizeof(guard_host), cudaMemcpyDeviceToHost);
    float n = losses_host[LOSS_N];
    if (n > 0) {
        float inv_n = 1.0f / n;
        losses_dict["policy"] = losses_host[LOSS_PG] * inv_n;
        losses_dict["value"] = losses_host[LOSS_VF] * inv_n;
        losses_dict["entropy"] = losses_host[LOSS_ENT] * inv_n;
        losses_dict["total"] = losses_host[LOSS_TOTAL] * inv_n;
        losses_dict["old_kl"] = losses_host[LOSS_OLD_APPROX_KL] * inv_n;
        losses_dict["kl"] = losses_host[LOSS_APPROX_KL] * inv_n;
        losses_dict["clipfrac"] = losses_host[LOSS_CLIPFRAC] * inv_n;
        losses_dict["kl_skipped"] = guard_host[KL_GUARD_SKIPPED];
        losses_dict["vf_skipped"] = guard_host[KL_GUARD_SKIPPED_VF];
    }
    cudaMemset(pufferl.losses_puf.data, 0, numel(pufferl.losses_puf.shape) * sizeof(float));
    // SKIPPED and SKIPPED_VF are adjacent — zero both counters
    cudaMemset(pufferl.kl_guard_puf.data + KL_GUARD_SKIPPED, 0, 2 * sizeof(float));
    result["loss"] = losses_dict;

    // Profile
    pybind11::dict perf_dict;
    float train_total = 0;
    for (int i = 0; i < NUM_PROF; i++) {
        float sec = pufferl.profile.accum[i] / 1000.0f;
        perf_dict[PROF_NAMES[i]] = sec;
        if (i >= PROF_TRAIN_MISC) train_total += sec;
    }
    perf_dict["train"] = train_total;
    memset(pufferl.profile.accum, 0, sizeof(pufferl.profile.accum));
    result["perf"] = perf_dict;

    // Utilization
    pybind11::dict util_dict;
    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(pufferl.nvml_device, &util);
    util_dict["gpu_percent"] = (float)util.gpu;

    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(pufferl.nvml_device, &mem);
    util_dict["gpu_mem"] = 100.0f * (float)mem.used / (float)mem.total;

    size_t cuda_free, cuda_total;
    cudaMemGetInfo(&cuda_free, &cuda_total);
    util_dict["vram_used_gb"] = (float)(cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
    util_dict["vram_total_gb"] = (float)cuda_total / (1024.0f * 1024.0f * 1024.0f);

    long rss_kb = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
        }
        fclose(f);
    }
    util_dict["cpu_mem_gb"] = (float)rss_kb / (1024.0f * 1024.0f);
    result["util"] = util_dict;

    return result;
}

pybind11::dict puf_eval_log(pybind11::object pufferl_obj) {
    auto& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::dict result;

    double now = wall_clock();
    pufferl.last_log_time = now;
    pufferl.last_log_step = pufferl.global_step;
 
    pybind11::dict env_dict;
    // Capacity sized for per-bank keys (ptcg league_w_<b>/league_n_<b>,
    // chess hist_score_bank/hist_n_bank) on top of base env-log fields.
    Dict* env_out = create_dict(128);
    static_vec_eval_log(pufferl.vec, env_out);
    for (int i = 0; i < env_out->size; i++) {
        env_dict[env_out->items[i].key] = env_out->items[i].value;
    }
    result["env"] = env_dict;

    return result;
}

void python_vec_recv(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

void python_vec_send(pybind11::object pufferl_obj, int buf) {
    // Not used in static/OMP path
}

void render(pybind11::object pufferl_obj, int env_id) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    static_vec_render(pufferl.vec, env_id);
}

void rollouts(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    pybind11::gil_scoped_release no_gil;
    double t0 = wall_clock();

    // Zero state buffers (primary + every frozen bank, so all banks see fresh
    // state symmetrically — otherwise frozen banks accumulate indefinitely while
    // primary resets, giving primary an unfair in-distribution advantage).
    if (pufferl.hypers.reset_state) {
        for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
            puf_zero(&pufferl.buffer_states[i], pufferl.default_stream);
        }
        for (int b = 0; b < pufferl.num_frozen_banks; b++) {
            for (int i = 0; i < pufferl.hypers.num_buffers; i++) {
                puf_zero(&pufferl.frozen_banks[b].buffer_states[i], pufferl.default_stream);
            }
        }
    }

    static_vec_omp_step(pufferl.vec);
    float sec = (float)(wall_clock() - t0);
    pufferl.profile.accum[PROF_ROLLOUT] += sec * 1000.0f;  // store as ms

    float eval_prof[NUM_EVAL_PROF];
    static_vec_read_profile(pufferl.vec, eval_prof);
    pufferl.profile.accum[PROF_EVAL_GPU] += eval_prof[EVAL_GPU];
    pufferl.profile.accum[PROF_EVAL_ENV] += eval_prof[EVAL_ENV_STEP];
    pufferl.global_step += pufferl.hypers.horizon * pufferl.hypers.total_agents;
}

pybind11::dict train(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    {
        pybind11::gil_scoped_release no_gil;
        train_impl(pufferl);
    }
    pybind11::dict losses;
    return losses;
}

void puf_close(pybind11::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    close_impl(pufferl);
}

// Weight files are PACKED fp32 blobs (torch-converter format); the live
// arena is 16B-aligned per tensor — every save/load goes through the
// packed<->arena helpers in pufferlib.cu.
void save_weights(pybind11::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t arena_n = numel(pufferl.master_weights.shape);
    int64_t packed_n = pufferl.params_alloc.total_elems;
    std::vector<float> arena_img(arena_n);
    cudaMemcpy(arena_img.data(), pufferl.master_weights.data,
               arena_n * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> packed(packed_n);
    arena_host_to_packed_blob(&pufferl.params_alloc, arena_img.data(), packed.data());
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("Failed to open " + path + " for writing");
    fwrite(packed.data(), sizeof(float), packed_n, f);
    fclose(f);
}

// Parity/debug entry: run the train-path policy forward on a caller-supplied
// obs batch and return the decoder output (for ptcg: (B, 65) = [64 raw option
// scores || oracle value]). Batch size must equal the minibatch the trainer
// was created with — activations are sized for exactly that shape.
pybind11::array_t<float> debug_forward(pybind11::object pufferl_obj,
        pybind11::array_t<float, pybind11::array::c_style | pybind11::array::forcecast> obs) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    if (sizeof(precision_t) != sizeof(float))
        throw std::runtime_error("debug_forward: fp32 builds only");
    TrainGraph& graph = pufferl.train_buf;
    long need = numel(graph.mb_obs.shape);
    if ((long)obs.size() != need)
        throw std::runtime_error("debug_forward: need exactly " +
            std::to_string(need) + " floats (minibatch " +
            std::to_string(graph.mb_obs.shape[0] * graph.mb_obs.shape[1]) +
            " x obs " + std::to_string(graph.mb_obs.shape[2]) + "), got " +
            std::to_string((long)obs.size()));
    cudaMemcpy(graph.mb_obs.data, obs.data(), need * sizeof(float),
               cudaMemcpyHostToDevice);
    puf_zero(&graph.mb_state, pufferl.default_stream);
    PrecisionTensor dec = policy_forward_train(&pufferl.policy, pufferl.weights,
        pufferl.train_activations, graph.mb_obs, graph.mb_state,
        pufferl.default_stream);
    cudaStreamSynchronize(pufferl.default_stream);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("debug_forward CUDA error: ") +
            cudaGetErrorString(e));
    long rows = dec.shape[0], cols = numel(dec.shape) / dec.shape[0];
    pybind11::array_t<float> out({rows, cols});
    cudaMemcpy(out.mutable_data(), dec.data, rows * cols * sizeof(float),
               cudaMemcpyDeviceToHost);
    return out;
}

// Full trainer state for crash resume: weights + muon momentum + step/epoch.
// Separate from save_weights: the step-stamped weight .bins stay bare fp32
// blobs (the eval converter and native_weights.py depend on that layout).
// Written atomically (tmp + rename) so a crash mid-write never eats the
// previous state.
struct PtcgStateHeader {
    char magic[8];       // "PTCGSTA1"
    long global_step;
    long epoch;
    long n_params;
};

void save_state(pybind11::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    long arena_n = numel(pufferl.master_weights.shape);
    long packed_n = pufferl.params_alloc.total_elems;
    // momentum mirrors the aligned arena layout — same gather applies
    std::vector<float> w_img(arena_n), m_img(arena_n);
    cudaMemcpy(w_img.data(), pufferl.master_weights.data,
               arena_n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(m_img.data(), pufferl.muon.mb_puf.data,
               arena_n * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> w_packed(packed_n), m_packed(packed_n);
    arena_host_to_packed_blob(&pufferl.params_alloc, w_img.data(), w_packed.data());
    arena_host_to_packed_blob(&pufferl.params_alloc, m_img.data(), m_packed.data());
    PtcgStateHeader hdr;
    memcpy(hdr.magic, "PTCGSTA1", 8);
    hdr.global_step = pufferl.global_step;
    hdr.epoch = pufferl.epoch;
    hdr.n_params = packed_n;
    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("Failed to open " + tmp + " for writing");
    fwrite(&hdr, sizeof(hdr), 1, f);
    fwrite(w_packed.data(), sizeof(float), packed_n, f);
    fwrite(m_packed.data(), sizeof(float), packed_n, f);
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    if (rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("Failed to rename " + tmp + " -> " + path);
}

void load_state(pybind11::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    long arena_n = numel(pufferl.master_weights.shape);
    long packed_n = pufferl.params_alloc.total_elems;
    int64_t packed_bytes = packed_n * (int64_t)sizeof(float);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Failed to open " + path + " for reading");
    PtcgStateHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || memcmp(hdr.magic, "PTCGSTA1", 8) != 0) {
        fclose(f);
        throw std::runtime_error(path + " is not a PTCGSTA1 state file");
    }
    if (hdr.n_params != packed_n) {
        fclose(f);
        throw std::runtime_error("State param count mismatch: file " +
            std::to_string(hdr.n_params) + ", model " + std::to_string(packed_n));
    }
    std::vector<float> w_packed(packed_n), m_packed(packed_n);
    if (fread(w_packed.data(), 1, packed_bytes, f) != (size_t)packed_bytes ||
        fread(m_packed.data(), 1, packed_bytes, f) != (size_t)packed_bytes) {
        fclose(f);
        throw std::runtime_error("Truncated state file " + path);
    }
    fclose(f);
    std::vector<float> w_img(arena_n, 0.0f), m_img(arena_n, 0.0f);
    packed_blob_to_arena_host(&pufferl.params_alloc, w_packed.data(), w_img.data());
    packed_blob_to_arena_host(&pufferl.params_alloc, m_packed.data(), m_img.data());
    cudaMemcpy(pufferl.master_weights.data, w_img.data(),
               arena_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pufferl.muon.mb_puf.data, m_img.data(),
               arena_n * sizeof(float), cudaMemcpyHostToDevice);
    if (USE_BF16) {
        int nn = numel(pufferl.param_puf.shape);
        cast<<<grid_size(nn), BLOCK_SIZE, 0, pufferl.default_stream>>>(
            pufferl.param_puf.data, pufferl.master_weights.data, nn);
    }
    cudaDeviceSynchronize();
    // epoch drives the cosine LR + prio-beta anneals in train_impl; restoring
    // it (not just global_step) is what makes the schedule resume-correct.
    pufferl.global_step = hdr.global_step;
    pufferl.epoch = hdr.epoch;
    pufferl.last_log_step = hdr.global_step;
}

void load_weights(pybind11::object pufferl_obj, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    int64_t packed_bytes = pufferl.params_alloc.total_elems * (int64_t)sizeof(float);
    int64_t arena_n = numel(pufferl.master_weights.shape);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Failed to open " + path + " for reading");
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size != packed_bytes) {
        fclose(f);
        throw std::runtime_error("Weight file size mismatch: expected " +
            std::to_string(packed_bytes) + " bytes, got " + std::to_string(file_size));
    }
    std::vector<float> packed(pufferl.params_alloc.total_elems);
    size_t nread = fread(packed.data(), 1, packed_bytes, f);
    fclose(f);
    if ((int64_t)nread != packed_bytes)
        throw std::runtime_error("Failed to read weight file");
    std::vector<float> arena_img(arena_n, 0.0f);
    packed_blob_to_arena_host(&pufferl.params_alloc, packed.data(), arena_img.data());
    cudaMemcpy(pufferl.master_weights.data, arena_img.data(),
               arena_n * sizeof(float), cudaMemcpyHostToDevice);
    if (USE_BF16) {
        int n = numel(pufferl.param_puf.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, pufferl.default_stream>>>(
            pufferl.param_puf.data, pufferl.master_weights.data, n);
    }
}

int py_add_frozen_bank(py::object pufferl_obj, int slice_size,
                       int hidden_size, int num_layers) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    return pufferl_add_frozen_bank(&pufferl, slice_size, hidden_size, num_layers);
}

void py_load_frozen_bank(py::object pufferl_obj, int bank_idx, const std::string& path) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    pufferl_load_frozen_bank(&pufferl, bank_idx, path.c_str());
}

void py_set_agent_perm(py::object pufferl_obj, py::array_t<int> perm) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto buf = perm.request();
    if (buf.ndim != 1) throw std::runtime_error("agent_perm must be 1-D");
    if ((int)buf.shape[0] != pufferl.vec->total_agents) {
        throw std::runtime_error("agent_perm length must equal total_agents");
    }
    pufferl_set_agent_perm(&pufferl, (const int*)buf.ptr);
}

void py_set_env_tags(py::object pufferl_obj, py::array_t<int> tags) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    auto buf = tags.request();
    if (buf.ndim != 1) throw std::runtime_error("env_tags must be 1-D");
    int num_envs = pufferl_num_envs(&pufferl);
    if ((int)buf.shape[0] != num_envs) {
        throw std::runtime_error("env_tags length must equal num_envs");
    }
    pufferl_set_env_tags(&pufferl, (const int*)buf.ptr);
}

int py_count_aligned(py::object pufferl_obj, int tag_value, int reset_flags) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    return pufferl_count_aligned(&pufferl, tag_value, reset_flags);
}

// Mid-run deck-pool append. tables.c is C, linked in via libstatic_ptcg.a.
extern "C" int pt_reload_decks(const char* path);

// Swap in a grown training deck pool. MUST be called between _C.train()
// returning and the next _C.rollouts(): the env threads are parked at the
// fork-join barrier there, which is the only reason a plain pointer publish is
// safe (docs/deck-pool.md).
// Returns the new deck count, or -1 if the blob was rejected (nothing mutated).
int py_reload_decks(const std::string& path) {
    return pt_reload_decks(path.c_str());
}

// Per-deck sampling weights. Names are resolved to ids on the PYTHON side (it
// owns the pool listing), so this takes a dense vector whose length must equal
// the live n_decks -- the length check is the guard against a caller that
// resolved against a stale listing.
//
// Unlike reload_decks this does not require parked callers: it publishes under
// the same lock game_start holds for reading. Returns the number of decks with
// weight > 0, or -1 if the vector was rejected (nothing mutated).
extern "C" int pt_set_deck_weights(const float* w, int n);
extern "C" void pt_clear_deck_weights(void);

int py_set_deck_weights(const std::vector<float>& w) {
    return pt_set_deck_weights(w.data(), (int)w.size());
}

void py_clear_deck_weights() { pt_clear_deck_weights(); }

int py_allreduce_i32(py::object pufferl_obj, int value, int op) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    return pufferl_allreduce_i32(&pufferl, value, op);
}

int py_num_envs(py::object pufferl_obj) {
    PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
    return pufferl_num_envs(&pufferl);
}

void py_puff_advantage(
        long long values_ptr, long long rewards_ptr,
        long long dones_ptr,  long long importance_ptr,
        long long advantages_ptr,
        int num_steps, int horizon,
        float gamma, float lambda, float rho_clip, float c_clip) {
    constexpr int N = 16 / sizeof(precision_t);
    int blocks = grid_size(num_steps);
    auto kernel = (horizon % N == 0) ? puff_advantage : puff_advantage_scalar;
    kernel<<<blocks, 256>>>(
        (const precision_t*)values_ptr, (const precision_t*)rewards_ptr,
        (const precision_t*)dones_ptr,  (const precision_t*)importance_ptr,
        (precision_t*)advantages_ptr,
        gamma, lambda, rho_clip, c_clip, num_steps, horizon);
}

double get_config(py::dict& kwargs, const char* key) {
    if (!kwargs.contains(key)) {
        throw std::runtime_error(std::string("Missing config key: ") + key);
    }
    try {
        return kwargs[key].cast<double>();
    } catch (const py::cast_error& e) {
        throw std::runtime_error(std::string("Failed to cast config key '") + key + "': " + e.what());
    }
}

// For keys added after callers already existed (parity/bench harnesses build
// their own train dicts) — missing means "take the default", not "crash".
double get_config_or(py::dict& kwargs, const char* key, double fallback) {
    if (!kwargs.contains(key)) {
        return fallback;
    }
    return get_config(kwargs, key);
}

Dict* py_dict_to_c_dict(py::dict py_dict) {
    Dict* c_dict = create_dict(py_dict.size());
    for (auto item : py_dict) {
        const char* key = PyUnicode_AsUTF8(item.first.ptr());
        try {
            dict_set(c_dict, key, item.second.cast<double>());
        } catch (const py::cast_error&) {
            // Skip non-numeric values
        }
    }
    return c_dict;
}

// ============================================================================
// Python-facing VecEnv: wraps StaticVec for use from python_pufferl.py.
// After vec_step(), GPU buffers are current — Python wraps them zero-copy
// with torch.from_blob(ptr, shape, dtype, device='cuda').
// ============================================================================

struct VecEnv {
    StaticVec* vec;
    int total_agents;
    int obs_size;
    int num_atns;
    std::vector<int> act_sizes;
    std::string obs_dtype;
    size_t obs_elem_size;
    int gpu;
};

std::unique_ptr<VecEnv> create_vec(py::dict args, int gpu) {
    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();

    int total_agents = (int)get_config(vec_kwargs, "total_agents");
    int num_buffers  = (int)get_config(vec_kwargs, "num_buffers");

    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs);
    Dict* env_dict = py_dict_to_c_dict(env_kwargs);

    auto ve = std::make_unique<VecEnv>();
    ve->gpu = gpu;
    {
        py::gil_scoped_release no_gil;
        ve->vec = create_static_vec(total_agents, num_buffers, gpu, vec_dict, env_dict);
    }
    ve->total_agents  = total_agents;
    ve->obs_size      = get_obs_size();
    ve->num_atns      = get_num_atns();
    {
        int* raw = get_act_sizes();
        int  n   = get_num_act_sizes();
        ve->act_sizes = std::vector<int>(raw, raw + n);
    }
    ve->obs_dtype     = std::string(get_obs_dtype());
    ve->obs_elem_size = get_obs_elem_size();
    return ve;
}

void vec_reset(VecEnv& ve) {
    py::gil_scoped_release no_gil;
    static_vec_reset(ve.vec);
}

void gpu_vec_step_py(VecEnv& ve, long long actions_ptr) {
    cudaMemcpy(ve.vec->gpu_actions, (void*)actions_ptr,
        (size_t)ve.total_agents * ve.num_atns * sizeof(float),
        cudaMemcpyDeviceToDevice);
    {
        py::gil_scoped_release no_gil;
        gpu_vec_step(ve.vec);
    }
}

void cpu_vec_step_py(VecEnv& ve, long long actions_ptr) {
    memcpy(ve.vec->actions, (void*)actions_ptr,
        (size_t)ve.total_agents * ve.num_atns * sizeof(float));
    {
        py::gil_scoped_release no_gil;
        cpu_vec_step(ve.vec);
    }
}

py::dict vec_log(VecEnv& ve) {
    Dict* out = create_dict(32);
    static_vec_log(ve.vec, out);
    py::dict result;
    for (int i = 0; i < out->size; i++) {
        result[out->items[i].key] = out->items[i].value;
    }
    free(out->items);
    free(out);
    return result;
}

void vec_close(VecEnv& ve) {
    static_vec_close(ve.vec);
    ve.vec = nullptr;
}

std::unique_ptr<PuffeRL> create_pufferl(py::dict args) {
    py::dict train_kwargs = args["train"].cast<py::dict>();
    py::dict vec_kwargs = args["vec"].cast<py::dict>();
    py::dict env_kwargs = args["env"].cast<py::dict>();
    py::dict policy_kwargs = args["policy"].cast<py::dict>();

    HypersT hypers;
    // Layout (total_agents and num_buffers come from vec config)
    hypers.total_agents = get_config(vec_kwargs, "total_agents");
    hypers.num_buffers = get_config(vec_kwargs, "num_buffers");
    hypers.num_threads = get_config(vec_kwargs, "num_threads");
    hypers.horizon = get_config(train_kwargs, "horizon");
    // Model architecture (num_atns computed from env in C++)
    hypers.hidden_size = get_config(policy_kwargs, "hidden_size");
    hypers.num_layers = get_config(policy_kwargs, "num_layers");
    // Learning rate
    hypers.lr = get_config(train_kwargs, "learning_rate");
    hypers.min_lr_ratio = get_config(train_kwargs, "min_lr_ratio");
    hypers.anneal_lr = get_config(train_kwargs, "anneal_lr");
    // defaulted: model_parity.py builds its own train dict (docs/training.md)
    hypers.lr_warmup_epochs = (int)get_config_or(train_kwargs, "lr_warmup_epochs", 0);
    hypers.start_epoch = (int)get_config_or(train_kwargs, "start_epoch", 0);
    hypers.lr_anneal_from_epoch = (int)get_config_or(train_kwargs, "lr_anneal_from_epoch", 0);
    // Optimizer
    hypers.beta1 = get_config(train_kwargs, "beta1");
    hypers.beta2 = get_config(train_kwargs, "beta2");
    hypers.eps = get_config(train_kwargs, "eps");
    // Training
    hypers.minibatch_size = get_config(train_kwargs, "minibatch_size");
    // defaulted: parity/eval tools build their own train dicts without it
    hypers.accum_minibatches = (int)get_config_or(train_kwargs, "accum_minibatches", 1);
    hypers.replay_ratio = get_config(train_kwargs, "replay_ratio");
    hypers.total_timesteps = get_config(train_kwargs, "total_timesteps");
    hypers.max_grad_norm = get_config(train_kwargs, "max_grad_norm");
    // PPO
    hypers.clip_coef = get_config(train_kwargs, "clip_coef");
    hypers.vf_clip_coef = get_config(train_kwargs, "vf_clip_coef");
    hypers.vf_coef = get_config(train_kwargs, "vf_coef");
    hypers.ent_coef = get_config(train_kwargs, "ent_coef");
    hypers.min_ent_coef_ratio = get_config(train_kwargs, "min_ent_coef_ratio");
    hypers.anneal_ent_coef = get_config(train_kwargs, "anneal_ent_coef");
    // GAE
    hypers.gamma = get_config(train_kwargs, "gamma");
    hypers.gae_lambda = get_config(train_kwargs, "gae_lambda");
    // VTrace
    hypers.vtrace_rho_clip = get_config(train_kwargs, "vtrace_rho_clip");
    hypers.vtrace_c_clip = get_config(train_kwargs, "vtrace_c_clip");
    // Priority
    hypers.prio_alpha = get_config(train_kwargs, "prio_alpha");
    hypers.prio_beta0 = get_config(train_kwargs, "prio_beta0");
    // KL/value guard + Muon 1-D weight decay (docs/training.md).
    // Defaulted so existing callers keep working — and default PROTECTIVE:
    // a run that forgets the flags still gets the guard. vf default 0.5 =
    // ~20x the healthy value loss (0.023) and well under the measured
    // ignition trajectory (0.5 -> 10 -> 160 within ~20 epochs).
    hypers.kl_skip_threshold = get_config_or(train_kwargs, "kl_skip_threshold", 0.05);
    hypers.vf_skip_threshold = get_config_or(train_kwargs, "vf_skip_threshold", 0.5);
    hypers.muon_wd_1d = get_config_or(train_kwargs, "muon_wd_1d", 0.01);
    hypers.league_routing = (int)get_config_or(train_kwargs, "league_routing", 0);
    hypers.reset_state = get_config(args, "reset_state");
    // Base-level config ([base] section becomes top-level in args)
    hypers.cudagraphs = get_config(args, "cudagraphs");
    hypers.profile = get_config(args, "profile");
    // Multi-GPU / device selection
    hypers.rank = get_config(args, "rank");
    hypers.world_size = get_config(args, "world_size");
    hypers.gpu_id = get_config(args, "gpu_id");
    hypers.nccl_id = args["nccl_id"].cast<std::string>();
    // Seed
    hypers.seed = get_config(args, "seed");

    int device_count = 0;
    int err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA is not available");
    }

    std::string env_name = args["env_name"].cast<std::string>();
    Dict* vec_dict = py_dict_to_c_dict(vec_kwargs.cast<py::dict>());
    Dict* env_dict = py_dict_to_c_dict(env_kwargs.cast<py::dict>());

    std::unique_ptr<PuffeRL> pufferl;
    {
        pybind11::gil_scoped_release no_gil;
        pufferl = create_pufferl_impl(hypers, env_name, vec_dict, env_dict);
    }

    if (!pufferl) {
        throw std::runtime_error("CUDA OOM: failed to allocate training buffers");
    }

    return pufferl;
}

PYBIND11_MODULE(_C, m) {
    // Multi-GPU: generate NCCL unique ID (call on rank 0, pass bytes to all ranks)
    m.def("get_nccl_id", []() {
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        return py::bytes(reinterpret_cast<char*>(&id), sizeof(id));
    });
    // Standalone utilization monitor (no PuffeRL instance needed)
    m.def("get_utilization", [](int gpu_id) {
        static bool nvml_inited = false;
        if (!nvml_inited) { nvmlInit(); nvml_inited = true; }

        py::dict util_dict;
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(gpu_id, &device);

        nvmlUtilization_t util;
        nvmlDeviceGetUtilizationRates(device, &util);
        util_dict["gpu_percent"] = (float)util.gpu;

        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(device, &mem);
        util_dict["gpu_mem"] = 100.0f * (float)mem.used / (float)mem.total;

        size_t cuda_free, cuda_total;
        cudaMemGetInfo(&cuda_free, &cuda_total);
        util_dict["vram_used_gb"] = (float)(cuda_total - cuda_free) / (1024.0f * 1024.0f * 1024.0f);
        util_dict["vram_total_gb"] = (float)cuda_total / (1024.0f * 1024.0f * 1024.0f);

        long rss_kb = 0;
        FILE* f = fopen("/proc/self/status", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (sscanf(line, "VmRSS: %ld", &rss_kb) == 1) break;
            }
            fclose(f);
        }
        util_dict["cpu_mem_gb"] = (float)rss_kb / (1024.0f * 1024.0f);

        return util_dict;
    });

    m.attr("precision_bytes") = (int)sizeof(precision_t);
    m.attr("env_name") = PUFFER_STRINGIFY(ENV_NAME);
    m.attr("gpu") = 1;

    // Core functions
    m.def("log", &puf_log);
    m.def("eval_log", &puf_eval_log);
    m.def("render", &render);
    m.def("rollouts", &rollouts);
    m.def("train", &train);
    m.def("close", &puf_close);
    m.def("save_weights", &save_weights);
    m.def("load_weights", &load_weights);
    m.def("save_state", &save_state);
    m.def("load_state", &load_state);
    // Weights-only restarts: stamp the checkpoint's per-rank step so
    // global_step/epoch (and everything derived: wandb x-axes, checkpoint
    // filenames, eval epoch numbering, epoch-driven schedules) continue from
    // the parent run instead of 0. load_state overrides this on real resumes.
    m.def("set_start", [](py::object pufferl_obj, long step) {
        PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
        long spe = (long)pufferl.hypers.total_agents * pufferl.hypers.horizon;
        pufferl.global_step = step;
        pufferl.epoch = spe > 0 ? step / spe : 0;
        pufferl.last_log_step = step;
    });
    m.def("debug_forward", &debug_forward);
    m.def("add_frozen_bank", &py_add_frozen_bank);
    m.def("load_frozen_bank", &py_load_frozen_bank);
    m.def("set_agent_perm", &py_set_agent_perm);
    m.def("set_env_tags", &py_set_env_tags);
    m.def("count_aligned", &py_count_aligned);
    m.def("reload_decks", &py_reload_decks);
    m.def("set_deck_weights", &py_set_deck_weights);
    m.def("clear_deck_weights", &py_clear_deck_weights);
    m.def("allreduce_i32", &py_allreduce_i32, py::arg("pufferl"),
          py::arg("value"), py::arg("op") = 0);
    m.def("num_envs", &py_num_envs);
    m.def("python_vec_recv", &python_vec_recv);
    m.def("python_vec_send", &python_vec_send);
    py::class_<Policy>(m, "Policy");
    py::class_<Muon>(m, "Muon");
    py::class_<Allocator>(m, "Allocator")
        .def(py::init<>());

    py::class_<HypersT>(m, "HypersT")
        .def_readwrite("horizon", &HypersT::horizon)
        .def_readwrite("total_agents", &HypersT::total_agents)
        .def_readwrite("num_buffers", &HypersT::num_buffers)
        .def_readwrite("num_atns", &HypersT::num_atns)
        .def_readwrite("hidden_size", &HypersT::hidden_size)

        .def_readwrite("replay_ratio", &HypersT::replay_ratio)
        .def_readwrite("num_layers", &HypersT::num_layers)
        .def_readwrite("lr", &HypersT::lr)
        .def_readwrite("min_lr_ratio", &HypersT::min_lr_ratio)
        .def_readwrite("anneal_lr", &HypersT::anneal_lr)
        .def_readwrite("lr_warmup_epochs", &HypersT::lr_warmup_epochs)
        .def_readwrite("lr_anneal_from_epoch", &HypersT::lr_anneal_from_epoch)
        .def_readwrite("beta1", &HypersT::beta1)
        .def_readwrite("beta2", &HypersT::beta2)
        .def_readwrite("eps", &HypersT::eps)
        .def_readwrite("total_timesteps", &HypersT::total_timesteps)
        .def_readwrite("max_grad_norm", &HypersT::max_grad_norm)
        .def_readwrite("clip_coef", &HypersT::clip_coef)
        .def_readwrite("vf_clip_coef", &HypersT::vf_clip_coef)
        .def_readwrite("vf_coef", &HypersT::vf_coef)
        .def_readwrite("ent_coef", &HypersT::ent_coef)
        .def_readwrite("min_ent_coef_ratio", &HypersT::min_ent_coef_ratio)
        .def_readwrite("anneal_ent_coef", &HypersT::anneal_ent_coef)
        .def_readwrite("gamma", &HypersT::gamma)
        .def_readwrite("gae_lambda", &HypersT::gae_lambda)
        .def_readwrite("vtrace_rho_clip", &HypersT::vtrace_rho_clip)
        .def_readwrite("vtrace_c_clip", &HypersT::vtrace_c_clip)
        .def_readwrite("prio_alpha", &HypersT::prio_alpha)
        .def_readwrite("prio_beta0", &HypersT::prio_beta0)
        .def_readwrite("cudagraphs", &HypersT::cudagraphs)
        .def_readwrite("profile", &HypersT::profile)
        .def_readwrite("rank", &HypersT::rank)
        .def_readwrite("world_size", &HypersT::world_size)
        .def_readwrite("gpu_id", &HypersT::gpu_id)
        .def_readwrite("nccl_id", &HypersT::nccl_id);

    py::class_<PrecisionTensor>(m, "PrecisionTensor")
        .def("__repr__", [](const PrecisionTensor& t) { return std::string(puf_repr(&t)); })
        .def("ndim", [](const PrecisionTensor& t) { return ndim(t.shape); })
        .def("numel", [](const PrecisionTensor& t) { return numel(t.shape); });
    py::class_<FloatTensor>(m, "FloatTensor")
        .def("__repr__", [](const FloatTensor& t) { return std::string(puf_repr(&t)); })
        .def("ndim", [](const FloatTensor& t) { return ndim(t.shape); })
        .def("numel", [](const FloatTensor& t) { return numel(t.shape); });

    py::class_<RolloutBuf>(m, "RolloutBuf")
        .def_readwrite("observations", &RolloutBuf::observations)
        .def_readwrite("actions", &RolloutBuf::actions)
        .def_readwrite("values", &RolloutBuf::values)
        .def_readwrite("logprobs", &RolloutBuf::logprobs)
        .def_readwrite("rewards", &RolloutBuf::rewards)
        .def_readwrite("terminals", &RolloutBuf::terminals)
        .def_readwrite("ratio", &RolloutBuf::ratio)
        .def_readwrite("importance", &RolloutBuf::importance);

    m.def("uptime", [](py::object pufferl_obj) -> double {
        PuffeRL& pufferl = pufferl_obj.cast<PuffeRL&>();
        double now = wall_clock();
        return now - pufferl.start_time;
    });
    m.def("puff_advantage", &py_puff_advantage);
    m.def("create_vec", &create_vec, py::arg("args"), py::arg("gpu") = 1);
    py::class_<VecEnv, std::unique_ptr<VecEnv>>(m, "VecEnv")
        .def_readonly("total_agents",  &VecEnv::total_agents)
        .def_readonly("obs_size",      &VecEnv::obs_size)
        .def_readonly("num_atns",      &VecEnv::num_atns)
        .def_readonly("act_sizes",     &VecEnv::act_sizes)
        .def_readonly("obs_dtype",     &VecEnv::obs_dtype)
        .def_readonly("obs_elem_size", &VecEnv::obs_elem_size)
        .def_readonly("gpu",           &VecEnv::gpu)
        // GPU buffer pointers — wrap with torch.from_blob(..., device='cuda')
        .def_property_readonly("gpu_obs_ptr",       [](VecEnv& ve) { return (long long)ve.vec->gpu_observations; })
        .def_property_readonly("gpu_rewards_ptr",   [](VecEnv& ve) { return (long long)ve.vec->gpu_rewards; })
        .def_property_readonly("gpu_terminals_ptr", [](VecEnv& ve) { return (long long)ve.vec->gpu_terminals; })
        // CPU buffer pointers (same as gpu_ in CPU mode since they alias)
        .def_property_readonly("obs_ptr",       [](VecEnv& ve) { return (long long)ve.vec->observations; })
        .def_property_readonly("rewards_ptr",   [](VecEnv& ve) { return (long long)ve.vec->rewards; })
        .def_property_readonly("terminals_ptr", [](VecEnv& ve) { return (long long)ve.vec->terminals; })
        .def("reset", &vec_reset)
        .def("gpu_step", &gpu_vec_step_py)
        .def("cpu_step", &cpu_vec_step_py)
        .def("render", [](VecEnv& ve, int env_id) { static_vec_render(ve.vec, env_id); })
        .def("log",   &vec_log)
        .def("close", &vec_close);

    m.def("create_pufferl", &create_pufferl);
    py::class_<PuffeRL, std::unique_ptr<PuffeRL>>(m, "PuffeRL")
        .def_readwrite("policy", &PuffeRL::policy)
        .def_readwrite("muon", &PuffeRL::muon)
        .def_readwrite("hypers", &PuffeRL::hypers)
        .def_readwrite("rollouts", &PuffeRL::rollouts)
        .def_readonly("epoch", &PuffeRL::epoch)
        .def_readonly("global_step", &PuffeRL::global_step)
        .def_readonly("last_log_time", &PuffeRL::last_log_time)
        .def("num_params", [](PuffeRL& self) -> int64_t {
            return numel(self.master_weights.shape);
        });
}
