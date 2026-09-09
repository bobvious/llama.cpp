#include "nvfp4-trace.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

// See nvfp4-trace.cuh for why this exists. Short version: every bug this feature has
// had was a SILENT fallback, so no output-shaped test could see it.

static int nvfp4_trace_level() {
    // Read once. getenv on every matmul would itself be a measurable cost.
    static const int level = [] {
        const char * s = getenv("GGML_CUDA_NVFP4_TRACE");
        return s ? atoi(s) : 0;
    }();
    return level;
}

bool ggml_cuda_nvfp4_trace_enabled() {
    return nvfp4_trace_level() > 0;
}

static const char * nvfp4_path_name(ggml_cuda_nvfp4_path p) {
    switch (p) {
        case GGML_CUDA_NVFP4_PATH_MMVQ:       return "MMVQ(q8_1)";
        case GGML_CUDA_NVFP4_PATH_MMQ_NATIVE: return "MMQ-native-fp4";
        case GGML_CUDA_NVFP4_PATH_MMQ_Q8:     return "MMQ(q8_1)";
        case GGML_CUDA_NVFP4_PATH_MMF:        return "MMF(float)";
        case GGML_CUDA_NVFP4_PATH_CUBLAS:     return "cuBLAS";
        default:                              return "?";
    }
}

namespace {
struct nvfp4_trace_state {
    std::mutex             mtx;
    std::set<std::string>  seen;      // distinct outcomes already reported
    size_t                 n_graphs         = 0;
    size_t                 n_nvfp4_mm       = 0;  // NVFP4 matmuls dispatched
    size_t                 n_with_scale     = 0;  // ... carrying a valid scale at the kernel
    size_t                 n_native_mmq     = 0;  // ... that actually reached the FP4 quantizer
    size_t                 n_native_scaled  = 0;  // ... native AND scaled = calibration in effect
    bool                   any_scale_ever   = false;
};
nvfp4_trace_state & trace_state() {
    static nvfp4_trace_state s;
    return s;
}
} // namespace

void ggml_cuda_nvfp4_trace_dispatch(
        const ggml_tensor *  dst,
        ggml_cuda_nvfp4_path path,
        bool                 mul_mat_id) {
    if (!ggml_cuda_nvfp4_trace_enabled()) {
        return;
    }
    const ggml_tensor * src0 = dst->src[0];
    if (src0 == nullptr || src0->type != GGML_TYPE_NVFP4) {
        return; // only this feature's weights are in scope
    }

    const bool scaled = ggml_cuda_nvfp4_act_scale_ptr(dst) != nullptr;
    const bool native = path == GGML_CUDA_NVFP4_PATH_MMQ_NATIVE;

    auto & st = trace_state();
    std::lock_guard<std::mutex> lock(st.mtx);

    st.n_nvfp4_mm++;
    st.n_with_scale    += scaled ? 1 : 0;
    st.n_native_mmq    += native ? 1 : 0;
    st.n_native_scaled += (native && scaled) ? 1 : 0;
    st.any_scale_ever  |= scaled;

    // One line per DISTINCT (weight, path, scaled) triple. Repeats stay silent, so a
    // long run is readable and a NEW behaviour mid-run is impossible to miss.
    const std::string name = src0->name[0] ? src0->name : "<unnamed>";
    const std::string key  = name + "|" + nvfp4_path_name(path) + (scaled ? "|s" : "|-")
                           + (mul_mat_id ? "|id" : "");
    if (!st.seen.insert(key).second) {
        return;
    }

    if (native && scaled) {
        GGML_LOG_WARN("NVFP4-TRACE: %-40s %-15s CALIBRATED SCALE IN EFFECT%s\n",
                      name.c_str(), nvfp4_path_name(path), mul_mat_id ? " (expert)" : "");
    } else if (native) {
        // Reached the modified quantizer with no scale. Expected for an uncalibrated
        // checkpoint; a RED FLAG once anything in this process has carried one, because
        // that is what a dropped carrier looks like.
        GGML_LOG_WARN("NVFP4-TRACE: %-40s %-15s no scale -> runtime amax%s%s\n",
                      name.c_str(), nvfp4_path_name(path),
                      mul_mat_id ? " (expert)" : "",
                      st.any_scale_ever ? "   <-- CARRIER LOST? other tensors DO have one" : "");
    } else {
        // Not a defect: this dispatch never consults an activation scale. It matters
        // because a measurement taken here cannot say anything about the feature.
        GGML_LOG_WARN("NVFP4-TRACE: %-40s %-15s bypasses the fp4 activation quantizer%s\n",
                      name.c_str(), nvfp4_path_name(path), mul_mat_id ? " (expert)" : "");
    }
}

void ggml_cuda_nvfp4_trace_graph(const ggml_cgraph * cgraph) {
    if (!ggml_cuda_nvfp4_trace_enabled()) {
        return;
    }

    // POST-scheduler count. libllama logs what it ATTACHED at build time; the delta
    // between the two is a carrier dropped by a copy, a split snapshot or a view.
    size_t n_mm = 0, n_carrier = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        if (node->src[0] == nullptr || node->src[0]->type != GGML_TYPE_NVFP4) {
            continue;
        }
        n_mm++;
        n_carrier += ggml_cuda_nvfp4_act_scale_ptr(node) != nullptr ? 1 : 0;
    }

    auto & st = trace_state();
    std::lock_guard<std::mutex> lock(st.mtx);
    st.n_graphs++;

    // Report the first submitted graph always — that is where a carrier loss shows up
    // before a single token is produced — then only when asked (level 2).
    if (st.n_graphs == 1 || nvfp4_trace_level() >= 2) {
        GGML_LOG_WARN("NVFP4-TRACE: graph #%zu submitted to CUDA: %zu NVFP4 matmul(s), "
                      "%zu carrying a valid scale POST-scheduler\n",
                      st.n_graphs, n_mm, n_carrier);
        if (n_mm > 0 && n_carrier == 0) {
            GGML_LOG_WARN("NVFP4-TRACE: no NVFP4 matmul in this graph carries a scale. Either the "
                          "checkpoint has none, or the carrier was dropped between graph build and "
                          "submission. Compare against libllama's attach count -- do NOT assume.\n");
        }
    }

    if (nvfp4_trace_level() >= 2) {
        GGML_LOG_WARN("NVFP4-TRACE: cumulative: %zu nvfp4 matmul dispatches, %zu with scale, "
                      "%zu reached native fp4 MMQ, %zu of those scaled\n",
                      st.n_nvfp4_mm, st.n_with_scale, st.n_native_mmq, st.n_native_scaled);
    }
}
