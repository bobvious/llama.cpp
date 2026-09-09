#include "nvfp4-trace.cuh"
#include "quantize.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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
    // 🔴 n_with_scale / n_native_scaled count a NON-NULL POINTER, nothing more. They are
    // incremented before the value readback below, so a stale-but-positive or an outright
    // bad scale still lands in them. "native AND scaled" therefore means "the carrier was
    // present at the kernel", NOT "calibration in effect" -- the stronger reading is what
    // the Astra review (F13) called out, and it is the reading these numbers were quoted
    // under. The value-verified counters below are the ones that support that claim, and
    // they only advance on DISTINCT (weight, path, scaled) triples because of the dedup.
    size_t                 n_with_scale     = 0;  // ... carrying a non-null scale POINTER
    size_t                 n_native_mmq     = 0;  // ... that actually reached the FP4 quantizer
    size_t                 n_native_scaled  = 0;  // ... native AND carrier present
    size_t                 n_value_ok       = 0;  // distinct tensors whose scale READ BACK finite > 0
    size_t                 n_value_bad      = 0;  // distinct tensors whose scale read back garbage
    size_t                 n_value_unread   = 0;  // readback attempted and FAILED (cudaMemcpy error)
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
    // long run stays readable and a NEW (weight, path, scaled) COMBINATION mid-run is loud.
    // 🔴 BUT THE DEDUP KEY EXCLUDES THE VALUE, so each scale is read back EXACTLY ONCE, ever.
    // A scale that is correct on its first dispatch and goes stale or garbage later is
    // INVISIBLE to this instrument -- which is precisely astra-20260909-F4's scenario, and
    // this comment used to say "a NEW behaviour mid-run is impossible to miss" without that
    // exception. To close F4 the key would need the value in it (or a periodic re-read);
    // deliberately not done, because it turns a once-per-tensor blocking D2H into a
    // per-dispatch one. Until then, "the scale is live" is a statement about the FIRST
    // dispatch of each tensor, not about the run.
    const std::string name = src0->name[0] ? src0->name : "<unnamed>";
    const std::string key  = name + "|" + nvfp4_path_name(path) + (scaled ? "|s" : "|-")
                           + (mul_mat_id ? "|id" : "");
    if (!st.seen.insert(key).second) {
        return;
    }

    if (native && scaled) {
        // 🔴 READ THE VALUE BACK, not just the pointer. The accessor validates F32 / 1 element /
        // non-null data and NOTHING about the number, but all three carrier defects Astra found
        // were silent VALUE corruption (a zeroed copy, a stale snapshot, a view's byte offset
        // reinterpreted as a float). Without this, "the scale reached the kernel" is equally
        // consistent with every scale being 0.0, and every placement-matrix PASS is vacuous.
        //
        // Blocking 4-byte D2H, once per distinct tensor because of the dedup above, and only
        // when the trace is enabled. Cost is irrelevant next to what it rules out.
        const float * dptr = ggml_cuda_nvfp4_act_scale_ptr(dst);
        float v = std::numeric_limits<float>::quiet_NaN();
        const cudaError_t err = cudaMemcpy(&v, dptr, sizeof(float), cudaMemcpyDeviceToHost);

        if (err != cudaSuccess) {
            st.n_value_unread++;
            GGML_LOG_WARN("NVFP4-TRACE: %-40s %-15s scale READBACK FAILED: %s\n",
                          name.c_str(), nvfp4_path_name(path), cudaGetErrorString(err));
        } else if (!std::isfinite(v) || v <= 0.0f) {
            // A pointer that survives every placement but carries garbage is the exact failure
            // the pointer-only check cannot see. Loud on purpose.
            st.n_value_bad++;
            GGML_LOG_WARN("NVFP4-TRACE: %-40s %-15s 🔴 SCALE VALUE IS BAD: %.9g%s\n",
                          name.c_str(), nvfp4_path_name(path), v, mul_mat_id ? " (expert)" : "");
        } else {
            // Value read back finite and > 0 FOR THIS TENSOR. That is engagement, not
            // agreement with the checkpoint: nothing here compares it to the exported
            // input_scale, so a stale positive still reads as a pass. (Astra F13.)
            st.n_value_ok++;
            // 🔴 THE SUBSTRING "CALIBRATED SCALE IN EFFECT" IS AN API. Three artifacts key on
            // it literally: runs/nvfp4-trace-20260909/run-split.ps1:79 (it feeds the PASS gate),
            // run-split-tensoronly.ps1:75, and runs/nvfp4-killswitch-20260909/compare.py:72 --
            // the artifact behind the "260 calibrated vs 0" kill-switch headline. I renamed this
            // to "SCALE VALUE OK" for honesty and swept the DLL for the NEW string, never for
            // readers of the OLD one. All three would then have counted 0, silently, exit 0 --
            // which reads as "the calibrated scale never reaches the kernel", the catastrophic
            // finding this whole program exists to hunt. Restored, with the qualifier appended
            // instead. To rename it, sweep by CONNECTION SITE and update all three in the same
            // commit (memory/reference_sweep_by_connection_site_not_by_string.md).
            GGML_LOG_WARN("NVFP4-TRACE: %-40s %-15s CALIBRATED SCALE IN EFFECT (value read back finite>0; NOT compared to the checkpoint) value=%.9g%s\n",
                          name.c_str(), nvfp4_path_name(path), v, mul_mat_id ? " (expert)" : "");
        }
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
        GGML_LOG_WARN("NVFP4-TRACE: cumulative: %zu nvfp4 matmul dispatches, %zu carrying a scale POINTER, "
                      "%zu reached native fp4 MMQ, %zu of those with a POINTER; of the DISTINCT tensors "
                      "read back: %zu good, %zu bad, %zu unreadable\n",
                      st.n_nvfp4_mm, st.n_with_scale, st.n_native_mmq, st.n_native_scaled,
                      st.n_value_ok, st.n_value_bad, st.n_value_unread);
    }
}
