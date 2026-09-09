#pragma once

#include "common.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// NVFP4 activation-scale TRACE — makes a SILENT FALLBACK LOUD.
//
// WHY THIS EXISTS. Every defect this feature has had failed by silently reverting
// to the previous behaviour, so a smoke test passed cleanly each time:
//   * the op_params carrier was zeroed by backend offload copies;
//   * the Meta backend snapshotted it before it was written (tensor-split);
//   * a VIEW's byte offset was read as if it were a calibration scale.
// All three produced plausible text. Output-shaped testing CANNOT distinguish
// "the calibrated scale reached the kernel" from "it was dropped and we fell back".
// Neither can a KLD number: below MMVQ_MAX_BATCH_SIZE the dispatch never reaches
// the modified quantizer at all, so a decode-only measurement is blind by
// construction (GPT-6 Astra review 2026-09-09, finding 7).
//
// ⇒ Do not interpret ANY quality or throughput number for this feature without a
//   trace line proving which path ran and whether a scale was present.
//
// This is a FORK-LOCAL diagnostic. It is deliberately not upstreamable and does
// not need to be: see D:\llm\FORK-PATCHES.md.
//
// USAGE
//   GGML_CUDA_NVFP4_TRACE=1   log each DISTINCT outcome once (recommended)
//   GGML_CUDA_NVFP4_TRACE=2   ... and a per-graph summary line
//   unset / 0                 off; one predictable branch, no allocation, no locking
//
// "Distinct outcome" = (weight tensor, dispatch path, scale present?). Repeats are
// silent, so a long run stays readable while every new behaviour is reported the
// first time it happens. That is what makes this safe to leave enabled.
// ─────────────────────────────────────────────────────────────────────────────

enum ggml_cuda_nvfp4_path {
    GGML_CUDA_NVFP4_PATH_MMVQ,        // quantizes activations to Q8_1 — IGNORES the scale
    GGML_CUDA_NVFP4_PATH_MMQ_NATIVE,  // native Blackwell FP4 MMQ — the only consumer
    GGML_CUDA_NVFP4_PATH_MMQ_Q8,      // MMQ, but Q8_1 activations — ignores the scale
    GGML_CUDA_NVFP4_PATH_MMF,         // float path
    GGML_CUDA_NVFP4_PATH_CUBLAS,      // dequant + cuBLAS
    GGML_CUDA_NVFP4_PATH_COUNT,
};

// Cheap, cached. The whole trace compiles to this test when disabled.
bool ggml_cuda_nvfp4_trace_enabled();

// Record one mul_mat / mul_mat_id dispatch. `mul_mat_id` distinguishes MoE expert
// matmuls, which take the gather/scatter quantizers and have their own coverage gap.
void ggml_cuda_nvfp4_trace_dispatch(
        const ggml_tensor *   dst,
        ggml_cuda_nvfp4_path  path,
        bool                  mul_mat_id);

// Walk a scheduler-submitted graph and count NVFP4 matmuls that still carry a valid
// scale. This is the POST-scheduler, POST-split view: comparing it with the count
// libllama attached at graph BUILD time is what detects a carrier that was dropped
// by an offload copy, a tensor-split snapshot or a view. That comparison is the
// whole point — one number alone proves nothing.
void ggml_cuda_nvfp4_trace_graph(const ggml_cgraph * cgraph);
