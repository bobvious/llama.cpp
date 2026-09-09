#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>
#include <cstdlib>

// Calibrated NVFP4 level-2 (per-tensor) ACTIVATION scale.
//
// 🔴 CARRIED AS A REAL GRAPH SOURCE (mul_mat src[3]), NOT as tensor metadata. An earlier
// version of this patch stashed it in the weight's op_params; a GPT-6 Astra review found three
// independent ways that silently fails, all verified in-tree:
//   * ggml_dup_tensor() inits op_params={0} (ggml.c:1815) and ggml_dup_tensor_layout copies
//     only nb[] (ggml-backend.cpp:749-754), so any backend OFFLOAD COPY drops the scale;
//   * the Meta backend snapshots op_params during buffer alloc, BEFORE a post-load write, so
//     LLAMA_SPLIT_MODE_TENSOR keeps the stale zero;
//   * ggml stores a VIEW's byte offset in op_params (ggml.c:3791) -- so a view of an NVFP4
//     weight yields a positive normal float that would be read as a calibration scale, which
//     breaks the fallback even for checkpoints carrying NO scale at all.
// A src is preserved by copies, splits and views by construction. Do not "simplify" this back
// into metadata. The device pointer is null when the checkpoint has no calibration.

// Device pointer to the 1-element F32 calibration scale carried on a mul_mat node as src[3],
// or nullptr when the checkpoint had none (=> kernel keeps the runtime-amax behaviour).
// NOTE src[3]: src[2] is the `ids` operand of GGML_MUL_MAT_ID and is read as such by the CUDA
// dispatch (ggml-cuda.cu:1905), so the scale cannot live there. See llama-graph.cpp.
// KILL-SWITCH: GGML_CUDA_NVFP4_NO_ACT_SCALE=1 makes every call return nullptr, forcing the
// runtime-amax path exactly as if the checkpoint carried no calibration. This is the CONTROL for
// "does the calibrated scale change any OUTPUT?" -- same build, same argv, one env var apart.
// Read once per process (getenv is not cheap and this is on the dispatch path).
static inline bool ggml_cuda_nvfp4_act_scale_disabled() {
    static const bool disabled = [] {
        const char * e = getenv("GGML_CUDA_NVFP4_NO_ACT_SCALE");
        return e != nullptr && *e != 0 && *e != '0';
    }();
    return disabled;
}

static inline const float * ggml_cuda_nvfp4_act_scale_ptr(const ggml_tensor * dst) {
    if (ggml_cuda_nvfp4_act_scale_disabled()) {
        return nullptr;
    }
    const ggml_tensor * s = dst->src[3];
    if (s == nullptr || s->type != GGML_TYPE_F32 || ggml_nelements(s) != 1 || s->data == nullptr) {
        return nullptr;
    }
    return (const float *) s->data;
}

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             const float *   input_scale,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   const float *   input_scale,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);
