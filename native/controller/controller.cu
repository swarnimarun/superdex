#include "common.cuh"
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

extern void ggml_cuda_op_rms_norm(ggml_backend_cuda_context &, ggml_tensor *);

namespace {
constexpr int norm_kind = 0x49434e33, down_kind = 0x49434433, merge_kind = 0x49434d33;
struct Inputs {
    const float *cur[3];
    const float *message;
    int message_stride;
};

// Preserve the exact CUDA RMSNorm binary and its launch parameters. A single
// pack puts all six disjoint rows in one array. The later merge reuses it.
__global__ void pack_rows(Inputs in, float *out, int width) {
    const int row = blockIdx.y, i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= width / 4)
        return;
    const float *src = row % 2 ? in.message + (row / 2) * in.message_stride : in.cur[row / 2];
    reinterpret_cast<float4 *>(out + row * width)[i] = reinterpret_cast<const float4 *>(src)[i];
}

struct Weights {
    const float *a[3], *b[3];
};

// GGML's FP32 MMVF: 256 threads / output when K=4608, float2 loads,
// two sequential FP32 MADs, then one warp and one CTA-level reduction.
__global__ void controller_down(const float *norm, Weights weights, float *out, int width) {
    constexpr int threads = 256;
    const int j = blockIdx.y, r = blockIdx.x, tid = threadIdx.x;
    const float2 *w = reinterpret_cast<const float2 *>(weights.a[j] + r * (2 * width));
    const float2 *x = reinterpret_cast<const float2 *>(norm + j * (2 * width));
    float sum = 0.f;
    for (int i = tid; i < width; i += threads) {
        const float2 wi = w[i], xi = x[i];
        ggml_cuda_mad(sum, wi.x, xi.x);
        ggml_cuda_mad(sum, wi.y, xi.y);
    }
    sum = warp_reduce_sum<32>(sum);
    __shared__ float partial[8];
    if (!(tid & 31))
        partial[tid >> 5] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = tid < 8 ? partial[tid] : 0.f;
        sum = warp_reduce_sum<32>(sum);
        if (!tid)
            out[j * 64 + r] = sum;
    }
}

// One warp per 64-coefficient output row, as in MMVF for K=64. Eight rows
// per CTA amortize the dual matrix reads and the residual/threshold work.
__global__ void controller_merge(const float *gate, const float *raw, Weights weights, float *out, int width) {
    const int j = blockIdx.y, lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= width)
        return;
    const float2 v = reinterpret_cast<const float2 *>(gate + j * 64)[lane];
    const float2 a = reinterpret_cast<const float2 *>(weights.a[j] + row * 64)[lane];
    const float2 b = reinterpret_cast<const float2 *>(weights.b[j] + row * 64)[lane];
    float gain = 0.f, forget = 0.f;
    ggml_cuda_mad(gain, a.x, v.x);
    ggml_cuda_mad(gain, a.y, v.y);
    ggml_cuda_mad(forget, b.x, v.x);
    ggml_cuda_mad(forget, b.y, v.y);
    gain = warp_reduce_sum<32>(gain);
    forget = warp_reduce_sum<32>(forget);
    if (!lane) {
        const float c = raw[(2 * j) * width + row], m = raw[(2 * j + 1) * width + row];
        out[j * width + row] = __fadd_rn(__fsub_rn(c, __fmul_rn(tanhf(forget), c)), __fmul_rn(tanhf(gain), m));
    }
}

void bind_launch(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    const auto stream = ctx.stream();
    constexpr int w = 2304;
    Weights weights = {};
    if (dst->op_params[15] == norm_kind) {
        Inputs in = {};
        for (int j = 0; j < 3; ++j)
            in.cur[j] = (const float *)dst->src[j]->data;
        in.message = (const float *)dst->src[3]->data;
        in.message_stride = dst->src[3]->nb[1] / sizeof(float);
        float *raw = (float *)dst->data + 6 * w;
        pack_rows<<<dim3((w / 4 + 255) / 256, 6), 256, 0, stream>>>(in, raw, w);
        ggml_tensor source = *dst, target = *dst;
        for (auto *x : {&source, &target}) {
            x->ne[0] = w;
            x->ne[1] = 6;
            x->ne[2] = x->ne[3] = 1;
            x->nb[0] = sizeof(float);
            x->nb[1] = w * sizeof(float);
            x->nb[2] = x->nb[3] = 6 * w * sizeof(float);
        }
        source.op = GGML_OP_NONE;
        source.data = raw;
        target.op = GGML_OP_RMS_NORM;
        target.src[0] = &source;
        const float eps = 1e-6f;
        std::memcpy(target.op_params, &eps, sizeof(eps));
        ggml_cuda_op_rms_norm(ctx, &target);
    } else if (dst->op_params[15] == down_kind) {
        for (int j = 0; j < 3; ++j)
            weights.a[j] = (const float *)dst->src[1 + j]->data;
        controller_down<<<dim3(64, 3), 256, 0, stream>>>((const float *)dst->src[0]->data, weights,
                                                         (float *)dst->data, w);
    } else {
        for (int j = 0; j < 3; ++j) {
            weights.a[j] = (const float *)dst->src[2 + j]->data;
            weights.b[j] = (const float *)dst->src[5 + j]->data;
        }
        controller_merge<<<dim3((w + 7) / 8, 3), 256, 0, stream>>>((const float *)dst->src[0]->data,
                                                                   (const float *)dst->src[1]->data + 6 * w,
                                                                   weights, (float *)dst->data, w);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

extern "C" __attribute__((visibility("default"))) bool inkling_controller_available() {
    static const bool enabled = [] {
        const char *s = std::getenv("INKLING_CONTROLLER_FUSION");
        if (!s || std::strcmp(s, "1") != 0)
            return false;
        int dev, major, minor;
        return cudaGetDevice(&dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
               major == 12 && minor == 0;
    }();
    return enabled;
}

__attribute__((visibility("default"))) void ggml_cuda_op_concat(ggml_backend_cuda_context &ctx,
                                                                ggml_tensor *dst) {
    const int kind = dst->op_params[15];
    if (kind == norm_kind || kind == down_kind || kind == merge_kind) {
        static std::once_flag printed;
        std::call_once(printed, [] {
            std::fprintf(stderr, "Inkling four-kernel QKV loop controllers active, FP32, D=2304, Q=1\n");
        });
        bind_launch(ctx, dst);
        return;
    }
    using fn = void (*)(ggml_backend_cuda_context &, ggml_tensor *);
    static auto next = [] {
        void *lib = dlopen("libggml-cuda.so", RTLD_LAZY | RTLD_NOLOAD);
        return lib ? reinterpret_cast<fn>(
                         dlsym(lib, "_Z19ggml_cuda_op_concatR25ggml_backend_cuda_contextP11ggml_tensor"))
                   : nullptr;
    }();
    if (!next || next == ggml_cuda_op_concat) {
        std::fprintf(stderr, "Could not resolve original ggml CUDA CONCAT\n");
        std::abort();
    }
    next(ctx, dst);
}
