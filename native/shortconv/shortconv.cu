#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
namespace {
constexpr int magic = 0x494b4334;
template <bool dynamic>
__global__ void fused_decode_conv(const float *x, const float *weights, const float *coefficients, int channels,
                                  int channels_per_head, float scale, float *result) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels)
        return;
    const float4 v = reinterpret_cast<const float4 *>(x)[c];
    const float4 w = reinterpret_cast<const float4 *>(weights)[c];
    // ggml's ssm_conv_f32<false,128,4> accumulates ascending taps with FMA.
    float sum = __fmaf_rn(v.x, w.x, 0.f);
    sum = __fmaf_rn(v.y, w.y, sum);
    sum = __fmaf_rn(v.z, w.z, sum);
    sum = __fadd_rn(__fmaf_rn(v.w, w.w, sum), 0.f);
    if constexpr (dynamic) {
        const float4 d = reinterpret_cast<const float4 *>(coefficients)[c / channels_per_head];
        // Separate ggml MUL and ADD for each descending temporal window.
        sum = __fadd_rn(sum, __fmul_rn(v.w, d.x));
        sum = __fadd_rn(sum, __fmul_rn(v.z, d.y));
        sum = __fadd_rn(sum, __fmul_rn(v.y, d.z));
        sum = __fadd_rn(sum, __fmul_rn(v.x, d.w));
    }
    result[c] = __fmaf_rn(scale, __fadd_rn(v.w, sum), 0.f);
}
} // namespace
extern "C" __attribute__((visibility("default"))) bool inkling_shortconv_available() {
    static const bool supported = [] {
        const char *opt = std::getenv("INKLING_FUSED_SHORTCONV");
        if (!opt || std::strcmp(opt, "1") != 0)
            return false;
        int dev, major, minor;
        return cudaGetDevice(&dev) == cudaSuccess && dev == 0 &&
               cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
               major == 12 && minor == 0;
    }();
    return supported;
}
__attribute__((visibility("default"))) void
ggml_cuda_op_ssm_conv(ggml_backend_cuda_context &ctx, ggml_tensor *t, ggml_tensor *bias, ggml_tensor *silu) {
    if (t->op_params[15] != magic) {
        using op = void (*)(ggml_backend_cuda_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *);
        static auto base = [] {
            auto *lib = dlopen("libggml-cuda.so", RTLD_LAZY | RTLD_NOLOAD);
            return lib ? reinterpret_cast<op>(dlsym(
                             lib, "_Z21ggml_cuda_op_ssm_convR25ggml_backend_cuda_contextP11ggml_tensorS2_S2_"))
                       : nullptr;
        }();
        if (!base || base == ggml_cuda_op_ssm_conv) {
            std::fputs("Failed to resolve stock CUDA SSM_CONV\n", stderr);
            std::abort();
        }
        base(ctx, t, bias, silu);
        return;
    }
    GGML_ASSERT(!bias && !silu);
    const int channels = t->ne[0];
    const auto *x = static_cast<const float *>(t->src[0]->data),
               *w = static_cast<const float *>(t->src[1]->data);
    float *out = static_cast<float *>(t->data), scale;
    std::memcpy(&scale, t->op_params + 1, sizeof(scale));
    if (t->src[2]) {
        int heads = t->src[2]->ne[0] / 4;
        fused_decode_conv<true><<<(channels + 127) / 128, 128, 0, ctx.stream()>>>(
            x, w, static_cast<const float *>(t->src[2]->data), channels, channels / heads, scale, out);
    } else {
        fused_decode_conv<false>
            <<<(channels + 127) / 128, 128, 0, ctx.stream()>>>(x, w, nullptr, channels, 0, scale, out);
    }
    CUDA_CHECK(cudaGetLastError());
    static std::once_flag reported;
    std::call_once(reported, [] {
        std::fputs("Inkling fused single-token short convolution executed, FP32 four-tap residual\n", stderr);
    });
}
