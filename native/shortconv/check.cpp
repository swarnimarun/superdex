#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>
int main() {
    auto available = reinterpret_cast<bool (*)()>(dlsym(RTLD_DEFAULT, "inkling_shortconv_available"));
    if (!available || !available())
        return 77;
    auto *gpu = ggml_backend_cuda_init(0);
    if (!gpu)
        return 77;
    std::mt19937 rng(1861791);
    std::uniform_real_distribution<float> uniform(-1, 1);
    size_t mismatch = 0, count = 0;
    double maxabs = 0;
    for (int channels : {1024, 2304})
        for (int heads : {0, 1, 8}) {
            auto *ctx = ggml_init({1u << 20, nullptr, true});
            auto *cur = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
            auto *past = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, channels);
            auto *weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, channels);
            auto *dyn = heads ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * heads) : nullptr;
            auto *x = ggml_concat(ctx, past, ggml_transpose(ctx, cur), 0);
            auto *base = ggml_reshape_2d(ctx, ggml_ssm_conv(ctx, x, weight), channels, 1);
            if (heads) {
                auto *taps = ggml_reshape_3d(ctx, dyn, 4, heads, 1);
                for (int lag = 0; lag < 4; ++lag) {
                    auto *tap =
                        ggml_view_3d(ctx, taps, 1, heads, 1, taps->nb[1], taps->nb[2], lag * taps->nb[0]);
                    auto *row = ggml_view_3d(ctx, x, 1, channels, 1, x->nb[1], x->nb[2], (3 - lag) * x->nb[0]);
                    auto *window = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, row)),
                                                   channels / heads, heads, 1);
                    base = ggml_add(ctx, base, ggml_reshape_2d(ctx, ggml_mul(ctx, window, tap), channels, 1));
                }
            }
            const float scale = 1.f / sqrtf((heads ? 2 : 1) * 4 + 1);
            auto *reference = ggml_scale(ctx, ggml_add(ctx, cur, base), scale);
            auto *fused = ggml_ssm_conv(ctx, x, weight);
            fused->src[2] = dyn;
            fused->op_params[15] = 0x494b4334;
            std::memcpy(fused->op_params + 1, &scale, sizeof(scale));
            auto *graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, reference);
            ggml_build_forward_expand(graph, fused);
            auto *memory = ggml_backend_alloc_ctx_tensors(ctx, gpu);
            if (!memory)
                return 2;
            for (int test = 0; test < 10; ++test) {
                const float amps[] = {0.f, 1.e-35f, 1.e-10f, 0.01f, 0.5f, 1.f, 2.f, 32.f, 512.f, 8192.f};
                auto set = [&](ggml_tensor *t, bool coefficients) {
                    std::vector<float> a(ggml_nelements(t));
                    for (size_t i = 0; i < a.size(); ++i) {
                        const float amp = coefficients && test == 1 ? 1.e-4f : amps[test];
                        a[i] = i % 17 ? amp * uniform(rng) : (i % 2 ? 0.f : -0.f);
                    }
                    ggml_backend_tensor_set(t, a.data(), 0, a.size() * sizeof(float));
                };
                set(past, false);
                set(cur, false);
                set(weight, true);
                if (dyn)
                    set(dyn, true);
                if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
                    return 3;
                std::vector<float> a(channels), b(channels);
                ggml_backend_tensor_get(reference, a.data(), 0, channels * 4);
                ggml_backend_tensor_get(fused, b.data(), 0, channels * 4);
                for (int i = 0; i < channels; ++i) {
                    ++count;
                    if (std::memcmp(&a[i], &b[i], sizeof(float))) {
                        if (mismatch < 4)
                            std::printf("mismatch c=%d h=%d test=%d i=%d reference=%a fused=%a\n", channels,
                                        heads, test, i, a[i], b[i]);
                        ++mismatch;
                    }
                    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
                        return 4;
                    maxabs = std::max(maxabs, std::abs(double(a[i]) - b[i]));
                }
            }
            ggml_backend_buffer_free(memory);
            ggml_free(ctx);
        }
    ggml_backend_free(gpu);
    std::printf("60 cases: widths 1024/2304, static & 1/8 dynamic heads; %zu float32 outputs, %zu bitwise "
                "differences, max_abs=%.9g\n",
                count, mismatch, maxabs);
    return mismatch != 0;
}
