#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <dlfcn.h>
#include <vector>

int main() {
    using available_fn = bool (*)();
    auto available = reinterpret_cast<available_fn>(dlsym(RTLD_DEFAULT, "inkling_controller_available"));
    if (!available || !available()) {
        std::fprintf(stderr, "Requires the SM120 controller plugin and INKLING_CONTROLLER_FUSION=1\n");
        return 77;
    }
    constexpr int width = 2304;
    ggml_init_params pars = {2u << 20, nullptr, true};
    auto *ctx = ggml_init(pars);
    auto *gpu = ggml_backend_cuda_init(0);
    if (!ctx || !gpu)
        return 2;
    ggml_tensor *cur[3], *g[3], *f[3], *stock_norm[3], *stock_mix[3], *stock_down[3], *wd[3], *wg[3], *wf[3];
    // Nonzero offset and extra materialized fourth message channel match the real graph.
    auto *all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 6);
    auto *messages = ggml_view_2d(ctx, all, width, 4, all->nb[1], all->nb[1]);
    auto *packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * width, 6);
    auto *down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
    auto *merge = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 3);
    packed->op = down->op = merge->op = GGML_OP_CONCAT;
    packed->op_params[15] = 0x49434e33;
    down->op_params[15] = 0x49434433;
    merge->op_params[15] = 0x49434d33;
    packed->src[3] = messages;
    down->src[0] = packed;
    merge->src[0] = down;
    merge->src[1] = packed;
    auto *graph = ggml_new_graph(ctx);
    for (int j = 0; j < 3; ++j) {
        cur[j] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        wd[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * width, 64);
        wg[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, width);
        wf[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, width);
        packed->src[j] = cur[j];
        down->src[1 + j] = wd[j];
        merge->src[2 + j] = wg[j];
        merge->src[5 + j] = wf[j];
        auto *message = ggml_view_1d(ctx, messages, width, j * messages->nb[1]);
        stock_norm[j] =
            ggml_concat(ctx, ggml_rms_norm(ctx, cur[j], 1e-6f), ggml_rms_norm(ctx, message, 1e-6f), 0);
        stock_down[j] = ggml_mul_mat(ctx, wd[j], stock_norm[j]);
        g[j] = ggml_mul_mat(ctx, wg[j], stock_down[j]);
        f[j] = ggml_mul_mat(ctx, wf[j], stock_down[j]);
        stock_mix[j] = ggml_add(ctx, ggml_sub(ctx, cur[j], ggml_mul(ctx, ggml_tanh(ctx, f[j]), cur[j])),
                                ggml_mul(ctx, ggml_tanh(ctx, g[j]), message));
        ggml_build_forward_expand(graph, stock_norm[j]);
        ggml_build_forward_expand(graph, stock_mix[j]);
    }
    ggml_build_forward_expand(graph, packed);
    ggml_build_forward_expand(graph, merge);
    auto *memory = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!memory)
        return 2;
    std::mt19937 rng(532836);
    std::uniform_real_distribution<float> uniform(-1.f, 1.f);
    auto set = [&](ggml_tensor *t, float scale) {
        std::vector<float> a(ggml_nelements(t));
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = i % 67 ? scale * uniform(rng) : (i % 2 ? -0.f : 0.f);
        ggml_backend_tensor_set(t, a.data(), 0, a.size() * sizeof(float));
    };
    int unequal_norm = 0, unequal_merge = 0, unequal_down = 0;
    double worst_down = 0;
    double worst_norm = 0, worst_merge = 0;
    for (int test = 0; test < 8; ++test) {
        const float amp[] = {0.f, 1e-6f, 1e-3f, 0.1f, 1.f, 4.f, 33.f, 1024.f};
        set(all, amp[test]);
        for (int j = 0; j < 3; ++j) {
            set(cur[j], amp[test]);
            set(wd[j], .023f);
            set(wg[j], .1f);
            set(wf[j], .15f);
        }
        if (ggml_backend_graph_compute(gpu, graph) != GGML_STATUS_SUCCESS)
            return 2;
        for (int j = 0; j < 3; ++j) {
            float a[64], b[64];
            ggml_backend_tensor_get(stock_down[j], a, 0, sizeof(a));
            ggml_backend_tensor_get(down, b, 64 * j * sizeof(float), sizeof(b));
            for (int i = 0; i < 64; i++) {
                const double d = std::abs(double(a[i]) - b[i]);
                if (!std::isfinite(d))
                    return 3;
                if (std::memcmp(a + i, b + i, sizeof(float)))
                    ++unequal_down;
                worst_down = std::max(worst_down, d);
            }
        }
        for (int op = 0; op < 2; ++op) {
            const int n = width * (op ? 1 : 2);
            std::vector<float> target(n), fused(n);
            for (int j = 0; j < 3; ++j) {
                ggml_backend_tensor_get(op ? stock_mix[j] : stock_norm[j], target.data(), 0, n * 4);
                ggml_backend_tensor_get(op ? merge : packed, fused.data(), j * n * 4, n * 4);
                for (int i = 0; i < n; ++i) {
                    const double d = std::abs(double(target[i]) - fused[i]);
                    if (!std::isfinite(d))
                        return 3;
                    if (std::memcmp(target.data() + i, fused.data() + i, sizeof(float)))
                        ++(op ? unequal_merge : unequal_norm);
                    auto &highest = op ? worst_merge : worst_norm;
                    highest = std::max(highest, d);
                }
            }
        }
    }
    std::printf("8 cases, 3 controller lanes, D=%d, nonzero message offset, bitwise FP32: norm scalars=%d "
                "unequal=%d max_abs=%.9g; merge scalars=%d unequal=%d max_abs=%.9g\n",
                width, 8 * 6 * width, unequal_norm, worst_norm, 8 * 3 * width, unequal_merge, worst_merge);
    std::printf("Down projection scalars=%d unequal=%d max_abs=%.9g\n", 8 * 3 * 64, unequal_down, worst_down);
    ggml_backend_buffer_free(memory);
    ggml_backend_free(gpu);
    ggml_free(ctx);
    return unequal_down || unequal_merge || unequal_norm ? 1 : 0;
}
