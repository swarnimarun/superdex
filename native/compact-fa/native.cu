// Instantiate only the SM120 MMA cases that Inkling's per-head bias uses.
// Separate symbol names avoid preempting any original CUDA backend template.
#define launch_fattn inkling_native_launch_fattn
#define ggml_cuda_flash_attn_ext_mma_f16_case inkling_native_fa_case
#include "fattn-mma-f16.cuh"
#define ggml_cuda_flash_attn_ext_vec_case_impl inkling_decode_vec_case
#include "fattn-vec.cuh"
#include <mutex>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>

template void inkling_native_fa_case<128, 128, 8, 1>(ggml_backend_cuda_context &, ggml_tensor *);
template void inkling_native_fa_case<128, 128, 16, 1>(ggml_backend_cuda_context &, ggml_tensor *);
template void inkling_native_fa_case<128, 128, 32, 1>(ggml_backend_cuda_context &, ggml_tensor *);
template void inkling_native_fa_case<128, 128, 64, 1>(ggml_backend_cuda_context &, ggml_tensor *);

extern "C" __attribute__((visibility("default"))) bool inkling_native_compact_decode_v1() { return true; }

static void launch_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto queries = dst->src[0]->ne[1];
    if (queries <= 8) {
        inkling_native_fa_case<128, 128, 8, 1>(ctx, dst);
    } else if (queries <= 16) {
        inkling_native_fa_case<128, 128, 16, 1>(ctx, dst);
    } else if (queries <= 32) {
        inkling_native_fa_case<128, 128, 32, 1>(ctx, dst);
    } else {
        inkling_native_fa_case<128, 128, 64, 1>(ctx, dst);
    }
}

static int64_t query_tile_size(const ggml_tensor * dst) {
    const auto queries = dst->src[0]->ne[1];
    const char * match = std::getenv("INKLING_NATIVE_FA_MATCH_TILES");
    if (!match || std::strcmp(match, "1") || ggml_nbytes(dst) >= (size_t(1) << 30)) {
        return queries;
    }
    const char * cells = std::getenv("INKLING_QUERY_TILE_CELLS");
    char * end = nullptr;
    const int64_t budget = cells ? std::strtoll(cells, &end, 10) : 512 * 1024;
    GGML_ASSERT(!cells || (end != cells && !*end && budget >= 0 && budget <= INT32_MAX));
    if (!budget) {
        return queries;
    }
    const int64_t floor = std::max<int64_t>(16, GGML_PAD((queries + 15) / 16, 8));
    return std::min(queries, std::max(floor, budget / dst->src[1]->ne[1] / 8 * 8));
}

// A zero threshold selects the ordinary one-head vector kernel at every depth.
static int grouped_decode_min_rows() {
    static const int threshold = [] {
        const char * text = std::getenv("INKLING_GQA_MIN_K");
        if (!text || !*text) {
            return 0;
        }
        char * end = nullptr;
        const long rows = std::strtol(text, &end, 10);
        return end != text && !*end && rows >= 1 && rows <= INT_MAX ? static_cast<int>(rows) : 0;
    }();
    return threshold;
}

extern "C" __attribute__((visibility("default"))) bool inkling_compact_native_launch(
        ggml_backend_cuda_context * ctx, ggml_tensor * dst) {
    const int64_t queries = dst->src[0]->ne[1];
    if (queries == 1 && dst->src[1]->ne[1] % FATTN_KQ_STRIDE == 0) {
        const auto * q = dst->src[0];
        const auto * k = dst->src[1];
        const auto * v = dst->src[2];
        const auto * table = dst->src[6];
        float max_bias;
        std::memcpy(&max_bias, dst->op_params + 1, sizeof(max_bias));
        const int minimum = grouped_decode_min_rows();
        if (minimum && k->ne[1] >= minimum && q->ne[2] == 3*k->ne[2] && q->ne[3] == 1 &&
                q->ne[0] == 128 && k->ne[0] == 128 && v->ne[0] == 128 && q->type == GGML_TYPE_F32 &&
                k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && table && table->nb[0] == 4 && max_bias == 0.0f) {
            // Keep the same split-K count and combine order as the one-head kernel.
            auto stock = flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, false>;
            auto grouped = flash_attn_ext_vec<128, 3, GGML_TYPE_F16, GGML_TYPE_F16, false>;
            inkling_native_launch_fattn<128, 1, 1>(*ctx, dst, grouped, 4, 0, 128, true, true,
                                                  false, false, 32, false, stock);
            static std::once_flag grouped_reported;
            std::call_once(grouped_reported, [&] {
                std::fprintf(stderr, "Inkling compact grouped-query decode executed: kv=%lld heads_per_group=3\n",
                             (long long)k->ne[1]);
            });
            return true;
        }

        static std::once_flag reported;
        std::call_once(reported, [&] {
            std::fprintf(stderr, "Inkling compact single-token vector attention executed: kv=%lld\n",
                         (long long)dst->src[1]->ne[1]);
        });
        ggml_cuda_flash_attn_ext_vec_case_impl<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, false>(*ctx, dst);
        return true;
    }
    const int64_t tile = query_tile_size(dst);
    for (int64_t offset = 0; offset < queries; offset += tile) {
        // K/V use their original layout. Q, bias and output use views of the selected
        // queries. The position vector still starts with the complete physical KV map.
        const int64_t count = std::min(tile, queries - offset);
        ggml_tensor q = *dst->src[0], table = *dst->src[6], out = *dst;
        q.ne[1] = table.ne[2] = out.ne[2] = count;
        q.data = static_cast<char *>(q.data) + offset*q.nb[1];
        table.data = static_cast<char *>(table.data) + offset*table.nb[2];
        out.data = static_cast<char *>(out.data) + offset*out.nb[2];
        out.src[0] = &q;
        out.src[6] = &table;
        out.op_params[8] = int32_t(offset);
        launch_tile(*ctx, &out);
    }
    return true;
}
