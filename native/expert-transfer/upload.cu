#define INKLING_BATCH_IMPL
#include "inkling-expert-upload.h"
#include "ggml-backend-impl.h"
#include "common.cuh"
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

extern "C" __attribute__((visibility("default"))) bool inkling_batch_expert_copies(ggml_backend_t backend, const inkling_expert_range *ranges, size_t count) {
    constexpr size_t capacity = 64;  // Inkling's top-8 x three projections needs at most 24.
    if (count < 2 || count > capacity || !backend || strcmp(ggml_backend_name(backend), "CUDA0")) {
        return false;
    }
    auto *ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    // The immediately preceding router D2H and synchronize have created this stream.
    auto stream = ctx->streams[ctx->device][ctx->curr_stream_no];
    if (!stream) return false;
    using submit_fn = cudaError_t (*)(void **, void **, size_t *, size_t, cudaMemcpyAttributes *, size_t *, size_t, size_t *, cudaStream_t);
    static auto submit = [] {
        void *lib = dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
        return lib ? reinterpret_cast<submit_fn>(dlsym(lib, "cudaMemcpyBatchAsync")) : nullptr;
    }();
    if (!submit) return false;

    std::array<void *, capacity> src, dst;
    std::array<size_t, capacity> sizes;
    for (size_t i=0; i<count; ++i) {
        const auto &r = ranges[i];
        const size_t bank_bytes = ggml_nbytes(r.bank);
        if (!r.bytes || r.offset > bank_bytes || r.bytes > bank_bytes-r.offset) {
            std::fputs("Inkling invalid expert transfer range\n", stderr);
            std::abort();
        }
        src[i] = const_cast<void *>(r.source);
        dst[i] = static_cast<char *>(r.bank->data) + r.offset;
        sizes[i] = r.bytes;
    }
    cudaMemcpyAttributes attrs{};
    attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    size_t first = 0, failed = size_t(-1);
    auto rc = submit(dst.data(), src.data(), sizes.data(), count, &attrs, &first, 1, &failed, stream);
    if (rc != cudaSuccess) {
        // The API can surface asynchronous failures from earlier work. Do not publish a slot map.
        std::fprintf(stderr, "Inkling batched expert upload failed: CUDA status=%d range=%zu count=%zu\n", int(rc), failed, count);
        std::abort();
    }
    static std::once_flag report;
    std::call_once(report,[&]{ std::fprintf(stderr,"Inkling same-stream batched expert H2D active: ranges=%zu\n",count); });
    return true;
}
