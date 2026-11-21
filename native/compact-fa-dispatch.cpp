// The b12602 CUDA plugin exports its FA entry point through a PLT relocation.
// Keep its normal dispatch for dense nodes and dispatch only the marked Inkling nodes.
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <mutex>

struct ggml_backend_cuda_context;
extern "C" bool inkling_compact_fa_launch(void *, const ggml_tensor *, const ggml_tensor *,
        const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, float) __attribute__((weak));
extern "C" void inkling_compact_fa_capture(void *, const ggml_tensor *, float) __attribute__((weak));

namespace {
constexpr const char * fa_symbol = "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor";
constexpr const char * blas_symbol = "_ZN25ggml_backend_cuda_context13cublas_handleEv";

template<typename T>
T lookup(void * library, const char * symbol) {
    return library ? reinterpret_cast<T>(dlsym(library, symbol)) : nullptr;
}

[[noreturn]] void fail(const char * why) {
    std::fprintf(stderr, "Inkling compact FA dispatch failed: %s\n", why);
    std::abort();
}

void * cuda_backend() {
    // b12602 opens the plugin by its full path with RTLD_LOCAL. It has no SONAME.
    static void * handle = [] {
        void * found = nullptr;
        dl_iterate_phdr([](dl_phdr_info * info, size_t, void * data) {
            const char * start = std::strrchr(info->dlpi_name, '/');
            if (std::strcmp(start ? start + 1 : info->dlpi_name, "libggml-cuda.so") != 0) {
                return 0;
            }
            *static_cast<void **>(data) = dlopen(info->dlpi_name, RTLD_NOLOAD | RTLD_NOW);
            return 1;
        }, &found);
        return found;
    }();
    return handle;
}

using native_fn = bool (*)(ggml_backend_cuda_context *, ggml_tensor *);

bool native_requested() {
    const char * path = std::getenv("INKLING_NATIVE_FA_LIBRARY");
    return path && *path;
}

native_fn native_kernel() {
    static const auto launch = []() -> native_fn {
        const char * path = std::getenv("INKLING_NATIVE_FA_LIBRARY");
        if (!native_requested()) {
            return nullptr;
        }
        void * library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!library) {
            std::fprintf(stderr, "Inkling native FA unavailable: %s\n", dlerror());
            return nullptr;
        }
        auto fn = lookup<native_fn>(library, "inkling_compact_native_launch");
        if (!fn) {
            std::fprintf(stderr, "Inkling native FA unavailable: missing inkling_compact_native_launch\n");
        }
        return fn;
    }();
    return launch;
}

void * current_stream(ggml_backend_cuda_context & ctx) {
    using handle_fn = void * (*)(ggml_backend_cuda_context *);
    static auto handle = lookup<handle_fn>(cuda_backend(), blas_symbol);
    static auto get_stream = lookup<int (*)(void *, void **)>(
            dlopen("libcublas.so.12", RTLD_NOLOAD | RTLD_NOW), "cublasGetStream_v2");
    void * stream = nullptr;
    if (!handle || !get_stream || get_stream(handle(&ctx), &stream)) {
        fail("cannot get the CUDA backend's current stream");
    }
    return stream;
}
}

extern "C" bool inkling_compact_fa_ready() {
    static const bool ready = [] {
        if (!cuda_backend()) {
            return false;
        }
        auto * driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        auto init = lookup<int (*)(unsigned)>(driver, "cuInit");
        auto device = lookup<int (*)(int *, int)>(driver, "cuDeviceGet");
        auto capability = lookup<int (*)(int *, int *, int)>(driver, "cuDeviceComputeCapability");
        int gpu, major, minor;
        return init && device && capability && init(0) == 0 && device(&gpu, 0) == 0 &&
               capability(&major, &minor, gpu) == 0 && major == 12 && minor == 0 &&
               (native_requested() ? native_kernel() != nullptr : inkling_compact_fa_launch != nullptr);
    }();
    return ready;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!dst->src[6] || !dst->src[7]) {
        using fa_fn = void (*)(ggml_backend_cuda_context &, ggml_tensor *);
        static auto dense = lookup<fa_fn>(cuda_backend(), fa_symbol);
        if (!dense || dense == ggml_cuda_flash_attn_ext) {
            fail("cannot resolve the original b12602 CUDA attention dispatcher");
        }
        dense(ctx, dst);
        return;
    }
    float scale;
    std::memcpy(&scale, dst->op_params, sizeof(scale));
    auto native = native_kernel();
    void * stream = !native || inkling_compact_fa_capture ? current_stream(ctx) : nullptr;
    bool accepted = native ? native(&ctx, dst) :
            inkling_compact_fa_launch && inkling_compact_fa_launch(stream, dst->src[0], dst->src[1],
                    dst->src[2], dst->src[6], dst->src[7], dst, scale);
    if (!accepted) {
        fail("the compact kernel rejected the marked graph node");
    }
    static std::once_flag reported;
    if (inkling_compact_fa_capture) {
        inkling_compact_fa_capture(stream, dst, scale);
    }
    if (dst->src[0]->ne[1] == 1) {
        return; // The native single-query kernel reports its own dispatch.
    }
    std::call_once(reported, [&] {
        std::fprintf(stderr, "Inkling compact global attention executed: q=%lld kv=%lld position_bytes=%zu kernel=%s\n",
                (long long) dst->src[0]->ne[1], (long long) dst->src[1]->ne[1],
                size_t(dst->src[7]->ne[0])*sizeof(int32_t), native ? "native" : "hand");
    });
}

extern "C" bool inkling_compact_decode_ready() {
    static const bool ready = [] {
        if (!native_requested() || !inkling_compact_fa_ready()) {
            return false;
        }
        void * handle = dlopen(std::getenv("INKLING_NATIVE_FA_LIBRARY"), RTLD_NOLOAD | RTLD_NOW);
        return handle && dlsym(handle, "inkling_native_compact_decode_v1");
    }();
    return ready;
}
