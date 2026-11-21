#include "launch.hpp"
#include "compact_ptx.inc"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <mutex>

namespace {
using Handle = void *;
using Proc = int;
struct Driver {
    Handle lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    template <class T> T fn(const char *s) const {
        return lib ? reinterpret_cast<T>(dlsym(lib, s)) : nullptr;
    }
    Proc (*ctx)(Handle *) = fn<decltype(ctx)>("cuCtxGetCurrent");
    Proc (*device)(int *) = fn<decltype(device)>("cuCtxGetDevice");
    Proc (*attribute)(int *, int, int) = fn<decltype(attribute)>("cuDeviceGetAttribute");
    Proc (*load)(Handle *, const void *, unsigned, void *,
                 void **) = fn<decltype(load)>("cuModuleLoadDataEx");
    Proc (*function)(Handle *, Handle, const char *) = fn<decltype(function)>("cuModuleGetFunction");
    Proc (*launch)(Handle, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, Handle,
                   void **, void **) = fn<decltype(launch)>("cuLaunchKernel");
    Proc (*errstr)(int, const char **) = fn<decltype(errstr)>("cuGetErrorString");
    bool err(int rc, const char *action) const {
        if (!rc)
            return false;
        const char *msg = nullptr;
        if (errstr)
            errstr(rc, &msg);
        std::fprintf(stderr, "Inkling compact attention: %s failed (%d, %s)\n", action, rc,
                     msg ? msg : "unknown");
        return true;
    }
};
} // namespace
extern "C" bool inkling_compact_fa_launch(void *stream, const ggml_tensor *q, const ggml_tensor *k,
                                          const ggml_tensor *v, const ggml_tensor *t,
                                          const ggml_tensor *p, ggml_tensor *dst, float scale) {
    if (!inkling_compact_fa_supported(q, k, v, t, p, dst, scale) || !q->data || !k->data || !v->data ||
        !t->data || !p->data || !dst->data || uintptr_t(k->data) % 16 || uintptr_t(v->data) % 16)
        return false;
    static Driver d;
    static std::mutex guard;
    static std::map<Handle, Handle> functions;
    if (!d.ctx || !d.device || !d.attribute || !d.load || !d.function || !d.launch) {
        std::fprintf(stderr, "Inkling compact attention: CUDA driver entry points unavailable\n");
        return false;
    }
    Handle context = nullptr, fn = nullptr;
    if (d.err(d.ctx(&context), "get context") || !context)
        return false;
    {
        std::lock_guard<std::mutex> lock(guard);
        const auto found = functions.find(context);
        if (found == functions.end()) {
            int dev = 0, major = 0, minor = 0;
            if (d.err(d.device(&dev), "get device") ||
                d.err(d.attribute(&major, 75, dev), "compute major") ||
                d.err(d.attribute(&minor, 76, dev), "compute minor") || major != 12 || minor != 0)
                return false;
            Handle mod = nullptr;
            char log[2048]{};
            unsigned opts[] = {5, 6}; // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
            void *vals[] = {log, reinterpret_cast<void *>(sizeof(log))};
            if (d.err(d.load(&mod, compact_ptx, 2, opts, vals), "JIT SM120 PTX")) {
                std::fprintf(stderr, "Inkling compact PTX: %s\n", log);
                return false;
            }
            if (d.err(d.function(&fn, mod, "inkling_compact_mma"), "resolve kernel"))
                return false;
            functions.emplace(context, fn);
        } else
            fn = found->second;
    }
    int nq = q->ne[1], nk = k->ne[1], heads = q->ne[2], hkv = k->ne[2], extent = t->ne[1] - 1;
    int qs1 = q->nb[1] / 4, qs2 = q->nb[2] / 4, ks1 = k->nb[1] / 2, ks2 = k->nb[2] / 2,
        vs1 = v->nb[1] / 2, vs2 = v->nb[2] / 2;
    int ts1 = t->nb[1] / 4, ts2 = t->nb[2] / 4, ds1 = dst->nb[1] / 4, ds2 = dst->nb[2] / 4;
    void *qp = q->data, *kp = k->data, *vp = v->data, *tp = t->data, *pp = p->data, *out = dst->data;
    void *args[] = {&qp,  &kp,  &vp,  &tp,  &pp,  &out, &nq,  &nk,  &heads, &hkv, &extent,
                    &qs1, &qs2, &ks1, &ks2, &vs1, &vs2, &ts1, &ts2, &ds1,   &ds2, &scale};
    constexpr unsigned shared = 4 * 32 * 136 * sizeof(uint16_t);
    return !d.err(d.launch(fn, (nq + 15) / 16, heads, 1, 128, 1, 1, shared, stream, args, nullptr),
                  "launch");
}
