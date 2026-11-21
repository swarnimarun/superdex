#define INK_ROUTE_IMPL
#include "common.cuh"
#include "ggml-backend-impl.h"
#include "inkling-device-routes.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <dlfcn.h>

namespace {
constexpr int nlayer = 48, n = 128, top = 8, classes = 8, maxslots = 3072;
struct State {
    int map[nlayer * n], owner[maxslots], prev[maxslots], next[maxslots];
    int cls[maxslots], head[classes], tail[classes], nslots;
    int64_t clock[nlayer], last[nlayer * n];
    float crf[nlayer * n];
    int miss_count, src_id[top], dst_id[top];
    ink_route_stats stats;
};
[[noreturn]] void fail(cudaError_t e, const char *op) {
    std::fprintf(stderr, "Inkling GPU expert routing failed at %s: %s\n", op, cudaGetErrorString(e));
    std::abort();
}
#define CK(x)                                                                                                \
    do {                                                                                                     \
        auto err = (x);                                                                                      \
        if (err != cudaSuccess)                                                                              \
            fail(err, #x);                                                                                   \
    } while (0)
__global__ void clear_state(State *s, int slots) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nlayer * n) {
        s->map[i] = slots;
        s->last[i] = -1;
        s->crf[i] = 0;
    }
    if (i < slots) {
        s->owner[i] = -1;
        s->prev[i] = i - 1;
        s->next[i] = i + 1 < slots ? i + 1 : -1;
        s->cls[i] = 0;
    }
    if (i < classes) {
        s->head[i] = i == 0 ? 0 : -1;
        s->tail[i] = i == 0 ? slots - 1 : -1;
    }
    if (i < nlayer)
        s->clock[i] = 0;
    if (!i) {
        s->nslots = slots;
        s->stats = {};
        s->miss_count = 0;
    }
}
__device__ void promote(State *s, int pos, int target) {
    const int from = s->cls[pos], prev = s->prev[pos], next = s->next[pos];
    if (from == target && s->tail[from] == pos)
        return;
    if (prev >= 0)
        s->next[prev] = next;
    else
        s->head[from] = next;
    if (next >= 0)
        s->prev[next] = prev;
    else
        s->tail[from] = prev;
    s->prev[pos] = s->tail[target];
    s->next[pos] = -1;
    if (s->tail[target] >= 0)
        s->next[s->tail[target]] = pos;
    else
        s->head[target] = pos;
    s->tail[target] = pos;
    s->cls[pos] = target;
}
__device__ bool protected_slot(int key, int layer, const int *ids) {
    if (key < 0 || key / n != layer)
        return false;
    for (int j = 0; j < top; ++j)
        if (key % n == ids[j])
            return true;
    return false;
}
__device__ int victim(State *s, int layer, const int *ids) {
    int best = -1;
    double score = -1;
    for (int c = 0; c < classes; ++c) {
        int p = s->head[c];
        while (p >= 0 && protected_slot(s->owner[p], layer, ids))
            p = s->next[p];
        if (p < 0)
            continue;
        const int k = s->owner[p];
        if (k < 0)
            return p;
        const double age = (double)(s->clock[k / n] - s->last[k]);
        const double val = age / (1 << c);
        if (val > score) {
            score = val;
            best = p;
        }
    }
    return best;
}
__device__ int level(float crf) { return min(classes - 1, ilogbf(crf)); }
template<bool selected_only> __global__ void resolve(State *s, int layer, const int *ids, int *dst, size_t stride) {
    if (!threadIdx.x) {
        int unique[top], num = 0;
        for (int j = 0; j < top; ++j) {
            const int e = ids[j];
            if (e < 0 || e >= n) {
                asm("trap;");
                return;
            }
            bool dup = false;
            for (int i = 0; i < num; ++i)
                dup |= e == unique[i];
            if (!dup)
                unique[num++] = e;
        }
        int missing[top], miss = 0;
        for (int j = 0; j < num; ++j) {
            const int e = unique[j], k = layer * n + e, pos = s->map[k];
            const float decay = exp2f(-float(s->clock[layer] - s->last[k]) / 128.f);
            s->crf[k] = __fadd_rn(__fmul_rn(s->crf[k], decay), 1.f);
            s->last[k] = s->clock[layer];
            if (pos < s->nslots) {
                promote(s, pos, level(s->crf[k]));
            } else {
                int i = miss++;
                while (i > 0 && missing[i - 1] > e) {
                    missing[i] = missing[i - 1];
                    --i;
                }
                missing[i] = e;
            }
        }
        for (int j = 0; j < miss; ++j) {
            const int e = missing[j], k = layer * n + e, p = victim(s, layer, ids);
            if (p < 0) {
                asm("trap;");
                return;
            }
            const int prior = s->owner[p];
            if (prior >= 0)
                s->map[prior] = s->nslots;
            s->owner[p] = k;
            s->map[k] = p;
            promote(s, p, level(s->crf[k]));
            s->src_id[j] = e;
            s->dst_id[j] = p;
        }
        s->miss_count = miss;
        ++s->clock[layer];
        ++s->stats.visits;
        s->stats.hits += num - miss;
        s->stats.misses += miss;
        s->stats.bytes += miss * stride;
        s->stats.all_hits += miss == 0;
    }
    __syncthreads();
    if constexpr (selected_only) {
        if (threadIdx.x < top) dst[threadIdx.x] = s->map[layer*n + ids[threadIdx.x]];
    } else {
        dst[threadIdx.x] = s->map[layer*n + threadIdx.x];
    }
}
__global__ void fill(const State *s, ink_dev_layer banks) {
    const int miss = s->miss_count;
    if (!miss)
        return;
    const int part = blockIdx.y, q = blockIdx.x % miss, words = banks.stride[part] / sizeof(uint4);
    const uint4 *src = static_cast<const uint4 *>(banks.source[part]) + size_t(s->src_id[q]) * words;
    uint4 *dst = static_cast<uint4 *>(banks.bank[part]) + size_t(s->dst_id[q]) * words;
    const int workers = (gridDim.x - 1 - q) / miss + 1;
    for (int i = (blockIdx.x / miss) * blockDim.x + threadIdx.x; i < words; i += workers * blockDim.x)
        dst[i] = src[i];
}
} // namespace
struct RouteObserver {
    cudaEvent_t ready{};
    ink_route_stats *host = nullptr;
    bool pending = false;
};

extern "C" bool ink_route_supported() {
    // The first release ships and validates SM120 only. Probe both the CUDA
    // image and mapped host memory before the host planner relinquishes ownership.
    int device = -1, major = 0, minor = 0, mapped = 0, uva = 0;
    cudaFuncAttributes attr{};
    const bool supported =
        cudaGetDevice(&device) == cudaSuccess && device == 0 &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
        major == 12 && minor == 0 &&
        cudaDeviceGetAttribute(&mapped, cudaDevAttrCanMapHostMemory, device) == cudaSuccess && mapped &&
        cudaDeviceGetAttribute(&uva, cudaDevAttrUnifiedAddressing, device) == cudaSuccess && uva &&
        cudaFuncGetAttributes(&attr, resolve<false>) == cudaSuccess &&
        cudaFuncGetAttributes(&attr, fill) == cudaSuccess &&
        cudaFuncGetAttributes(&attr, clear_state) == cudaSuccess;
    if (!supported)
        cudaGetLastError(); // discard a preflight error before the checked fallback
    return supported;
}

extern "C" void *ink_route_open_observer() {
    auto *observer = new (std::nothrow) RouteObserver;
    if (!observer)
        return nullptr;
    if (cudaEventCreateWithFlags(&observer->ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaMallocHost(&observer->host, sizeof(ink_route_stats)) != cudaSuccess) {
        cudaGetLastError();
        if (observer->ready)
            CK(cudaEventDestroy(observer->ready));
        delete observer;
        return nullptr;
    }
    return observer;
}

extern "C" bool ink_route_poll(void *handle, void *stream, void *state, ink_route_stats *out) {
    if (!handle)
        return false;
    auto *observer = static_cast<RouteObserver *>(handle);
    bool updated = false;
    if (observer->pending) {
        const auto ready = cudaEventQuery(observer->ready);
        if (ready == cudaErrorNotReady)
            return false;
        if (ready != cudaSuccess)
            fail(ready, "cudaEventQuery");
        *out = *observer->host;
        updated = true;
    }
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    CK(cudaMemcpyAsync(observer->host, &static_cast<State *>(state)->stats, sizeof(ink_route_stats),
                       cudaMemcpyDeviceToHost, cuda_stream));
    CK(cudaEventRecord(observer->ready, cuda_stream));
    observer->pending = true;
    return updated;
}

extern "C" void ink_route_close_observer(void *handle) {
    if (!handle)
        return;
    // Host calls after ink_route_finish has joined every stream consumer.
    auto *observer = static_cast<RouteObserver *>(handle);
    CK(cudaEventDestroy(observer->ready));
    CK(cudaFreeHost(observer->host));
    delete observer;
}

extern "C" size_t ink_route_bytes() { return sizeof(State); }
extern "C" void *ink_route_stream(ggml_backend_t be) {
    if (!be || std::strcmp(ggml_backend_name(be), "CUDA0"))
        return nullptr;
    auto *ctx = static_cast<ggml_backend_cuda_context *>(be->context);
    return ctx->device == 0 && ctx->curr_stream_no == 0 ? ctx->streams[0][0] : nullptr;
}
extern "C" bool ink_route_bind(ggml_backend_t be, ggml_tensor *const *weights, ggml_tensor *const *banks,
                               int slots, ink_dev_layer *out) {
    if (!ink_route_stream(be) || slots < top || slots > maxslots)
        return false;
    for (int i = 0; i < 2; ++i) {
        const auto *w = weights[i], *b = banks[i];
        if (!w || !b || !w->data || !b->data || w->type != b->type || w->ne[0] != b->ne[0] ||
            w->ne[1] != b->ne[1] || w->nb[1] != b->nb[1] || w->ne[2] != n || b->ne[2] != slots + 1 ||
            w->nb[2] != b->nb[2] || w->nb[2] % 16 || (uintptr_t(w->data) | uintptr_t(b->data)) % 16)
            return false;
        cudaPointerAttributes a{}, d{};
        if (cudaPointerGetAttributes(&a, w->data) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (cudaPointerGetAttributes(&d, b->data) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (a.type != cudaMemoryTypeHost || !a.devicePointer || uintptr_t(a.devicePointer) % 16 ||
            d.type != cudaMemoryTypeDevice || d.device != 0)
            return false;
        out->source[i] = a.devicePointer;
        out->bank[i] = b->data;
        out->stride[i] = w->nb[2];
    }
    return true;
}
extern "C" void ink_route_init(void *stream, void *state, int slots) {
    if (!state || !stream || slots < top || slots > maxslots)
        std::abort();
    clear_state<<<(nlayer * n + 127) / 128, 128, 0, (cudaStream_t)stream>>>((State *)state, slots);
    CK(cudaPeekAtLastError());
}
extern "C" bool ink_route_send(void *stream, void *state, int layer, const int *ids, int *dst,
                               const ink_dev_layer *b) {
    if (!stream || !state || layer < 0 || layer >= nlayer || !ids || !dst || !b)
        return false;
    resolve<false><<<1, n, 0, (cudaStream_t)stream>>>((State *)state, layer, ids, dst, b->stride[0] + b->stride[1]);
    CK(cudaPeekAtLastError());
    fill<<<dim3(256, 2), 256, 0, (cudaStream_t)stream>>>((State *)state, *b);
    CK(cudaPeekAtLastError());
    return true;
}
extern "C" void ink_route_finish(void *stream, void *state, ink_route_stats *out) {
    CK(cudaMemcpyAsync(out, &((State *)state)->stats, sizeof(*out), cudaMemcpyDeviceToHost,
                       (cudaStream_t)stream));
    CK(cudaStreamSynchronize((cudaStream_t)stream));
}

namespace {
constexpr int32_t route_magic = 0x49525437;
struct GraphRoute {
    State *state;
    ink_dev_layer projections;
    int32_t physical_layer, magic;
};
static_assert(sizeof(GraphRoute)==GGML_MAX_OP_PARAMS && offsetof(GraphRoute,magic)==60);
using GetRows = void (*)(ggml_backend_cuda_context &, ggml_tensor *);
const char * const get_rows_symbol = "_Z21ggml_cuda_op_get_rowsR25ggml_backend_cuda_contextP11ggml_tensor";
GetRows original_get_rows() {
    static const auto original=[] {
        auto *lib=dlopen("libggml-cuda.so",RTLD_NOW|RTLD_NOLOAD);
        return lib?reinterpret_cast<GetRows>(dlsym(lib,get_rows_symbol)):nullptr;
    }();
    return original;
}
}
__attribute__((visibility("default"))) void ggml_cuda_op_get_rows(ggml_backend_cuda_context &ctx,ggml_tensor *dst) {
    if (dst->op_params[15]!=route_magic) {
        auto original=original_get_rows();
        if(!original || original==ggml_cuda_op_get_rows) std::abort();
        original(ctx,dst);
        return;
    }
    GraphRoute descriptor;
    memcpy(&descriptor,dst->op_params,sizeof(descriptor));
    if (descriptor.physical_layer<0 || descriptor.physical_layer>=nlayer ||
        !descriptor.state || !dst->src[0] || !dst->src[1] || !dst->data || dst->type!=GGML_TYPE_I32 ||
        dst->src[0]->data!=descriptor.state->map+descriptor.physical_layer*n ||
        dst->ne[0]!=1 || dst->ne[1]!=top || dst->ne[2]!=1 || dst->ne[3]!=1 ||
        ctx.device!=0) std::abort();
    auto stream=ctx.stream();
    resolve<true><<<1,n,0,stream>>>(descriptor.state,descriptor.physical_layer,
        static_cast<const int *>(dst->src[1]->data),static_cast<int *>(dst->data),
        descriptor.projections.stride[0]+descriptor.projections.stride[1]);
    CK(cudaPeekAtLastError());
    fill<<<dim3(256,2),256,0,stream>>>(descriptor.state,descriptor.projections);
    CK(cudaPeekAtLastError());
    static const bool logged=[] { std::fputs("Inkling GPU expert graph dispatch observed\n",stderr);return true;}();
    (void)logged;
}
extern "C" bool ink_route_graph_available() {
    auto original=original_get_rows();
    // Check the winning interposer by DSO identity. Taking the address of
    // our public function alone is insufficient when LD_PRELOAD preempts it.
    Dl_info local{}, active{};
    void *selected=dlsym(RTLD_DEFAULT,get_rows_symbol);
    if (!original || !selected || !dladdr(selected,&active) ||
        !dladdr(reinterpret_cast<void *>(&original_get_rows),&local) ||
        active.dli_fbase!=local.dli_fbase || reinterpret_cast<GetRows>(selected)==original) return false;
    cudaFuncAttributes attr{};
    auto err=cudaFuncGetAttributes(&attr,resolve<true>);
    if(err!=cudaSuccess) cudaGetLastError();
    return err==cudaSuccess;
}
extern "C" void *ink_route_graph_map(void *state,int physical_layer) {
    return static_cast<State *>(state)->map + physical_layer*n;
}
extern "C" void ink_route_graph_mark(ggml_tensor *dst,void *state,int physical_layer,const ink_dev_layer *layout) {
    const GraphRoute descriptor{static_cast<State *>(state),*layout,physical_layer,route_magic};
    memcpy(dst->op_params,&descriptor,sizeof(descriptor));
}
