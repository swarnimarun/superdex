// Optional synthetic-input diagnostic. Outside the hot path unless this DSO is
// preloaded and INKLING_COMPACT_CAPTURE points to a new output file.
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace {
// Stable Linux x86-64 driver ABI for cuMemcpy2D_v2 (CUDA_MEMCPY2D).
struct copy2d {
    size_t sx, sy; int source_type; const void * host_source; uint64_t device_source; void * source_array; size_t source_pitch;
    size_t dx, dy; int dest_type; void * host_dest; uint64_t device_dest; void * dest_array; size_t dest_pitch;
    size_t width, height;
};
static_assert(sizeof(copy2d) == 128);
struct driver {
    void * lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    template<class T> T sym(const char * name) { return lib ? reinterpret_cast<T>(dlsym(lib, name)) : nullptr; }
    int (*copy)(const copy2d *) = sym<decltype(copy)>("cuMemcpy2D_v2");
    int (*sync)(void *) = sym<decltype(sync)>("cuStreamSynchronize");
    int (*capture)(void *, int *) = sym<decltype(capture)>("cuStreamIsCapturing");
};
struct snapshot {
    const char * name;
    const char * dtype;
    const ggml_tensor * src;
    size_t width, height, layers, bytes_per_element;
    std::vector<unsigned char> data;
};

bool read_tensor(const driver & d, snapshot & t) {
    const size_t row_bytes = t.width * t.bytes_per_element;
    t.data.resize(row_bytes * t.height * t.layers);
    for (size_t h = 0; h < t.layers; ++h) {
        copy2d c{};
        c.source_type = 2; // CU_MEMORYTYPE_DEVICE
        c.device_source = reinterpret_cast<uintptr_t>(t.src->data) + h*t.src->nb[2];
        c.source_pitch = t.src->nb[1];
        c.dest_type = 1; // CU_MEMORYTYPE_HOST
        c.host_dest = t.data.data() + h*t.height*row_bytes;
        c.dest_pitch = row_bytes;
        c.width = row_bytes;
        c.height = t.height;
        if (d.copy(&c)) return false;
    }
    return true;
}
}

extern "C" void inkling_compact_fa_capture(void * stream, const ggml_tensor * op, float scale) {
    const char * path = std::getenv("INKLING_COMPACT_CAPTURE");
    if (!path || !*path) return;
    const auto * q = op->src[0], * k = op->src[1], * v = op->src[2];
    const auto * table = op->src[6], * positions = op->src[7];
    const char * min_str = std::getenv("INKLING_COMPACT_CAPTURE_MIN_K");
    const long min_k = min_str ? std::strtol(min_str, nullptr, 10) : 4096;
    if (min_k <= 0 || k->ne[1] < min_k || k->ne[1] > 8192 || q->ne[1] < 16 || q->ne[0] != 128 ||
        k->ne[0] != 128 || v->ne[0] != 128 || q->ne[2] != 3*k->ne[2] || table->ne[1] != 1025) return;
    static std::atomic_flag attempted = ATOMIC_FLAG_INIT;
    if (attempted.test_and_set()) return;
    driver d;
    int capturing = 0;
    if (!d.copy || !d.sync || !d.capture || d.capture(stream, &capturing) || capturing || d.sync(stream)) {
        std::fprintf(stderr, "Inkling compact capture unavailable; use GGML_CUDA_DISABLE_GRAPHS=1 on a dedicated worker\n");
        return;
    }
    const size_t n_kv = k->ne[1], n_q = std::min<int64_t>(q->ne[1], 64);
    std::vector<snapshot> tensors{
        {"q","f32",q,128,n_q,3,4,{}}, {"k","f16",k,128,n_kv,1,2,{}},
        {"v","f16",v,128,n_kv,1,2,{}}, {"table","f32",table,3,1025,n_q,4,{}},
        {"positions","i32",positions,n_kv+n_q,1,1,4,{}}, {"out","f32",op,128,3,n_q,4,{}}};
    for (auto & t : tensors) {
        if (!read_tensor(d, t)) {
            std::fprintf(stderr, "Inkling compact capture failed reading %s\n", t.name);
            return;
        }
    }
    std::string temp = std::string(path) + ".tmp.XXXXXX";
    int fd = mkstemp(temp.data()); // owner-only, same directory as final path
    if (fd < 0) { std::perror("Inkling compact capture create"); return; }
    FILE * f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(temp.c_str()); return; }
    std::fprintf(f, "{\"format\":\"inkling-compact-v1\",\"layer_node\":\"%s\",\"scale\":%.9g,"
            "\"effective_sink\":0,\"sink_subtracted_from_table\":true,\"query_start\":0,\"q_head_start\":0,"
            "\"kv_head_start\":0,\"original_q_count\":%lld,\"original_q_heads\":%lld,\"tensors\":[",
            op->name, scale, (long long)q->ne[1], (long long)q->ne[2]);
    size_t offset = 0;
    for (auto & t : tensors) {
        std::fprintf(f, "%s{\"name\":\"%s\",\"type\":\"%s\",\"shape_ggml\":[%zu,%zu,%zu],"
                "\"source_strides\":[%zu,%zu,%zu],\"offset\":%zu,\"bytes\":%zu}",
                offset ? "," : "",t.name,t.dtype,t.width,t.height,t.layers,t.src->nb[0],t.src->nb[1],t.src->nb[2],
                offset,t.data.size());
        offset += t.data.size();
    }
    std::fputs("]}\n", f);
    for (auto & t : tensors) std::fwrite(t.data.data(), 1, t.data.size(), f);
    bool ok = !std::ferror(f);
    ok = std::fclose(f) == 0 && ok;
    if (ok && ::link(temp.c_str(), path) == 0) {
        std::fprintf(stderr, "Inkling compact capture wrote %zu bytes plus metadata to %s\n", offset, path);
    } else {
        std::fprintf(stderr, "Inkling compact capture write failed for %s (%s)\n", path, std::strerror(errno));
    }
    unlink(temp.c_str());
}
