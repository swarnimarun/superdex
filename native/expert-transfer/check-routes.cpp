#define INK_ROUTE_IMPL
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include "inkling-device-routes.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <functional>
#include <vector>

#include "inkling-reference-lru.inc"
#define CK(x)                                                                                                \
    do {                                                                                                     \
        auto e = x;                                                                                          \
        if (e != cudaSuccess) {                                                                              \
            fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));                                          \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)
struct Pool {
    int slots;
    size_t size[2];
    uint8_t *host[2]{}, *banks[2]{};
    void *state = nullptr;
    int *route = nullptr, *dst = nullptr;
    moe_cache_lru lru;
    std::array<std::array<int, 128>, 48> maps{};
    std::array<ink_dev_layer, 48> layer{};
    ink_route_stats exp{};
    void init(cudaStream_t s, int c, size_t bytes) {
        slots = c;
        size[0] = bytes * 2;
        size[1] = bytes;
        for (int j = 0; j < 2; ++j) {
            CK(cudaMallocHost(reinterpret_cast<void **>(&host[j]), 3 * 128 * size[j]));
            CK(cudaMalloc(reinterpret_cast<void **>(&banks[j]), (slots + 1) * size[j]));
            CK(cudaMemsetAsync(banks[j], 0xad, (slots + 1) * size[j], s));
            for (size_t b = 0; b < 3 * 128 * size[j]; ++b)
                host[j][b] = uint8_t(b * 3 + (b >> 6) + (b >> 11) + j * 91);
            cudaPointerAttributes a{};
            CK(cudaPointerGetAttributes(&a, host[j]));
            assert(a.devicePointer);
            for (int l = 0; l < 3; ++l) {
                layer[l].source[j] = static_cast<char *>(a.devicePointer) + l * 128 * size[j];
                layer[l].bank[j] = banks[j];
                layer[l].stride[j] = size[j];
            }
        }
        CK(cudaMalloc(&state, ink_route_bytes()));
        CK(cudaMalloc(reinterpret_cast<void **>(&route), 32));
        CK(cudaMalloc(reinterpret_cast<void **>(&dst), 512));
        reset(s);
    }
    void reset(cudaStream_t s) {
        lru = moe_cache_lru{};
        lru.init(48, 128, slots);
        for (int l = 0; l < 48; ++l) {
            maps[l].fill(slots);
            lru.slot_map[l] = maps[l].data();
        }
        ink_route_init(s, state, slots);
        exp = {};
    }
    bool step(cudaStream_t s, int il, const std::array<int, 8> &ids) {
        std::vector<moe_cache_lru::fill> fills, cpu;
        size_t hits = 0;
        if (!lru.plan(
                il, ids.data(), ids.size(), 1, 8, 0, [](size_t) { return size_t(0); }, fills, cpu, hits))
            return false;
        CK(cudaMemcpyAsync(route, ids.data(), 32, cudaMemcpyHostToDevice, s));
        if (!ink_route_send(s, state, il, route, dst, &layer[il]))
            return false;
        std::array<int, 128> result;
        CK(cudaMemcpyAsync(result.data(), dst, 512, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        if (result != maps[il]) {
            fprintf(stderr, "slot map differs from stock ClassLRU at visit %llu layer=%d\n",
                    (unsigned long long)exp.visits, il);
            return false;
        }
        for (int j = 0; j < 2; ++j) {
            std::vector<uint8_t> contents((slots + 1) * size[j]);
            CK(cudaMemcpyAsync(contents.data(), banks[j], contents.size(), cudaMemcpyDeviceToHost, s));
            CK(cudaStreamSynchronize(s));
            for (int pos = 0; pos < slots; ++pos) {
                const int owner = lru.key_of[pos];
                if (owner < 0)
                    continue;
                if (memcmp(contents.data() + pos * size[j], host[j] + owner * size[j], size[j]))
                    return false;
            }
            if (!std::all_of(contents.end() - size[j], contents.end(), [](uint8_t x) { return x == 0xad; }))
                return false;
        }
        ++exp.visits;
        exp.hits += hits;
        exp.misses += fills.size();
        exp.bytes += fills.size() * (size[0] + size[1]);
        exp.all_hits += fills.empty();
        return true;
    }
    bool finish(cudaStream_t s) {
        ink_route_stats got{};
        ink_route_finish(s, state, &got);
        return got.visits == exp.visits && got.misses == exp.misses && got.hits == exp.hits &&
               got.bytes == exp.bytes && got.all_hits == exp.all_hits;
    }
    void release() {
        CK(cudaFree(route));
        CK(cudaFree(dst));
        CK(cudaFree(state));
        for (int j = 0; j < 2; ++j) {
            CK(cudaFree(banks[j]));
            CK(cudaFreeHost(host[j]));
        }
    }
};
// Exercise the Q4_K_M tail projections with the actual ggml CUDA backend and
// original byte strides. Only nine expert IDs are selected in this fixture.
static bool check_mixed_projections() {
    constexpr int slots = 8, rows = 9;
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend)
        return false;
    ggml_backend_synchronize(backend); // initialize ggml's main CUDA stream
    auto stream = static_cast<cudaStream_t>(ink_route_stream(backend));
    if (!stream)
        return false;
    const ggml_type types[2] = {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0};
    const int64_t dims[2][2] = {{3072, 1728}, {864, 3072}};
    const size_t strides[2] = {3649536, 2820096};
    ggml_tensor weights[2]{}, banks[2]{};
    ggml_tensor *source[2] = {&weights[0], &weights[1]}, *destination[2] = {&banks[0], &banks[1]};
    for (int i = 0; i < 2; ++i) {
        auto &w = weights[i];
        w.type = types[i];
        w.ne[0] = dims[i][0];
        w.ne[1] = dims[i][1];
        w.ne[2] = 128;
        w.ne[3] = 1;
        w.nb[0] = ggml_type_size(w.type);
        w.nb[1] = ggml_row_size(w.type, w.ne[0]);
        w.nb[2] = w.ne[1] * w.nb[1];
        w.nb[3] = w.ne[2] * w.nb[2];
        if (w.nb[2] != strides[i])
            return false;
        banks[i] = w;
        banks[i].ne[2] = slots + 1;
        banks[i].nb[3] = banks[i].ne[2] * w.nb[2];
        CK(cudaMallocHost(&w.data, rows * strides[i]));
        CK(cudaMalloc(&banks[i].data, (slots + 1) * strides[i]));
        CK(cudaMemsetAsync(banks[i].data, 0xAD, (slots + 1) * strides[i], stream));
        auto *host = static_cast<uint8_t *>(w.data);
        for (int row = 0; row < rows; ++row) {
            for (size_t off = 0; off < strides[i]; ++off)
                host[row * strides[i] + off] = static_cast<uint8_t>(row * 11 + i * 103 + off + (off >> 10));
        }
    }
    ink_dev_layer layout{};
    if (!ink_route_bind(backend, source, destination, slots, &layout))
        return false;
    banks[0].type = GGML_TYPE_Q8_0;
    if (ink_route_bind(backend, source, destination, slots, &layout))
        return false;
    banks[0].type = weights[0].type;
    void *state = nullptr, *ids_device = nullptr, *map_device = nullptr;
    CK(cudaMalloc(&state, ink_route_bytes()));
    CK(cudaMalloc(&ids_device, 8 * sizeof(int)));
    CK(cudaMalloc(&map_device, 128 * sizeof(int)));
    ink_route_init(stream, state, slots);
    void *observer = ink_route_open_observer();
    if (!observer)
        return false;
    std::array<int, 128> map{};
    std::array<int, 8> ids{};
    for (int step = 0; step < 3; ++step) {
        for (int j = 0; j < 8; ++j)
            ids[j] = j + (step == 2 ? 1 : 0);
        CK(cudaMemcpyAsync(ids_device, ids.data(), sizeof(ids), cudaMemcpyHostToDevice, stream));
        if (!ink_route_send(stream, state, 0, static_cast<const int *>(ids_device),
                            static_cast<int *>(map_device), &layout))
            return false;
        CK(cudaMemcpyAsync(map.data(), map_device, sizeof(map), cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        for (int i = 0; i < 2; ++i) {
            std::vector<uint8_t> result(strides[i]);
            const auto *data = static_cast<const uint8_t *>(banks[i].data);
            for (int id : ids) {
                if (map[id] < 0 || map[id] >= slots)
                    return false;
                CK(cudaMemcpyAsync(result.data(), data + map[id] * strides[i], strides[i],
                                   cudaMemcpyDeviceToHost, stream));
                CK(cudaStreamSynchronize(stream));
                const auto *expected = static_cast<const uint8_t *>(weights[i].data) + id * strides[i];
                if (std::memcmp(result.data(), expected, result.size()) != 0)
                    return false;
            }
            CK(cudaMemcpyAsync(result.data(), data + slots * strides[i], strides[i], cudaMemcpyDeviceToHost,
                               stream));
            CK(cudaStreamSynchronize(stream));
            if (!std::all_of(result.begin(), result.end(), [](uint8_t v) { return v == 0xAD; }))
                return false;
        }
        ink_route_stats sampled{};
        const bool complete = ink_route_poll(observer, stream, state, &sampled);
        if (complete != (step > 0) || (complete && sampled.visits != (unsigned)step))
            return false;
    }
    ink_route_stats final{};
    ink_route_finish(stream, state, &final);
    ink_route_close_observer(observer);
    CK(cudaFree(state));
    CK(cudaFree(ids_device));
    CK(cudaFree(map_device));
    for (int i = 0; i < 2; ++i) {
        CK(cudaFree(banks[i].data));
        CK(cudaFreeHost(weights[i].data));
    }
    ggml_backend_free(backend);
    if (final.visits != 3 || final.misses != 9 || final.hits != 15 || final.all_hits != 1 ||
        final.bytes != 9 * (strides[0] + strides[1]))
        return false;
    std::puts("Q4_K_M Q5_K gate-up + Q8_0 down: nine misses, full hit, eviction, guards and snapshots exact");
    return true;
}

int main() {
    if (!ink_route_supported()) {
        std::fprintf(stderr, "CUDA0 requires SM120 and mapped host memory for device-owned expert routing\n");
        return 77;
    }
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    Pool p[2];
    p[0].init(s, 8, 128);
    p[1].init(s, 23, 256);
    for (int epoch = 0; epoch < 2; ++epoch) {
        for (int i = 0; i < 256; ++i) {
            const int group = (i / 2) % 2;
            const int tick = i / 4, il = (tick / 2) % 3;
            std::array<int, 8> ids;
            for (int j = 0; j < 8; ++j)
                ids[j] = ((tick / 2) * 11 + j * 13 + group * 31) % 128;
            if (tick % 7 == 0)
                ids[7] = ids[0];
            if (!p[group].step(s, il, ids))
                return 2;
        }
        for (auto &v : p) {
            if (!v.finish(s))
                return 3;
            printf("epoch=%d slots=%d visits=%llu hits=%llu misses=%llu all_hits=%llu "
                   "maps+owners+data+guard=exact\n",
                   epoch, v.slots, (unsigned long long)v.exp.visits, (unsigned long long)v.exp.hits,
                   (unsigned long long)v.exp.misses, (unsigned long long)v.exp.all_hits);
            if (epoch == 0)
                v.reset(s); // preserve dirty banks, reset ownership
        }
    }
    for (auto &v : p)
        v.release();
    CK(cudaStreamDestroy(s));
    return check_mixed_projections() ? 0 : 4;
}
