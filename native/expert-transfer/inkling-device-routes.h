#pragma once
#include "ggml-backend.h"
#include <cstdint>
// v1 for b12602 Inkling Q4XL only, single CUDA0; all entry points mandatory for ownership.
struct ink_dev_layer {
    const void *source[2];
    void *bank[2];
    size_t stride[2];
};
struct ink_route_stats {
    uint64_t visits, hits, misses, bytes, all_hits;
};
#ifdef INK_ROUTE_IMPL
#define INK_EXPORT __attribute__((visibility("default")))
#else
#define INK_EXPORT __attribute__((weak))
#endif
extern "C" {
// A successful preflight guarantees support for this kernel image and mapped
// host memory on the selected device. Launches run on ggml's main CUDA stream.
INK_EXPORT bool ink_route_supported();
INK_EXPORT bool ink_route_graph_available();
INK_EXPORT void *ink_route_graph_map(void *state, int physical_layer);
INK_EXPORT void ink_route_graph_mark(ggml_tensor *node, void *state, int physical_layer, const ink_dev_layer *layout);
INK_EXPORT size_t ink_route_bytes();
INK_EXPORT void *ink_route_stream(ggml_backend_t);
INK_EXPORT bool ink_route_bind(ggml_backend_t, ggml_tensor *const *, ggml_tensor *const *, int,
                               ink_dev_layer *);
INK_EXPORT void ink_route_init(void *, void *, int);
INK_EXPORT bool ink_route_send(void *, void *, int, const int *, int *, const ink_dev_layer *);
// Phase counters use asynchronous, nonblocking sampling. A null observer is
// valid and final counters remain available through ink_route_finish.
INK_EXPORT void *ink_route_open_observer();
INK_EXPORT bool ink_route_poll(void *, void *, void *, ink_route_stats *);
INK_EXPORT void ink_route_close_observer(void *);
INK_EXPORT void ink_route_finish(void *, void *, ink_route_stats *);
}
