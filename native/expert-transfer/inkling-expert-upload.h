#pragma once
#include "ggml-backend.h"
struct inkling_expert_range {
    ggml_tensor *bank;
    const void *source;
    size_t offset;
    size_t bytes;
};
// Returns false only before queuing any bytes, for unsupported configurations.
extern "C" bool inkling_batch_expert_copies(ggml_backend_t, const inkling_expert_range *, size_t)
#ifndef INKLING_BATCH_IMPL
__attribute__((weak))
#endif
;
