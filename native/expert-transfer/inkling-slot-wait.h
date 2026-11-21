#pragma once
#include "ggml-backend.h"

// For a single-buffer host-weight input on the selected CUDA backend, this
// callback may enqueue a read of prior router output before waiting for reuse.
// False means it did not use or modify any tensor data. True means it waited for
// all previous work on this backend stream before touching any reusable host
// or device data and handled the whole copy. Replacing the normal copy callback
// drops this contract, so other callback implementations retain the original wait.
extern "C" { GGML_API void inkling_set_waiting_copy_callback(
        ggml_backend_sched_t sched, ggml_backend_sched_copy_callback copy)
#ifndef INKLING_SLOT_WAIT_IMPL
        __attribute__((weak))
#endif
        ;
}
