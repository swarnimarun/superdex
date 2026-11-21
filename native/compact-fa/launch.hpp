#pragma once
#include "ggml.h"
extern "C" bool inkling_compact_fa_supported(const ggml_tensor *q, const ggml_tensor *k,
                                             const ggml_tensor *v, const ggml_tensor *table,
                                             const ggml_tensor *positions, const ggml_tensor *dst,
                                             float scale);
extern "C" bool inkling_compact_fa_launch(void *stream, const ggml_tensor *q, const ggml_tensor *k,
                                          const ggml_tensor *v, const ggml_tensor *table,
                                          const ggml_tensor *positions, ggml_tensor *dst, float scale);
