#include "launch.hpp"
#include <climits>
#include <cmath>

static bool dimensions(const ggml_tensor *t, ggml_type type, int width) {
    if (!t || t->type != type || t->ne[3] != 1 || t->nb[0] != size_t(width))
        return false;
    size_t last = 0;
    for (int i = 0; i < 3; i++) {
        if (t->ne[i] < 1 || t->ne[i] > INT_MAX || t->nb[i] % width || t->nb[i] / width > INT_MAX)
            return false;
        const size_t step = t->nb[i] / width;
        if (step > size_t(INT_MAX - last) / size_t(t->ne[i]))
            return false;
        last += step * (t->ne[i] - 1);
    }
    return true;
}
extern "C" bool inkling_compact_fa_supported(const ggml_tensor *q, const ggml_tensor *k,
                                             const ggml_tensor *v, const ggml_tensor *t,
                                             const ggml_tensor *p, const ggml_tensor *dst, float scale) {
    if (!dimensions(q, GGML_TYPE_F32, 4) || !dimensions(k, GGML_TYPE_F16, 2) ||
        !dimensions(v, GGML_TYPE_F16, 2) || !dimensions(t, GGML_TYPE_F32, 4) ||
        !dimensions(p, GGML_TYPE_I32, 4) || !dimensions(dst, GGML_TYPE_F32, 4))
        return false;
    const auto nq = q->ne[1], nk = k->ne[1], heads = q->ne[2], hkv = k->ne[2];
    return nq >= 1 && nk <= INT_MAX / 4 && nk + nq <= INT_MAX && heads <= 65535 && heads % hkv == 0 &&
           q->ne[0] == 128 && k->ne[0] == 128 && v->ne[0] == 128 && v->ne[1] == nk && v->ne[2] == hkv &&
           t->ne[0] == heads && t->ne[1] >= 1 && t->ne[2] == nq && p->ne[0] == nk + nq &&
           p->ne[1] == 1 && p->ne[2] == 1 && dst->ne[0] == 128 && dst->ne[1] == heads &&
           dst->ne[2] == nq && k->nb[1] % 16 == 0 && k->nb[2] % 16 == 0 && v->nb[1] % 16 == 0 &&
           v->nb[2] % 16 == 0 && std::isfinite(scale) && scale > 0;
}
