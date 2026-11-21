#include "launch.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {
int cuInit(unsigned);
int cuDeviceGet(int *, int);
int cuCtxCreate_v2(void **, unsigned, int);
int cuCtxDestroy_v2(void *);
int cuStreamCreate(void **, unsigned);
int cuStreamDestroy_v2(void *);
int cuStreamSynchronize(void *);
int cuMemAlloc_v2(uint64_t *, size_t);
int cuMemFree_v2(uint64_t);
int cuMemcpyHtoD_v2(uint64_t, const void *, size_t);
int cuMemcpyDtoH_v2(void *, uint64_t, size_t);
int cuGetErrorString(int, const char **);
}
static void check(int rc) {
    if (rc) {
        const char *msg = nullptr;
        cuGetErrorString(rc, &msg);
        std::fprintf(stderr, "CUDA (%d): %s\n", rc, msg ? msg : "");
        std::exit(3);
    }
}
struct Tensor {
    ggml_tensor g{};
    uint64_t gpu = 0;
    Tensor(ggml_type type, int d0, int d1, int d2, int scalar_bytes, bool permute = false) {
        g.type = type;
        g.ne[0] = d0;
        g.ne[1] = d1;
        g.ne[2] = d2;
        g.ne[3] = 1;
        g.nb[0] = scalar_bytes;
        g.nb[1] = d0 * scalar_bytes * (permute ? d2 : 1);
        g.nb[2] = d0 * scalar_bytes * (permute ? 1 : d1);
        g.nb[3] = size_t(d0) * d1 * d2 * scalar_bytes;
        check(cuMemAlloc_v2(&gpu, g.nb[3]));
        g.data = (void *)gpu;
    }
    void set(const void *p) { check(cuMemcpyHtoD_v2(gpu, p, g.nb[3])); }
    ~Tensor() {
        if (gpu)
            cuMemFree_v2(gpu);
    }
};
static float r16(float x) { return float((_Float16)x); }
static float wave(unsigned i, unsigned j) {
    unsigned t = (i + 31) * 2654435761u + j * 1597334677u;
    t ^= t >> 16;
    t *= 2246822507u;
    t ^= t >> 13;
    return (float(t % 2001) - 1000.0f) * .001f;
}
static bool trial(void *stream, int nq, int nk, int h, int hk, int extent, bool timed) {
    constexpr int d = 128;
    Tensor Q(GGML_TYPE_F32, d, nq, h, 4, true), K(GGML_TYPE_F16, d, nk, hk, 2, true),
        V(GGML_TYPE_F16, d, nk, hk, 2, true);
    Tensor T(GGML_TYPE_F32, h, extent + 1, nq, 4), P(GGML_TYPE_I32, nk + nq, 1, 1, 4),
        O(GGML_TYPE_F32, d, h, nq, 4);
    std::vector<float> q(size_t(nq) * h * d), table(size_t(nq) * h * (extent + 1)),
        out(size_t(nq) * h * d);
    std::vector<_Float16> k(size_t(nk) * hk * d), v(k.size());
    std::vector<int> pos(nk + nq);
    for (size_t i = 0; i < q.size(); i++)
        q[i] = wave(i, 1) * 1.7f;
    for (size_t i = 0; i < k.size(); i++) {
        k[i] = (_Float16)(wave(i, 2) * 2.1f);
        v[i] = (_Float16)(wave(i, 3) * .9f);
    }
    for (int i = 0; i < nq; i++)
        for (int e = 0; e <= extent; e++)
            for (int j = 0; j < h; j++) {
                // The remote table row is generally -sink, and it varies by query/head.
                const float sink = wave(i, j + 25) * 1.25f;
                table[(i * (extent + 1) + e) * h + j] =
                    (e == extent ? 0.0f : wave(37 * e + i, j + 22) * 1.5f) - sink;
            }
    table[0] = -0.0f;
    if (table.size() > 1)
        table[1] = 1.00048828125f;
    for (int j = 0; j < nk; j++)
        pos[j] = (nk > 2 && j % 11 == 7) ? -1 : 200000 + (j * 17) % nk;
    for (int i = 0; i < nq; i++)
        pos[nk + i] =
            timed ? 200000 + nk - nq + i : (i == 0 && nq > 1 ? 199999 : 200000 + (i * 31) % nk);
    Q.set(q.data());
    K.set(k.data());
    V.set(v.data());
    T.set(table.data());
    P.set(pos.data());
    const float scale = 1.0f / std::sqrt(float(d));
    auto run = [&] {
        if (!inkling_compact_fa_launch(stream, &Q.g, &K.g, &V.g, &T.g, &P.g, &O.g, scale)) {
            std::fprintf(stderr, "launch returned false\n");
            std::exit(2);
        }
        check(cuStreamSynchronize(stream));
    };
    run();
    if (timed) {
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < 3; r++)
            run();
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 3;
        std::printf("Fused global layer Q=%d K=%d H=%d: %.3f ms, %.3f effective TFLOP/s\n", nq, nk, h,
                    1000 * elapsed, (4. * d * nq * nk * h) / (elapsed * 1e12));
        return true;
    }
    check(cuMemcpyDtoH_v2(out.data(), O.gpu, O.g.nb[3]));
    std::vector<double> logits(nk), expected(d);
    double maxerr = 0, err2 = 0;
    size_t worst = 0, count = 0;
    bool exact_blocked = true;
    for (int i = 0; i < nq; i++)
        for (int j = 0; j < h; j++) {
            const int kh = j / (h / hk), qp = pos[nk + i];
            double m = 0.0;
            for (int r = 0; r < nk; r++) {
                const int kp = pos[r];
                if (kp < 0 || kp > qp) {
                    logits[r] = -INFINITY;
                    continue;
                }
                float dot = 0;
                for (int z = 0; z < d; z++)
                    dot +=
                        r16(r16(q[(i * h + j) * d + z]) * r16(scale)) * float(k[(r * hk + kh) * d + z]);
                const int bucket = std::min(qp - kp, extent);
                logits[r] = dot + r16(table[(i * (extent + 1) + bucket) * h + j] + 0.0f);
                m = std::max(m, logits[r]);
            }
            std::fill(expected.begin(), expected.end(), 0.0);
            double den = std::exp(-m);
            for (int r = 0; r < nk; r++) {
                double p = std::exp(logits[r] - m);
                den += p;
                for (int z = 0; z < d; z++)
                    expected[z] += p * float(v[(r * hk + kh) * d + z]);
            }
            for (int z = 0; z < d; z++) {
                size_t ix = (i * h + j) * d + z;
                double y = out[ix], err = std::abs(y - expected[z] / den);
                if (!std::isfinite(y)) {
                    std::fprintf(stderr, "nonfinite at %zu\n", ix);
                    return false;
                }
                if (qp == 199999 && y != 0)
                    exact_blocked = false;
                if (err > maxerr) {
                    maxerr = err;
                    worst = ix;
                }
                err2 += err * err;
                count++;
            }
        }
    const double rms = std::sqrt(err2 / count);
    std::printf("Q=%d K=%d H=%d:%d E=%d max_abs=%.8f rms=%.8f worst_i=%zu all_blocked_exact=%d\n", nq,
                nk, h, hk, extent, maxerr, rms, worst, exact_blocked);
    return maxerr < 0.0025 && rms < 0.00035 && exact_blocked;
}
int main(int argc, char **argv) {
    const bool timed = argc == 2 && !std::strcmp(argv[1], "--timing");
    if (argc > 1 && !timed)
        return 2;
    check(cuInit(0));
    int dev;
    void *ctx, *stream;
    check(cuDeviceGet(&dev, 0));
    check(cuCtxCreate_v2(&ctx, 0, dev));
    check(cuStreamCreate(&stream, 1));
    bool ok = trial(stream, 1, 1, 3, 1, 0, false);
    ok = trial(stream, 19, 73, 6, 2, 4, false) && ok;
    ok = trial(stream, 37, 263, 24, 8, 11, false) && ok;
    ok = trial(stream, 32, 129, 3, 1, 1024, false) && ok;
    if (timed && ok)
        ok = trial(stream, 256, 32768, 24, 8, 1024, true);
    check(cuStreamDestroy_v2(stream));
    check(cuCtxDestroy_v2(ctx));
    return ok ? 0 : 1;
}
