// Device-only CUDA for NVIDIA SM120. The single model-specific operation expects
// FP32 queries and learned relative bias, FP16 keys/values, and absolute positions.
// It uses Clang's NVPTX target without toolkit headers or external device libraries.
#define DEVICE __attribute__((device, always_inline)) inline
#define KERNEL extern "C" __attribute__((global))
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
struct F4 {
    float x, y, z, w;
};
struct A4 {
    u32 x, y, z, w;
};
struct U4 {
    u32 x, y, z, w;
} __attribute__((aligned(16)));

DEVICE int tid() {
    unsigned x;
    asm("mov.u32 %0, %%tid.x;" : "=r"(x));
    return x;
}
DEVICE int bx() {
    unsigned x;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(x));
    return x;
}
DEVICE int by() {
    unsigned x;
    asm("mov.u32 %0, %%ctaid.y;" : "=r"(x));
    return x;
}
DEVICE u16 rn16(float x) {
    u16 y;
    asm("cvt.rn.f16.f32 %0,%1;" : "=h"(y) : "f"(x));
    return y;
}
DEVICE float f32(u16 x) {
    float y;
    asm("cvt.f32.f16 %0,%1;" : "=f"(y) : "h"(x));
    return y;
}
DEVICE u32 pack(u16 lo, u16 hi) { return u32(lo) | (u32(hi) << 16); }
DEVICE u32 mulh2(u32 a, u32 b) {
    u32 c;
    asm("mul.rn.f16x2 %0,%1,%2;" : "=r"(c) : "r"(a), "r"(b));
    return c;
}
DEVICE float mx(float a, float b) {
    float c;
    asm("max.f32 %0,%1,%2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}
DEVICE float epx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0,%1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
    return y;
}
DEVICE float sx(float x, int n) {
    float y;
    asm("shfl.sync.bfly.b32 %0,%1,%2,31,0xffffffff;" : "=f"(y) : "f"(x), "r"(n));
    return y;
}
DEVICE float row_max(float x) {
    x = mx(x, sx(x, 1));
    return mx(x, sx(x, 2));
}
DEVICE float row_sum(float x) {
    x += sx(x, 1);
    return x + sx(x, 2);
}
DEVICE unsigned active(int b) {
    unsigned v;
    asm("{ .reg .pred p; setp.ne.s32 p,%1,0; vote.sync.ballot.b32 %0,p,0xffffffff; }"
        : "=r"(v)
        : "r"(b));
    return v;
}
DEVICE u32 saddr(const void *p) {
    u64 u;
    asm("cvta.to.shared.u64 %0,%1;" : "=l"(u) : "l"(p));
    return u32(u);
}
DEVICE void stage16(void *s, const void *g, int n) {
    asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" ::"r"(saddr(s)), "l"(g), "r"(n)
                 : "memory");
}
DEVICE void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
DEVICE void wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
DEVICE int warp_min(int x) {
    int z;
    asm("redux.sync.min.s32 %0,%1,0xffffffff;" : "=r"(z) : "r"(x));
    return z;
}
DEVICE int warp_max(int x) {
    int z;
    asm("redux.sync.max.s32 %0,%1,0xffffffff;" : "=r"(z) : "r"(x));
    return z;
}
DEVICE void ldm(u32 &x, u32 &y, const u16 *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
                 : "=r"(x), "=r"(y)
                 : "r"(saddr(p))
                 : "memory");
}
DEVICE void ldmt(u32 &x, u32 &y, const u16 *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];"
                 : "=r"(x), "=r"(y)
                 : "r"(saddr(p))
                 : "memory");
}
DEVICE void warp_sync() { asm volatile("bar.warp.sync 0xffffffff;" ::: "memory"); }
DEVICE void block_sync() { asm volatile("bar.sync 0;" ::: "memory"); }
DEVICE float normalized_mask(float x) {
    float kept;
    asm("add.rn.f32 %0,%1,0f00000000;" : "=f"(kept) : "f"(x));
    return f32(rn16(kept));
}
// PTX m16n8k16 FP16 fragment mapping as in ggml-cuda/mma.cuh. Rows of A are
// queries, rows of physical K are the columns of B; accumulation stays FP32.
DEVICE void mma(F4 &d, A4 a, u32 b0, u32 b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d.x), "+f"(d.y), "+f"(d.z), "+f"(d.w)
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}
DEVICE u32 qpair(const float *p, int n, u32 scale) {
    return mulh2(pack(rn16(p[n]), rn16(p[n + 1])), scale);
}
DEVICE float mbias(const float *table, int dstep, int extent, int keypos, int querypos, float remote) {
    if (keypos < 0 || keypos > querypos)
        return -__builtin_inff();
    const int dist = querypos - keypos;
    return dist >= extent ? remote : normalized_mask(table[dist * dstep]);
}
DEVICE u32 load_pair(const u16 *p, int n) { return *reinterpret_cast<const u32 *>(p + n); }

// One CTA covers 16 queries of a single head. Its four warps process separate
// consecutive key ranges, then combine via log-sum-exp exactly once per row.
KERNEL void inkling_compact_mma(const float *__restrict q, const u16 *__restrict k,
                                const u16 *__restrict v, const float *__restrict t,
                                const int *__restrict pos, float *__restrict dst, int nq, int nk,
                                int heads, int heads_kv, int extent, int qs1, int qs2, int ks1, int ks2,
                                int vs1, int vs2, int ts1, int ts2, int ds1, int ds2, float scale) {
    extern __attribute__((shared, aligned(16))) unsigned char smem[];
    constexpr int warps = 4, D = 128, pitch = 136;
    constexpr float max_offset = 3.0f * 0.6931f;
    const int thread = tid(), warp = thread >> 5, lane = thread & 31, group = lane >> 2, sub = lane & 3;
    const int q0 = 16 * bx(), head = by(), kvhead = head / (heads / heads_kv);
    const int row[2] = {q0 + group, q0 + group + 8};
    const int qp[2] = {row[0] < nq ? pos[nk + row[0]] : -1, row[1] < nq ? pos[nk + row[1]] : -1};
    const float *tables[2] = {t + head + (row[0] < nq ? row[0] : 0) * ts2,
                              t + head + (row[1] < nq ? row[1] : 0) * ts2};
    const float remote[2] = {normalized_mask(tables[0][extent * ts1]),
                             normalized_mask(tables[1][extent * ts1])};
    const float *qr[2] = {q + head * qs2 + (row[0] < nq ? row[0] : 0) * qs1,
                          q + head * qs2 + (row[1] < nq ? row[1] : 0) * qs1};
    const u32 hs = pack(rn16(scale), rn16(scale));
    A4 qa[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        int d = j * 16 + 2 * sub;
        qa[j] = {qpair(qr[0], d, hs), qpair(qr[1], d, hs), qpair(qr[0], d + 8, hs),
                 qpair(qr[1], d + 8, hs)};
    }
    F4 numerator[16] = {};
    float maximum[2] = {-1e30f, -1e30f}, denominator[2] = {0, 0};
    const int blocks = (nk + 15) / 16, begin = (blocks * warp / warps) * 16,
              end = (blocks * (warp + 1) / warps) * 16;
    const u16 *kh = k + kvhead * ks2, *vh = v + kvhead * vs2;
    u16 *kt = reinterpret_cast<u16 *>(smem) + warp * 32 * pitch;
    u16 *vt = kt + 16 * pitch;
    const int pmin = warp_min((row[0] < nq && row[1] < nq) ? (qp[0] < qp[1] ? qp[0] : qp[1])
                              : row[0] < nq                ? qp[0]
                                                           : 2147483647);
    const int pmax = warp_max(qp[0] > qp[1] ? qp[0] : qp[1]);
    for (int key = begin; key < end; key += 16) {
        const int qi = key + (lane & 15);
        const int kp = qi < nk ? pos[qi] : -1;
        const int low = warp_min(kp), high = warp_max(kp);
        if (high < 0 || low > pmax)
            continue;
#pragma unroll
        for (int z = 0; z < 8; z++) {
            int r = 2 * z + (lane >> 4), col = (lane & 15) * 8;
            const int ok = key + r < nk, i = ok ? key + r : 0;
            stage16(kt + r * pitch + col, kh + i * ks1 + col, ok ? 16 : 0);
            stage16(vt + r * pitch + col, vh + i * vs1 + col, ok ? 16 : 0);
        }
        commit();
        F4 bias[2];
        if (low >= 0 && ((long long)pmin - high) >= extent) {
            bias[0] = bias[1] = {remote[0], remote[0], remote[1], remote[1]};
        } else {
#pragma unroll
            for (int b = 0; b < 2; b++) {
                int i = key + 8 * b + sub * 2;
                int p0 = i < nk ? pos[i] : -1, p1 = i + 1 < nk ? pos[i + 1] : -1;
                bias[b] = {mbias(tables[0], ts1, extent, p0, qp[0], remote[0]),
                           mbias(tables[0], ts1, extent, p1, qp[0], remote[0]),
                           mbias(tables[1], ts1, extent, p0, qp[1], remote[1]),
                           mbias(tables[1], ts1, extent, p1, qp[1], remote[1])};
            }
        }
        wait0();
        warp_sync();
        F4 score[2] = {};
#pragma unroll
        for (int j = 0; j < 8; j++) {
#pragma unroll
            for (int b = 0; b < 2; b++) {
                const u16 *kb = kt + (8 * b + (lane & 7)) * pitch + 16 * j + 8 * ((lane >> 3) & 1);
                u32 b0, b1;
                ldm(b0, b1, kb);
                mma(score[b], qa[j], b0, b1);
            }
        }
#pragma unroll
        for (int b = 0; b < 2; b++) {
            score[b].x += bias[b].x;
            score[b].y += bias[b].y;
            score[b].z += bias[b].z;
            score[b].w += bias[b].w;
        }
        float newmax[2] = {row_max(mx(mx(score[0].x, score[0].y), mx(score[1].x, score[1].y))),
                           row_max(mx(mx(score[0].z, score[0].w), mx(score[1].z, score[1].w)))};
        float alpha[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            newmax[i] = mx(maximum[i], newmax[i] + max_offset);
            float delta = maximum[i] - newmax[i];
            alpha[i] = delta < -20.0f ? 0.0f : epx(delta);
            maximum[i] = newmax[i];
            denominator[i] *= alpha[i];
        }
#pragma unroll
        for (int b = 0; b < 2; b++) {
            score[b].x = epx(score[b].x - newmax[0]);
            score[b].y = epx(score[b].y - newmax[0]);
            score[b].z = epx(score[b].z - newmax[1]);
            score[b].w = epx(score[b].w - newmax[1]);
            denominator[0] += score[b].x + score[b].y;
            denominator[1] += score[b].z + score[b].w;
        }
        A4 prob = {pack(rn16(score[0].x), rn16(score[0].y)), pack(rn16(score[0].z), rn16(score[0].w)),
                   pack(rn16(score[1].x), rn16(score[1].y)), pack(rn16(score[1].z), rn16(score[1].w))};
#pragma unroll
        for (int j = 0; j < 16; j++) {
            numerator[j].x *= alpha[0];
            numerator[j].y *= alpha[0];
            numerator[j].z *= alpha[1];
            numerator[j].w *= alpha[1];
            const u16 *column = vt + (lane & 15) * pitch + 8 * j;
            u32 b0, b1;
            ldmt(b0, b1, column);
            mma(numerator[j], prob, b0, b1);
        }
        warp_sync();
    }
    // Shared V scratch is no longer live. Store disjoint warp partials and
    // combine them with one zero-valued attention sink (denominator only).
    block_sync();
    float *partials = reinterpret_cast<float *>(smem);
    float *stats = partials + warps * 16 * D;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const float sum = row_sum(denominator[i]);
        if (sub == 0) {
            int r = 16 * warp + group + 8 * i;
            stats[2 * r] = maximum[i];
            stats[2 * r + 1] = sum;
        }
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
        float *p = partials + (16 * warp + group) * D + j * 8 + sub * 2;
        p[0] = numerator[j].x;
        p[1] = numerator[j].y;
        p[8 * D] = numerator[j].z;
        p[8 * D + 1] = numerator[j].w;
    }
    block_sync();
    for (int i = thread; i < 16 * D; i += warps * 32) {
        int r = i / D, d = i % D;
        if (q0 + r >= nq)
            break;
        float m = max_offset;
#pragma unroll
        for (int w = 0; w < warps; w++)
            m = mx(m, stats[(16 * w + r) * 2]);
        float den = epx(-m), num = 0;
#pragma unroll
        for (int w = 0; w < warps; w++) {
            float weight = epx(stats[(16 * w + r) * 2] - m);
            den += stats[(16 * w + r) * 2 + 1] * weight;
            num += partials[(16 * w + r) * D + d] * weight;
        }
        dst[(q0 + r) * ds2 + head * ds1 + d] = num / den;
    }
}
