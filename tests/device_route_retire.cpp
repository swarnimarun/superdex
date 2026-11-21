#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define GGML_ASSERT assert
struct ink_route_stats { uint64_t visits{}, hits{}, misses{}, bytes{}, all_hits{}; };
static std::string calls;
static bool backend_alive = true;
static int stream;
static size_t ggml_backend_buffer_get_size(void *) { return 148024; }
static void *ggml_backend_buffer_get_base(void *b) { return b; }
static void ink_route_finish(void *s, void *b, ink_route_stats *stats) {
    assert(s == &stream && backend_alive && b);
    stats->visits = 3;
    calls += 'F';
}
static void ink_route_close_observer(void *) {
    assert(!calls.empty() && calls.back() == 'F');
    calls += 'C';
}
struct Buffer {
    bool allocated = true;
    explicit operator bool() const { return allocated; }
    void *get() const { return const_cast<Buffer *>(this); }
    void reset() {
        assert(allocated && !calls.empty() && (calls.back() == 'C' || calls.back() == 'F'));
        allocated = false;
        calls += 'R';
    }
};

struct Fixture {
    struct Group {
        int id = 0;
        Buffer route_buf;
        void *route_observer = nullptr;
        struct { int n_filled = 0; } lru;
    };
    std::vector<Group> groups{3};
    struct { size_t buf_size = 4096 + 2 * 148024; } devices[1];
    void *route_stream = &stream;
    bool route_observed = true;
    unsigned long long route_phase = 1;
#include "retire.inc"
};
int main() {
    Fixture f;
    f.groups[0].route_observer = &stream;
    f.groups[1].route_buf.allocated = false;
    f.stop_device_decode();
    assert(calls == "FCRFR" && f.devices[0].buf_size == 4096);
    assert(!f.route_observed && !f.route_stream && !f.groups[0].route_observer);
    // Retiring in the context destructor must make the later member destructor
    // safe even after ggml has released its CUDA backend and streams.
    backend_alive = false;
    f.stop_device_decode();
    assert(calls == "FCRFR");
}
