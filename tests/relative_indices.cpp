#include "inkling-relative-indices.h"

#include <cmath>
#include <cstdio>
#include <cstring>

struct cell {
    int32_t position;
    unsigned sequences;
};

// The CPU helper supplies gather indices. Compare the FP32 values that the
// two mask graphs convert to FP16, including negative zero and blocked keys.
static bool check(uint32_t keys, uint32_t query_count, int32_t origin, bool fragmented) {
    constexpr int32_t extent = 11;
    const float blocked = -std::numeric_limits<float>::infinity();
    std::vector<std::vector<cell>> streams(2, std::vector<cell>(keys));
    for (unsigned stream = 0; stream < streams.size(); ++stream) {
        for (uint32_t j = 0; j < keys; ++j) {
            const uint32_t offset = fragmented ? (j * 113 + stream) % keys : j;
            const bool empty = fragmented && j % 7 == 0;
            streams[stream][j] = {empty ? -1 : int32_t(origin + offset),
                                  fragmented ? 1u + (j + stream) % 7 : 7u};
        }
    }
    std::vector<int32_t> positions(query_count);
    std::vector<unsigned> sequences(query_count);
    for (uint32_t i = 0; i < query_count; ++i) {
        positions[i] = origin + (fragmented ? (i * 79) % keys : keys - query_count + i);
        sequences[i] = fragmented ? i % 3 : 0;
    }

    std::vector<int32_t> output(size_t(keys) * query_count);
    for (bool packed : {false, true}) {
        std::fill(output.begin(), output.end(), -1);
        for (unsigned seq = 0; seq < 3; ++seq) {
            std::vector<std::pair<uint32_t, int32_t>> queries;
            for (uint32_t i = 0; i < query_count; ++i) {
                if (sequences[i] == seq) {
                    queries.emplace_back(i, positions[i]);
                }
            }
            const auto & cells = streams[seq % 2];
            inkling_relative_indices(output.data(), keys, extent, packed, queries, [&](uint32_t j) {
                return packed && !(cells[j].sequences & (1u << seq)) ? -1 : cells[j].position;
            });
        }

        for (uint32_t i = 0; i < query_count; ++i) {
            const unsigned seq = sequences[i];
            for (uint32_t j = 0; j < keys; ++j) {
                const auto & key = streams[seq % 2][j];
                const int64_t distance = key.position == -1 ? -1 : int64_t(positions[i]) - key.position;
                const int32_t bucket = distance >= 0 && distance < extent ? int32_t(distance) : extent;
                const bool visible = distance >= 0 && (key.sequences & (1u << seq));
                const int32_t actual = output[size_t(i) * keys + j];
                if (actual != (packed && !visible ? extent + 1 : bucket)) {
                    return false;
                }
                if (packed) {
                    auto bias = [i](int32_t index) { return index % 4 == 0 ? -0.0f : float(index) - float(i); };
                    const float reference = bias(bucket) + (visible ? 0.0f : blocked);
                    const float candidate = actual == extent + 1 ? blocked : bias(actual) + 0.0f;
                    if (std::memcmp(&reference, &candidate, sizeof(float)) != 0) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    const int32_t high = std::numeric_limits<int32_t>::max() - 131072;
    return check(257, 1, 0, true) && check(257, 32, 4096, true) &&
           check(131072, 4, high, false) && check(131072, 4, high, true) ? 0 : 1;
}
