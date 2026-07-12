// tcgen05_helpers.cpp — shared fragment-arithmetic implementations.
//
// Extracted from processTcgen05Mma in src/ptxsim/instructions/tcgen05.cpp
// (lines 333-371 of pre-Phase 2.5 code) per Oracle 2026-07-08 Q4-recommendation.

#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/utils/half_utils.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ptxsim {
namespace {

// Pre-load existing C slot for accumulation (per Oracle 2026-07-11 Q4).
// T is the storage type (uint16_t for f16 in Phase 1, float for f32 in H2).
template <typename T>
void load_c_slot(Tmem& tmem, size_t c_slot, T* c_frag, size_t count) {
    alignas(T) std::array<uint8_t, Tmem::kSlotSize> buf{};
    tmem.read(c_slot, buf.data(), Tmem::kSlotSize);
    std::memcpy(c_frag, buf.data(), count * sizeof(T));
}

} // anonymous namespace

void tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id, bool accumulate) {
    if (warp_id < 0) {
        throw std::invalid_argument(
            "tcgen05_fragment_mma_f16: warp_id must be >= 0 (got " +
            std::to_string(warp_id) + ")");
    }

    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t a_slot = static_cast<size_t>(lane_id) * 2;
        size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
        size_t c_slot = static_cast<size_t>(warp_id) * 32
                      + static_cast<size_t>(64)
                      + static_cast<size_t>(lane_id);

        std::array<uint8_t, Tmem::kSlotSize> a_buf{};
        tmem.read(a_slot, a_buf.data(), Tmem::kSlotSize);
        const uint16_t* a_raw =
            reinterpret_cast<const uint16_t*>(a_buf.data());

        std::array<uint8_t, Tmem::kSlotSize> b_buf{};
        tmem.read(b_slot, b_buf.data(), Tmem::kSlotSize);
        const uint16_t* b_raw =
            reinterpret_cast<const uint16_t*>(b_buf.data());

        float a_flat[ROWS * COLS_A];
        float b_flat[ROWS * COLS_B];
        for (int k = 0; k < ROWS * COLS_A; ++k)
            a_flat[k] = f16_to_f32(a_raw[k]);
        for (int k = 0; k < ROWS * COLS_B; ++k)
            b_flat[k] = f16_to_f32(b_raw[k]);

        std::array<float, ROWS * COLS_B> c_frag{};  // f32 storage (H2, PTX ISA §9.7.16)

        if (accumulate) {
            load_c_slot<float>(tmem, c_slot, c_frag.data(),
                               ROWS * COLS_B);
        }

        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                float sum = 0.0f;
                if (accumulate) {
                    sum += c_frag[i * COLS_B + j];  // direct f32 add (no f16→f32 round-trip)
                }
                for (int k = 0; k < COLS_A; ++k) {
                    sum += a_flat[i * COLS_A + k] *
                           b_flat[k * COLS_B + j];
                }
                c_frag[i * COLS_B + j] = sum;  // direct f32 store (no f32→f16 truncation)
            }
        }

        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        std::memcpy(c_buf.data(), c_frag.data(),
                    c_frag.size() * sizeof(float));
        tmem.write(c_slot, c_buf.data(), Tmem::kSlotSize);
    }
}

} // namespace ptxsim