// tcgen05_helpers.cpp — shared fragment-arithmetic implementations.
//
// Extracted from processTcgen05Mma in src/ptxsim/instructions/tcgen05.cpp
// (lines 333-371 of pre-Phase 2.5 code) per Oracle 2026-07-08 Q4-recommendation.

#include "ptxsim/instructions/tcgen05_helpers.h"
#include "ptxsim/utils/half_utils.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace ptxsim {

void tcgen05_fragment_mma_f16(Tmem& tmem) {
    constexpr int ROWS = 8;
    constexpr int COLS_A = 8;
    constexpr int COLS_B = 4;

    for (int lane_id = 0; lane_id < 32; ++lane_id) {
        size_t a_slot = static_cast<size_t>(lane_id) * 2;
        size_t b_slot = static_cast<size_t>(lane_id) * 2 + 1;
        size_t c_slot = static_cast<size_t>(64) + static_cast<size_t>(lane_id);

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

        std::array<uint16_t, ROWS * COLS_B> c_frag{};
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS_B; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < COLS_A; ++k) {
                    sum += a_flat[i * COLS_A + k] *
                           b_flat[k * COLS_B + j];
                }
                c_frag[i * COLS_B + j] = f32_to_f16(sum);
            }
        }

        std::array<uint8_t, Tmem::kSlotSize> c_buf{};
        std::memcpy(c_buf.data(), c_frag.data(),
                    c_frag.size() * sizeof(uint16_t));
        tmem.write(c_slot, c_buf.data(), Tmem::kSlotSize);
    }
}

} // namespace ptxsim