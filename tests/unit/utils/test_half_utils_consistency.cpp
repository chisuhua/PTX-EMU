// test_half_utils_consistency.cpp
// =============================================================================
// Bit-perfect consistency check between
//   include/ptxsim/utils/half_utils.h::f16_to_f32
// and
//   src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float
//
// Iterates over all 65536 possible half-precision inputs and asserts that
// the two implementations produce bit-identical float32 output.  NaNs are
// compared bit-by-bit too (so a quiet vs. signaling NaN mismatch is
// detected, not papered over by std::isnan).
//
// This is the gate that proves Task 1 of the
// openspec/changes/phase3-half-precision-bugfix change succeeded: after the
// fix, T2-6 Step 2 (reuse half_utils.h for cvt_helpers) is safe to attempt.
// =============================================================================

#include "catch_amalgamated.hpp"
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include "ptxsim/utils/half_utils.h"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

union FloatBits {
    float f;
    uint32_t u;
};

} // namespace

TEST_CASE("half_utils vs cvt_helpers f16_to_f32 bit-perfect equivalence",
          "[half][utils][cvt][consistency]") {
    int mismatches = 0;
    uint32_t first_mismatch_h = 0;
    FloatBits first_a{}, first_b{};
    for (uint32_t h = 0; h <= 0xFFFFu; ++h) {
        const uint16_t hh = static_cast<uint16_t>(h);
        const float a = f16_to_f32(hh);
        const float b = ptxsim::cvt_helpers::half_to_float(hh);
        const FloatBits ba{a};
        const FloatBits bb{b};
        if (ba.u != bb.u) {
            if (mismatches == 0) {
                first_mismatch_h = h;
                first_a = ba;
                first_b = bb;
            }
            ++mismatches;
        }
    }
    INFO("Mismatches: " << mismatches);
    INFO("First mismatch h=0x" << std::hex << first_mismatch_h
                               << " half_utils=0x" << first_a.u
                               << " cvt_helpers=0x" << first_b.u);
    REQUIRE(mismatches == 0);
}

TEST_CASE("half_utils vs cvt_helpers specific denormal samples match",
          "[half][utils][cvt][consistency]") {
    // Targeted samples: smallest denormal, mid-range denormals, NaN payloads,
    // infinities.  This is a quick spot-check that complements the exhaustive
    // sweep above and gives useful failure messages if a specific bucket
    // regresses.
    const uint16_t samples[] = {
        0x0001, // smallest positive denormal
        0x0002,
        0x0100, // mid-range denormal
        0x0200,
        0x03FF, // largest denormal
        0x8001, // smallest negative denormal
        0x83FF, // largest negative denormal
        0x7C00, // +infinity
        0xFC00, // -infinity
        0x7E00, // quiet NaN
        0xFE00, // signaling NaN
        0x7FFF, // NaN with all mantissa bits set
        0x3C00, // 1.0
        0x4000, // 2.0
        0x7BFF, // largest normal
    };
    for (uint16_t h : samples) {
        const FloatBits a{f16_to_f32(h)};
        const FloatBits b{ptxsim::cvt_helpers::half_to_float(h)};
        INFO("h=0x" << std::hex << h << " half_utils=0x" << a.u
                    << " cvt_helpers=0x" << b.u);
        REQUIRE(a.u == b.u);
    }
}