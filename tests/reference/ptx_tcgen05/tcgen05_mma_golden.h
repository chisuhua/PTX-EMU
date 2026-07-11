#pragma once
// =============================================================================
// Hand-computed reference values for tcgen05.mma fragment arithmetic.
//
// Source: PTX ISA §9.7.16 specification (Blackwell sm_100+).
// Layout: 8 rows × 4 cols = 32 f32 elements (per-lane fragment output).
// Storage format: f32 (per PTX ISA §9.7.16, mma output dtype is f32).
// Previously stored as f16 with f16→f32 readback; storage changed in
// fix-tcgen05-mma-accumulator-and-f32-storage Phase 2 commit (Oracle H2).
//
//
// Inputs (per fragment multiplication test):
//   A[8][1] = {1.0f16, 2.0f16, 3.0f16, 4.0f16, 5.0f16, 6.0f16, 7.0f16, 8.0f16}
//   B[1][4] = {1.0f16, 2.0f16, 3.0f16, 4.0f16}
//
// Expected output C[i][j] = A[i][0] * B[0][j], f16→f32 conversion.
//
// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16 manual computation.
// Each element can be hand-verified by the reviewer:
//   C[0][0] = 1*1 = 1.0,  C[0][1] = 1*2 = 2.0,  C[0][2] = 1*3 = 3.0,  C[0][3] = 1*4 = 4.0
//   C[1][0] = 2*1 = 2.0,  C[1][1] = 2*2 = 4.0,  C[1][2] = 2*3 = 6.0,  C[1][3] = 2*4 = 8.0
//   ...
//   C[7][0] = 8*1 = 8.0,  C[7][1] = 8*2 = 16.0, C[7][2] = 8*3 = 24.0, C[7][3] = 8*4 = 32.0
//
// Note: processTcgen05Mma currently does full SUM_k accumulation (8×8 A * 8×4 B).
// For the test inputs above (A[i][k≠0] = 0, B[k≠0][j] = 0), this collapses to
// the simple product A[i][0] * B[0][j]. See src/ptxsim/instructions/tcgen05.cpp:332-364.
// =============================================================================

#include <array>
#include <cstdint>

namespace ptxsim::reference::tcgen05 {

// 8 rows × 4 cols = 32 f32 elements (per lane's fragment output).
// Stored row-major: index = i * 4 + j.
constexpr std::array<float, 32> GOLDEN_MMA_F16_F16_F32 = {
    1.0f,  2.0f,  3.0f,  4.0f,    // i=0: 1*[1..4]
    2.0f,  4.0f,  6.0f,  8.0f,    // i=1: 2*[1..4]
    3.0f,  6.0f,  9.0f, 12.0f,    // i=2: 3*[1..4]
    4.0f,  8.0f, 12.0f, 16.0f,    // i=3: 4*[1..4]
    5.0f, 10.0f, 15.0f, 20.0f,    // i=4: 5*[1..4]
    6.0f, 12.0f, 18.0f, 24.0f,    // i=5: 6*[1..4]
    7.0f, 14.0f, 21.0f, 28.0f,    // i=6: 7*[1..4]
    8.0f, 16.0f, 24.0f, 32.0f,    // i=7: 8*[1..4]
};

}  // namespace ptxsim::reference::tcgen05