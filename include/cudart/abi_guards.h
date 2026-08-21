#ifndef PTX_EMU_ABI_GUARDS_H
#define PTX_EMU_ABI_GUARDS_H

// PTX-EMU ABI 一致性静态断言
// 从 cpptlm_bridge.h 迁移（G-D4 12 端点 + 签名级 + cudaStream_t 宽度 = 17 断言）
// 保留 ABI 大门锁，删除 bridge 专用锁（PtxEmuDriverApi sizeof 断言）
// per cleanup-cudart-cpptlm-bridge-coupling Phase 3 (2026-08-21)

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "cudart/cudart_intrinsics.h"  // cudaStream_t
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"

static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t),
              "cudaStream_t wider than uint64_t — ABI guard must be enlarged");

namespace abi_guards_g_d4 {

// PipelineId (6 端点)
static_assert(static_cast<uint32_t>(PipelineId::P0_INT_FP32) == 0,
              "G-D4 ABI drift: PipelineId::P0_INT_FP32 != 0");
static_assert(static_cast<uint32_t>(PipelineId::V_SIMD) == 1,
              "G-D4 ABI drift: PipelineId::V_SIMD != 1");
static_assert(static_cast<uint32_t>(PipelineId::P1_FP64) == 2,
              "G-D4 ABI drift: PipelineId::P1_FP64 != 2");
static_assert(static_cast<uint32_t>(PipelineId::P2_SFU) == 3,
              "G-D4 ABI drift: PipelineId::P2_SFU != 3");
static_assert(static_cast<uint32_t>(PipelineId::P3_LSU) == 4,
              "G-D4 ABI drift: PipelineId::P3_LSU != 4");
static_assert(static_cast<uint32_t>(PipelineId::P4_TC) == 5,
              "G-D4 ABI drift: PipelineId::P4_TC != 5");

// TcPrecision (6 端点)
static_assert(static_cast<uint32_t>(TcPrecision::FP4) == 0,
              "G-D4 ABI drift: TcPrecision::FP4 != 0");
static_assert(static_cast<uint32_t>(TcPrecision::FP6) == 1,
              "G-D4 ABI drift: TcPrecision::FP6 != 1");
static_assert(static_cast<uint32_t>(TcPrecision::FP8) == 2,
              "G-D4 ABI drift: TcPrecision::FP8 != 2");
static_assert(static_cast<uint32_t>(TcPrecision::FP16) == 3,
              "G-D4 ABI drift: TcPrecision::FP16 != 3");
static_assert(static_cast<uint32_t>(TcPrecision::BF16) == 4,
              "G-D4 ABI drift: TcPrecision::BF16 != 4");
static_assert(static_cast<uint32_t>(TcPrecision::TF32) == 5,
              "G-D4 ABI drift: TcPrecision::TF32 != 5");

// 签名级 ABI 验证 — 防 silent enum 替换 / 函数签名漂移
static_assert(std::is_same_v<
              decltype(std::declval<IScoreboard&>().allocate(
                  uint32_t{}, uint32_t{})),
              bool>,
              "G-D4 ABI drift: IScoreboard::allocate signature");

static_assert(std::is_same_v<
              decltype(std::declval<IPipelineLatencyProvider&>()
                           .get_fractional_cycles_by_type(
                               int{}, PipelineId::P0_INT_FP32)),
              double>,
              "G-D4 ABI drift: IPipelineLatencyProvider::get_fractional_cycles_by_type");

static_assert(std::is_same_v<
              decltype(std::declval<ITensorCoreTiming&>().get_latency(
                  TcPrecision::FP16)),
              uint32_t>,
              "G-D4 ABI drift: ITensorCoreTiming::get_latency");

static_assert(std::is_same_v<
              decltype(std::declval<ITensorCoreTiming&>().get_latency_mnk(
                  TcPrecision::FP16, uint32_t{}, uint32_t{}, uint32_t{})),
              uint32_t>,
              "G-D4 ABI drift: ITensorCoreTiming::get_latency_mnk");

}  // namespace abi_guards_g_d4

#endif  // PTX_EMU_ABI_GUARDS_H