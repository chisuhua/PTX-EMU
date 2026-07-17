/**
 * Phase 8.B injection interfaces ABI 真值源测试 (TDD Green).
 *
 * 覆盖 3 个纯虚接口:
 *   - IScoreboard (4 方法 ABI + mock 实现)
 *   - IPipelineLatencyProvider (2 方法 ABI + PipelineId enum 0-5)
 *   - ITensorCoreTiming (3 方法 ABI + TcPrecision enum 0-5 + 默认 get_latency_mnk)
 *
 * 21 assertions 验证 ABI 编译 + 方法签名 + enum 值；CppTLM 端 Adapter
 * 通过 static_assert 验证双方一致性 (RFC-P1-001~004).
 *
 * Phase 8.B PTX-1/2/3 — cpptlm-phase8b-injection-points
 * Ref: ADR-0020, design.md §3
 */
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"

#include "catch_amalgamated.hpp"

namespace {

// Mock implementations for ABI compilation check
struct MockScoreboard : IScoreboard {
    int tick_count = 0;
    bool free_entry = true;

    bool has_free_entry() const override { return free_entry; }
    bool allocate(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        return true;
    }
    bool release(uint32_t /*reg_id*/, uint32_t /*warp_id*/) override {
        return true;
    }
    void tick() override { tick_count++; }
};

struct MockPipelineProvider : IPipelineLatencyProvider {
    double get_fractional_cycles(const std::string& /*instr*/,
                                 PipelineId /*pipe_id*/) const override {
        return 0.0;
    }
    double get_fractional_cycles_by_type(int /*stmt_type*/,
                                         PipelineId /*pipe_id*/) const override {
        return 0.0;
    }
};

struct MockTensorCoreTiming : ITensorCoreTiming {
    uint32_t get_latency(TcPrecision /*prec*/) const override { return 10; }
    uint32_t get_throughput_cycles(TcPrecision /*prec*/) const override {
        return 5;
    }
    // get_latency_mnk inherits default impl from ITensorCoreTiming
};

}  // namespace

TEST_CASE("IScoreboard ABI: mock can override all 4 pure virtual methods",
          "[unit][cpptlm][inject][scoreboard]") {
    MockScoreboard sb;
    REQUIRE(sb.has_free_entry() == true);
    REQUIRE(sb.allocate(0, 0) == true);
    REQUIRE(sb.release(0, 0) == true);
    sb.tick();
    REQUIRE(sb.tick_count == 1);
}

TEST_CASE("IPipelineLatencyProvider ABI: mock can override and returns fractional",
          "[unit][cpptlm][inject][pipeline]") {
    MockPipelineProvider p;
    REQUIRE(p.get_fractional_cycles("add.f32", PipelineId::P0_INT_FP32) == 0.0);
    REQUIRE(p.get_fractional_cycles_by_type(0, PipelineId::P4_TC) == 0.0);
}

TEST_CASE("ITensorCoreTiming ABI: mock can override and default get_latency_mnk",
          "[unit][cpptlm][inject][tensor_core]") {
    MockTensorCoreTiming tc;
    REQUIRE(tc.get_latency(TcPrecision::FP16) == 10);
    REQUIRE(tc.get_throughput_cycles(TcPrecision::BF16) == 5);
    // get_latency_mnk should default-delegate to get_latency(prec)
    REQUIRE(tc.get_latency_mnk(TcPrecision::FP8, 16, 16, 16) == 10);
}

TEST_CASE("PipelineId enum values match CppTLM tlm::PipelineId (0-5)",
          "[unit][cpptlm][inject][enum]") {
    // Per ADR-0020 + CppTLM RFC-P1-003 §3.1
    // CppTLM Adapter uses static_assert to verify these match at compile time
    REQUIRE(static_cast<uint32_t>(PipelineId::P0_INT_FP32) == 0);
    REQUIRE(static_cast<uint32_t>(PipelineId::V_SIMD) == 1);
    REQUIRE(static_cast<uint32_t>(PipelineId::P1_FP64) == 2);
    REQUIRE(static_cast<uint32_t>(PipelineId::P2_SFU) == 3);
    REQUIRE(static_cast<uint32_t>(PipelineId::P3_LSU) == 4);
    REQUIRE(static_cast<uint32_t>(PipelineId::P4_TC) == 5);
}

TEST_CASE("TcPrecision enum values match CppTLM tlm::TcPrecision (0-5)",
          "[unit][cpptlm][inject][enum]") {
    // Per ADR-0020 + CppTLM RFC-P1-003 §3.2
    REQUIRE(static_cast<uint32_t>(TcPrecision::FP4) == 0);
    REQUIRE(static_cast<uint32_t>(TcPrecision::FP6) == 1);
    REQUIRE(static_cast<uint32_t>(TcPrecision::FP8) == 2);
    REQUIRE(static_cast<uint32_t>(TcPrecision::FP16) == 3);
    REQUIRE(static_cast<uint32_t>(TcPrecision::BF16) == 4);
    REQUIRE(static_cast<uint32_t>(TcPrecision::TF32) == 5);
}
