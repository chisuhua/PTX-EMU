// test_mock_injection_slow_path.cpp
// Phase 3.4 G4: exe_once slow-path injection verification

#include "catch_amalgamated.hpp"
#include <memory>

#include "ptx_ir/ptx_types.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/tensor_core_interface.h"

using namespace ptxsim;

namespace {

struct MockSlowScoreboard : IScoreboard {
    mutable int check_calls = 0;
    int alloc_calls = 0, release_calls = 0;
    bool has_free_entry() const override { check_calls++; return true; }
    bool allocate(uint32_t, uint32_t) override { alloc_calls++; return true; }
    bool release(uint32_t, uint32_t) override { release_calls++; return true; }
    void tick() override {}
};

struct MockSlowPipeline : IPipelineLatencyProvider {
    mutable int cycles_calls = 0;
    double get_fractional_cycles(const std::string&, PipelineId) const override {
        cycles_calls++;
        return 4.22;
    }
    double get_fractional_cycles_by_type(int, PipelineId) const override {
        cycles_calls++;
        return 4.22;
    }
};

struct MockSlowTensorCore : ITensorCoreTiming {
    mutable int latency_calls = 0;
    uint32_t get_latency(TcPrecision) const override { latency_calls++; return 8; }
    uint32_t get_throughput_cycles(TcPrecision) const override { return 1; }
};

} // namespace

TEST_CASE("G4 slow-path: setter/getter round-trip", "[integration][cpptlm][g4][mock][slow_path]") {
    auto scoreboard = std::make_unique<MockSlowScoreboard>();
    auto pipeline = std::make_unique<MockSlowPipeline>();
    auto tc = std::make_unique<MockSlowTensorCore>();

    auto gpu = std::make_unique<GPUContext>("configs/ampere_a100.json");
    gpu->init();
    SMContext* sm = gpu->get_sm(0);

    sm->set_scoreboard(scoreboard.get());
    sm->set_pipeline_latency_provider(pipeline.get());
    sm->set_tensor_core_timing(tc.get());

    REQUIRE(sm->get_scoreboard() == scoreboard.get());
    REQUIRE(sm->get_pipeline_latency_provider() == pipeline.get());
    REQUIRE(sm->get_tensor_core_timing() == tc.get());

    sm->set_scoreboard(nullptr);
    sm->set_pipeline_latency_provider(nullptr);
    sm->set_tensor_core_timing(nullptr);

    REQUIRE(sm->get_scoreboard() == nullptr);
    REQUIRE(sm->get_pipeline_latency_provider() == nullptr);
    REQUIRE(sm->get_tensor_core_timing() == nullptr);
}

TEST_CASE("G4 slow-path: mock returns non-zero latency", "[integration][cpptlm][g4][mock][slow_path]") {
    MockSlowPipeline pipeline;
    REQUIRE(pipeline.get_fractional_cycles_by_type(
        static_cast<int>(ptxemu::ir::StatementType::S_FMA),
        PipelineId::P0_INT_FP32) == 4.22);

    MockSlowTensorCore tc;
    REQUIRE(tc.get_latency(TcPrecision::FP16) == 8);
    REQUIRE(tc.get_throughput_cycles(TcPrecision::FP16) == 1);
}

TEST_CASE("G4 slow-path: pipeline mapping helpers", "[integration][cpptlm][g4][mock][slow_path]") {
    ptxemu::ir::StatementContext sin_stmt;
    sin_stmt.type = ptxemu::ir::StatementType::S_SIN;
    REQUIRE(SMContext::map_instruction_to_pipeline(sin_stmt) == PipelineId::P2_SFU);

    ptxemu::ir::StatementContext cos_stmt;
    cos_stmt.type = ptxemu::ir::StatementType::S_COS;
    REQUIRE(SMContext::map_instruction_to_pipeline(cos_stmt) == PipelineId::P2_SFU);

    ptxemu::ir::StatementContext atom_stmt;
    atom_stmt.type = ptxemu::ir::StatementType::S_ATOM;
    REQUIRE(SMContext::map_instruction_to_pipeline(atom_stmt) == PipelineId::P3_LSU);
}

TEST_CASE("G4 slow-path: is_tensor_core_instruction", "[integration][cpptlm][g4][mock][slow_path]") {
    ptxemu::ir::StatementContext tc_stmt;
    tc_stmt.type = ptxemu::ir::StatementType::S_TCGEN05_MMA;
    REQUIRE(SMContext::is_tensor_core_instruction(tc_stmt) == true);

    ptxemu::ir::StatementContext mov_stmt;
    mov_stmt.type = ptxemu::ir::StatementType::S_MOV;
    REQUIRE(SMContext::is_tensor_core_instruction(mov_stmt) == false);
}