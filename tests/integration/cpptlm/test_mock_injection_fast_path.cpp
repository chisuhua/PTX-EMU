// test_mock_injection_fast_path.cpp
// Phase 3.4 G4: exe_once fast-path injection verification

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

struct CallCounterScoreboard : IScoreboard {
    int check_calls = 0, alloc_calls = 0, release_calls = 0;
    bool has_free_entry() const override { const_cast<CallCounterScoreboard*>(this)->check_calls++; return true; }
    bool allocate(uint32_t, uint32_t) override { alloc_calls++; return true; }
    bool release(uint32_t, uint32_t) override { release_calls++; return true; }
    void tick() override {}
};

struct CallCounterPipeline : IPipelineLatencyProvider {
    int cycles_calls = 0;
    double get_fractional_cycles(const std::string&, PipelineId) const override {
        const_cast<CallCounterPipeline*>(this)->cycles_calls++;
        return 0.0;
    }
    double get_fractional_cycles_by_type(int, PipelineId) const override {
        const_cast<CallCounterPipeline*>(this)->cycles_calls++;
        return 0.0;
    }
};

struct CallCounterTensorCore : ITensorCoreTiming {
    int latency_calls = 0;
    uint32_t get_latency(TcPrecision) const override { const_cast<CallCounterTensorCore*>(this)->latency_calls++; return 0; }
    uint32_t get_throughput_cycles(TcPrecision) const override { return 0; }
};

} // namespace

TEST_CASE("G4 fast-path: nullptr injection is no-op", "[integration][cpptlm][g4][mock][fast_path]") {
    auto gpu = std::make_unique<GPUContext>("configs/ampere_a100.json");
    gpu->init();
    SMContext* sm = gpu->get_sm(0);
    REQUIRE(sm->get_scoreboard() == nullptr);
    REQUIRE(sm->get_pipeline_latency_provider() == nullptr);
    REQUIRE(sm->get_tensor_core_timing() == nullptr);
}

TEST_CASE("G4 fast-path: mock interfaces callable", "[integration][cpptlm][g4][mock][fast_path]") {
    CallCounterScoreboard sb;
    REQUIRE(sb.has_free_entry() == true);
    REQUIRE(sb.allocate(1, 0) == true);
    REQUIRE(sb.release(1, 0) == true);

    CallCounterPipeline pl;
    REQUIRE(pl.get_fractional_cycles_by_type(0, PipelineId::P0_INT_FP32) == 0.0);

    CallCounterTensorCore tc;
    REQUIRE(tc.get_latency(TcPrecision::FP16) == 0);
    REQUIRE(tc.get_throughput_cycles(TcPrecision::FP16) == 0);
}

TEST_CASE("G4 fast-path: pipeline mapping helpers", "[integration][cpptlm][g4][mock][fast_path]") {
    StatementContext ld_stmt;
    ld_stmt.type = StatementType::S_LD;
    REQUIRE(SMContext::map_instruction_to_pipeline(ld_stmt) == PipelineId::P3_LSU);

    StatementContext st_stmt;
    st_stmt.type = StatementType::S_ST;
    REQUIRE(SMContext::map_instruction_to_pipeline(st_stmt) == PipelineId::P3_LSU);

    StatementContext mov_stmt;
    mov_stmt.type = StatementType::S_MOV;
    REQUIRE(SMContext::map_instruction_to_pipeline(mov_stmt) == PipelineId::P0_INT_FP32);
}