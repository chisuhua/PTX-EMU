#include "ptx_ir/instruction_latency_table.h"
#include "ptx_ir/instruction_latency.h"
#include "ptx_ir/ptx_types.h"
#include <catch_amalgamated.hpp>

using namespace ptxsim;

TEST_CASE("instruction latency table: load instructions use long delay",
          "[ptx][latency]") {
    REQUIRE(getLatency(S_LD).cycles == LD_GLOBAL_LATENCY.cycles);
    REQUIRE(getLatency(S_LD).is_long_delay == true);
}

TEST_CASE("instruction latency table: stores are non-blocking",
          "[ptx][latency]") {
    REQUIRE(getLatency(S_ST).cycles == ST_GLOBAL_LATENCY.cycles);
    REQUIRE(getLatency(S_ST).is_long_delay == false);
}

TEST_CASE("instruction latency table: multi-cycle ALU instructions",
          "[ptx][latency]") {
    REQUIRE(getLatency(S_MUL).cycles == MUL_LATENCY.cycles);
    REQUIRE(getLatency(S_MUL24).cycles == MUL_LATENCY.cycles);
    REQUIRE(getLatency(S_MAD).cycles == MUL_LATENCY.cycles);
    REQUIRE(getLatency(S_MAD24).cycles == MUL_LATENCY.cycles);
    REQUIRE(getLatency(S_FMA).cycles == MUL_LATENCY.cycles);
}

TEST_CASE("instruction latency table: division is long delay",
          "[ptx][latency]") {
    REQUIRE(getLatency(S_DIV).cycles == DIV_LATENCY.cycles);
    REQUIRE(getLatency(S_DIV).is_long_delay == true);
    REQUIRE(getLatency(S_REM).cycles == DIV_LATENCY.cycles);
}

TEST_CASE("instruction latency table: barrier is single cycle",
          "[ptx][latency]") {
    REQUIRE(getLatency(S_BAR).cycles == BAR_SYNC_LATENCY.cycles);
    REQUIRE(getLatency(S_BAR_WARP_SYNC).cycles == BAR_SYNC_LATENCY.cycles);
    REQUIRE(getLatency(S_MEMBAR).cycles == BAR_SYNC_LATENCY.cycles);
    REQUIRE(getLatency(S_FENCE).cycles == BAR_SYNC_LATENCY.cycles);
}

TEST_CASE("instruction latency table: unknown ops use default",
          "[ptx][latency]") {
    // Use a control flow op that has no specific override
    REQUIRE(getLatency(S_BRA).cycles == DEFAULT_LATENCY.cycles);
    REQUIRE(getLatency(S_ADD).cycles == DEFAULT_LATENCY.cycles);
    REQUIRE(getLatency(S_RET).cycles == DEFAULT_LATENCY.cycles);
}
