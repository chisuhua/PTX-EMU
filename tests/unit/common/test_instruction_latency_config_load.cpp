// test_instruction_latency_config_load.cpp
// =============================================================================
// Validates that InstructionLatencyConfig values from the JSON config
// correctly override the constexpr defaults in InstructionLatencyTable.
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/instruction_latency.h"
#include "ptx_ir/instruction_latency_config.h"
#include "ptx_ir/instruction_latency_table.h"
#include "ptx_ir/ptx_types.h"

#include <cstdint>

using ptxsim::InstructionLatency;
using ptxsim::InstructionLatencyTable;
using ptxsim::LD_GLOBAL_LATENCY;
using ptxsim::ST_GLOBAL_LATENCY;
using ptxsim::MUL_LATENCY;
using ptxsim::DIV_LATENCY;
using ptxsim::BAR_SYNC_LATENCY;
using ptxsim::DEFAULT_LATENCY;
using ptxsim::getLatency;

namespace {

// RAII helper: save table state on construction, restore on destruction so
// tests don't leak overrides into other test cases.
struct TableScope {
    InstructionLatencyTable& t;
    TableScope() : t(InstructionLatencyTable::instance()) {
        t.reset_to_defaults();
    }
    ~TableScope() { t.reset_to_defaults(); }
};

} // namespace

TEST_CASE("InstructionLatencyConfig: defaults expose constexpr values",
          "[latency][config]") {
    TableScope s;
    auto info = s.t.get(S_LD);
    REQUIRE(info.cycles == LD_GLOBAL_LATENCY.cycles);
    REQUIRE(info.is_long_delay == LD_GLOBAL_LATENCY.is_long_delay);
}

TEST_CASE("InstructionLatencyTable::load overrides LD_GLOBAL only",
          "[latency][config]") {
    TableScope s;
    InstructionLatencyConfig cfg;
    cfg.ld_global_cycles = 42;
    cfg.ld_global_long_delay = false;
    s.t.load(cfg);

    REQUIRE(s.t.get(S_LD).cycles == 42);
    REQUIRE(s.t.get(S_LD).is_long_delay == false);

    // Other classes unaffected
    REQUIRE(s.t.get(S_ST).cycles == ST_GLOBAL_LATENCY.cycles);
    REQUIRE(s.t.get(S_MUL).cycles == MUL_LATENCY.cycles);
    REQUIRE(s.t.get(S_DIV).cycles == DIV_LATENCY.cycles);
}

TEST_CASE("InstructionLatencyTable::load overrides ST/MUL/DIV/BAR classes",
          "[latency][config]") {
    TableScope s;
    InstructionLatencyConfig cfg;
    cfg.st_global_cycles = 2;
    cfg.mul_cycles = 7;
    cfg.div_cycles = 99;
    cfg.bar_sync_cycles = 3;
    s.t.load(cfg);

    REQUIRE(s.t.get(S_ST).cycles == 2);
    REQUIRE(s.t.get(S_MUL).cycles == 7);
    REQUIRE(s.t.get(S_MUL24).cycles == 7);
    REQUIRE(s.t.get(S_MAD).cycles == 7);
    REQUIRE(s.t.get(S_MAD24).cycles == 7);
    REQUIRE(s.t.get(S_FMA).cycles == 7);

    REQUIRE(s.t.get(S_DIV).cycles == 99);
    REQUIRE(s.t.get(S_REM).cycles == 99);

    REQUIRE(s.t.get(S_BAR).cycles == 3);
    REQUIRE(s.t.get(S_BAR_WARP_SYNC).cycles == 3);
    REQUIRE(s.t.get(S_MEMBAR).cycles == 3);
    REQUIRE(s.t.get(S_FENCE).cycles == 3);
}

TEST_CASE("InstructionLatencyTable::load with cycles<=0 keeps current value",
          "[latency][config]") {
    TableScope s;

    InstructionLatencyConfig cfg1;
    cfg1.ld_global_cycles = 50;
    s.t.load(cfg1);
    REQUIRE(s.t.get(S_LD).cycles == 50);

    // Second load that doesn't set ld_global must keep 50
    InstructionLatencyConfig cfg2;
    cfg2.st_global_cycles = 9;
    s.t.load(cfg2);
    REQUIRE(s.t.get(S_LD).cycles == 50);
    REQUIRE(s.t.get(S_ST).cycles == 9);
}

TEST_CASE("InstructionLatencyTable::load default_cycles overrides DEFAULT only",
          "[latency][config]") {
    TableScope s;
    InstructionLatencyConfig cfg;
    cfg.default_cycles = 6;
    s.t.load(cfg);

    // Default-class entries (e.g. S_BRA, S_ADD, S_RET) take the new value
    REQUIRE(s.t.get(S_BRA).cycles == 6);
    REQUIRE(s.t.get(S_ADD).cycles == 6);
    REQUIRE(s.t.get(S_RET).cycles == 6);

    // Specifically-classified entries unaffected
    REQUIRE(s.t.get(S_LD).cycles == LD_GLOBAL_LATENCY.cycles);
    REQUIRE(s.t.get(S_MUL).cycles == MUL_LATENCY.cycles);
}

TEST_CASE("InstructionLatencyTable::reset_to_defaults restores constexpr values",
          "[latency][config]") {
    TableScope s;

    InstructionLatencyConfig cfg;
    cfg.ld_global_cycles = 999;
    cfg.div_cycles = 888;
    s.t.load(cfg);
    REQUIRE(s.t.get(S_LD).cycles == 999);
    REQUIRE(s.t.get(S_DIV).cycles == 888);

    s.t.reset_to_defaults();
    REQUIRE(s.t.get(S_LD).cycles == LD_GLOBAL_LATENCY.cycles);
    REQUIRE(s.t.get(S_DIV).cycles == DIV_LATENCY.cycles);
}

TEST_CASE("getLatency free function matches InstructionLatencyTable::get",
          "[latency][config]") {
    TableScope s;
    InstructionLatencyConfig cfg;
    cfg.ld_global_cycles = 17;
    s.t.load(cfg);

    REQUIRE(getLatency(S_LD).cycles == s.t.get(S_LD).cycles);
    REQUIRE(getLatency(S_LD).cycles == 17);
}