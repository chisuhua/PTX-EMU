#include "ptx_ir/instruction_latency_table.h"
#include "ptx_ir/instruction_latency.h"
#include "ptx_ir/ptx_types.h"
#include <catch_amalgamated.hpp>

using namespace ptxsim;

TEST_CASE("ld.global latency reports 100 cycles and long delay flag",
          "[ptx][latency][memory]") {
    auto info = getLatency(S_LD);
    REQUIRE(info.cycles == 100);
    REQUIRE(info.is_long_delay == true);
}

TEST_CASE("st.global latency is 1 cycle and not long delay",
          "[ptx][latency][memory]") {
    auto info = getLatency(S_ST);
    REQUIRE(info.cycles == 1);
    REQUIRE(info.is_long_delay == false);
}
