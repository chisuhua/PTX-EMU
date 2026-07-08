// test_tcgen05_dispatch.cpp
// =============================================================================
// Integration test (类型二): S_TCGEN05_* handler dispatch verification.
//
// Per design.md D1 + spec.md "S_TCGEN05_* handlers SHALL be registered
// in handler_map" + "Tcgen05Handler class SHALL provide processTcgen05Operation".
//
// Verifies that InstructionFactory::initialize() registers all 11
// S_TCGEN05_* enum values with non-null handlers (per X-Macro in
// instruction_factory.cpp:16-19 + Tcgen05Handler registered for all 11
// via single class dispatched by instr.op_kind).
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_factory.h"

#include <cstdio>

namespace {
void ensure_init() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}
}  // namespace

TEST_CASE("S_TCGEN05_MMA dispatch returns non-null handler",
          "[integration][tcgen05][dispatch]") {
    ensure_init();
    auto *h = InstructionFactory::get_handler(S_TCGEN05_MMA);
    INFO("S_TCGEN05_MMA handler = " << h);
    REQUIRE(h != nullptr);
}

TEST_CASE("All 11 S_TCGEN05_* dispatch correctly (no nullptr handler)",
          "[integration][tcgen05][dispatch][all]") {
    ensure_init();

    // 11 S_TCGEN05_* enum values registered via ptx_op.def X-Macro.
    // Each must resolve to a non-null handler (per spec scenario).
    const StatementType tcgen05_types[] = {
        S_TCGEN05_ALLOC,
        S_TCGEN05_DEALLOC,
        S_TCGEN05_RELINQUISH,
        S_TCGEN05_LD,
        S_TCGEN05_ST,
        S_TCGEN05_CP,
        S_TCGEN05_MMA,
        S_TCGEN05_MMA_WS,
        S_TCGEN05_COMMIT,
        S_TCGEN05_WAIT,
        S_TCGEN05_FENCE,
    };

    int null_count = 0;
    for (auto t : tcgen05_types) {
        auto *h = InstructionFactory::get_handler(t);
        INFO("StatementType=" << static_cast<int>(t) << " handler=" << h);
        if (h == nullptr) ++null_count;
    }
    REQUIRE(null_count == 0);
}

TEST_CASE("get_handler returns same instance for repeat calls (registration is consistent)",
          "[integration][tcgen05][dispatch][consistent]") {
    ensure_init();
    auto *h1 = InstructionFactory::get_handler(S_TCGEN05_MMA);
    auto *h2 = InstructionFactory::get_handler(S_TCGEN05_MMA);
    REQUIRE(h1 != nullptr);
    REQUIRE(h1 == h2);  // Same instance — registration is idempotent
}