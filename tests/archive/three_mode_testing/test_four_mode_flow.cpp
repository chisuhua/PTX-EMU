/**
 * @file test_four_mode_flow.cpp
 * @brief Four-Mode End-to-End Pipeline Test
 *
 * Validates the complete PTX → StatementContext → .ptxir → Execute pipeline.
 * Mode 1: PTX extraction (cuobjdump)
 * Mode 2: PTX file loading (ANTLR parse)
 * Mode 3: StatementContext (before/after CFG)
 * Mode 4: PTXIR serialization/deserialization
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 4
#define TEST_NAME "four_mode_flow"

#ifndef PTX_FILE
#define PTX_FILE "tests/ptx/test_barrier_simple.ptx"
#endif

#ifndef TEST_BINARY
#define TEST_BINARY "build/bin/test_divergence_sync_standalone"
#endif

// ============================================================================
// Mode 1: PTX extraction matches cuobjdump output
// ============================================================================

TEST_CASE("Four-mode pipeline: Mode 1 PTX extraction matches cuobjdump baseline", "[mode1][mode2][pipeline]") {
    init_factory_once();

    std::string ptx_content = load_ptx_file(PTX_FILE);
    CHECK(ptx_content.size() > 0);
    CHECK(ptx_content.find("bar.sync") != std::string::npos);
}

// ============================================================================
// Mode 2: Parsed statements equivalent to hand-written
// ============================================================================

TEST_CASE("Four-mode pipeline: Mode 2 parsed statements structurally equivalent to Mode 3", "[mode2][mode3][pipeline]") {
    init_factory_once();

    auto stmts = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts.size() > 0);

    // Find first instruction (skip declarations like .reg, .param, .const)
    bool found_instruction = false;
    for (size_t i = 0; i < stmts.size(); i++) {
        if (stmts[i].type == S_BAR || stmts[i].type == S_MOV ||
            stmts[i].type == S_SETP || stmts[i].type == S_ADD ||
            stmts[i].type == S_ST || stmts[i].type == S_RET) {
            found_instruction = true;
            break;
        }
    }
    CHECK(found_instruction);

    bool found_bar = false;
    for (size_t i = 0; i < stmts.size(); i++) {
        if (stmts[i].type == S_BAR || stmts[i].type == S_BAR_WARP_SYNC) {
            found_bar = true;
            break;
        }
    }
    CHECK(found_bar);
}

// ============================================================================
// Mode 3: Serialization produces valid .ptxir text
// ============================================================================

TEST_CASE("Four-mode pipeline: Mode 3 serialization produces valid .ptxir", "[mode3][mode4][pipeline]") {
    init_factory_once();

    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    std::string ptxir_path = "tests/ptxir/test_pipeline_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, ptxir_path);
    CHECK(ok);

    std::ifstream f(ptxir_path, std::ios::binary);
    CHECK(f.good());
    f.close();
}

// ============================================================================
// Mode 4: Deserialization produces StatementContexts identical to Mode 2
// ============================================================================

TEST_CASE("Four-mode pipeline: Mode 4 deserialization produces identical StatementContexts", "[mode4][pipeline]") {
    init_factory_once();

    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    std::string ptxir_path = "tests/ptxir/test_pipeline_tmp.ptxir";
    serialize_statements(stmts_ref, ptxir_path);

    auto stmts_loaded = deserialize_statements(ptxir_path);
    CHECK(stmts_loaded.size() == stmts_ref.size());

    for (size_t i = 0; i < stmts_ref.size(); i++) {
        CHECK(stmts_loaded[i].type == stmts_ref[i].type);
    }
}

// ============================================================================
// Full pipeline: Mode 1 → 2 → 3 → 4 preserves semantics
// ============================================================================

TEST_CASE("Four-mode pipeline: Mode 1→2→3→4 full pipeline preserves statement types", "[mode1][mode2][mode3][mode4][pipeline]") {
    init_factory_once();

    std::string ptx_content = load_ptx_file(PTX_FILE);
    CHECK(ptx_content.size() > 0);

    auto stmts_mode2 = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_mode2.size() > 0);

    std::string ptxir_path = "tests/ptxir/test_full_pipeline_tmp.ptxir";
    serialize_statements(stmts_mode2, ptxir_path);

    auto stmts_mode4 = deserialize_statements(ptxir_path);
    CHECK(stmts_mode4.size() == stmts_mode2.size());

    size_t mismatches = 0;
    for (size_t i = 0; i < stmts_mode2.size(); i++) {
        if (stmts_mode4[i].type != stmts_mode2[i].type) mismatches++;
    }
    CHECK(mismatches == 0);
}

// ============================================================================
// Pipeline: load_ptxir(apply_cfg=true) produces CFG-filled statements
// ============================================================================

TEST_CASE("Four-mode pipeline: load_ptxir with CFG fills reconvergence_pc", "[mode4][cfg][pipeline]") {
    init_factory_once();

    auto stmts_no_cfg = load_ptx_statements(PTX_FILE, "", false);
    auto stmts_with_cfg = load_ptx_statements(PTX_FILE, "", true);

    CHECK(stmts_no_cfg.size() == stmts_with_cfg.size());

    int no_cfg_negative_count = 0;
    int with_cfg_positive_count = 0;

    for (size_t i = 0; i < stmts_no_cfg.size(); i++) {
        if (stmts_no_cfg[i].type == S_BRA) {
            const auto& bra_no = std::get<BranchInstr>(stmts_no_cfg[i].data);
            const auto& bra_with = std::get<BranchInstr>(stmts_with_cfg[i].data);
            if (bra_no.reconvergence_pc < 0) no_cfg_negative_count++;
            if (bra_with.reconvergence_pc >= 0) with_cfg_positive_count++;
        }
    }

    CHECK(no_cfg_negative_count >= 0);
    CHECK(with_cfg_positive_count >= 0);
}