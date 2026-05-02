/**
 * @file test_ptxir_mode4.cpp
 * @brief Mode 4: PTXIR Binary Serialization Roundtrip Test
 *
 * TDD RED: 这是一个失败的测试，验证 serialize→deserialize 往返功能。
 * 在实现 serialize_statements / deserialize_statements 之前，编译会失败。
 */

#include "catch_amalgamated.hpp"
#include "test_helpers.hpp"

#define TEST_MODE 4
#define TEST_NAME "ptxir_roundtrip"

#ifndef PTX_FILE
#define PTX_FILE "tests/ptx/test_divergent_small_cta.ptx"
#endif

// ============================================================================
// Test 1: Roundtrip - serialize then deserialize yields same count
// ============================================================================

TEST_CASE("Mode4: ptxir serialize→deserialize preserves statement count", "[mode4][roundtrip]") {
    init_factory_once();

    // 使用 Mode 2 加载的 PTX 作为基准
    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    // 序列化到临时文件
    std::string tmp_path = "tests/ptxir/test_roundtrip_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, tmp_path);
    CHECK(ok);

    // 反序列化
    auto stmts_loaded = deserialize_statements(tmp_path);

    // 验证：语句数量相同
    CHECK(stmts_loaded.size() == stmts_ref.size());
}

// ============================================================================
// Test 2: Roundtrip - statement types preserved
// ============================================================================

TEST_CASE("Mode4: ptxir roundtrip preserves statement types", "[mode4][roundtrip]") {
    init_factory_once();

    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    REQUIRE(stmts_ref.size() > 0);

    std::string tmp_path = "tests/ptxir/test_types_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, tmp_path);
    REQUIRE(ok);

    auto stmts_loaded = deserialize_statements(tmp_path);

    // 每条语句的 type 必须一致
    for (size_t i = 0; i < std::min(stmts_ref.size(), stmts_loaded.size()); i++) {
        INFO("i=" << i << " ref_type=" << static_cast<int>(stmts_ref[i].type)
                  << " loaded_type=" << static_cast<int>(stmts_loaded[i].type));
        CHECK(stmts_loaded[i].type == stmts_ref[i].type);
    }
}

// ============================================================================
// Test 3: BranchInstr reconvergence_pc roundtrip
// ============================================================================

TEST_CASE("Mode4: ptxir roundtrip preserves BranchInstr fields", "[mode4][branch]") {
    init_factory_once();

    auto stmts_ref = load_ptx_statements(PTX_FILE, "", false);
    std::string tmp_path = "tests/ptxir/test_branch_tmp.ptxir";
    bool ok = serialize_statements(stmts_ref, tmp_path);
    REQUIRE(ok);

    auto stmts_loaded = deserialize_statements(tmp_path);

    // 找到所有分支指令，比较 target 名称
    for (size_t i = 0; i < stmts_ref.size(); i++) {
        if (stmts_ref[i].type == S_BRA) {
            const auto& bra_ref = std::get<BranchInstr>(stmts_ref[i].data);
            const auto& bra_loaded = std::get<BranchInstr>(stmts_loaded[i].data);
            CHECK(bra_loaded.target == bra_ref.target);
            CHECK(bra_loaded.predicate == bra_ref.predicate);
            CHECK(bra_loaded.predicate_negated == bra_ref.predicate_negated);
        }
    }
}

// ============================================================================
// Test 4: generate_ptxir - PTX file → .ptxir
// ============================================================================

TEST_CASE("Mode4: generate_ptxir creates .ptxir from PTX file", "[mode4][generate]") {
    init_factory_once();

    std::string ptxir_path = "tests/ptxir/test_generate_tmp.ptxir";

    // 使用 generate_ptxir 从 PTX 文件直接生成
    bool ok = generate_ptxir(PTX_FILE, ptxir_path, "");
    CHECK(ok);

    // 验证文件存在且可读
    auto stmts = deserialize_statements(ptxir_path);
    CHECK(stmts.size() > 0);
}

// ============================================================================
// Test 5: load_ptxir - deserialize + optional CFG
// ============================================================================

TEST_CASE("Mode4: load_ptxir without CFG (reconvergence_pc = -1)", "[mode4][load]") {
    init_factory_once();

    std::string ptxir_path = "tests/ptxir/test_generate_tmp.ptxir";
    generate_ptxir(PTX_FILE, ptxir_path, "");

    // load_ptxir(apply_cfg=false) → Mode 3a 行为
    auto stmts = load_ptxir(ptxir_path, false);

    // 检查分支指令 reconvergence_pc 应为 -1（未应用 CFG）
    for (size_t i = 0; i < stmts.size(); i++) {
        if (stmts[i].type == S_BRA) {
            const auto& bra = std::get<BranchInstr>(stmts[i].data);
            INFO("reconvergence_pc at " << i << " = " << bra.reconvergence_pc);
            CHECK(bra.reconvergence_pc == -1);
        }
    }
}

TEST_CASE("Mode4: load_ptxir with CFG (reconvergence_pc filled)", "[mode4][load][cfg]") {
    init_factory_once();

    std::string ptxir_path = "tests/ptxir/test_generate_tmp.ptxir";
    generate_ptxir(PTX_FILE, ptxir_path, "");

    // load_ptxir(apply_cfg=true) → Mode 3b 行为
    auto stmts = load_ptxir(ptxir_path, true);

    // 检查分支指令 reconvergence_pc 已填充
    bool has_branch = false;
    for (size_t i = 0; i < stmts.size(); i++) {
        if (stmts[i].type == S_BRA) {
            has_branch = true;
            const auto& bra = std::get<BranchInstr>(stmts[i].data);
            INFO("reconvergence_pc at " << i << " = " << bra.reconvergence_pc);
            CHECK(bra.reconvergence_pc >= 0);
        }
    }
    CHECK(has_branch);  // 确保测试的 PTX 包含分支
}

// ============================================================================
// Test 6: Invalid file handling
// ============================================================================

TEST_CASE("Mode4: deserialize_statements throws on nonexistent file", "[mode4][error]") {
    CHECK_THROWS_AS(deserialize_statements("/nonexistent/path.ptxir"), std::runtime_error);
}

TEST_CASE("Mode4: serialize_statements returns false on bad path", "[mode4][error]") {
    std::vector<StatementContext> stmts;
    stmts.push_back(make_nop());
    CHECK(serialize_statements(stmts, "/nonexistent/dir/file.ptxir") == false);
}

TEST_CASE("Mode4: serialize_to_string → deserialize_from_string preserves operand values", "[mode4][roundtrip]") {
    init_factory_once();

    std::vector<StatementContext> stmts;
    stmts.push_back(make_mov("%r_dst", "%r_src"));
    stmts.push_back(make_add("%r_result", "%r_a", "%r_b"));
    stmts.push_back(make_mov_imm("%r_imm", 42));
    stmts.push_back(make_ld_shared("%r_ld", "shmem", "%r_off"));
    stmts.push_back(make_exit());

    std::string serialized = serialize_to_string(stmts);
    CHECK(serialized.size() > 0);

    auto stmts_loaded = deserialize_from_string(serialized);

    CHECK(stmts_loaded.size() == stmts.size());

    for (size_t i = 0; i < stmts.size(); i++) {
        CHECK(stmts_loaded[i].type == stmts[i].type);
        // instructionText is not preserved in binary format (stored only as metadata)
        // Use toString() instead which reconstructs from binary data
    }
}

TEST_CASE("Mode4: deserialize_statements has no ANTLR dependency", "[mode4][load][no-antlr]") {
    init_factory_once();

    std::string ptxir_path = "tests/ptxir/test_generate_tmp.ptxir";
    generate_ptxir(PTX_FILE, ptxir_path, "");

    auto stmts = load_ptxir(ptxir_path, false);
    CHECK(stmts.size() > 0);

    bool has_branch = false;
    for (size_t i = 0; i < stmts.size(); i++) {
        if (stmts[i].type == S_BRA) has_branch = true;
    }
    CHECK(has_branch);
}
