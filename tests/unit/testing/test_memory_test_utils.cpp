// test_memory_test_utils.cpp
// =============================================================================
// Unit test for ptxsim::testing helpers in memory_test_utils.h.
//
// Verifies each helper produces a StatementContext with the expected:
//   - ctx.type
//   - qualifier
//   - instruction text
//
// This is a pure data-shape test -- it does NOT execute the statements
// (that's what the integration tests in tests/integration/ are for).
// =============================================================================

#include "catch_amalgamated.hpp"

#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/utils/qualifier_utils.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"

#include <string>
#include <type_traits>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_ld_local_addr;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_local_decl;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_local_addr;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::read_reg_u32;
using ptxsim::testing::setup_block;

TEST_CASE("make_shared_decl sets SHARED kind and b32 type",
          "[unit][testing][memory_test_utils]") {
    auto ctx = make_shared_decl("buf", 32);

    REQUIRE(ctx.type == S_SHARED);
    auto *d = std::get_if<DeclarationInstr>(&ctx.data);
    REQUIRE(d != nullptr);
    REQUIRE(d->kind == DeclarationInstr::Kind::SHARED);
    REQUIRE(d->name == "buf");
    REQUIRE(d->array_size == 32);
    REQUIRE(d->dataType == Qualifier::Q_B32);
    REQUIRE(ctx.instructionText == ".shared .b32 buf[32];");
}

TEST_CASE("make_local_decl sets LOCAL kind and b32 type",
          "[unit][testing][memory_test_utils]") {
    auto ctx = make_local_decl("arr", 16);

    REQUIRE(ctx.type == S_LOCAL);
    auto *d = std::get_if<DeclarationInstr>(&ctx.data);
    REQUIRE(d != nullptr);
    REQUIRE(d->kind == DeclarationInstr::Kind::LOCAL);
    REQUIRE(d->name == "arr");
    REQUIRE(d->array_size == 16);
    REQUIRE(d->dataType == Qualifier::Q_B32);
    REQUIRE(ctx.instructionText == ".local .b32 arr[16];");
}

TEST_CASE("make_st_shared_addr uses Q_SHARED b8 AddrOperand",
          "[unit][testing][memory_test_utils]") {
    auto ctx = make_st_shared_addr("buf", "r1", "r2");

    REQUIRE(ctx.type == S_ST);
    auto *instr = std::get_if<GenericInstr>(&ctx.data);
    REQUIRE(instr != nullptr);
    REQUIRE(instr->qualifiers.size() == 2);
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_SHARED));
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_B8));
    REQUIRE(instr->operands.size() == 2);
    auto *addr = std::get_if<AddrOperand>(&instr->operands[0].data);
    REQUIRE(addr != nullptr);
    REQUIRE(addr->space == AddrOperand::Space::SHARED);
    REQUIRE(addr->baseSymbol == "buf");
    REQUIRE(addr->offsetType == AddrOperand::OffsetType::REGISTER);
    REQUIRE(ctx.instructionText == "st.shared.b8 [buf+r1], r2;");
}

TEST_CASE("make_st_local_addr uses Q_LOCAL b32 AddrOperand",
          "[unit][testing][memory_test_utils]") {
    auto ctx = make_st_local_addr("arr", "r0", "r0");

    REQUIRE(ctx.type == S_ST);
    auto *instr = std::get_if<GenericInstr>(&ctx.data);
    REQUIRE(instr != nullptr);
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_LOCAL));
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_B32));
    REQUIRE(instr->operands.size() == 2);
    auto *addr = std::get_if<AddrOperand>(&instr->operands[0].data);
    REQUIRE(addr != nullptr);
    REQUIRE(addr->space == AddrOperand::Space::LOCAL);
    REQUIRE(addr->offsetType == AddrOperand::OffsetType::REGISTER);
    REQUIRE(ctx.instructionText == "st.local.b32 [arr+r0], r0;");
}

TEST_CASE("make_ld_shared_addr uses Q_SHARED b8 AddrOperand",
          "[unit][testing][memory_test_utils]") {
    auto ctx = make_ld_shared_addr("r2", "buf", "r1");

    REQUIRE(ctx.type == S_LD);
    auto *instr = std::get_if<GenericInstr>(&ctx.data);
    REQUIRE(instr != nullptr);
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_SHARED));
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_B8));
    REQUIRE(instr->operands.size() == 2);
    auto *dst = std::get_if<RegOperand>(&instr->operands[0].data);
    REQUIRE(dst != nullptr);
    REQUIRE(dst->name == "r2");
    auto *addr = std::get_if<AddrOperand>(&instr->operands[1].data);
    REQUIRE(addr != nullptr);
    REQUIRE(addr->space == AddrOperand::Space::SHARED);
    REQUIRE(ctx.instructionText == "ld.shared.b8 r2, [buf+r1];");
}

TEST_CASE("make_ld_local_addr uses Q_LOCAL b32 AddrOperand",
          "[unit][testing][memory_test_utils]") {
    auto ctx = make_ld_local_addr("r1", "arr", "r0");

    REQUIRE(ctx.type == S_LD);
    auto *instr = std::get_if<GenericInstr>(&ctx.data);
    REQUIRE(instr != nullptr);
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_LOCAL));
    REQUIRE(QvecHasQ(instr->qualifiers, Qualifier::Q_B32));
    REQUIRE(instr->operands.size() == 2);
    auto *addr = std::get_if<AddrOperand>(&instr->operands[1].data);
    REQUIRE(addr != nullptr);
    REQUIRE(addr->space == AddrOperand::Space::LOCAL);
    REQUIRE(addr->offsetType == AddrOperand::OffsetType::REGISTER);
    REQUIRE(ctx.instructionText == "ld.local.b32 r1, [arr+r0];");
}

TEST_CASE("init_instruction_factory_once is callable multiple times",
          "[unit][testing][memory_test_utils]") {
    init_instruction_factory_once();
    init_instruction_factory_once();
    SUCCEED("callable multiple times without error");
}

TEST_CASE("read_reg_u32 returns uint32_t (compile-time signature check)",
          "[unit][testing][memory_test_utils]") {
    static_assert(std::is_same_v<decltype(read_reg_u32(nullptr, "r", 0)),
                                  uint32_t>,
                  "read_reg_u32 must return uint32_t");
    SUCCEED("signature verified at compile time");
}

TEST_CASE("setup_block creates warp on minimal CTA",
          "[unit][testing][memory_test_utils]") {
    init_instruction_factory_once();
    ResourceManager::instance().initialize(1, 8192);

    std::vector<StatementContext> stmts;
    stmts.push_back(make_shared_decl("buf", 32));

    SMContext sm(4, 128, 4096, 0);
    WarpContext *w = setup_block(sm, stmts);
    REQUIRE(w != nullptr);
}
