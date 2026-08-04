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

#include "memory/resource_manager.h"
#include "ptxsim/testing/memory_test_utils.h"
#include "ptxsim/utils/qualifier_utils.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"

#include <string>
#include <type_traits>

using ptxsim::testing::init_instruction_factory_once;
using ptxsim::testing::make_ld_local_addr;
using ptxsim::testing::make_ld_shared_addr;
using ptxsim::testing::make_ld_shared_addr_v2;
using ptxsim::testing::make_ld_shared_addr_v4;
using ptxsim::testing::make_local_decl;
using ptxsim::testing::make_shared_decl;
using ptxsim::testing::make_st_local_addr;
using ptxsim::testing::make_st_shared_addr;
using ptxsim::testing::make_st_shared_addr_v2;
using ptxsim::testing::make_st_shared_addr_v4;
using ptxsim::testing::make_setp_eq;
using ptxsim::testing::make_setp_ne;
using ptxsim::testing::make_setp_gt;
using ptxsim::testing::make_setp_ge;
using ptxsim::testing::make_setp_le;
using ptxsim::testing::make_setp_eq_imm;
using ptxsim::testing::make_setp_ne_imm;
using ptxsim::testing::make_setp_lt_imm;
using ptxsim::testing::make_setp_gt_imm;
using ptxsim::testing::make_setp_le_imm;
using ptxsim::testing::make_setp_ge_imm;
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

TEST_CASE("All new shared memory helpers compile and produce valid StatementContext",
           "[unit][testing][memory_test_utils][smoke]") {
    auto ld_b16 = make_ld_shared_addr("r1", "buf", "r0", Qualifier::Q_B16);
    REQUIRE(std::get_if<GenericInstr>(&ld_b16.data) != nullptr);
    
    auto ld_b32 = make_ld_shared_addr("r1", "buf", "r0", Qualifier::Q_B32);
    REQUIRE(std::get_if<GenericInstr>(&ld_b32.data) != nullptr);
    
    auto ld_b64 = make_ld_shared_addr("r1", "buf", "r0", Qualifier::Q_B64);
    REQUIRE(std::get_if<GenericInstr>(&ld_b64.data) != nullptr);
    
    auto st_b32 = make_st_shared_addr("buf", "r0", "r1", Qualifier::Q_B32);
    REQUIRE(std::get_if<GenericInstr>(&st_b32.data) != nullptr);
    
    auto ld_v2 = make_ld_shared_addr_v2("r1", "r2", "buf", "r0");
    REQUIRE(std::get_if<GenericInstr>(&ld_v2.data) != nullptr);
    
    auto st_v2 = make_st_shared_addr_v2("buf", "r0", "r1", "r2");
    REQUIRE(std::get_if<GenericInstr>(&st_v2.data) != nullptr);
    
    auto ld_v4 = make_ld_shared_addr_v4("r1", "r2", "r3", "r4", "buf", "r0");
    REQUIRE(std::get_if<GenericInstr>(&ld_v4.data) != nullptr);
    
    auto st_v4 = make_st_shared_addr_v4("buf", "r0", "r1", "r2", "r3", "r4");
    REQUIRE(std::get_if<GenericInstr>(&st_v4.data) != nullptr);
    
    auto decl_b16 = make_shared_decl("buf", 32, Qualifier::Q_B16);
    REQUIRE(std::get_if<DeclarationInstr>(&decl_b16.data) != nullptr);
    
    auto decl_2d = make_shared_decl("buf", 32, 33);
    REQUIRE(std::get_if<DeclarationInstr>(&decl_2d.data) != nullptr);
    
    auto setp_eq = make_setp_eq("%p1", "r0", "r1");
    REQUIRE(std::get_if<GenericInstr>(&setp_eq.data) != nullptr);
    
    auto setp_ne = make_setp_ne("%p1", "r0", "r1");
    REQUIRE(std::get_if<GenericInstr>(&setp_ne.data) != nullptr);
    
    auto setp_gt = make_setp_gt("%p1", "r0", "r1");
    REQUIRE(std::get_if<GenericInstr>(&setp_gt.data) != nullptr);
    
    auto setp_ge = make_setp_ge("%p1", "r0", "r1");
    REQUIRE(std::get_if<GenericInstr>(&setp_ge.data) != nullptr);
    
    auto setp_le = make_setp_le("%p1", "r0", "r1");
    REQUIRE(std::get_if<GenericInstr>(&setp_le.data) != nullptr);
    
    auto setp_eq_imm = make_setp_eq_imm("%p1", "r0", 16);
    REQUIRE(std::get_if<GenericInstr>(&setp_eq_imm.data) != nullptr);
    
    auto setp_ne_imm = make_setp_ne_imm("%p1", "r0", 16);
    REQUIRE(std::get_if<GenericInstr>(&setp_ne_imm.data) != nullptr);
    
    auto setp_lt_imm = make_setp_lt_imm("%p1", "r0", 16);
    REQUIRE(std::get_if<GenericInstr>(&setp_lt_imm.data) != nullptr);
    
    auto setp_gt_imm = make_setp_gt_imm("%p1", "r0", 16);
    REQUIRE(std::get_if<GenericInstr>(&setp_gt_imm.data) != nullptr);
    
    auto setp_le_imm = make_setp_le_imm("%p1", "r0", 16);
    REQUIRE(std::get_if<GenericInstr>(&setp_le_imm.data) != nullptr);
    
    auto setp_ge_imm = make_setp_ge_imm("%p1", "r0", 16);
    REQUIRE(std::get_if<GenericInstr>(&setp_ge_imm.data) != nullptr);
}
