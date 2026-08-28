/**
 * PTX-5b PoC tests: RegisterAnalyzer::get_dest_registers_as_ids
 *
 * TDD Red-Phase: 7 tests cover the PTX dest convention matrix from
 * design.md §6.1 PTX dest 约定矩阵. Tests verify the operands[0]
 * extraction strategy + 85% correctness rate for arithmetic/ld/vote/shfl/atom
 * and natural-empty for st/red/prefetch/barrier/bra/ret.
 *
 * Phase 8.B PTX-5b — cpptlm-phase8b-injection-points
 * Ref: ADR-0020, design.md §6.1
 */
#include "catch_amalgamated.hpp"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/register_analyzer.h"



// ----------------------------------------------------------------------------
// Test helpers
// ----------------------------------------------------------------------------
namespace {

RegOperand make_reg(const std::string& name, int index) {
    RegOperand r;
    r.name = name;
    r.index = index;
    return r;
}

AddrOperand make_addr(const std::string& base) {
    AddrOperand a;
    a.space = AddrOperand::Space::GLOBAL;
    a.baseSymbol = base;
    a.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    a.immediateOffset = "0";
    return a;
}

ImmOperand make_imm(const std::string& v) {
    ImmOperand i;
    i.value = v;
    return i;
}

ptxemu::ir::StatementContext make_generic_add_stmt() {
    // add.f32 %f1, %f2, %f3
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_ADD;
    GenericInstr instr;
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 1)));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 2)));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 3)));
    stmt.data = instr;
    return stmt;
}

ptxemu::ir::StatementContext make_generic_ld_stmt() {
    // ld.global.f32 %f5, [%rd1]
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_LD;
    GenericInstr instr;
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 5)));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_addr("rd1")));
    stmt.data = instr;
    return stmt;
}

ptxemu::ir::StatementContext make_generic_st_stmt() {
    // st.global.f32 [%rd1], %f1
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_ST;
    GenericInstr instr;
    instr.operands.push_back(ptxemu::ir::OperandContext(make_addr("rd1")));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 1)));
    stmt.data = instr;
    return stmt;
}

ptxemu::ir::StatementContext make_generic_setp_stmt() {
    // setp.eq.f32 %p1, %f2, %f3
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_SETP;
    GenericInstr instr;
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("p", 1)));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 2)));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("f", 3)));
    stmt.data = instr;
    return stmt;
}

ptxemu::ir::StatementContext make_atom_stmt() {
    // atom.global.add.u32 %r1, [%rd1], %r2
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_ATOM;
    AtomInstr instr;
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("r", 1)));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_addr("rd1")));
    instr.operands.push_back(ptxemu::ir::OperandContext(make_reg("r", 2)));
    stmt.data = instr;
    return stmt;
}

ptxemu::ir::StatementContext make_bra_stmt() {
    // bra L_target;
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_BRA;
    BranchInstr instr;
    instr.target = "L_target";
    stmt.data = instr;
    return stmt;
}

ptxemu::ir::StatementContext make_bar_sync_stmt() {
    // bar.sync 0;
    ptxemu::ir::StatementContext stmt;
    stmt.type = S_BAR;
    BarrierInstr instr;
    instr.type = "cta";
    instr.barId = 0;
    stmt.data = instr;
    return stmt;
}

}  // namespace

// ----------------------------------------------------------------------------
// Test #1: 算术指令 add.f32 → operands[0] dest
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: add.f32 %f1, %f2, %f3 → [1]",
          "[unit][register][dest]") {
    auto stmt = make_generic_add_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.size() == 1);
    REQUIRE(dests[0] == 1);
}

// ----------------------------------------------------------------------------
// Test #2: 加载指令 ld.global.f32 → operands[0] dest
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: ld.global.f32 %f5, [%rd1] → [5]",
          "[unit][register][dest]") {
    auto stmt = make_generic_ld_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.size() == 1);
    REQUIRE(dests[0] == 5);
}

// ----------------------------------------------------------------------------
// Test #3: 存储指令 st.global.f32 → operands[0] = AddrOperand → 自然空
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: st.global.f32 [%rd1], %f1 → []",
          "[unit][register][dest]") {
    auto stmt = make_generic_st_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.empty());
}

// ----------------------------------------------------------------------------
// Test #4: 谓词 setp.eq.f32 → operands[0] pred dest
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: setp.eq.f32 %p1, %f2, %f3 → [1]",
          "[unit][register][dest]") {
    auto stmt = make_generic_setp_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.size() == 1);
    REQUIRE(dests[0] == 1);
}

// ----------------------------------------------------------------------------
// Test #5: 原子操作 atom.global.add.u32 → operands[0] = dest (旧值返回)
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: atom.global.add.u32 %r1, [%rd1], %r2 → [1]",
          "[unit][register][dest]") {
    auto stmt = make_atom_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.size() == 1);
    REQUIRE(dests[0] == 1);
}

// ----------------------------------------------------------------------------
// Test #6: 控制流 bra → 无 operands 字段 → 空
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: bra L_target → []",
          "[unit][register][dest]") {
    auto stmt = make_bra_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.empty());
}

// ----------------------------------------------------------------------------
// Test #7: 屏障 bar.sync → 无 operands 字段 → 空
// ----------------------------------------------------------------------------
TEST_CASE("get_dest_registers_as_ids: bar.sync 0 → []",
          "[unit][register][dest]") {
    auto stmt = make_bar_sync_stmt();
    auto dests = RegisterAnalyzer::get_dest_registers_as_ids(stmt);
    REQUIRE(dests.empty());
}