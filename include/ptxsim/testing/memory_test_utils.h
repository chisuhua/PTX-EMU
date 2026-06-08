// memory_test_utils.h
// =============================================================================
// Memory and CTA setup helpers for type-2 integration tests.
//
// Consolidates 9 inline helpers that were previously copy-pasted across
// tests/integration/{memory,ptx}/test_*.cpp. Following the convention of
// the existing testing library (instruction_helpers.h, shared_memory.h),
// all functions are `inline` in the ptxsim::testing namespace.
//
// Coverage map (which test file each helper originated from):
//   - make_shared_decl       test_ld_st_shared, test_shared_memory_layout
//   - make_local_decl        test_local_memory
//   - make_st_shared_addr    test_ld_st_shared
//   - make_st_local_addr     test_local_memory
//   - make_ld_shared_addr    test_ld_st_shared, test_shared_memory_layout
//   - make_ld_local_addr     test_local_memory
//   - setup_block            all 3
//   - init_instruction_factory_once  all 3
//   - read_reg_u32           test_shared_memory_layout, test_local_memory
// =============================================================================

#ifndef PTXSIM_TESTING_MEMORY_TEST_UTILS_H
#define PTXSIM_TESTING_MEMORY_TEST_UTILS_H

#include "catch_amalgamated.hpp"

#include "ptxsim/cta_context.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/sm_context.h"
#include "ptxsim/warp_context.h"

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"

#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ptxsim::testing {

// ============================================================================
// Factory Initialization
// ============================================================================

inline void init_instruction_factory_once() {
    static bool done = false;
    if (!done) {
        InstructionFactory::initialize();
        done = true;
    }
}

// ============================================================================
// Memory Declarations (S_SHARED, S_LOCAL)
// ============================================================================

inline StatementContext make_shared_decl(const std::string &name,
                                          int array_size) {
    StatementContext ctx;
    ctx.type = S_SHARED;
    DeclarationInstr d;
    d.kind = DeclarationInstr::Kind::SHARED;
    d.name = name;
    d.dataType = Qualifier::Q_B32;
    d.array_size = array_size;
    ctx.data = d;
    ctx.instructionText =
        ".shared .b32 " + name + "[" + std::to_string(array_size) + "];";
    return ctx;
}

inline StatementContext make_local_decl(const std::string &name,
                                         int array_size) {
    StatementContext ctx;
    ctx.type = S_LOCAL;
    DeclarationInstr d;
    d.kind = DeclarationInstr::Kind::LOCAL;
    d.name = name;
    d.dataType = Qualifier::Q_B32;
    d.array_size = array_size;
    ctx.data = d;
    ctx.instructionText =
        ".local .b32 " + name + "[" + std::to_string(array_size) + "];";
    return ctx;
}

// ============================================================================
// Addressed Loads / Stores (AddrOperand form, not VariableOperand)
// ============================================================================

inline StatementContext make_st_shared_addr(const std::string &base_sym,
                                             const std::string &offset_reg,
                                             const std::string &src_reg) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B8};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "st.shared.b8 [" + base_sym + "+" + offset_reg + "], " + src_reg + ";";
    return ctx;
}

inline StatementContext make_st_local_addr(const std::string &base_sym,
                                            const std::string &offset_reg,
                                            const std::string &src_reg) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_LOCAL, Qualifier::Q_B32};
    AddrOperand addr;
    addr.space = AddrOperand::Space::LOCAL;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "st.local.b32 [" + base_sym + "+" + offset_reg + "], " + src_reg + ";";
    return ctx;
}

inline StatementContext make_ld_shared_addr(const std::string &dst_reg,
                                             const std::string &base_sym,
                                             const std::string &offset_reg) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B8};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    ctx.instructionText =
        "ld.shared.b8 " + dst_reg + ", [" + base_sym + "+" + offset_reg + "];";
    return ctx;
}

inline StatementContext make_ld_local_addr(const std::string &dst_reg,
                                            const std::string &base_sym,
                                            const std::string &offset_reg) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_LOCAL, Qualifier::Q_B32};
    AddrOperand addr;
    addr.space = AddrOperand::Space::LOCAL;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    ctx.instructionText =
        "ld.local.b32 " + dst_reg + ", [" + base_sym + "+" + offset_reg + "];";
    return ctx;
}

// ============================================================================
// CTA / Warp Setup
// ============================================================================

inline WarpContext *setup_block(SMContext &sm,
                                 std::vector<StatementContext> &stmts) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{32, 1, 1};
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, Symtable *> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

// Create a 32-thread CTA with dynamic shared memory, attach to SM, return warp 0.
//
// Pre-conditions:
//   - InstructionFactory must be initialized (call init_instruction_factory_once())
//   - ResourceManager must be initialized
inline WarpContext *setup_block_with_dynamic_shared(SMContext &sm,
                                                     std::vector<StatementContext> &stmts,
                                                     size_t dynamic_bytes) {
    auto blk = std::make_unique<CTAContext>();
    Dim3 g{1, 1, 1};
    Dim3 b{32, 1, 1};
    Dim3 bi{0, 0, 0};
    std::map<std::string, int> l2pc;
    std::map<std::string, Symtable *> n2s;
    blk->init(g, b, bi, stmts, &n2s, l2pc, nullptr, 0, dynamic_bytes);
    bool ok = sm.add_block(std::move(blk));
    REQUIRE(ok);
    return sm.get_warp(0);
}

// ============================================================================
// Register Read
// ============================================================================

inline uint32_t read_reg_u32(WarpContext *w, const std::string &reg, int lane) {
    auto rbm = w->get_register_bank_manager();
    void *p = rbm->get_register(reg, 0, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint32_t *>(p);
}

// ============================================================================
// Multi-Width Load/Store (Qualifier Overloads)
// ============================================================================

inline StatementContext make_ld_shared_addr(const std::string &dst_reg,
                                            const std::string &base_sym,
                                            const std::string &offset_reg,
                                            Qualifier q) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, q};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    std::string qStr = Q2s(q);
    ctx.instructionText =
        "ld.shared." + qStr + " " + dst_reg + ", [" + base_sym + "+" + offset_reg + "];";
    return ctx;
}

inline StatementContext make_st_shared_addr(const std::string &base_sym,
                                            const std::string &offset_reg,
                                            const std::string &src_reg,
                                            Qualifier q) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, q};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    std::string qStr = Q2s(q);
    ctx.instructionText =
        "st.shared." + qStr + " [" + base_sym + "+" + offset_reg + "], " + src_reg + ";";
    return ctx;
}

// ============================================================================
// Vector Load/Store (v2/v4)
// ============================================================================

inline StatementContext make_ld_shared_addr_v2(const std::string &dst1,
                                               const std::string &dst2,
                                               const std::string &base_sym,
                                               const std::string &offset_reg,
                                               Qualifier q = Qualifier::Q_B32) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, q, Qualifier::Q_V2};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{dst2, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    std::string qStr = Q2s(q);
    ctx.instructionText = "ld.shared.v2." + qStr + " {" + dst1 + "," + dst2 +
                          "}, [" + base_sym + "+" + offset_reg + "];";
    return ctx;
}

inline StatementContext make_st_shared_addr_v2(const std::string &base_sym,
                                               const std::string &offset_reg,
                                               const std::string &src1,
                                               const std::string &src2,
                                               Qualifier q = Qualifier::Q_B32) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, q, Qualifier::Q_V2};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    std::string qStr = Q2s(q);
    ctx.instructionText = "st.shared.v2." + qStr + " [" + base_sym + "+" +
                          offset_reg + "], {" + src1 + "," + src2 + "};";
    return ctx;
}

inline StatementContext make_ld_shared_addr_v4(const std::string &dst1,
                                               const std::string &dst2,
                                               const std::string &dst3,
                                               const std::string &dst4,
                                               const std::string &base_sym,
                                               const std::string &offset_reg,
                                               Qualifier q = Qualifier::Q_B32) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, q, Qualifier::Q_V4};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{RegOperand{dst1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{dst2, -1}});
    instr.operands.push_back(OperandContext{RegOperand{dst3, -1}});
    instr.operands.push_back(OperandContext{RegOperand{dst4, -1}});
    instr.operands.push_back(OperandContext{addr});
    ctx.data = instr;
    std::string qStr = Q2s(q);
    ctx.instructionText = "ld.shared.v4." + qStr + " {" + dst1 + "," + dst2 +
                          "," + dst3 + "," + dst4 + "}, [" + base_sym + "+" +
                          offset_reg + "];";
    return ctx;
}

inline StatementContext make_st_shared_addr_v4(const std::string &base_sym,
                                               const std::string &offset_reg,
                                               const std::string &src1,
                                               const std::string &src2,
                                               const std::string &src3,
                                               const std::string &src4,
                                               Qualifier q = Qualifier::Q_B32) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, q, Qualifier::Q_V4};
    AddrOperand addr;
    addr.space = AddrOperand::Space::SHARED;
    addr.baseSymbol = base_sym;
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{offset_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src3, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src4, -1}});
    ctx.data = instr;
    std::string qStr = Q2s(q);
    ctx.instructionText = "st.shared.v4." + qStr + " [" + base_sym + "+" +
                          offset_reg + "], {" + src1 + "," + src2 + "," + src3 +
                          "," + src4 + "};";
    return ctx;
}

// ============================================================================
// Setp Comparison Variants (Register operands)
// ============================================================================

inline StatementContext make_setp_eq(const std::string &pred,
                                      const std::string &src1,
                                      const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_EQ};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.eq.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_setp_ne(const std::string &pred,
                                      const std::string &src1,
                                      const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_NE};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.ne.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_setp_gt(const std::string &pred,
                                      const std::string &src1,
                                      const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_GT};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.gt.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_setp_ge(const std::string &pred,
                                      const std::string &src1,
                                      const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_GE};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.ge.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_setp_le(const std::string &pred,
                                      const std::string &src1,
                                      const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_LE};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.le.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

// ============================================================================
// Setp Comparison Variants (Immediate operand)
// ============================================================================

inline StatementContext make_setp_eq_imm(const std::string &pred,
                                          const std::string &src1,
                                          int32_t imm_value) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_EQ};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm_value)}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.eq.u32 " + pred + ", " + src1 + ", " + std::to_string(imm_value) + ";";
    return ctx;
}

inline StatementContext make_setp_ne_imm(const std::string &pred,
                                          const std::string &src1,
                                          int32_t imm_value) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_NE};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm_value)}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.ne.u32 " + pred + ", " + src1 + ", " + std::to_string(imm_value) + ";";
    return ctx;
}

inline StatementContext make_setp_lt_imm(const std::string &pred,
                                          const std::string &src1,
                                          int32_t imm_value) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_LT};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm_value)}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.lt.u32 " + pred + ", " + src1 + ", " + std::to_string(imm_value) + ";";
    return ctx;
}

inline StatementContext make_setp_gt_imm(const std::string &pred,
                                          const std::string &src1,
                                          int32_t imm_value) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_GT};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm_value)}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.gt.u32 " + pred + ", " + src1 + ", " + std::to_string(imm_value) + ";";
    return ctx;
}

inline StatementContext make_setp_le_imm(const std::string &pred,
                                          const std::string &src1,
                                          int32_t imm_value) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_LE};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm_value)}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.le.u32 " + pred + ", " + src1 + ", " + std::to_string(imm_value) + ";";
    return ctx;
}

inline StatementContext make_setp_ge_imm(const std::string &pred,
                                          const std::string &src1,
                                          int32_t imm_value) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32, Qualifier::Q_GE};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm_value)}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.ge.u32 " + pred + ", " + src1 + ", " + std::to_string(imm_value) + ";";
    return ctx;
}

// ============================================================================
// Shared Declaration with Qualifier and Multi-Dim
// ============================================================================

inline StatementContext make_shared_decl(const std::string &name, int array_size,
                                         Qualifier q) {
    StatementContext ctx;
    ctx.type = S_SHARED;
    DeclarationInstr d;
    d.kind = DeclarationInstr::Kind::SHARED;
    d.name = name;
    d.dataType = q;
    d.array_size = array_size;
    ctx.data = d;
    std::string qStr = Q2s(q);
    ctx.instructionText =
        ".shared " + qStr + " " + name + "[" + std::to_string(array_size) + "];";
    return ctx;
}

inline StatementContext make_shared_decl(const std::string &name, int dim1,
                                         int dim2,
                                         Qualifier q = Qualifier::Q_B32) {
    StatementContext ctx;
    ctx.type = S_SHARED;
    DeclarationInstr d;
    d.kind = DeclarationInstr::Kind::SHARED;
    d.name = name;
    d.dataType = q;
    d.array_size = dim1 * dim2;
    ctx.data = d;
    std::string qStr = Q2s(q);
    ctx.instructionText = ".shared " + qStr + " " + name + "[" +
                          std::to_string(dim1) + "][" + std::to_string(dim2) + "];";
    return ctx;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_MEMORY_TEST_UTILS_H