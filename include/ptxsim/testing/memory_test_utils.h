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
#include "ptx_ir/statement_context.h"

#include "memory/resource_manager.h"
#include "register/register_bank_manager.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ptxsim::testing {

// ============================================================================
// Factory Initialization
// ============================================================================

// One-shot guard for InstructionFactory::initialize().
//
// All 3 tests require the factory to be initialized before any
// S_LD/S_ST/etc. statement executes, but the initializer is not idempotent
// in the current codebase. The static-bool-guard pattern ensures the call
// happens exactly once per process.
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

// `.shared .b32 <name>[<size>];` declaration.
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

// `.local .b32 <name>[<size>];` declaration.
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
//
// IMPORTANT: these helpers use AddrOperand with REGISTER offset. The
// VariableOperand form (used in older test helpers in instruction_helpers.h)
// SEGFAULTs the handler per KNOWN_ISSUES.md section "Pre-P0 Baseline Red".
//
// The b8 qualifier on shared variants avoids per-lane overlap on 32 lanes:
// a b32 write per lane (4 bytes) at offset=lane_id would cause inter-lane
// overlap because lane N writes buf[N..N+3] and lane N+1 writes buf[N+1..N+4].

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

// Create a 32-thread CTA, attach to SM, return warp 0.
//
// Pre-conditions:
//   - InstructionFactory must be initialized (call init_instruction_factory_once())
//   - ResourceManager must be initialized (call ResourceManager::instance().initialize(...))
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

// ============================================================================
// Register Read
// ============================================================================

// Read a u32 register from a specific lane.
//
// Fails the test if the register is not allocated for that lane.
inline uint32_t read_reg_u32(WarpContext *w, const std::string &reg, int lane) {
    auto rbm = w->get_register_bank_manager();
    void *p = rbm->get_register(reg, 0, lane);
    REQUIRE(p != nullptr);
    return *static_cast<uint32_t *>(p);
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_MEMORY_TEST_UTILS_H
