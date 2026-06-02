// instruction_helpers.h
#ifndef PTXSIM_TESTING_INSTRUCTION_HELPERS_H
#define PTXSIM_TESTING_INSTRUCTION_HELPERS_H

#include "ptx_ir/ptx_types.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ptxsim::testing {

// ============================================================================
// Instruction Construction Helpers
// Build StatementContext objects for PTX simulation testing
// ============================================================================

inline StatementContext make_bar_warp_sync(uint32_t mask, int reconvergence_pc) {
    StatementContext ctx;
    ctx.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(mask)}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(reconvergence_pc)}});
    ctx.data = instr;
    ctx.instructionText = "bar.warp.sync.b32 0x" + std::to_string(mask) + ", " + std::to_string(reconvergence_pc) + ";";
    return ctx;
}

inline StatementContext make_bar_sync(int bar_id = 0) {
    StatementContext ctx;
    ctx.type = S_BAR;
    BarrierInstr instr;
    instr.barId = bar_id;
    ctx.data = instr;
    ctx.instructionText = "bar.sync " + std::to_string(bar_id) + ";";
    return ctx;
}

inline StatementContext make_mov(const std::string& dst, const std::string& src) {
    StatementContext ctx;
    ctx.type = S_MOV;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "mov.b32 " + dst + ", " + src + ";";
    return ctx;
}

inline StatementContext make_mov_imm(const std::string& dst, int64_t imm) {
    StatementContext ctx;
    ctx.type = S_MOV;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(imm)}});
    ctx.data = instr;
    ctx.instructionText = "mov.b32 " + dst + ", " + std::to_string(imm) + ";";
    return ctx;
}

inline StatementContext make_add(const std::string& dst, const std::string& src1, const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_ADD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "add.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_mul(const std::string& dst, const std::string& src1, const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_MUL;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "mul.lo.s32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_ld_shared(const std::string& dst_reg, const std::string& shared_var, const std::string& offset_reg) {
    StatementContext ctx;
    ctx.type = S_LD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B32};
    std::string addr = "[" + shared_var + "+" + offset_reg + "]";
    instr.operands.push_back(OperandContext{RegOperand{dst_reg, -1}});
    instr.operands.push_back(OperandContext{VariableOperand{addr}});
    ctx.data = instr;
    ctx.instructionText = "ld.shared.b32 " + dst_reg + ", " + addr + ";";
    return ctx;
}

inline StatementContext make_st_shared(const std::string& shared_var, const std::string& offset_reg, const std::string& src_reg) {
    StatementContext ctx;
    ctx.type = S_ST;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_SHARED, Qualifier::Q_B32};
    std::string addr = "[" + shared_var + "+" + offset_reg + "]";
    instr.operands.push_back(OperandContext{VariableOperand{addr}});
    instr.operands.push_back(OperandContext{RegOperand{src_reg, -1}});
    ctx.data = instr;
    ctx.instructionText = "st.shared.b32 " + addr + ", " + src_reg + ";";
    return ctx;
}

inline StatementContext make_setp_lt(const std::string& pred, const std::string& src1, const std::string& src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "setp.lt.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_bra(const std::string& target) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr instr;
    instr.target = target;
    ctx.data = instr;
    ctx.instructionText = "bra " + target + ";";
    return ctx;
}

inline StatementContext make_bra_pred(const std::string& target, const std::string& pred, bool neg = false) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr instr;
    instr.target = target;
    instr.predicate = pred;
    instr.predicate_negated = neg;
    ctx.data = instr;
    ctx.instructionText = (neg ? "@!" : "@") + pred + " bra " + target + ";";
    return ctx;
}

inline StatementContext make_label(const std::string& name) {
    StatementContext ctx;
    ctx.type = S_LABEL;
    ctx.data = LabelInstr{name};
    ctx.instructionText = name + ":;";
    return ctx;
}

inline StatementContext make_nop() {
    StatementContext ctx;
    ctx.type = S_PRAGMA;
    ctx.data = PragmaInstr{"nop"};
    ctx.instructionText = "nop;";
    return ctx;
}

inline StatementContext make_exit() {
    StatementContext ctx;
    ctx.type = S_EXIT;
    ctx.data = VoidInstr{};
    ctx.instructionText = "exit;";
    return ctx;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_INSTRUCTION_HELPERS_H