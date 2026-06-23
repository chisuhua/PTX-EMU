// instruction_helpers.h
#ifndef PTXSIM_TESTING_INSTRUCTION_HELPERS_H
#define PTXSIM_TESTING_INSTRUCTION_HELPERS_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ptxsim::testing {

// ============================================================================
// Instruction Construction Helpers
// Build StatementContext objects for PTX simulation testing
// ============================================================================

inline StatementContext make_bar_warp_sync(uint32_t mask,
                                           int reconvergence_pc) {
    StatementContext ctx;
    ctx.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{ImmOperand{std::to_string(mask)}});
    instr.operands.push_back(
        OperandContext{ImmOperand{std::to_string(reconvergence_pc)}});
    ctx.data = instr;
    ctx.instructionText = "bar.warp.sync.b32 0x" + std::to_string(mask) + ", " +
                          std::to_string(reconvergence_pc) + ";";
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

inline StatementContext make_mov(const std::string &dst,
                                 const std::string &src) {
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

inline StatementContext make_mov_imm(const std::string &dst, int64_t imm) {
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

inline StatementContext make_add(const std::string &dst,
                                 const std::string &src1,
                                 const std::string &src2) {
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

inline StatementContext make_mul(const std::string &dst,
                                 const std::string &src1,
                                 const std::string &src2) {
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

inline StatementContext make_sub(const std::string &dst,
                                 const std::string &src1,
                                 const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SUB;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "sub.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_and(const std::string &dst,
                                 const std::string &src1,
                                 const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_AND;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "and.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_or(const std::string &dst, const std::string &src1,
                                const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_OR;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "or.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_xor(const std::string &dst,
                                 const std::string &src1,
                                 const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_XOR;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "xor.b32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_shl(const std::string &dst, const std::string &src,
                                 const std::string &shift) {
    StatementContext ctx;
    ctx.type = S_SHL;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    instr.operands.push_back(OperandContext{RegOperand{shift, -1}});
    ctx.data = instr;
    ctx.instructionText = "shl.b32 " + dst + ", " + src + ", " + shift + ";";
    return ctx;
}

inline StatementContext make_shr(const std::string &dst, const std::string &src,
                                 const std::string &shift) {
    StatementContext ctx;
    ctx.type = S_SHR;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    instr.operands.push_back(OperandContext{RegOperand{shift, -1}});
    ctx.data = instr;
    ctx.instructionText = "shr.b32 " + dst + ", " + src + ", " + shift + ";";
    return ctx;
}

inline StatementContext make_not(const std::string &dst,
                                 const std::string &src) {
    StatementContext ctx;
    ctx.type = S_NOT;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "not.b32 " + dst + ", " + src + ";";
    return ctx;
}

// Bit Field Extract: bfe.type dst, src, pos, len
inline StatementContext make_bfe_u32(const std::string &dst,
                                     const std::string &src,
                                     const std::string &pos,
                                     const std::string &len) {
    StatementContext ctx;
    ctx.type = S_BFE;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_U32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    instr.operands.push_back(OperandContext{RegOperand{pos, -1}});
    instr.operands.push_back(OperandContext{RegOperand{len, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "bfe.u32 " + dst + ", " + src + ", " + pos + ", " + len + ";";
    return ctx;
}

// Atomic add on global memory: atom.global.add.u32 dst, [addr], src
// Address is read from a 64-bit register (matches ld.global / st.global
// pattern: register holds the host pointer).
inline StatementContext make_atom_global_add_u32(const std::string &dst,
                                                 const std::string &addr_reg,
                                                 const std::string &src) {
    StatementContext ctx;
    ctx.type = S_ATOM;
    AtomInstr instr;
    instr.qualifiers = {Qualifier::Q_U32, Qualifier::Q_GLOBAL,
                        Qualifier::Q_ADD_ATOM};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    AddrOperand addr;
    addr.space = AddrOperand::Space::GLOBAL;
    addr.baseSymbol = "";
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{addr_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "atom.global.add.u32 " + dst + ", [" + addr_reg + "], " + src + ";";
    return ctx;
}

// Atomic exchange on global memory: atom.global.exch.u32 dst, [addr], src
// Returns the old value at addr to dst, writes src to addr.
inline StatementContext make_atom_global_exch_u32(const std::string &dst,
                                                  const std::string &addr_reg,
                                                  const std::string &src) {
    StatementContext ctx;
    ctx.type = S_ATOM;
    AtomInstr instr;
    instr.qualifiers = {Qualifier::Q_U32, Qualifier::Q_GLOBAL,
                        Qualifier::Q_EXCH_ATOM};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    AddrOperand addr;
    addr.space = AddrOperand::Space::GLOBAL;
    addr.baseSymbol = "";
    addr.offsetType = AddrOperand::OffsetType::REGISTER;
    addr.registerOffset =
        std::make_shared<OperandContext>(RegOperand{addr_reg, -1});
    instr.operands.push_back(OperandContext{addr});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "atom.global.exch.u32 " + dst + ", [" + addr_reg + "], " + src + ";";
    return ctx;
}

inline StatementContext
make_cvt(const std::string &dst, const std::string &src, Qualifier dst_dtype,
         Qualifier src_dtype, const std::vector<Qualifier> &extra_quals = {}) {
    StatementContext ctx;
    ctx.type = S_CVT;
    GenericInstr instr;
    instr.qualifiers = {dst_dtype, src_dtype};
    for (auto q : extra_quals) {
        instr.qualifiers.push_back(q);
    }
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    auto qual_name = [](Qualifier q) -> std::string {
        switch (q) {
        case Qualifier::Q_S32:
            return "s32";
        case Qualifier::Q_F32:
            return "f32";
        case Qualifier::Q_F64:
            return "f64";
        case Qualifier::Q_S64:
            return "s64";
        case Qualifier::Q_B32:
            return "b32";
        case Qualifier::Q_B64:
            return "b64";
        case Qualifier::Q_RP:
            return "rp";
        case Qualifier::Q_RPI:
            return "rpi";
        case Qualifier::Q_RN:
            return "rn";
        case Qualifier::Q_RNI:
            return "rni";
        case Qualifier::Q_RZ:
            return "rz";
        case Qualifier::Q_RZI:
            return "rzi";
        case Qualifier::Q_RM:
            return "rm";
        case Qualifier::Q_RMI:
            return "rmi";
        case Qualifier::Q_SAT:
            return "sat";
        default:
            return "";
        }
    };
    std::string text =
        "cvt." + qual_name(dst_dtype) + "." + qual_name(src_dtype);
    for (auto q : extra_quals) {
        std::string name = qual_name(q);
        if (!name.empty()) {
            text += "." + name;
        }
    }
    text += " " + dst + ", " + src + ";";
    ctx.instructionText = text;
    return ctx;
}

// cvt with saturation (.sat) — emits e.g. "cvt.u32.f32.sat"
inline StatementContext make_cvt_sat(const std::string &dst,
                                     const std::string &src,
                                     Qualifier dst_dtype, Qualifier src_dtype) {
    StatementContext ctx;
    ctx.type = S_CVT;
    GenericInstr instr;
    instr.qualifiers = {dst_dtype, src_dtype, Qualifier::Q_SAT};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    auto qual_name = [](Qualifier q) -> std::string {
        switch (q) {
        case Qualifier::Q_S32:
            return "s32";
        case Qualifier::Q_U32:
            return "u32";
        case Qualifier::Q_F32:
            return "f32";
        case Qualifier::Q_F64:
            return "f64";
        case Qualifier::Q_S64:
            return "s64";
        case Qualifier::Q_B32:
            return "b32";
        case Qualifier::Q_B64:
            return "b64";
        default:
            return "b32";
        }
    };
    ctx.instructionText = "cvt." + qual_name(dst_dtype) + "." +
                          qual_name(src_dtype) + ".sat " + dst + ", " + src +
                          ";";
    return ctx;
}

inline StatementContext make_fadd(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_ADD; // Float reuses integer opcode; Q_F32 selects float path
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "add.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_fsub(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SUB;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "sub.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_fmul(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_MUL;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "mul.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_fdiv(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_DIV;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "div.f32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_ffma(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2,
                                  const std::string &src3) {
    StatementContext ctx;
    ctx.type = S_FMA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_F32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src3, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "fma.rn.f32 " + dst + ", " + src1 + ", " + src2 + ", " + src3 + ";";
    return ctx;
}

inline StatementContext make_addc(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_ADDC;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "addc.u32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_subc(const std::string &dst,
                                  const std::string &src1,
                                  const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SUBC;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "subc.u32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_mad(const std::string &dst,
                                 const std::string &src1,
                                 const std::string &src2,
                                 const std::string &src3) {
    StatementContext ctx;
    ctx.type = S_MAD;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src3, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "mad.lo.s32 " + dst + ", " + src1 + ", " + src2 + ", " + src3 + ";";
    return ctx;
}

inline StatementContext make_mul24(const std::string &dst,
                                   const std::string &src1,
                                   const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_MUL24;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText = "mul.lo.u32 " + dst + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_cvta_to_global(const std::string &dst,
                                            const std::string &src) {
    StatementContext ctx;
    ctx.type = S_CVTA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_U64, Qualifier::Q_GLOBAL};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "cvta.to.global.u64 " + dst + ", " + src + ";";
    return ctx;
}

inline StatementContext make_cvta_to_shared(const std::string &dst,
                                            const std::string &src) {
    StatementContext ctx;
    ctx.type = S_CVTA;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_U64, Qualifier::Q_SHARED};
    instr.operands.push_back(OperandContext{RegOperand{dst, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src, -1}});
    ctx.data = instr;
    ctx.instructionText = "cvta.to.shared.u64 " + dst + ", " + src + ";";
    return ctx;
}

// NOTE: make_ld_shared / make_st_shared (VariableOperand form) have been
// removed. Use memory_test_utils.h::make_ld_shared_addr / make_st_shared_addr
// with AddrOperand form instead.

inline StatementContext make_setp_lt(const std::string &pred,
                                     const std::string &src1,
                                     const std::string &src2) {
    StatementContext ctx;
    ctx.type = S_SETP;
    GenericInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{RegOperand{pred, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src1, -1}});
    instr.operands.push_back(OperandContext{RegOperand{src2, -1}});
    ctx.data = instr;
    ctx.instructionText =
        "setp.lt.u32 " + pred + ", " + src1 + ", " + src2 + ";";
    return ctx;
}

inline StatementContext make_bra(const std::string &target,
                                 int reconvergence_pc = -1) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr instr;
    instr.target = target;
    if (reconvergence_pc >= 0) {
        instr.reconvergence_pc = reconvergence_pc;
    }
    ctx.data = instr;
    ctx.instructionText = "bra " + target + ";";
    return ctx;
}

inline StatementContext make_bra_pred(const std::string &target,
                                      const std::string &pred, bool neg = false,
                                      int reconvergence_pc = -1) {
    StatementContext ctx;
    ctx.type = S_BRA;
    BranchInstr instr;
    instr.target = target;
    instr.predicate = pred;
    instr.predicate_negated = neg;
    if (reconvergence_pc >= 0) {
        instr.reconvergence_pc = reconvergence_pc;
    }
    ctx.data = instr;
    ctx.instructionText = (neg ? "@!" : "@") + pred + " bra " + target + ";";
    return ctx;
}

inline StatementContext make_label(const std::string &name) {
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

inline StatementContext make_ret() {
    StatementContext ctx;
    ctx.type = S_RET;
    ctx.data = VoidInstr{};
    ctx.instructionText = "ret;";
    return ctx;
}

} // namespace ptxsim::testing

#endif // PTXSIM_TESTING_INSTRUCTION_HELPERS_H