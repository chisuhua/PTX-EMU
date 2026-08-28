// statement_factory.h
// 功能: StatementContext 集中化工厂函数
// 作者: AI Agent
// 最后修改日期: 2026-05-08

#ifndef STATEMENT_FACTORY_H
#define STATEMENT_FACTORY_H

#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_types.h"
#include <string>
#include <vector>
#include <optional>

namespace ptxir::factory {

// Phase 1.5c+d: operand types live in the canonical ptxemu::ir
// namespace. The ptxir::factory namespace needs them in scope to build
// OperandContext from bare RegOperand/ImmOperand/etc.
using ptxemu::ir::OperandContext;
using ptxemu::ir::RegOperand;
using ptxemu::ir::VariableOperand;
using ptxemu::ir::ImmOperand;
using ptxemu::ir::Predicate;
using ptxemu::ir::AddrOperand;
using ptxemu::ir::VecOperand;

// =============================================================================
// 1. 底层工厂: 直接利用已有模板构造函数
// =============================================================================

/**
 * @brief 创建 StatementContext（通用工厂）
 * @param type StatementType 枚举值
 * @param instr 指令结构体（任意 InstrVariant 兼容类型）
 * @param text 原始 PTX 文本（用于调试）
 * @return 构造完成的 StatementContext
 */
template <typename InstrType>
inline ptxemu::ir::StatementContext makeStatementContext(ptxemu::ir::StatementType type,
                                              InstrType &&instr,
                                              const std::string &text = "") {
    ptxemu::ir::StatementContext ctx(type, std::forward<InstrType>(instr));
    ctx.instructionText = text;
    return ctx;
}

// =============================================================================
// 2. 便捷工厂: 为常用指令类型提供简化构造
// =============================================================================

// --- 2.1 无操作数指令 (VoidInstr) ---
inline ptxemu::ir::StatementContext makeVoidInstr(ptxemu::ir::StatementType type,
                                       const std::string &text = "") {
    return makeStatementContext(type, VoidInstr{}, text);
}

// --- 2.2 通用指令 (GenericInstr) ---
inline ptxemu::ir::StatementContext makeGenericInstr(
    ptxemu::ir::StatementType type,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    GenericInstr instr;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    return makeStatementContext(type, std::move(instr), text);
}

// --- 2.3 分支指令 (BranchInstr) ---
inline ptxemu::ir::StatementContext makeBranchInstr(
    ptxemu::ir::StatementType type,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &target,
    const std::string &predicate,
    bool predicate_negated,
    const std::string &text = "") {
    BranchInstr instr;
    instr.qualifiers = qualifiers;
    instr.target = target;
    instr.predicate = predicate;
    instr.predicate_negated = predicate_negated;
    instr.reconvergence_pc = -1;
    return makeStatementContext(type, std::move(instr), text);
}

// --- 2.4 屏障指令 (BarrierInstr) ---
inline ptxemu::ir::StatementContext makeBarrierInstr(
    ptxemu::ir::StatementType type,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::optional<int> &barId,
    const std::string &type_str,
    const std::string &text = "") {
    BarrierInstr instr;
    instr.qualifiers = qualifiers;
    instr.barId = barId;
    instr.type = type_str;
    instr.reconvergence_pc = -1;
    return makeStatementContext(type, std::move(instr), text);
}

// --- 2.5 Warp 屏障指令 (BarWarpSyncInstr) ---
inline ptxemu::ir::StatementContext makeBarWarpSyncInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    BarWarpSyncInstr instr;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    return makeStatementContext(S_BAR_WARP_SYNC, std::move(instr), text);
}

/**
 * @brief 便捷工厂: 创建 S_BAR_WARP_SYNC 屏障指令
 * @param mask 线程掩码（十六进制字符串，如 "0xFF"）
 * @param reconv_pc 重汇聚 PC
 * @param text 原始 PTX 文本（用于调试）
 */
inline ptxemu::ir::StatementContext makeBarWarpSyncInstr(
    const std::string &mask,
    int reconv_pc,
    const std::string &text = "") {
    BarWarpSyncInstr instr;
    instr.operands.push_back(ptxemu::ir::OperandContext{ImmOperand{mask}});
    instr.operands.push_back(ptxemu::ir::OperandContext{ImmOperand{std::to_string(reconv_pc)}});
    return makeStatementContext(S_BAR_WARP_SYNC, std::move(instr), text);
}

/**
 * @brief 便捷工厂: 创建 S_BAR_WARP_SYNC 屏障指令（mask 以 uint32_t 传入）
 * @param mask 线程掩码（uint32_t，会转换为十六进制）
 * @param reconv_pc 重汇聚 PC
 * @param text 原始 PTX 文本（用于调试）
 */
inline ptxemu::ir::StatementContext makeBarWarpSyncInstr(
    uint32_t mask,
    int reconv_pc,
    const std::string &text = "") {
    std::ostringstream oss;
    oss << "0x" << std::hex << mask;
    return makeBarWarpSyncInstr(oss.str(), reconv_pc, text);
}

// --- 2.6 调用指令 (CallInstr) ---
inline ptxemu::ir::StatementContext makeCallInstr(
    ptxemu::ir::StatementType type,
    const std::string &funcName,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    CallInstr instr;
    instr.funcName = funcName;
    instr.instructionText = text;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    return makeStatementContext(type, std::move(instr), text);
}

// --- 2.7 声明指令 (DeclarationInstr) ---
inline ptxemu::ir::StatementContext makeDeclarationInstr(
    ptxemu::ir::StatementType type,
    DeclarationInstr::Kind kind,
    const std::string &name,
    ptxemu::ir::Qualifier dataType,
    int array_size,
    const std::string &text = "") {
    DeclarationInstr instr;
    instr.kind = kind;
    instr.name = name;
    instr.dataType = dataType;
    instr.array_size = array_size;
    return makeStatementContext(type, std::move(instr), text);
}

// --- 2.8 标签指令 (LabelInstr) ---
inline ptxemu::ir::StatementContext makeLabelInstr(
    const std::string &labelName,
    const std::string &text = "") {
    LabelInstr instr;
    instr.labelName = labelName;
    return makeStatementContext(S_LABEL, std::move(instr), text);
}

// --- 2.9 内联寄存器指令 (DollarNameInstr) ---
inline ptxemu::ir::StatementContext makeDollarNameInstr(
    const std::string &name,
    const std::string &text = "") {
    DollarNameInstr instr;
    instr.name = name;
    return makeStatementContext(S_DOLLOR, std::move(instr), text);
}

// --- 2.10 Pragma 指令 (PragmaInstr) ---
inline ptxemu::ir::StatementContext makePragmaInstr(
    const std::string &content,
    const std::string &text = "") {
    PragmaInstr instr;
    instr.content = content;
    return makeStatementContext(S_PRAGMA, std::move(instr), text);
}

// --- 2.11 ABI 指令 (AbiDirective) ---
inline ptxemu::ir::StatementContext makeAbiDirective(
    int regNumber,
    const std::string &text = "") {
    AbiDirective instr;
    instr.regNumber = regNumber;
    return makeStatementContext(S_ABI_PRESERVE, std::move(instr), text);
}

// --- 2.12 Membar 指令 (MembarInstr) ---
inline ptxemu::ir::StatementContext makeMembarInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &scope,
    const std::string &text = "") {
    MembarInstr instr;
    instr.qualifiers = qualifiers;
    instr.scope = scope;
    return makeStatementContext(S_MEMBAR, std::move(instr), text);
}

// --- 2.13 Fence 指令 (FenceInstr) ---
inline ptxemu::ir::StatementContext makeFenceInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &memoryOrder,
    const std::string &scope,
    const std::string &text = "") {
    FenceInstr instr;
    instr.qualifiers = qualifiers;
    instr.memoryOrder = memoryOrder;
    instr.scope = scope;
    return makeStatementContext(S_FENCE, std::move(instr), text);
}

// --- 2.14 ReduxSync 指令 (ReduxSyncInstr) ---
inline ptxemu::ir::StatementContext makeReduxSyncInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &operation,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    ReduxSyncInstr instr;
    instr.qualifiers = qualifiers;
    instr.operation = operation;
    instr.operands = operands;
    return makeStatementContext(S_REDUX_SYNC, std::move(instr), text);
}

// --- 2.15 Mbarrier 指令 (MbarrierInstr) ---
inline ptxemu::ir::StatementContext makeMbarrierInstr(
    ptxemu::ir::StatementType type,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &operation,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    MbarrierInstr instr;
    instr.qualifiers = qualifiers;
    instr.operation = operation;
    instr.operands = operands;
    return makeStatementContext(type, std::move(instr), text);
}

// --- 2.16 谓词前缀 (PredicatePrefix) ---
inline ptxemu::ir::StatementContext makePredicatePrefix(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &target,
    const std::string &text = "") {
    PredicatePrefix instr;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    instr.target = target;
    return makeStatementContext(static_cast<ptxemu::ir::StatementType>(0), std::move(instr), text);
}

// --- 2.17.1 Blackwell tcgen05 指令 (Tcgen05Instr) ---
inline ptxemu::ir::StatementContext makeTcgen05Instr(
    ptxemu::ir::Tcgen05OpKind op_kind,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "",
    uint32_t cta_group = 1) {
    ptxemu::ir::Tcgen05Instr instr;
    instr.op_kind = op_kind;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    instr.instructionText = text;
    instr.cta_group = cta_group;  // Oracle C3 fix — route cta_group from visitor
    // Map op_kind to correct S_TCGEN05_* StatementType (fix-tcgen05-grammar-mr3 B2)
    ptxemu::ir::StatementType stmt_type;
    switch (op_kind) {
        case ptxemu::ir::Tcgen05OpKind::ALLOC:      stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_ALLOC); break;
        case ptxemu::ir::Tcgen05OpKind::DEALLOC:    stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_DEALLOC); break;
        case ptxemu::ir::Tcgen05OpKind::RELINQUISH: stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_RELINQUISH); break;
        case ptxemu::ir::Tcgen05OpKind::LD:         stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_LD); break;
        case ptxemu::ir::Tcgen05OpKind::ST:         stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_ST); break;
        case ptxemu::ir::Tcgen05OpKind::CP:         stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_CP); break;
        case ptxemu::ir::Tcgen05OpKind::MMA:        stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_MMA); break;
        case ptxemu::ir::Tcgen05OpKind::MMA_WS:     stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_MMA_WS); break;
        case ptxemu::ir::Tcgen05OpKind::COMMIT:     stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_COMMIT); break;
        case ptxemu::ir::Tcgen05OpKind::WAIT:       stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_WAIT); break;
        case ptxemu::ir::Tcgen05OpKind::FENCE:      stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_FENCE); break;
        default:                        stmt_type = static_cast<ptxemu::ir::StatementType>(S_TCGEN05_MMA); break;
    }
    return makeStatementContext(stmt_type, std::move(instr), text);
}

// --- 2.18 原子指令 (AtomInstr) ---
inline ptxemu::ir::StatementContext makeAtomInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    int operandNum,
    const std::string &text = "") {
    AtomInstr instr;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    instr.operandNum = operandNum;
    return makeStatementContext(S_ATOM, std::move(instr), text);
}

// --- 2.19 Vote 指令 (VoteInstr) ---
inline ptxemu::ir::StatementContext makeVoteInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &mode,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    VoteInstr instr;
    instr.qualifiers = qualifiers;
    instr.mode = mode;
    instr.operands = operands;
    return makeStatementContext(S_VOTE, std::move(instr), text);
}

// --- 2.20 Shuffle指令 (ShflInstr) ---
inline ptxemu::ir::StatementContext makeShflInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &mode,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    ShflInstr instr;
    instr.qualifiers = qualifiers;
    instr.mode = mode;
    instr.operands = operands;
    return makeStatementContext(S_SHFL, std::move(instr), text);
}

// --- 2.21 Texture/Surface/Reduction/Prefetch 指令 ---
template <typename InstrType>
inline ptxemu::ir::StatementContext makeInstrWithQualifiersAndOperands(
    ptxemu::ir::StatementType type,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    InstrType instr;
    instr.qualifiers = qualifiers;
    instr.operands = operands;
    return makeStatementContext(type, std::move(instr), text);
}

// 专用别名函数（提高代码可读性）
inline ptxemu::ir::StatementContext makeTextureInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    return makeInstrWithQualifiersAndOperands<TextureInstr>(
        S_TEX, qualifiers, operands, text);
}

inline ptxemu::ir::StatementContext makeSurfaceInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    return makeInstrWithQualifiersAndOperands<SurfaceInstr>(
        S_SURF, qualifiers, operands, text);
}

inline ptxemu::ir::StatementContext makeReductionInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::string &operation,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    ReductionInstr instr;
    instr.qualifiers = qualifiers;
    instr.operation = operation;
    instr.operands = operands;
    return makeStatementContext(S_RED, std::move(instr), text);
}

inline ptxemu::ir::StatementContext makePrefetchInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    return makeInstrWithQualifiersAndOperands<PrefetchInstr>(
        S_PREFETCH, qualifiers, operands, text);
}

// --- 2.22 异步指令 ---
inline ptxemu::ir::StatementContext makeCpAsyncInstr(
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<ptxemu::ir::OperandContext> &operands,
    const std::string &text = "") {
    return makeInstrWithQualifiersAndOperands<CpAsyncInstr>(
        S_CP_ASYNC, qualifiers, operands, text);
}

} // namespace ptxir::factory

#endif // STATEMENT_FACTORY_H