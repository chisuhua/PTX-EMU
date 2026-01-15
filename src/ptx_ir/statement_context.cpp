// statement_context.cpp
#include "ptx_ir/statement_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"

std::string S2s(StatementType s) {
    switch (s) {
#define X(enum_val, struct_name, str, opcount, _)                              \
    case enum_val:                                                             \
        return #str;
#include "ptx_ir/ptx_op.def"
#undef X
    case S_UNKNOWN:
        return "unknown";
    default:
        assert(0 && "Unsupported statement type");
        return "";
    }
}

// 析构函数
StatementContext::~StatementContext() {
    if (!statement) {
        return;
    }

#define X(enum_val, struct_name, str, opcount, struct_kind)                    \
    case enum_val: {                                                           \
        delete static_cast<struct_name *>(statement);                          \
        break;                                                                 \
    }

    switch (statementType) {
#include "ptx_ir/ptx_op.def"
    }
#undef X

    // S_UNKNOWN: do nothing (statement should be null, but no harm in check)
    statement = nullptr;
}

StatementContext::StatementContext(const StatementContext &other)
    : statement(nullptr), statementType(S_UNKNOWN),
      state(InstructionState::READY), instructionText(other.instructionText) {
    if (!other.statement)
        return;
    statementType = other.statementType;
    state = other.state;

// 宏重载：统一接口为 (Name, OpCount)
#define COPY_IMPL_OPERAND_REG(Name, OpCount) COPY_IMPL_OPERAND_REG_IMPL(Name)
#define COPY_IMPL_OPERAND_CONST(Name, OpCount)                                 \
    COPY_IMPL_OPERAND_CONST_IMPL(Name)
#define COPY_IMPL_OPERAND_MEMORY(Name, OpCount)                                \
    COPY_IMPL_OPERAND_MEMORY_IMPL(Name)
#define COPY_IMPL_SIMPLE_NAME(Name, OpCount) COPY_IMPL_SIMPLE_NAME_IMPL(Name)
#define COPY_IMPL_SIMPLE_STRING(Name, OpCount)                                 \
    COPY_IMPL_SIMPLE_STRING_IMPL(Name)
#define COPY_IMPL_VOID_INSTR(Name, OpCount) COPY_IMPL_VOID_INSTR_IMPL(Name)
#define COPY_IMPL_BRANCH(Name, OpCount) COPY_IMPL_BRANCH_IMPL(Name)
#define COPY_IMPL_BARRIER(Name, OpCount) COPY_IMPL_BARRIER_IMPL(Name)
#define COPY_IMPL_PREDICATE_PREFIX(Name, OpCount)                              \
    COPY_IMPL_PREDICATE_PREFIX_IMPL(Name, OpCount)
#define COPY_IMPL_GENERIC_INSTR(Name, OpCount)                                 \
    COPY_IMPL_GENERIC_INSTR_IMPL(Name, OpCount)
#define COPY_IMPL_WMMA_INSTR(Name, OpCount) COPY_IMPL_WMMA_INSTR_IMPL(Name)
#define COPY_IMPL_ATOM_INSTR(Name, OpCount) COPY_IMPL_ATOM_INSTR_IMPL(Name)

    // 自动生成所有 case
#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    case enum_val: {                                                           \
        COPY_IMPL_##struct_kind(type_name, op_count);                          \
        break;                                                                 \
    }
    switch (statementType) {
#include "ptx_ir/ptx_op.def"
    }
#undef X

// 清理宏
#undef COPY_IMPL_OPERAND_REG
#undef COPY_IMPL_OPERAND_CONST
#undef COPY_IMPL_OPERAND_MEMORY
#undef COPY_IMPL_SIMPLE_NAME
#undef COPY_IMPL_SIMPLE_STRING
#undef COPY_IMPL_VOID_INSTR
#undef COPY_IMPL_BRANCH
#undef COPY_IMPL_BARRIER
#undef COPY_IMPL_PREDICATE_PREFIX
#undef COPY_IMPL_GENERIC_INSTR
#undef COPY_IMPL_WMMA_INSTR
#undef COPY_IMPL_ATOM_INSTR
}