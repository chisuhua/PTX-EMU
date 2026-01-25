// statement_context.h
#ifndef STATEMENT_CONTEXT_H
#define STATEMENT_CONTEXT_H

#include "operand_context.h"
#include "ptx_types.h"
#include "ptxsim/execution_types.h"
#include <vector>

class StatementContext {
public:
    StatementType statementType;
    void *statement;
    InstructionState state;
    std::vector<Qualifier> *qualifier; // qualifer for collect operand
    std::string instructionText;       // 存储原始指令文本

// =============================================================================
// 1. 操作数描述结构体
// =============================================================================
#define DEFINE_OPERAND_REG(Name, _)                                            \
    struct Name {                                                              \
        int regNum = 0;                                                        \
        std::vector<Qualifier> regDataType;                                    \
        std::string regName;                                                   \
    };

#define DEFINE_OPERAND_CONST(Name, _)                                          \
    struct Name {                                                              \
        int constAlign = 0;                                                    \
        int constSize = 1;                                                     \
        std::vector<Qualifier> constDataType;                                  \
        std::string constName;                                                 \
    };

#define DEFINE_OPERAND_MEMORY(Name, _)                                         \
    struct Name {                                                              \
        int align = 0;                                                         \
        int size = 1;                                                          \
        std::vector<Qualifier> dataType;                                       \
        std::string name;                                                      \
    };

    // =============================================================================
    // 2. 简单结构体
    // =============================================================================

#define DEFINE_SIMPLE_NAME(Name, _)                                            \
    struct Name {                                                              \
        std::string dollorName;                                                \
    };

#define DEFINE_SIMPLE_STRING(Name, _)                                          \
    struct Name {                                                              \
        std::string pragmaString;                                              \
    };

#define DEFINE_VOID_INSTR(Name, _)                                             \
    struct Name {};

    // =============================================================================
    // 3. 控制流结构体
    // =============================================================================

#define DEFINE_BRANCH(Name, _)                                                 \
    struct Name {                                                              \
        std::vector<Qualifier> qualifier;                                      \
        std::string braTarget;                                                 \
    };

#define DEFINE_BARRIER(Name, _)                                                \
    struct Name {                                                              \
        std::vector<Qualifier> qualifier;                                      \
        std::string barType;                                                   \
        int barId;                                                             \
    };

    // =============================================================================
    // 4. 通用指令结构体
    // =============================================================================
    struct BASE_INSTR {
        std::vector<Qualifier> qualifier;
        std::vector<OperandContext> operands;
    };

#define DEFINE_GENERIC_INSTR(Name, OpCount)                                    \
    struct Name : BASE_INSTR {                                                 \
        static constexpr int op_count = OpCount;                               \
    };

#define DEFINE_PREDICATE_PREFIX(Name, OpCount)                                 \
    struct Name : BASE_INSTR {                                                 \
        static constexpr int op_count = OpCount;                               \
        std::string atLabelName;                                               \
    };

#define DEFINE_WMMA_INSTR(Name, OpCount)                                       \
    struct Name : BASE_INSTR {                                                 \
        static constexpr int op_count = OpCount;                               \
        WmmaType wmmaType;                                                     \
    };

#define DEFINE_ATOM_INSTR(Name, OpCount)                                       \
    struct Name : BASE_INSTR {                                                 \
        static constexpr int op_count = OpCount;                               \
        int operandNum = 0;                                                    \
    };

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DEFINE_##struct_kind(type_name, op_count)
#include "ptx_op.def"
#undef X

    StatementContext()
        : statementType(S_REG), statement(nullptr),
          state{InstructionState::READY}, instructionText("") {}
    ~StatementContext();

    // 深拷贝方法
    StatementContext(const StatementContext &other);
};

// =============================================================================
// 内部辅助：深拷贝 OperandContext 数组
// =============================================================================
template <std::size_t N>
inline void deepCopyOperandArray(OperandContext (&dst)[N],
                                 const OperandContext (&src)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        dst[i] = OperandContext(
            src[i]); // 调用 OperandContext 的拷贝构造函数（深拷贝）
    }
}

inline void deepCopyOperandArray(std::vector<OperandContext> &dst,
                                 const std::vector<OperandContext> &src) {
    uint32_t op_count = src.size();
    dst.resize(op_count);
    for (std::size_t i = 0; i < op_count; ++i) {
        dst[i] = src[i]; // 调用 OperandContext 的拷贝构造函数（深拷贝）
    }
}

// =============================================================================
// 1. 操作数描述符（Operand Kinds）
// =============================================================================
#define COPY_IMPL_OPERAND_REG_IMPL(Name)                                       \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->regNum = source->regNum;                                         \
        dest->regDataType = source->regDataType;                               \
        dest->regName = source->regName;                                       \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_OPERAND_CONST_IMPL(Name)                                     \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->constAlign = source->constAlign;                                 \
        dest->constSize = source->constSize;                                   \
        dest->constDataType = source->constDataType;                           \
        dest->constName = source->constName;                                   \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_OPERAND_MEMORY_IMPL(Name)                                    \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->align = source->align;                                           \
        dest->size = source->size;                                             \
        dest->dataType = source->dataType;                                     \
        dest->name = source->name;                                             \
        statement = dest;                                                      \
    } while (0)

// =============================================================================
// 2. 简单标识符
// =============================================================================
#define COPY_IMPL_SIMPLE_NAME_IMPL(Name)                                       \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->dollorName = source->dollorName;                                 \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_SIMPLE_STRING_IMPL(Name)                                     \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->pragmaString = source->pragmaString;                             \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_VOID_INSTR_IMPL(Name)                                        \
    do {                                                                       \
        statement = new Name();                                                \
    } while (0)

// =============================================================================
// 3. 控制流指令
// =============================================================================
#define COPY_IMPL_BRANCH_IMPL(Name)                                            \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->qualifier = source->qualifier;                                   \
        dest->braTarget = source->braTarget;                                   \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_BARRIER_IMPL(Name)                                           \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->qualifier = source->qualifier;                                   \
        dest->barType = source->barType;                                       \
        dest->barId = source->barId;                                           \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_PREDICATE_PREFIX_IMPL(Name, _)                               \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->atLabelName = source->atLabelName;                               \
        dest->qualifier = source->qualifier;                                   \
        deepCopyOperandArray(dest->operands, source->operands);                \
        statement = dest;                                                      \
    } while (0)

// =============================================================================
// 4. 通用指令（GENERIC_INSTR）
// =============================================================================
#define COPY_IMPL_GENERIC_INSTR_IMPL(Name, OpCount)                            \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->qualifier = source->qualifier;                                   \
        deepCopyOperandArray(dest->operands, source->operands);                \
        statement = dest;                                                      \
    } while (0)

// =============================================================================
// 5. 特殊指令
// =============================================================================
#define COPY_IMPL_WMMA_INSTR_IMPL(Name)                                        \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->wmmaType = source->wmmaType;                                     \
        dest->qualifier = source->qualifier;                                   \
        deepCopyOperandArray(dest->operands, source->operands);                \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_ATOM_INSTR_IMPL(Name)                                        \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->qualifier = source->qualifier;                                   \
        deepCopyOperandArray(dest->operands, source->operands);                \
        dest->operandNum = source->operandNum;                                 \
        statement = dest;                                                      \
    } while (0)

#endif // STATEMENT_CONTEXT_H