// statement_context.h
#ifndef STATEMENT_CONTEXT_H
#define STATEMENT_CONTEXT_H

#include "operand_context.h"
#include "ptx_types.h"
#include <vector>

class StatementContext {
public:
    StatementType statementType;
    void *statement;

// =============================================================================
// 1. 操作数描述结构体
// =============================================================================
#define DEFINE_OPERAND_REG(Name)                                               \
    struct Name {                                                              \
        int regNum = 1;                                                        \
        std::vector<Qualifier> regDataType;                                    \
        std::string regName;                                                   \
    };

#define DEFINE_OPERAND_CONST(Name)                                             \
    struct Name {                                                              \
        int constAlign = 0;                                                    \
        int constSize = 1;                                                     \
        std::vector<Qualifier> constDataType;                                  \
        std::string constName;                                                 \
    };

#define DEFINE_OPERAND_MEMORY(Name)                                            \
    struct Name {                                                              \
        int align = 0;                                                         \
        int size = 1;                                                          \
        std::vector<Qualifier> dataType;                                       \
        std::string name;                                                      \
    };

    // =============================================================================
    // 2. 简单结构体
    // =============================================================================

#define DEFINE_SIMPLE_NAME(Name)                                               \
    struct Name {                                                              \
        std::string dollorName;                                                \
    };

#define DEFINE_SIMPLE_STRING(Name)                                             \
    struct Name {                                                              \
        std::string pragmaString;                                              \
    };

#define DEFINE_VOID_INSTR(Name)                                                \
    struct Name {};

    // =============================================================================
    // 3. 控制流结构体
    // =============================================================================

#define DEFINE_BRANCH(Name)                                                    \
    struct Name {                                                              \
        std::vector<Qualifier> braQualifier;                                   \
        std::string braTarget;                                                 \
    };

#define DEFINE_BARRIER(Name)                                                   \
    struct Name {                                                              \
        std::vector<Qualifier> braQualifier;                                   \
        std::string barType;                                                   \
        int barId;                                                             \
    };

#define DEFINE_PREDICATE_PREFIX(Name)                                          \
    struct Name {                                                              \
        OperandContext atPred;                                                 \
        std::string atLabelName;                                               \
    };

    // =============================================================================
    // 4. 通用指令结构体
    // =============================================================================

#define DEFINE_GENERIC_INSTR(Name, OpCount)                                    \
    struct Name {                                                              \
        static constexpr int op_count = OpCount;                               \
        std::vector<Qualifier> qualifier;                                      \
        OperandContext op[OpCount];                                            \
    };

#define DEFINE_WMMA_INSTR(Name, OpCount)                                       \
    struct Name {                                                              \
        static constexpr int op_count = OpCount;                               \
        WmmaType wmmaType;                                                     \
        std::vector<Qualifier> qualifier;                                      \
        OperandContext op[OpCount];                                            \
    };

#define DEFINE_ATOM_INSTR(Name, OpCount)                                       \
    struct Name {                                                              \
        static constexpr int op_count = OpCount;                               \
        std::vector<Qualifier> qualifier;                                      \
        OperandContext op[OpCount];                                            \
        int operandNum = 0;                                                    \
    };

    // =============================================================================
    // 5. 主分发宏
    // =============================================================================
// Overloads for kinds that don't use OpCount
#define DISPATCH_OPERAND_REG(Name, _) DEFINE_OPERAND_REG(Name)
#define DISPATCH_OPERAND_CONST(Name, _) DEFINE_OPERAND_CONST(Name)
#define DISPATCH_OPERAND_MEMORY(Name, _) DEFINE_OPERAND_MEMORY(Name)
#define DISPATCH_SIMPLE_NAME(Name, _) DEFINE_SIMPLE_NAME(Name)
#define DISPATCH_SIMPLE_STRING(Name, _) DEFINE_SIMPLE_STRING(Name)
#define DISPATCH_VOID_INSTR(Name, _) DEFINE_VOID_INSTR(Name)
#define DISPATCH_BRANCH(Name, _) DEFINE_BRANCH(Name)
#define DISPATCH_BARRIER(Name, _) DEFINE_BARRIER(Name)
#define DISPATCH_PREDICATE_PREFIX(Name, _) DEFINE_PREDICATE_PREFIX(Name)
#define DISPATCH_GENERIC_INSTR(Name, cnt) DEFINE_GENERIC_INSTR(Name, cnt)
#define DISPATCH_WMMA_INSTR(Name, cnt) DEFINE_WMMA_INSTR(Name, cnt)
#define DISPATCH_ATOM_INSTR(Name, cnt) DEFINE_ATOM_INSTR(Name, cnt)

    // =============================================================================
    // 6. 生成所有结构体
    // =============================================================================

#define DISPATCH_STRUCT(struct_kind, Name, OpCount)                            \
    DISPATCH_##struct_kind(Name, OpCount)

#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    DISPATCH_STRUCT(struct_kind, type_name, op_count)
#include "ptx_op.def"
#undef X

    StatementContext() : statementType(S_REG), statement(nullptr) {}
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
        dest->braQualifier = source->braQualifier;                             \
        dest->braTarget = source->braTarget;                                   \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_BARRIER_IMPL(Name)                                           \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->braQualifier = source->braQualifier;                             \
        dest->barType = source->barType;                                       \
        dest->barId = source->barId;                                           \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_PREDICATE_PREFIX_IMPL(Name)                                  \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->atPred = OperandContext(source->atPred);                         \
        dest->atLabelName = source->atLabelName;                               \
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
        deepCopyOperandArray(dest->op, source->op);                            \
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
        deepCopyOperandArray(dest->op, source->op);                            \
        statement = dest;                                                      \
    } while (0)

#define COPY_IMPL_ATOM_INSTR_IMPL(Name)                                        \
    do {                                                                       \
        auto source = static_cast<const Name *>(other.statement);              \
        auto dest = new Name();                                                \
        dest->qualifier = source->qualifier;                                   \
        deepCopyOperandArray(dest->op, source->op);                            \
        dest->operandNum = source->operandNum;                                 \
        statement = dest;                                                      \
    } while (0)

#endif // STATEMENT_CONTEXT_H