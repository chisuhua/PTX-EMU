// operand_context.h
#ifndef OPERAND_CONTEXT_H
#define OPERAND_CONTEXT_H

#include "ptx_types.h"
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

struct RegOperand {
    std::string name; // e.g., "r", "pred"
    int index = -1;   // e.g., 5 → %r5

    [[nodiscard]] std::string fullName() const {
        if (index < 0)
            return name;
        return name + std::to_string(index);
    }
};

struct VariableOperand {
    std::string name; // e.g., "my_var"
};

struct ImmOperand {
    std::string value; // textual form: "0x123", "3.14f", "-42"
};

struct Predicate {
    bool negated = false;
    std::shared_ptr<class OperandContext> source; // usually a Register
};

struct AddrOperand {
    enum class Space { CONST, PARAM, GLOBAL, LOCAL, SHARED };
    enum class OffsetType { IMMEDIATE, REGISTER };

    Space space;
    std::string baseSymbol; // e.g., "buf", "param_0"
    OffsetType offsetType;
    std::string immediateOffset; // if offsetType == IMMEDIATE
    std::shared_ptr<OperandContext> registerOffset; // if REGISTER

    std::string id; // optional unique ID for alias analysis
};

struct VecOperand {
    std::vector<OperandContext> elements;
};

class OperandContext {
public:
    using Data = std::variant<RegOperand, VariableOperand, ImmOperand,
                              VecOperand, AddrOperand, Predicate>;

    Data data;
    mutable void *operand_phy_addr = nullptr;

    // Constructors
    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, OperandContext>>>
    OperandContext(T &&t) : data(std::forward<T>(t)) {}
    OperandContext(const OperandContext &) = default;
    OperandContext &operator=(const OperandContext &) = default;
    OperandContext(OperandContext &&) = default;
    OperandContext &operator=(OperandContext &&) = default;
    void setPhyAddr(void *addr) { operand_phy_addr = addr; }

    [[nodiscard]] OperandKind kind() const {
        return std::visit(
            [](const auto &x) -> OperandKind {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, RegOperand>)
                    return OperandKind::REG;
                if constexpr (std::is_same_v<T, VariableOperand>)
                    return OperandKind::VAR;
                if constexpr (std::is_same_v<T, ImmOperand>)
                    return OperandKind::IMM;
                if constexpr (std::is_same_v<T, VecOperand>)
                    return OperandKind::VEC;
                if constexpr (std::is_same_v<T, AddrOperand>)
                    return OperandKind::ADDR;
                if constexpr (std::is_same_v<T, Predicate>)
                    return OperandKind::PRED;
                __builtin_unreachable();
            },
            data);
    }

    [[nodiscard]] std::string toString(int bytes = 0) const;
};

#endif // OPERAND_CONTEXT_H