#include "ptx_ir/operand_context.h"
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace ptxemu {
namespace ir {

std::string OperandContext::toString(int bytes) const {
    std::ostringstream oss;
    std::visit(
        [&oss, bytes](const auto &op) {
            using T = std::decay_t<decltype(op)>;

            if constexpr (std::is_same_v<T, RegOperand>) {
                oss << "%" << op.fullName();
            } else if constexpr (std::is_same_v<T, VariableOperand>) {
                oss << op.name;
            } else if constexpr (std::is_same_v<T, ImmOperand>) {
                oss << op.value;
            } else if constexpr (std::is_same_v<T, VecOperand>) {
                oss << "{";
                for (size_t i = 0; i < op.elements.size(); ++i) {
                    if (i > 0)
                        oss << ", ";
                    oss << op.elements[i].toString(bytes);
                }
                oss << "}";
            } else if constexpr (std::is_same_v<T, AddrOperand>) {
                const char *spaceStr = "";
                switch (op.space) {
                case AddrOperand::Space::CONST: spaceStr = "const"; break;
                case AddrOperand::Space::PARAM: spaceStr = "param"; break;
                case AddrOperand::Space::GLOBAL: spaceStr = "global"; break;
                case AddrOperand::Space::LOCAL: spaceStr = "local"; break;
                case AddrOperand::Space::SHARED: spaceStr = "shared"; break;
                }
                oss << "[";
                if (spaceStr[0])
                    oss << spaceStr << "::";
                oss << op.baseSymbol;

                if (op.offsetType == AddrOperand::OffsetType::IMMEDIATE &&
                    !op.immediateOffset.empty()) {
                    oss << " + " << op.immediateOffset;
                } else if (op.offsetType == AddrOperand::OffsetType::REGISTER &&
                           op.registerOffset) {
                    oss << " + " << op.registerOffset->toString();
                }
                oss << "]";
            } else if constexpr (std::is_same_v<T, Predicate>) {
                if (op.negated)
                    oss << "!";
                if (op.source)
                    oss << op.source->toString();
                else
                    oss << "%p<unknown>";
            } else {
                oss << "<invalid>";
            }
        },
        data);

    return oss.str();
}

}  // namespace ir
}  // namespace ptxemu