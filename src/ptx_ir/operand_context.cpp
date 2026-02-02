#include "ptx_ir/operand_context.h"
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>

std::string OperandContext::toString(int bytes) const {
    std::ostringstream oss;

    // Step 1: Print operand representation (e.g., %r1, { %r1, %r2 },
    // [shared::buf + %r3])
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
                    oss << op.elements[i].toString(
                        bytes); // propagate bytes for recursive value print
                }
                oss << "}";
            } else if constexpr (std::is_same_v<T, AddrOperand>) {
                const char *spaceStr = "";
                switch (op.space) {
                case AddrOperand::Space::CONST:
                    spaceStr = "const";
                    break;
                case AddrOperand::Space::PARAM:
                    spaceStr = "param";
                    break;
                case AddrOperand::Space::GLOBAL:
                    spaceStr = "global";
                    break;
                case AddrOperand::Space::LOCAL:
                    spaceStr = "local";
                    break;
                case AddrOperand::Space::SHARED:
                    spaceStr = "shared";
                    break;
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

    // Step 2: Append physical addrOperand and value (if available)
    if (operand_phy_addr != nullptr) {
        oss << " phy_addr:0x" << std::hex
            << reinterpret_cast<uintptr_t>(operand_phy_addr);

        // Only print value if 'bytes' is valid (1,2,4,8)
        if (bytes == 1) {
            uint8_t val = *static_cast<const uint8_t *>(operand_phy_addr);
            oss << " value:0x" << std::setfill('0') << std::setw(2)
                << static_cast<unsigned>(val);
        } else if (bytes == 2) {
            uint16_t val = *static_cast<const uint16_t *>(operand_phy_addr);
            oss << " value:0x" << std::setfill('0') << std::setw(4) << val;
        } else if (bytes == 4) {
            uint32_t val = *static_cast<const uint32_t *>(operand_phy_addr);
            oss << " value:0x" << std::setfill('0') << std::setw(8) << val;
        } else if (bytes == 8) {
            uint64_t val = *static_cast<const uint64_t *>(operand_phy_addr);
            oss << " value:0x" << std::setfill('0') << std::setw(16) << val;
        }
        oss << std::dec; // restore dec for future use
    }

    return oss.str();
}