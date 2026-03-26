#include "ptx_ir/ptx_syntax_utils.h"
#include <cctype>
#include <string>
#include <cstddef>
#include <cstdlib>
#include <unordered_set>

namespace ptx {
namespace syntax {

bool parseRegisterFromText(const std::string &raw, RegOperand &regOut) {
    if (raw.empty()) {
        return false;
    }

    std::string text = raw;
    if (!text.empty() && (text.front() == '%' || text.front() == '$')) {
        text.erase(text.begin());
    }

    size_t split = 0;
    while (split < text.size() &&
           std::isalpha(static_cast<unsigned char>(text[split]))) {
        ++split;
    }
    if (split == 0 || split >= text.size()) {
        return false;
    }

    for (size_t i = split; i < text.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(text[i]))) {
            return false;
        }
    }

    const std::string prefix = text.substr(0, split);
    if (prefix != "r" && prefix != "rd" && prefix != "rs" && prefix != "f" &&
        prefix != "fd" && prefix != "p" && prefix != "b" && prefix != "h") {
        return false;
    }

    regOut.name = prefix;
    // Use std::atoi since we already validated the string contains only digits
    regOut.index = std::atoi(text.substr(split).c_str());
    return true;
}

bool isSpecialRegister(const std::string &text) {
    if (text.empty()) {
        return false;
    }

    std::string cleanText = text;
    if (!cleanText.empty() && (cleanText.front() == '%' || cleanText.front() == '$')) {
        cleanText.erase(cleanText.begin());
    }

    // Use explicit special register names to avoid false positives
    // from substring matches (e.g., "tidxxx" should not match "tid")
    static const std::unordered_set<std::string> specialRegisters = {
        // Thread index registers
        "tid", "tid.x", "tid.y", "tid.z",
        // CTA index registers
        "ctaid", "ctaid.x", "ctaid.y", "ctaid.z",
        // Grid dimension registers
        "gridDim", "gridDim.x", "gridDim.y", "gridDim.z",
        // Block dimension registers
        "blockDim", "blockDim.x", "blockDim.y", "blockDim.z",
        // Other special registers
        "nregs", "pc", "pm0", "pm1", "pm2", "pm3"
    };

    // Also check for known prefixes with dot notation (e.g., "tid.x")
    if (cleanText.find('.') != std::string::npos) {
        return specialRegisters.count(cleanText) > 0;
    }

    // For names without dot, check if it's a known prefix
    if (cleanText == "tid" || cleanText == "ctaid" ||
        cleanText == "gridDim" || cleanText == "blockDim" ||
        cleanText == "nregs" || cleanText == "pc" ||
        cleanText.find("pm") == 0) {
        return specialRegisters.count(cleanText) > 0 || cleanText == "pm0" || cleanText == "pm1";
    }

    return false;
}

bool parseSpecialRegister(const std::string &text, std::string &nameOut, 
                          std::string &componentOut) {
    if (text.empty()) {
        return false;
    }

    std::string cleanText = text;
    if (!cleanText.empty() && (cleanText.front() == '%' || cleanText.front() == '$')) {
        cleanText.erase(cleanText.begin());
    }

    size_t dotPos = cleanText.find('.');
    if (dotPos != std::string::npos) {
        nameOut = cleanText.substr(0, dotPos);
        componentOut = cleanText.substr(dotPos + 1);
    } else {
        nameOut = cleanText;
        componentOut = "";
    }

    return isSpecialRegister(text);
}

} // namespace syntax
} // namespace ptx
