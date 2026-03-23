#include "ptx_ir/ptx_syntax_utils.h"
#include <cctype>
#include <string>
#include <cstddef>

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
    try {
        regOut.index = std::stoi(text.substr(split));
    } catch (...) {
        return false;
    }
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

    if (cleanText.find("tid") == 0 ||
        cleanText.find("ctaid") == 0 ||
        cleanText.find("gridDim") == 0 ||
        cleanText.find("blockDim") == 0 ||
        cleanText.find("nregs") == 0 ||
        cleanText.find("pc") == 0 ||
        cleanText.find("pm") == 0) {
        return true;
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
