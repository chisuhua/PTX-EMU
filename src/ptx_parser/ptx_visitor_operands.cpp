// ============================================================================
// Operand / ptxemu::ir::Qualifier Helpers (ADR-0007, lessons-learned §13)
// ============================================================================

#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_factory.h"
#include "utils/logger.h"

using namespace ptxir::factory;

// File-local copy of parseRegisterFromText (originally in ptx_visitor.cpp).
namespace {

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
    regOut.index = std::stoi(text.substr(split));
    return true;
}

} // namespace

// Logging shims (parent ptx_visitor.cpp provides these via #define wrappers;
// this file needs them locally for self-containment).
#define PTX_ERROR(fmt, ...) PTX_ERROR_EMU(fmt, ##__VA_ARGS__)
#define PTX_WARN(fmt, ...) PTX_WARN_EMU(fmt, ##__VA_ARGS__)
#define PTX_DEBUG(fmt, ...) PTX_DEBUG_EMU(fmt, ##__VA_ARGS__)

ptxemu::ir::Qualifier PtxVisitor::tokenToQualifier(antlr4::Token *token) {
    if (!token) return ptxemu::ir::Qualifier::Q_UNKNOWN;

    std::string text = token->getText();

    // 使用宏来处理各种情况
#define X(enum_val, enum_name, str_val)                                        \
    if (text == str_val || text == std::string(str_val).substr(1)) {           \
        return ptxemu::ir::Qualifier::enum_val;                                            \
    }

#include "ptx_ir/ptx_qualifier.def"
#undef X

    return ptxemu::ir::Qualifier::Q_UNKNOWN;
}

std::vector<ptxemu::ir::Qualifier> PtxVisitor::extractQualifiersFromContext(antlr4::ParserRuleContext *ctx) {
    std::vector<ptxemu::ir::Qualifier> qualifiers;
    if (!ctx) return qualifiers;

    std::function<void(antlr4::tree::ParseTree *)> visitNode =
        [&](antlr4::tree::ParseTree *node) {
            if (!node) {
                return;
            }

            auto *terminal = dynamic_cast<antlr4::tree::TerminalNode *>(node);
            if (terminal) {
                auto qual = tokenToQualifier(terminal->getSymbol());
                if (qual != ptxemu::ir::Qualifier::Q_UNKNOWN) {
                    qualifiers.push_back(qual);
                }
                return;
            }

            const size_t childCount = node->children.size();
            for (size_t i = 0; i < childCount; ++i) {
                visitNode(node->children[i]);
            }
        };

    visitNode(ctx);

    return qualifiers;
}

ptxemu::ir::OperandContext PtxVisitor::createOperandFromContext(ptxparser::ptxParser::OperandContext *ctx) {
    if (!ctx) {
        PTX_WARN("createOperandFromContext received null context; defaulting to immediate 0");
        // Return an empty ptxemu::ir::OperandContext
        return ptxemu::ir::OperandContext{ImmOperand{"0"}};
    }

    // 根据语法规则，operand可以是register, immediate, address, specialRegister, 或ID
    // 我们需要检查ctx中的具体内容
    // 由于ANTLR生成的代码，我们可以通过检查各个子规则来确定类型

    // 首先检查register
    if (ctx->register_()) {
        auto regCtx = ctx->register_();
        auto anyResult = visitRegister(regCtx);
        try {
            return std::any_cast<ptxemu::ir::OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast register operand: %s", e.what());
            return ptxemu::ir::OperandContext{ImmOperand{"0"}};
        }
    }

    // 检查immediate
    if (ctx->immediate()) {
        auto immCtx = ctx->immediate();
        auto anyResult = visitImmediate(immCtx);
        try {
            return std::any_cast<ptxemu::ir::OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast immediate operand: %s", e.what());
            return ptxemu::ir::OperandContext{ImmOperand{"0"}};
        }
    }

    // 检查address
    if (ctx->address()) {
        auto addrCtx = ctx->address();
        auto anyResult = visitAddress(addrCtx);
        try {
            return std::any_cast<ptxemu::ir::OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast address operand: %s", e.what());
            return ptxemu::ir::OperandContext{ImmOperand{"0"}};
        }
    }

    // 检查specialRegister
    if (ctx->specialRegister()) {
        auto specRegCtx = ctx->specialRegister();
        auto anyResult = visitSpecialRegister(specRegCtx);
        try {
            return std::any_cast<ptxemu::ir::OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast special register operand: %s", e.what());
            return ptxemu::ir::OperandContext{ImmOperand{"0"}};
        }
    }

    // 检查vectorRegister (braced register list like {%r5, %r1, tmp})
    if (ctx->vectorRegister()) {
        auto vecCtx = ctx->vectorRegister();
        std::vector<ptxemu::ir::OperandContext> elements;
        for (auto virtRegCtx : vecCtx->virtRegister()) {
            // virtRegister is either a register_() or a bare ID
            if (virtRegCtx->register_()) {
                auto anyResult = visitRegister(virtRegCtx->register_());
                try {
                    elements.push_back(std::any_cast<ptxemu::ir::OperandContext>(anyResult));
                } catch (const std::bad_any_cast& e) {
                    PTX_ERROR("Failed to cast vector register element: %s", e.what());
                }
            } else if (virtRegCtx->ID()) {
                // Bare ID in vector register (e.g., {tmp, %r2})
                VariableOperand var;
                var.name = virtRegCtx->ID()->getText();
                elements.push_back(ptxemu::ir::OperandContext{var});
            } else {
                PTX_ERROR("virtRegister has neither register_() nor ID()");
            }
        }
        if (!elements.empty()) {
            VecOperand vecOp;
            vecOp.elements = std::move(elements);
            return ptxemu::ir::OperandContext{vecOp};
        }
    }

    // 检查ID（变量名）
    if (ctx->ID()) {
        const std::string text = ctx->ID()->getText();
        RegOperand reg;
        if (parseRegisterFromText(text, reg)) {
            return ptxemu::ir::OperandContext{reg};
        }

        VariableOperand var;
        var.name = text;
        return ptxemu::ir::OperandContext{var};
    }

    // 兜底：尽量保留原始文本，避免把符号名误降级为立即数0
    std::string raw = ctx->getText();
    if (!raw.empty()) {
        PTX_WARN("Fallback operand parsing path hit: raw=%s", raw.c_str());
        VariableOperand var;
        var.name = raw;
        return ptxemu::ir::OperandContext{var};
    }

    // 默认返回一个立即数0
    PTX_WARN("Fallback operand parsing produced empty raw text; defaulting to immediate 0");
    return ptxemu::ir::OperandContext{ImmOperand{"0"}};
}

int PtxVisitor::extractIntFromToken(antlr4::Token *token) {
    if (!token) return 0;
    try {
        return std::stoi(token->getText());
    } catch (...) {
        return 0;
    }
}

std::string PtxVisitor::extractStringFromToken(antlr4::Token *token) {
    if (!token) return "";
    return token->getText();
}

// ============================================================================
// Operand Visitors
// ============================================================================

std::any PtxVisitor::visitOperand(ptxparser::ptxParser::OperandContext *ctx) {
    return createOperandFromContext(ctx);
}

std::any PtxVisitor::visitSpecialRegister(ptxparser::ptxParser::SpecialRegisterContext *ctx) {
    // 特殊寄存器可以视为一种特殊的寄存器
    RegOperand reg;
    reg.name = ctx->getText();
    if (!reg.name.empty() && reg.name.front() == '%') {
        reg.name.erase(0, 1);
    }
    // 特殊寄存器通常没有索引
    reg.index = -1;
    return std::any{ptxemu::ir::OperandContext{reg}};
}

std::any PtxVisitor::visitRegister(ptxparser::ptxParser::RegisterContext *ctx) {
    RegOperand reg;

    // 寄存器名称：去掉$或%前缀
    std::string fullName = ctx->ID()->getText();

    // 提取寄存器类型和索引
    // 寄存器格式通常是：r0, pred0, %r1, $p2等
    // 首先去掉前缀字符
    std::string namePart = fullName;
    if (!namePart.empty() && (namePart[0] == '$' || namePart[0] == '%')) {
        namePart = namePart.substr(1);
    }

    // 分离字母部分和数字部分
    size_t i = 0;
    while (i < namePart.length() && std::isalpha(namePart[i])) {
        i++;
    }

    if (i > 0) {
        reg.name = namePart.substr(0, i);
        if (i < namePart.length()) {
            try {
                reg.index = std::stoi(namePart.substr(i));
            } catch (...) {
                reg.index = -1;
            }
        } else {
            reg.index = -1;
        }
    } else {
        reg.name = namePart;
        reg.index = -1;
    }

    return std::any{ptxemu::ir::OperandContext{reg}};
}

std::any PtxVisitor::visitImmediate(ptxparser::ptxParser::ImmediateContext *ctx) {
    ImmOperand imm;
    if (ctx->MINUS()) {
        imm.value = "-" + ctx->IMMEDIATE()->getText();
    } else {
        imm.value = ctx->IMMEDIATE()->getText();
    }
    return std::any{ptxemu::ir::OperandContext{imm}};
}

std::any PtxVisitor::visitAddress(ptxparser::ptxParser::AddressContext *ctx) {
    AddrOperand addr;

    // 默认空间
    addr.space = AddrOperand::Space::GLOBAL;

    addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;
    addr.immediateOffset = "0";

    // 获取地址表达式
    auto addrExprCtx = ctx->addressExpr();
    if (addrExprCtx) {
        // 获取基址操作数
        auto baseAny = visitOperand(addrExprCtx->operand());
        // 安全地提取OperandContext
        try {
            auto baseOperand = std::any_cast<ptxemu::ir::OperandContext>(baseAny);
            // 检查基址操作数的类型
            if (baseOperand.kind() == ptxemu::ir::OperandKind::VAR) {
                const auto& var = std::get<VariableOperand>(baseOperand.data);
                RegOperand reg;
                if (parseRegisterFromText(var.name, reg)) {
                    addr.baseSymbol = reg.fullName();
                    addr.id = reg.fullName();
                    addr.offsetType = AddrOperand::OffsetType::REGISTER;
                    addr.registerOffset =
                        std::make_shared<ptxemu::ir::OperandContext>(ptxemu::ir::OperandContext{reg});
                } else {
                    addr.baseSymbol = var.name;
                    addr.id = var.name;
                }
            } else if (baseOperand.kind() == ptxemu::ir::OperandKind::REG) {
                const auto& reg = std::get<RegOperand>(baseOperand.data);
                addr.baseSymbol = reg.fullName();
                addr.id = reg.fullName();
                addr.offsetType = AddrOperand::OffsetType::REGISTER;
                addr.registerOffset =
                    std::make_shared<ptxemu::ir::OperandContext>(baseOperand);
            } else if (baseOperand.kind() == ptxemu::ir::OperandKind::ADDR) {
                const auto &inner = std::get<AddrOperand>(baseOperand.data);
                addr.baseSymbol = inner.baseSymbol;
                addr.id = inner.id.empty() ? inner.baseSymbol : inner.id;
                addr.offsetType = inner.offsetType;
                addr.immediateOffset = inner.immediateOffset;
                addr.registerOffset = inner.registerOffset;
            }
        } catch (const std::bad_any_cast& e) {
            // 处理转换失败的情况
            PTX_ERROR("Failed to cast operand in address expression: %s", e.what());
        }

        // 检查是否有偏移量
        if (addrExprCtx->immediate()) {
            addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;
            addr.registerOffset.reset();
            auto immCtx = addrExprCtx->immediate();
            if (immCtx->MINUS()) {
                addr.immediateOffset = "-" + immCtx->IMMEDIATE()->getText();
            } else {
                addr.immediateOffset = immCtx->IMMEDIATE()->getText();
            }
        }
    }

    return std::any{ptxemu::ir::OperandContext{addr}};
}