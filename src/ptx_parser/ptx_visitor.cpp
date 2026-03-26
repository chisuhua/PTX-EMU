#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "utils/logger.h"
#include <cassert>
#include <cctype>
#include <sstream>
#include <string>
#include <algorithm>
#include <any>
#include <functional>

namespace {

Qualifier qualifierFromText(const std::string &text) {
#define X(enum_val, enum_name, str_val)                                        \
    if (text == str_val || text == std::string(str_val).substr(1)) {          \
        return Qualifier::enum_val;                                            \
    }
#include "ptx_ir/ptx_qualifier.def"
#undef X
    return Qualifier::Q_UNKNOWN;
}

int parseArraySizeFromRegDecl(
    ptxparser::ptxParser::RegDeclContext *regDeclCtx) {
    if (!regDeclCtx || !regDeclCtx->arraySize() ||
        regDeclCtx->arraySize()->IMMEDIATE().empty()) {
        return -1;
    }

    try {
        return std::stoi(regDeclCtx->arraySize()->IMMEDIATE(0)->getText());
    } catch (const std::invalid_argument&) {
        return -1;
    } catch (const std::out_of_range&) {
        return -1;
    }
}

// 检测存储类型（.param/.shared/.const/.local/.global）
StatementType detectStorageClass(const std::string &text) {
    if (text.find(".param") != std::string::npos) {
        return S_PARAM;
    }
    if (text.find(".shared") != std::string::npos) {
        return S_SHARED;
    }
    if (text.find(".const") != std::string::npos) {
        return S_CONST;
    }
    if (text.find(".local") != std::string::npos) {
        return S_LOCAL;
    }
    if (text.find(".global") != std::string::npos) {
        return S_GLOBAL;
    }
    return S_REG;
}

// 从文本检测数据类型（按长度从长到短排序，避免 .b16 误匹配 .b1）
Qualifier detectDataTypeFromText(const std::string &text) {
    // 按长度从长到短排序，避免前缀误匹配
    if (text.find(".f64") != std::string::npos) return Qualifier::Q_F64;
    if (text.find(".f32") != std::string::npos) return Qualifier::Q_F32;
    if (text.find(".s64") != std::string::npos) return Qualifier::Q_S64;
    if (text.find(".s32") != std::string::npos) return Qualifier::Q_S32;
    if (text.find(".s16") != std::string::npos) return Qualifier::Q_S16;
    if (text.find(".s8") != std::string::npos) return Qualifier::Q_S8;
    if (text.find(".u64") != std::string::npos) return Qualifier::Q_U64;
    if (text.find(".u32") != std::string::npos) return Qualifier::Q_U32;
    if (text.find(".u16") != std::string::npos) return Qualifier::Q_U16;
    if (text.find(".u8") != std::string::npos) return Qualifier::Q_U8;
    if (text.find(".b64") != std::string::npos) return Qualifier::Q_B64;
    if (text.find(".b32") != std::string::npos) return Qualifier::Q_B32;
    if (text.find(".b16") != std::string::npos) return Qualifier::Q_B16;
    if (text.find(".b8") != std::string::npos) return Qualifier::Q_B8;
    return Qualifier::Q_UNKNOWN;
}

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
    // Restrict to known PTX register families to avoid misclassifying symbols.
    if (prefix != "r" && prefix != "rd" && prefix != "rs" && prefix != "f" &&
        prefix != "fd" && prefix != "p" && prefix != "b" && prefix != "h") {
        return false;
    }

    regOut.name = prefix;
    regOut.index = std::stoi(text.substr(split));
    return true;
}

} // namespace

// 定义通用的日志宏
#define PTX_ERROR(fmt, ...) PTX_ERROR_EMU(fmt, ##__VA_ARGS__)
#define PTX_WARN(fmt, ...) PTX_WARN_EMU(fmt, ##__VA_ARGS__)
#define PTX_DEBUG(fmt, ...) PTX_DEBUG_EMU(fmt, ##__VA_ARGS__)

// 添加命名空间别名以简化代码
namespace ptx = ptxparser;

// ============================================================================
// Helper Methods Implementation
// ============================================================================

Qualifier PtxVisitor::tokenToQualifier(antlr4::Token *token) {
    if (!token) return Qualifier::Q_UNKNOWN;
    
    std::string text = token->getText();
    
    // 使用宏来处理各种情况
#define X(enum_val, enum_name, str_val)                                        \
    if (text == str_val || text == std::string(str_val).substr(1)) {           \
        return Qualifier::enum_val;                                            \
    }
    
#include "ptx_ir/ptx_qualifier.def"
#undef X
    
    return Qualifier::Q_UNKNOWN;
}

std::vector<Qualifier> PtxVisitor::extractQualifiersFromContext(antlr4::ParserRuleContext *ctx) {
    std::vector<Qualifier> qualifiers;
    if (!ctx) return qualifiers;

    std::function<void(antlr4::tree::ParseTree *)> visitNode =
        [&](antlr4::tree::ParseTree *node) {
            if (!node) {
                return;
            }

            auto *terminal = dynamic_cast<antlr4::tree::TerminalNode *>(node);
            if (terminal) {
                auto qual = tokenToQualifier(terminal->getSymbol());
                if (qual != Qualifier::Q_UNKNOWN) {
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

OperandContext PtxVisitor::createOperandFromContext(ptxparser::ptxParser::OperandContext *ctx) {
    if (!ctx) {
        PTX_WARN("createOperandFromContext received null context; defaulting to immediate 0");
        // Return an empty OperandContext
        return OperandContext{ImmOperand{"0"}};
    }
    
    // 根据语法规则，operand可以是register, immediate, address, specialRegister, 或ID
    // 我们需要检查ctx中的具体内容
    // 由于ANTLR生成的代码，我们可以通过检查各个子规则来确定类型
    
    // 首先检查register
    if (ctx->register_()) {
        auto regCtx = ctx->register_();
        auto anyResult = visitRegister(regCtx);
        try {
            return std::any_cast<OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast register operand: %s", e.what());
            return OperandContext{ImmOperand{"0"}};
        }
    }
    
    // 检查immediate
    if (ctx->immediate()) {
        auto immCtx = ctx->immediate();
        auto anyResult = visitImmediate(immCtx);
        try {
            return std::any_cast<OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast immediate operand: %s", e.what());
            return OperandContext{ImmOperand{"0"}};
        }
    }
    
    // 检查address
    if (ctx->address()) {
        auto addrCtx = ctx->address();
        auto anyResult = visitAddress(addrCtx);
        try {
            return std::any_cast<OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast address operand: %s", e.what());
            return OperandContext{ImmOperand{"0"}};
        }
    }
    
    // 检查specialRegister
    if (ctx->specialRegister()) {
        auto specRegCtx = ctx->specialRegister();
        auto anyResult = visitSpecialRegister(specRegCtx);
        try {
            return std::any_cast<OperandContext>(anyResult);
        } catch (const std::bad_any_cast& e) {
            PTX_ERROR("Failed to cast special register operand: %s", e.what());
            return OperandContext{ImmOperand{"0"}};
        }
    }
    
    // 检查ID（变量名）
    if (ctx->ID()) {
        const std::string text = ctx->ID()->getText();
        RegOperand reg;
        if (parseRegisterFromText(text, reg)) {
            return OperandContext{reg};
        }

        VariableOperand var;
        var.name = text;
        return OperandContext{var};
    }
    
    // 兜底：尽量保留原始文本，避免把符号名误降级为立即数0
    std::string raw = ctx->getText();
    if (!raw.empty()) {
        PTX_WARN("Fallback operand parsing path hit: raw=%s", raw.c_str());
        VariableOperand var;
        var.name = raw;
        return OperandContext{var};
    }

    // 默认返回一个立即数0
    PTX_WARN("Fallback operand parsing produced empty raw text; defaulting to immediate 0");
    return OperandContext{ImmOperand{"0"}};
}

void PtxVisitor::processFunctionAttributes(ptxparser::ptxParser::FunctionAttributeContext *ctx) {
    if (!ctx || !currentKernel) return;
    
    // TODO: Implement based on new grammar
    // For now, just log
    PTX_DEBUG("Processing function attributes");
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

size_t PtxVisitor::calculateTypeSize(const std::vector<Qualifier> &types) {
    // TODO: Implement proper type size calculation
    return 4; // Default to 4 bytes for now
}

// ============================================================================
// Top-level Visitors
// ============================================================================

std::any PtxVisitor::visitPtxFile(ptxparser::ptxParser::PtxFileContext *ctx) {
    PTX_DEBUG("Visiting PTX file");
    
    // 访问所有声明
    for (auto decl : ctx->declaration()) {
        visit(decl);
    }
    
    for (auto funcDecl : ctx->functionDecl()) {
        visitFunctionDecl(funcDecl);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitDeclaration(ptxparser::ptxParser::DeclarationContext *ctx) {
    // 根据声明类型分发到具体的访问器
    if (ctx->versionDirective()) {
        return visitVersionDirective(ctx->versionDirective());
    }
    else if (ctx->targetDirective()) {
        return visitTargetDirective(ctx->targetDirective());
    }
    else if (ctx->addressSizeDirective()) {
        return visitAddressSizeDirective(ctx->addressSizeDirective());
    }
    else if (ctx->variableDecl()) {
        return visitVariableDecl(ctx->variableDecl());
    }
    else if (ctx->abiPreserveDirective()) {
        return visitAbiPreserveDirective(ctx->abiPreserveDirective());
    }
    // TODO: Add extern function declaration handling
    
    return nullptr;
}

std::any PtxVisitor::visitVersionDirective(ptxparser::ptxParser::VersionDirectiveContext *ctx) {
    std::string ver = ctx->getText();
    // Remove "VERSION" and semicolon, parse the version number
    size_t start = 0;
    while (start < ver.size() && !isdigit(ver[start])) start++;
    size_t end = start;
    while (end < ver.size() && (isdigit(ver[end]) || ver[end] == '.')) end++;
    std::string verNum = ver.substr(start, end - start);
    
    size_t dotPos = verNum.find('.');
    if (dotPos != std::string::npos) {
        try {
            this->ctx.ptxMajorVersion = std::stoi(verNum.substr(0, dotPos));
            this->ctx.ptxMinorVersion = std::stoi(verNum.substr(dotPos + 1));
        } catch (...) {
            this->ctx.ptxMajorVersion = 7;
            this->ctx.ptxMinorVersion = 0;
        }
    } else {
        try {
            this->ctx.ptxMajorVersion = std::stoi(verNum);
            this->ctx.ptxMinorVersion = 0;
        } catch (...) {
            this->ctx.ptxMajorVersion = 7;
            this->ctx.ptxMinorVersion = 0;
        }
    }
    PTX_DEBUG("PTX version: %d.%d", this->ctx.ptxMajorVersion, this->ctx.ptxMinorVersion);
    return nullptr;
}

std::any PtxVisitor::visitTargetDirective(ptxparser::ptxParser::TargetDirectiveContext *ctx) {
    if (!ctx->SM_TARGET().empty()) {
        std::string target = ctx->SM_TARGET(0)->getText();
        // 提取sm_后面的数字
        if (target.length() >= 4 && target.substr(0, 3) == "sm_") {
            try {
                this->ctx.ptxTarget = std::stoi(target.substr(3));
            } catch (...) {
                this->ctx.ptxTarget = 0;
            }
        }
        PTX_DEBUG("PTX target: sm_%d", this->ctx.ptxTarget);
    }
    return nullptr;
}

std::any PtxVisitor::visitAddressSizeDirective(ptxparser::ptxParser::AddressSizeDirectiveContext *ctx) {
    if (ctx->IMMEDIATE()) {
        this->ctx.ptxAddressSize = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());
        PTX_DEBUG("Address size: %d", this->ctx.ptxAddressSize);
    }
    return nullptr;
}

std::any PtxVisitor::visitVariableDecl(ptxparser::ptxParser::VariableDeclContext *ctx) {
    StatementContext stmtCtx;
    stmtCtx.instructionText = ctx->getText();
    
    std::string text = ctx->getText();

    stmtCtx.type = detectStorageClass(text);
    
    DeclarationInstr decl;
    decl.kind = DeclarationInstr::Kind::REG;
    
    if (ctx->ID()) {
        decl.name = ctx->ID()->getText();
    }
    if (decl.name.empty()) {
        decl.name = "TODO";
    }
    
    decl.dataType = detectDataTypeFromText(text);
    if (decl.dataType == Qualifier::Q_UNKNOWN) {
        decl.dataType = Qualifier::Q_U32;
    }

    decl.array_size = 1;
    size_t bracketPos = text.find('[');
    if (bracketPos != std::string::npos) {
        size_t closeBracket = text.find(']', bracketPos);
        if (closeBracket != std::string::npos) {
            std::string sizeStr = text.substr(bracketPos + 1, closeBracket - bracketPos - 1);
            try {
                decl.array_size = std::stoi(sizeStr);
            } catch (const std::invalid_argument&) {
                decl.array_size = 1;
            } catch (const std::out_of_range&) {
                decl.array_size = 1;
            }
        }
    }
    
    stmtCtx.data = decl;
    
    if (currentKernel) {
        currentKernel->kernelStatements.push_back(stmtCtx);
    } else {
        this->ctx.ptxStatements.push_back(stmtCtx);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitFunctionDecl(ptxparser::ptxParser::FunctionDeclContext *ctx) {
    // 创建新的kernel上下文
    currentKernel = new KernelContext();
    
    // 函数名
    if (ctx->functionHeader()->ID()) {
        currentKernel->kernelName = ctx->functionHeader()->ID()->getText();
    }
    
    // 可见性
    if (ctx->visibility()) {
        if (ctx->visibility()->VISIBLE()) {
            currentKernel->ifVisibleKernel = true;
        } else if (ctx->visibility()->EXTERN()) {
            currentKernel->ifVisibleKernel = false; // extern函数
        }
    }
    
    // 入口函数
    if (ctx->ENTRY()) {
        currentKernel->ifEntryKernel = true;
    } else {
        currentKernel->ifEntryKernel = false;
    }
    
    if (ctx->functionHeader() && ctx->functionHeader()->paramList()) {
        for (auto *paramDecl : ctx->functionHeader()->paramList()->paramDecl()) {
            if (!paramDecl) {
                continue;
            }

            ParamContext param;
            if (paramDecl->ID()) {
                param.paramName = paramDecl->ID()->getText();
            }

            if (paramDecl->paramTypeSpec() &&
                paramDecl->paramTypeSpec()->typeSpecifier()) {
                auto q = qualifierFromText(
                    paramDecl->paramTypeSpec()->typeSpecifier()->getText());
                if (q != Qualifier::Q_UNKNOWN) {
                    param.paramTypes.push_back(q);
                    param.byteSize = Q2bytes(q);
                }

                if (paramDecl->paramTypeSpec()->PTR()) {
                    param.isPtr = true;
                    param.paramTypes.push_back(Qualifier::Q_PTR);
                    param.byteSize = this->ctx.ptxAddressSize == 32 ? 4 : 8;
                }

                if (paramDecl->paramTypeSpec()->alignClause() &&
                    paramDecl->paramTypeSpec()->alignClause()->IMMEDIATE()) {
                    try {
                        size_t alignValue = static_cast<size_t>(std::stoi(
                            paramDecl->paramTypeSpec()->alignClause()->IMMEDIATE()->getText()));
                        param.align = alignValue;
                    } catch (...) {
                    }
                }
            } else if (paramDecl->typeSpecifier()) {
                auto q = qualifierFromText(paramDecl->typeSpecifier()->getText());
                if (q != Qualifier::Q_UNKNOWN) {
                    param.paramTypes.push_back(q);
                    param.byteSize = Q2bytes(q);
                }
            }

            if (paramDecl->vectorSpec()) {
                std::string v = paramDecl->vectorSpec()->getText();
                if (v == ".v2") {
                    param.paramNum = 2;
                } else if (v == ".v4") {
                    param.paramNum = 4;
                }
            }

            // Handle array size in parameter declaration (e.g., .param .b8 x[2])
            if (paramDecl->arraySize() && !paramDecl->arraySize()->IMMEDIATE().empty()) {
                try {
                    int array_size = std::stoi(paramDecl->arraySize()->IMMEDIATE(0)->getText());
                    if (param.byteSize > 0) {
                        param.byteSize *= array_size;
                    }
                    param.paramNum = array_size;
                } catch (...) {
                    // Failed to parse array size, keep default
                }
            }

            // If byteSize is still 0 here, the parameter size is unknown.
            // Leave it as 0 so setupKernelArguments() can detect and reject it.
            if (param.byteSize == 0) {
                PTX_WARN("visitFunctionDecl: unknown byte size for param '%s'; "
                         "leaving as 0 for runtime rejection. "
                         "Ensure the parameter has an explicit type with a "
                         "known size (e.g. .u32, .u64, .b64).",
                         param.paramName.c_str());
            }
            param.paramAlign = static_cast<int>(param.effectiveAlignment());
            currentKernel->kernelParams.push_back(param);
        }
    }

    // TODO: Process function attributes

    // 访问函数体：先处理寄存器声明，再处理指令，保证寄存器可预分配
    if (ctx->funcBody()) {
        for (auto *regDeclCtx : ctx->funcBody()->regDecl()) {
            if (!regDeclCtx) {
                continue;
            }

            StatementContext stmtCtx;
            stmtCtx.instructionText = regDeclCtx->getText();

            std::string text = regDeclCtx->getText();
            stmtCtx.type = detectStorageClass(text);

            DeclarationInstr decl;
            decl.kind = DeclarationInstr::Kind::REG;
            decl.name = regDeclCtx->ID() ? regDeclCtx->ID()->getText() : "";
            decl.array_size = parseArraySizeFromRegDecl(regDeclCtx);
            decl.dataType = Qualifier::Q_U32;

            if (regDeclCtx->typeSpecifier()) {
                auto q = qualifierFromText(regDeclCtx->typeSpecifier()->getText());
                if (q != Qualifier::Q_UNKNOWN) {
                    decl.dataType = q;
                }
            }

            stmtCtx.data = decl;
            currentKernel->kernelStatements.push_back(stmtCtx);
        }
        
        for (auto *varDeclCtx : ctx->funcBody()->variableDecl()) {
            if (!varDeclCtx) {
                continue;
            }

            StatementContext stmtCtx;
            stmtCtx.instructionText = varDeclCtx->getText();

            std::string text = varDeclCtx->getText();
            stmtCtx.type = varDeclCtx->storageClass() ? detectStorageClass(text) : S_REG;

            DeclarationInstr decl;
            decl.kind = DeclarationInstr::Kind::REG;

            if (varDeclCtx->ID()) {
                decl.name = varDeclCtx->ID()->getText();
            }

            decl.array_size = 1;
            if (varDeclCtx->arraySize()) {
                for (auto *sizeCtx : varDeclCtx->arraySize()->IMMEDIATE()) {
                    if (sizeCtx) {
                        try {
                            decl.array_size = std::stoi(sizeCtx->getText());
                        } catch (const std::invalid_argument&) {
                            decl.array_size = 1;
                        } catch (const std::out_of_range&) {
                            decl.array_size = 1;
                        }
                        break;
                    }
                }
            }

            decl.dataType = Qualifier::Q_U32;
            if (varDeclCtx->typeSpecifier()) {
                auto q = qualifierFromText(varDeclCtx->typeSpecifier()->getText());
                if (q != Qualifier::Q_UNKNOWN) {
                    decl.dataType = q;
                }
            }

            stmtCtx.data = decl;
            currentKernel->kernelStatements.push_back(stmtCtx);
        }

        for (auto *instrCtx : ctx->funcBody()->instruction()) {
            if (instrCtx) {
                visit(instrCtx);
            }
        }
    }
    
    // 将kernel添加到上下文
    this->ctx.ptxKernels.push_back(*currentKernel);
    
    // 清理
    delete currentKernel;
    currentKernel = nullptr;
    
    return nullptr;
}

std::any PtxVisitor::visitAbiPreserveDirective(ptxparser::ptxParser::AbiPreserveDirectiveContext *ctx) {
    // ABI保留指令
    AbiDirective abiDir;
    abiDir.regNumber = 0; // TODO: Extract from context
    
    StatementContext stmtCtx;
    stmtCtx.type = S_ABI_PRESERVE;
    stmtCtx.instructionText = ctx->getText();
    stmtCtx.data = abiDir;
    
    // 添加到全局语句或当前kernel
    if (currentKernel) {
        currentKernel->kernelStatements.push_back(stmtCtx);
        
        // 同时添加到 KernelContext 的 abiPreservedRegisters
        // 解析寄存器名称: .abi_preserve %r15
        std::string regName = ctx->ID()->getText();
        currentKernel->addAbiPreservedRegister(regName, 32); // 默认32位，可后续改进
    } else {
        this->ctx.ptxStatements.push_back(stmtCtx);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitInstructionList(ptxparser::ptxParser::InstructionListContext *ctx) {
    // 访问所有指令
    for (auto instr : ctx->instruction()) {
        visit(instr);
    }
    return nullptr;
}

std::any PtxVisitor::visitInstruction(ptxparser::ptxParser::InstructionContext *ctx) {
    // 根据指令类型分发到具体的访问器
    // 这里使用宏来减少重复代码
    //
#define  VISITOR_IMPL_ABI_DIRECTIVE(opstr, opname, instr_kind)

#define  VISITOR_IMPL_structure(opstr, opname, instr_kind)

#define  VISITOR_IMPL_OPERAND_CONST(opstr, opname, instr_kind)
#define  VISITOR_IMPL_OPERAND_MEMORY(opstr, opname, instr_kind)
#define  VISITOR_IMPL_SIMPLE_NAME(opstr, opname, instr_kind)
#define  VISITOR_IMPL_SIMPLE_STRING(opstr, opname, instr_kind)
#define  VISITOR_IMPL_VOID_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_PREDICATE_PREFIX(opstr, opname, instr_kind)
#define  VISITOR_IMPL_BRANCH(opstr, opname, instr_kind)
#define  VISITOR_IMPL_WMMA_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_BARRIER(opstr, opname, instr_kind)
#define  VISITOR_IMPL_CALL_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_LABEL_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_MEMBAR_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_MBARRIER_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_FENCE_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_REDUX_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_VOTE_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_SHFL_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_TEXTURE_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_SURFACE_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_REDUCTION_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_PREFETCH_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_ASYNC_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_ASYNC_STORE(opstr, opname, instr_kind)
#define  VISITOR_IMPL_ASYNC_REDUCE(opstr, opname, instr_kind)
#define  VISITOR_IMPL_TCGEN_INSTR(opstr, opname, instr_kind)
#define  VISITOR_IMPL_TENSORMAP_INSTR(opstr, opname, instr_kind)

#define  VISITOR_IMPL_dataMovement(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) {                                                  \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst());                         \
    }
#define  VISITOR_IMPL_CP_ASYNC_INSTR(opstr, opname, instr_kind)

#define  VISITOR_IMPL_controlFlow(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_arithmetic(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_logical(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_warpLevel(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_textureSurface(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_reductionPrefetch(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_tcgen(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_Abi(opstr, opname, instr_kind) \

#define  VISITOR_IMPL_parallelSync(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_atomic(opstr, opname, instr_kind) \
    if (ctx->instr_kind##Inst() && ctx->instr_kind##Inst()->opstr##Inst()) { \
        return visit##opname##Inst(ctx->instr_kind##Inst()->opstr##Inst()); \
    }
#define  VISITOR_IMPL_matrix(opstr, opname, instr_kind)

#define X(openum, opstr, opname, opcount, _, instr_kind)                         \
    VISITOR_IMPL_##instr_kind(opstr, opname, instr_kind)
    
#include "ptx_ir/ptx_op.def"
#undef X
    
    if (ctx->label()) {
        if (!currentKernel) return nullptr;

        StatementContext stmtCtx;
        stmtCtx.instructionText = ctx->getText();
        stmtCtx.type = S_DOLLOR;

        DollarNameInstr dollar;
        if (ctx->label()->ID()) {
            dollar.name = ctx->label()->ID()->getText();
        } else {
            return nullptr;
        }
        stmtCtx.data = dollar;

        currentKernel->kernelStatements.push_back(stmtCtx);
        return nullptr;
    }
    
    return nullptr;
}

// ============================================================================
// 包含各个类别的指令访问器实现
// ============================================================================

// 包含通用指令实现
#include "ptx_visitor_generic.cpp"

// 包含原子指令实现
#include "ptx_visitor_atom.cpp"

// 包含调用指令实现
#include "ptx_visitor_call.cpp"

// 包含WMMA指令实现
#include "ptx_visitor_wmma.cpp"

// 包含分支指令实现
#include "ptx_visitor_branch.cpp"

// 包含屏障指令实现
#include "ptx_visitor_barrier.cpp"

// 包含简单指令实现
#include "ptx_visitor_simple.cpp"

// 包含特殊指令实现
#include "ptx_visitor_special.cpp"

// 包含Warp相关指令实现
#include "ptx_visitor_warp.cpp"

// 包含内存相关指令实现
#include "ptx_visitor_memory.cpp"

// 包含异步指令实现
#include "ptx_visitor_async.cpp"

// 包含Tensor相关指令实现
#include "ptx_visitor_tensor.cpp"

// 包含ABI指令实现
#include "ptx_visitor_abi.cpp"

#define X(openum, opstr, opname, opcount, struct_kind, instr_kind)                         \
    VISITOR_##struct_kind(openum, opstr, opname, opcount);
    
#include "ptx_ir/ptx_op.def"
#undef X
 
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
    return std::any{OperandContext{reg}};
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
    
    return std::any{OperandContext{reg}};
}

std::any PtxVisitor::visitImmediate(ptxparser::ptxParser::ImmediateContext *ctx) {
    ImmOperand imm;
    if (ctx->MINUS()) {
        imm.value = "-" + ctx->IMMEDIATE()->getText();
    } else {
        imm.value = ctx->IMMEDIATE()->getText();
    }
    return std::any{OperandContext{imm}};
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
            auto baseOperand = std::any_cast<OperandContext>(baseAny);
            // 检查基址操作数的类型
            if (baseOperand.kind() == OperandKind::VAR) {
                const auto& var = std::get<VariableOperand>(baseOperand.data);
                RegOperand reg;
                if (parseRegisterFromText(var.name, reg)) {
                    addr.baseSymbol = reg.fullName();
                    addr.id = reg.fullName();
                    addr.offsetType = AddrOperand::OffsetType::REGISTER;
                    addr.registerOffset =
                        std::make_shared<OperandContext>(OperandContext{reg});
                } else {
                    addr.baseSymbol = var.name;
                    addr.id = var.name;
                }
            } else if (baseOperand.kind() == OperandKind::REG) {
                const auto& reg = std::get<RegOperand>(baseOperand.data);
                addr.baseSymbol = reg.fullName();
                addr.id = reg.fullName();
                addr.offsetType = AddrOperand::OffsetType::REGISTER;
                addr.registerOffset =
                    std::make_shared<OperandContext>(baseOperand);
            } else if (baseOperand.kind() == OperandKind::ADDR) {
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
    
    return std::any{OperandContext{addr}};
}
