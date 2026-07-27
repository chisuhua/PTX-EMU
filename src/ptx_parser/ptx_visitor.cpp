#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "ptx_ir/statement_factory.h"
#include "ptxsim/ptx_exceptions.h"
#include "utils/logger.h"

using namespace ptxir::factory;
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

#include "ptx_visitor_operands.cpp"

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
    // function decl 由 visitFunctionDecl 直接处理（不在 declaration 上下文中）

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
    std::string text = ctx->getText();

    StatementType type = detectStorageClass(text);

    DeclarationInstr decl;
    decl.kind = DeclarationInstr::Kind::REG;

    if (ctx->ID()) {
        decl.name = ctx->ID()->getText();
    }
    if (decl.name.empty()) {
        // 禁止使用 "TODO" 作为标识符——会创建跨函数冲突的匿名声明。
        // Throw explicitly so callers see the parse failure.
        throw PTXParseException(
            "Variable declaration missing identifier (ctx text: '" +
            ctx->getText() + "')");
    }

    decl.dataType = detectDataTypeFromText(text);
    if (decl.dataType == Qualifier::Q_UNKNOWN) {
        decl.dataType = Qualifier::Q_U32;
    }

    decl.array_size = 1;

    bool is_extern = (text.find(".extern") != std::string::npos);

    size_t bracketPos = text.find('[');
    if (bracketPos != std::string::npos) {
        size_t closeBracket = text.find(']', bracketPos);
        if (closeBracket != std::string::npos) {
            std::string sizeStr = text.substr(bracketPos + 1, closeBracket - bracketPos - 1);

            if (sizeStr.empty() || is_extern) {
                decl.array_size = 0;
            } else {
                try {
                    decl.array_size = std::stoi(sizeStr);
                } catch (const std::invalid_argument&) {
                    decl.array_size = 0;
                } catch (const std::out_of_range&) {
                    decl.array_size = 0;
                }
            }
        }
    }

    if (ctx->initializer() && ctx->initializer()->initializerValue()) {
        auto initValue = ctx->initializer()->initializerValue();
        auto initList = initValue->initializerList();
        if (initList) {
            for (size_t i = 0; i < initList->initializerValue().size(); ++i) {
                auto iv = initList->initializerValue(i);
                if (iv && iv->IMMEDIATE()) {
                    try {
                        decl.initValues.push_back(std::stoi(iv->IMMEDIATE()->getText()));
                    } catch (...) {
                    }
                }
            }
        }
    }

    auto stmtCtx = makeDeclarationInstr(type, decl.kind, decl.name, decl.dataType, decl.array_size, text);
    stmtCtx.get<DeclarationInstr>().initValues = std::move(decl.initValues);

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
    
    // 函数名 - handle both functionHeader and extern function forms
    if (ctx->functionHeader() && ctx->functionHeader()->ID()) {
        currentKernel->kernelName = ctx->functionHeader()->ID()->getText();
    } else if (ctx->ID()) {
        // extern function form: .extern .func (.param ...) funcName
        currentKernel->kernelName = ctx->ID()->getText();
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

    // 访问函数体：先处理寄存器声明，再处理指令，保证寄存器可预分配
    if (ctx->funcBody()) {
        for (auto *regDeclCtx : ctx->funcBody()->regDecl()) {
            if (!regDeclCtx) {
                continue;
            }

            std::string text = regDeclCtx->getText();
            StatementType type = detectStorageClass(text);

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

            auto stmtCtx = makeDeclarationInstr(type, decl.kind, decl.name, decl.dataType, decl.array_size, text);
            currentKernel->kernelStatements.push_back(stmtCtx);
        }

        for (auto *varDeclCtx : ctx->funcBody()->variableDecl()) {
            if (!varDeclCtx) {
                continue;
            }

            std::string text = varDeclCtx->getText();
            StatementType type = varDeclCtx->storageClass() ? detectStorageClass(text) : S_REG;

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

            auto stmtCtx = makeDeclarationInstr(type, decl.kind, decl.name, decl.dataType, decl.array_size, text);
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
    std::string regName = ctx->ID() ? ctx->ID()->getText() : "";
    auto stmtCtx = ptxir::factory::makeAbiDirective(0, ctx->getText());

    if (currentKernel) {
        currentKernel->kernelStatements.push_back(stmtCtx);
        currentKernel->addAbiPreservedRegister(regName, 32);
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
// Tcgen05: 11 S_TCGEN05_* enums share a single visitTcgen05Inst handler
// (grammar has 1 tcgen05Inst rule). X-Macro expansion is a no-op here;
// the dispatch from visitTcgen05Inst to instr.op_kind happens inside
// PtxVisitor::visitTcgen05Inst (ptx_visitor_wmma.cpp:38-).
#define  VISITOR_TCGEN05_INSTR(openum, opstr, opname, opcount)  /* no-op */
#define  VISITOR_IMPL_TCGEN05_INSTR(openum, opstr, opname, opcount)  /* no-op */
#define  VISITOR_IMPL_TENSORMAP_INSTR(opstr, opname, instr_kind)
// 'tensor' instr_kind is used for tcgen05.* (Blackwell tensor cores).
// Like 'matrix' (WMMA), it has no per-instruction ANTLR grammar rule.
#define  VISITOR_IMPL_tensor(opstr, opname, instr_kind)  /* no-op: tcgen05 uses single tcgen05Inst rule */

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

    // Handle call.uni instruction (not in ptx_op.def)
    if (ctx->controlFlowInst() && ctx->controlFlowInst()->callUniInst()) {
        return visitCallUniInst(ctx->controlFlowInst()->callUniInst());
    }

    // Handle call instruction (includes CALL and CALL UNI via callInst grammar)
    if (ctx->controlFlowInst() && ctx->controlFlowInst()->callInst()) {
        return visitCallInst(ctx->controlFlowInst()->callInst());
    }

#define X(openum, opstr, opname, opcount, _, instr_kind)                         \
    VISITOR_IMPL_##instr_kind(opstr, opname, instr_kind)
    
#include "ptx_ir/ptx_op.def"
#undef X
    
    if (ctx->label()) {
        if (!currentKernel) return nullptr;

        std::string name;
        if (ctx->label()->ID()) {
            name = ctx->label()->ID()->getText();
        } else {
            return nullptr;
        }

        // 使用 makeLabelInstr 存储为 S_LABEL 类型，使 setupLabels 能正确识别
        auto stmtCtx = ptxir::factory::makeLabelInstr(name, ctx->getText());
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

// 包含 Blackwell tcgen05 visitor 实现 (ADR-0016)
#include "ptx_visitor_tcgen05.cpp"

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

// 包含 ABI 指令实现
#include "ptx_visitor_abi.cpp"

#define X(openum, opstr, opname, opcount, struct_kind, instr_kind)                         \
    VISITOR_##struct_kind(openum, opstr, opname, opcount);
    
#include "ptx_ir/ptx_op.def"
#undef X
 
// ============================================================================
// Operand Visitors
// ============================================================================
