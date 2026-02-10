#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/operand_context.h"
#include "ptx_ir/statement_context.h"
#include "utils/logger.h"
#include <cassert>
#include <sstream>
#include <string>
#include <algorithm>

// 定义通用的日志宏
#define PTX_ERROR(fmt, ...) PTX_ERROR_EMU(fmt, ##__VA_ARGS__)
#define PTX_DEBUG(fmt, ...) PTX_DEBUG_EMU(fmt, ##__VA_ARGS__)

// ============================================================================
// Helper Methods Implementation
// ============================================================================

Qualifier PtxVisitor::tokenToQualifier(antlr4::Token *token) {
    if (!token) return Qualifier::Q_UNKNOWN;
    
    std::string text = token->getText();
    
    // 使用宏来处理各种情况
#define X(enum_val, enum_name, str_val)                                        \
    if (text == std::string(str_val).substr(1)) {                              \
        return Qualifier::enum_val;                                            \
    }
    
#include "ptx_ir/ptx_qualifier.def"
#undef X
    
    return Qualifier::Q_UNKNOWN;
}

std::vector<Qualifier> PtxVisitor::extractQualifiersFromContext(antlr4::ParserRuleContext *ctx) {
    std::vector<Qualifier> qualifiers;
    if (!ctx) return qualifiers;
    
    // 遍历所有子节点，提取token
    for (auto child : ctx->children) {
        auto terminal = dynamic_cast<antlr4::tree::TerminalNode*>(child);
        if (terminal) {
            auto qual = tokenToQualifier(terminal->getSymbol());
            if (qual != Qualifier::Q_UNKNOWN) {
                qualifiers.push_back(qual);
            }
        }
    }
    
    return qualifiers;
}

OperandContext PtxVisitor::createOperandFromContext(ptxParser::OperandContext *ctx) {
    OperandContext oc;
    
    if (!ctx) return oc;
    
    // TODO: Implement proper operand creation based on new grammar
    // For now, create a simple placeholder
    oc.operandType = O_REG;
    auto reg = new OperandContext::REG();
    reg->regName = "TODO";
    reg->regIdx = 0;
    oc.operand = reg;
    
    return oc;
}

void PtxVisitor::processFunctionAttributes(ptxParser::FunctionAttributeContext *ctx) {
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

std::any PtxVisitor::visitPtxFile(ptxParser::PtxFileContext *ctx) {
    PTX_DEBUG("Visiting PTX file");
    
    // 访问所有声明
    for (auto decl : ctx->declaration()) {
        visit(decl);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitDeclaration(ptxParser::DeclarationContext *ctx) {
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
    else if (ctx->functionDecl()) {
        return visitFunctionDecl(ctx->functionDecl());
    }
    else if (ctx->abiPreserveDirective()) {
        return visitAbiPreserveDirective(ctx->abiPreserveDirective());
    }
    // TODO: Add extern function declaration handling
    
    return nullptr;
}

std::any PtxVisitor::visitVersionDirective(ptxParser::VersionDirectiveContext *ctx) {
    if (ctx->IMMEDIATE().size() >= 2) {
        this->ctx.ptxMajorVersion = extractIntFromToken(ctx->IMMEDIATE(0)->getSymbol());
        this->ctx.ptxMinorVersion = extractIntFromToken(ctx->IMMEDIATE(1)->getSymbol());
        PTX_DEBUG("PTX version: %d.%d", this->ctx.ptxMajorVersion, this->ctx.ptxMinorVersion);
    }
    return nullptr;
}

std::any PtxVisitor::visitTargetDirective(ptxParser::TargetDirectiveContext *ctx) {
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

std::any PtxVisitor::visitAddressSizeDirective(ptxParser::AddressSizeDirectiveContext *ctx) {
    if (ctx->IMMEDIATE()) {
        this->ctx.ptxAddressSize = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());
        PTX_DEBUG("Address size: %d", this->ctx.ptxAddressSize);
    }
    return nullptr;
}

std::any PtxVisitor::visitVariableDecl(ptxParser::VariableDeclContext *ctx) {
    StatementContext stmtCtx;
    stmtCtx.instructionText = ctx->getText();
    
    // TODO: Implement proper variable declaration parsing based on new grammar
    // For now, create a simple placeholder
    stmtCtx.type = S_REG;
    DeclarationInstr decl;
    decl.kind = DeclarationInstr::Kind::REG;
    decl.name = "TODO";
    decl.dataType = Qualifier::Q_U32;
    decl.array_size = 1;
    
    stmtCtx.data = decl;
    
    // 添加到适当的上下文
    if (currentKernel) {
        currentKernel->kernelStatements.push_back(stmtCtx);
    } else {
        this->ctx.ptxStatements.push_back(stmtCtx);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitFunctionDecl(ptxParser::FunctionDeclContext *ctx) {
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
    
    // TODO: Process parameters based on new grammar
    
    // TODO: Process function attributes
    
    // 访问函数体
    if (ctx->funcBody()) {
        visit(ctx->funcBody());
    }
    
    // 将kernel添加到上下文
    this->ctx.ptxKernels.push_back(*currentKernel);
    
    // 清理
    delete currentKernel;
    currentKernel = nullptr;
    
    return nullptr;
}

std::any PtxVisitor::visitAbiPreserveDirective(ptxParser::AbiPreserveDirectiveContext *ctx) {
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
    } else {
        this->ctx.ptxStatements.push_back(stmtCtx);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitInstructionList(ptxParser::InstructionListContext *ctx) {
    // 访问所有指令
    for (auto instr : ctx->instruction()) {
        visit(instr);
    }
    return nullptr;
}

std::any PtxVisitor::visitInstruction(ptxParser::InstructionContext *ctx) {
    // 根据指令类型分发到具体的访问器
    // 这里使用宏来减少重复代码
    
#define X(openum, opname, opstr, opcount, struct_kind)                         \
    if (ctx->opstr##Inst()) {                                                  \
        return visit##opstr##Inst(ctx->opstr##Inst());                         \
    }
    
#include "ptx_ir/ptx_op.def"
#undef X
    
    // TODO: Handle label instructions
    if (ctx->label()) {
        // Handle label
        StatementContext stmtCtx;
        stmtCtx.type = S_LABEL;
        LabelInstr label;
        label.labelName = ctx->label()->ID()->getText();
        stmtCtx.data = label;
        stmtCtx.instructionText = ctx->getText();
        
        if (currentKernel) {
            currentKernel->kernelStatements.push_back(stmtCtx);
        }
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

// ============================================================================
// Operand Visitors
// ============================================================================

std::any PtxVisitor::visitOperand(ptxParser::OperandContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitSpecialRegister(ptxParser::SpecialRegisterContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitRegister(ptxParser::RegisterContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitImmediate(ptxParser::ImmediateContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitAddress(ptxParser::AddressContext *ctx) {
    return nullptr;
}
