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
    if (!ctx) {
        // Return an empty OperandContext
        return OperandContext{ImmOperand{"0"}};
    }
    
    // 根据语法规则，operand可以是register, immediate, address, specialRegister, 或ID
    // 我们需要检查ctx中的具体内容
    // 由于ANTLR生成的代码，我们可以通过检查各个子规则来确定类型
    
    // 首先检查register
    if (ctx->register_()) {
        auto regCtx = ctx->register_();
        return visitRegister(regCtx).as<OperandContext>();
    }
    
    // 检查immediate
    if (ctx->immediate()) {
        auto immCtx = ctx->immediate();
        return visitImmediate(immCtx).as<OperandContext>();
    }
    
    // 检查address
    if (ctx->address()) {
        auto addrCtx = ctx->address();
        return visitAddress(addrCtx).as<OperandContext>();
    }
    
    // 检查specialRegister
    if (ctx->specialRegister()) {
        auto specRegCtx = ctx->specialRegister();
        return visitSpecialRegister(specRegCtx).as<OperandContext>();
    }
    
    // 检查ID（变量名）
    if (ctx->ID()) {
        VariableOperand var;
        var.name = ctx->ID()->getText();
        return OperandContext{var};
    }
    
    // 默认返回一个立即数0
    return OperandContext{ImmOperand{"0"}};
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
    return createOperandFromContext(ctx);
}

std::any PtxVisitor::visitSpecialRegister(ptxParser::SpecialRegisterContext *ctx) {
    // 特殊寄存器可以视为一种特殊的寄存器
    RegOperand reg;
    reg.name = ctx->getText();
    // 特殊寄存器通常没有索引
    reg.index = -1;
    return std::any{OperandContext{reg}};
}

std::any PtxVisitor::visitRegister(ptxParser::RegisterContext *ctx) {
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

std::any PtxVisitor::visitImmediate(ptxParser::ImmediateContext *ctx) {
    ImmOperand imm;
    if (ctx->MINUS()) {
        imm.value = "-" + ctx->IMMEDIATE()->getText();
    } else {
        imm.value = ctx->IMMEDIATE()->getText();
    }
    return std::any{OperandContext{imm}};
}

std::any PtxVisitor::visitAddress(ptxParser::AddressContext *ctx) {
    AddrOperand addr;
    
    // 默认空间
    addr.space = AddrOperand::Space::GLOBAL;
    
    // 获取地址表达式
    auto addrExprCtx = ctx->addressExpr();
    if (addrExprCtx) {
        // 获取基址操作数
        auto baseOperand = visitOperand(addrExprCtx->operand()).as<OperandContext>();
        
        // 检查基址操作数的类型
        if (baseOperand.kind() == OperandKind::VAR) {
            const auto& var = std::get<VariableOperand>(baseOperand.data);
            addr.baseSymbol = var.name;
        } else if (baseOperand.kind() == OperandKind::REG) {
            const auto& reg = std::get<RegOperand>(baseOperand.data);
            addr.baseSymbol = reg.fullName();
        }
        
        // 检查是否有偏移量
        if (addrExprCtx->immediate()) {
            addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;
            auto immCtx = addrExprCtx->immediate();
            if (immCtx->MINUS()) {
                addr.immediateOffset = "-" + immCtx->IMMEDIATE()->getText();
            } else {
                addr.immediateOffset = immCtx->IMMEDIATE()->getText();
            }
        } else if (addrExprCtx->PLUS()) {
            // 如果有PLUS但没有immediate，可能语法有变化
            // 这里简单处理
            addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;
            addr.immediateOffset = "0";
        } else {
            addr.offsetType = AddrOperand::OffsetType::IMMEDIATE;
            addr.immediateOffset = "0";
        }
    }
    
    return std::any{OperandContext{addr}};
}
