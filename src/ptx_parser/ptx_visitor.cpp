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
// Instruction Visitors (Generated using macros)
// ============================================================================

// 定义通用指令访问器的宏
#define VISITOR_GENERIC_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    GenericInstr instr;                                                        \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取操作数 */                                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

#define VISITOR_ATOM_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    AtomInstr instr;                                                           \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取操作数 */                                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
    instr.operandNum = std::min((int)operands.size(), (int)opcount);           \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

#define VISITOR_CALL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    CallInstr instr;                                                           \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取操作数 */                                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

#define VISITOR_WMMA_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    WmmaInstr instr;                                                           \
                                                                               \
    /* 确定WMMA类型 */                                                         \
    if (ctx->LOAD()) {                                                         \
        instr.wmmaType = WMMA_LOAD;                                            \
    } else if (ctx->STORE()) {                                                 \
        instr.wmmaType = WMMA_STORE;                                           \
    } else if (ctx->WMMA()) {                                                  \
        instr.wmmaType = WMMA_MMA;                                             \
    }                                                                          \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取操作数 */                                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

#define VISITOR_BRANCH(opstr, opname, opcount)                                 \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    BranchInstr instr;                                                         \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取跳转目标 */                                                         \
    if (ctx->ID()) {                                                           \
        instr.target = ctx->ID()->getText();                                   \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

#define VISITOR_BARRIER(opstr, opname, opcount)                                \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
                                                                               \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
                                                                               \
    BarrierInstr instr;                                                        \
                                                                               \
    /* 提取限定符 */                                                           \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    instr.qualifiers = qualifiers;                                             \
                                                                               \
    /* 提取barrier ID */                                                       \
    if (ctx->IMMEDIATE()) {                                                    \
        instr.barId = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());      \
    }                                                                          \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

// 对于不需要特殊处理的指令类型，使用默认实现
#define VISITOR_OPERAND_REG(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DeclarationInstr decl;                                                     \
    decl.kind = DeclarationInstr::Kind::REG;                                   \
    decl.name = "";                                                            \
    stmtCtx.data = decl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_OPERAND_CONST(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DeclarationInstr decl;                                                     \
    decl.kind = DeclarationInstr::Kind::CONST;                                 \
    decl.name = "";                                                            \
    stmtCtx.data = decl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_OPERAND_MEMORY(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DeclarationInstr decl;                                                     \
    if (opstr == "SHARED") decl.kind = DeclarationInstr::Kind::SHARED;         \
    else if (opstr == "LOCAL") decl.kind = DeclarationInstr::Kind::LOCAL;      \
    else if (opstr == "GLOBAL") decl.kind = DeclarationInstr::Kind::GLOBAL;    \
    else if (opstr == "PARAM") decl.kind = DeclarationInstr::Kind::PARAM;      \
    else decl.kind = DeclarationInstr::Kind::GLOBAL;                           \
    decl.name = "";                                                            \
    stmtCtx.data = decl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SIMPLE_NAME(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    DollarNameInstr dollar;                                                    \
    dollar.name = "";                                                          \
    stmtCtx.data = dollar;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SIMPLE_STRING(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PragmaInstr pragma;                                                        \
    pragma.content = "";                                                       \
    stmtCtx.data = pragma;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_VOID_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    VoidInstr voidInstr;                                                       \
    stmtCtx.data = voidInstr;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_PREDICATE_PREFIX(opstr, opname, opcount)                       \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PredicatePrefix pred;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    pred.qualifiers = qualifiers;                                              \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        pred.operands.push_back(oc);                                           \
    }                                                                          \
    pred.target = "";                                                          \
    stmtCtx.data = pred;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// 添加其他缺失的指令类型访问器
#define VISITOR_MEMBAR_INSTR(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    MembarInstr membar;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    membar.qualifiers = qualifiers;                                            \
    membar.scope = "";                                                         \
    stmtCtx.data = membar;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_FENCE_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    FenceInstr fence;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    fence.qualifiers = qualifiers;                                             \
    fence.memoryOrder = "";                                                    \
    fence.scope = "";                                                          \
    stmtCtx.data = fence;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUX_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ReduxSyncInstr redux;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    redux.qualifiers = qualifiers;                                             \
    redux.operation = "";                                                      \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        redux.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = redux;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MBARRIER_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    MbarrierInstr mbarrier;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    mbarrier.qualifiers = qualifiers;                                          \
    mbarrier.operation = "";                                                   \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        mbarrier.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = mbarrier;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_VOTE_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    VoteInstr vote;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    vote.qualifiers = qualifiers;                                              \
    vote.mode = "";                                                            \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        vote.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = vote;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SHFL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ShflInstr shfl;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    shfl.qualifiers = qualifiers;                                              \
    shfl.mode = "";                                                            \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        shfl.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = shfl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TEXTURE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TextureInstr tex;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tex.qualifiers = qualifiers;                                               \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tex.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = tex;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SURFACE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    SurfaceInstr surf;                                                         \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    surf.qualifiers = qualifiers;                                              \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        surf.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = surf;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUCTION_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ReductionInstr red;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    red.qualifiers = qualifiers;                                               \
    red.operation = "";                                                        \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        red.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = red;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_PREFETCH_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PrefetchInstr prefetch;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    prefetch.qualifiers = qualifiers;                                          \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        prefetch.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = prefetch;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_CP_ASYNC_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    CpAsyncInstr cpAsync;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    cpAsync.qualifiers = qualifiers;                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        cpAsync.operands.push_back(oc);                                        \
    }                                                                          \
    stmtCtx.data = cpAsync;                                                    \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ASYNC_STORE(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AsyncStoreInstr asyncStore;                                                \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncStore.qualifiers = qualifiers;                                        \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncStore.operands.push_back(oc);                                     \
    }                                                                          \
    stmtCtx.data = asyncStore;                                                 \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ASYNC_REDUCE(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AsyncReduceInstr asyncReduce;                                              \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncReduce.qualifiers = qualifiers;                                       \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncReduce.operands.push_back(oc);                                    \
    }                                                                          \
    stmtCtx.data = asyncReduce;                                                \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TCGEN_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TcgenInstr tcgen;                                                          \
    tcgen.opName = #opstr;                                                     \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tcgen.qualifiers = qualifiers;                                             \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tcgen.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = tcgen;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TENSORMAP_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TensormapInstr tensormap;                                                  \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tensormap.qualifiers = qualifiers;                                         \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), (int)opcount); ++i) {   \
        auto oc = createOperandFromContext(operands[i]);                       \
        tensormap.operands.push_back(oc);                                      \
    }                                                                          \
    stmtCtx.data = tensormap;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ABI_DIRECTIVE(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AbiDirective abiDir;                                                       \
    abiDir.regNumber = 0;                                                      \
    stmtCtx.data = abiDir;                                                     \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

// 使用宏生成所有指令的访问器
#define X(openum, opname, opstr, opcount, struct_kind)                         \
    VISITOR_##struct_kind(opstr, opname, opcount)
#include "ptx_ir/ptx_op.def"
#undef X

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
