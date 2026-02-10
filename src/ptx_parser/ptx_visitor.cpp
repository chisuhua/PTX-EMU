#include "ptx_parser/ptx_visiter.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/ptx_context.h"
#include "utils/logger.h"
#include <cassert>
#include <sstream>
#include <string>

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

OperandContext PtxVisitor::createOperandFromContext(PtxParser::OperandContext *ctx) {
    OperandContext oc;
    
    if (!ctx) return oc;
    
    // 检查不同类型的操作数
    if (ctx->register_()) {
        auto regCtx = ctx->register_();
        oc.operandType = O_REG;
        auto reg = new OperandContext::REG();
        
        std::string regName = regCtx->ID()->getText();
        if (regCtx->DOLLAR()) {
            regName = "$" + regName;
        } else if (regCtx->PERCENT()) {
            regName = "%" + regName;
        }
        
        extractREG(regName, reg->regIdx, reg->regName);
        oc.operand = reg;
    }
    else if (ctx->immediate()) {
        auto immCtx = ctx->immediate();
        oc.operandType = O_IMM;
        auto imm = new OperandContext::IMM();
        
        std::string immVal = immCtx->IMMEDIATE()->getText();
        if (immCtx->MINUS()) {
            immVal = "-" + immVal;
        }
        imm->immVal = immVal;
        oc.operand = imm;
    }
    else if (ctx->specialRegister()) {
        auto specRegCtx = ctx->specialRegister();
        oc.operandType = O_REG;
        auto reg = new OperandContext::REG();
        
        // 构建特殊寄存器名称
        std::string regName;
        if (specRegCtx->TID()) regName = "%tid";
        else if (specRegCtx->NTID()) regName = "%ntid";
        else if (specRegCtx->CTAID()) regName = "%ctaid";
        else if (specRegCtx->NCTAID()) regName = "%nctaid";
        else if (specRegCtx->LANEID()) regName = "%laneid";
        else if (specRegCtx->CLOCK()) regName = "%clock";
        else if (specRegCtx->CLOCK64()) regName = "%clock64";
        else if (specRegCtx->LANEMASK_EQ()) regName = "%lanemask_eq";
        else if (specRegCtx->LANEMASK_LE()) regName = "%lanemask_le";
        else if (specRegCtx->LANEMASK_LT()) regName = "%lanemask_lt";
        else if (specRegCtx->LANEMASK_GE()) regName = "%lanemask_ge";
        else if (specRegCtx->LANEMASK_GT()) regName = "%lanemask_gt";
        else if (specRegCtx->PM0()) regName = "%pm0";
        else if (specRegCtx->PM1()) regName = "%pm1";
        else if (specRegCtx->PM2()) regName = "%pm2";
        else if (specRegCtx->PM3()) regName = "%pm3";
        else if (specRegCtx->PM4()) regName = "%pm4";
        else if (specRegCtx->PM5()) regName = "%pm5";
        else if (specRegCtx->PM6()) regName = "%pm6";
        else if (specRegCtx->PM7()) regName = "%pm7";
        else if (specRegCtx->SP()) regName = "%sp";
        
        // 添加组件后缀
        if (specRegCtx->component()) {
            auto compCtx = specRegCtx->component();
            if (compCtx->X_COMP()) regName += ".x";
            else if (compCtx->Y_COMP()) regName += ".y";
            else if (compCtx->Z_COMP()) regName += ".z";
            else if (compCtx->W_COMP()) regName += ".w";
        }
        
        extractREG(regName, reg->regIdx, reg->regName);
        oc.operand = reg;
    }
    else if (ctx->address()) {
        auto addrCtx = ctx->address();
        oc.operandType = O_FA;
        auto fa = new OperandContext::FA();
        
        // 这里简化处理，实际需要解析地址表达式
        fa->ID = "TODO"; // 需要从addressExpr中提取
        fa->offsetVal = "0";
        oc.operand = fa;
    }
    else if (ctx->ID()) {
        oc.operandType = O_VAR;
        auto var = new OperandContext::VAR();
        var->varName = ctx->ID()->getText();
        oc.operand = var;
    }
    
    return oc;
}

void PtxVisitor::processFunctionAttributes(PtxParser::FunctionAttributeContext *ctx) {
    if (!ctx || !currentKernel) return;
    
    if (ctx->MAXNREG()) {
        currentKernel->maxRegisters = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());
    }
    else if (ctx->REQNTID()) {
        auto threadDimCtx = ctx->threadDim();
        auto digits = threadDimCtx->IMMEDIATE();
        if (digits.size() >= 1) {
            currentKernel->reqntid.x = extractIntFromToken(digits[0]->getSymbol());
        }
        if (digits.size() >= 2) {
            currentKernel->reqntid.y = extractIntFromToken(digits[1]->getSymbol());
        }
        if (digits.size() >= 3) {
            currentKernel->reqntid.z = extractIntFromToken(digits[2]->getSymbol());
        }
    }
    else if (ctx->MINNCTAPERSM()) {
        currentKernel->minnctapersm = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());
    }
}

int PtxVisitor::extractIntFromToken(antlr4::Token *token) {
    if (!token) return 0;
    try {
        return std::stoi(token->getText());
    } catch (...) {
        return 0;
    }
}

// ============================================================================
// Top-level Visitors
// ============================================================================

std::any PtxVisitor::visitPtxFile(PtxParser::PtxFileContext *ctx) {
    PTX_DEBUG("Visiting PTX file");
    
    // 访问所有声明
    for (auto decl : ctx->declaration()) {
        visit(decl);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitDeclaration(PtxParser::DeclarationContext *ctx) {
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
    
    return nullptr;
}

std::any PtxVisitor::visitVersionDirective(PtxParser::VersionDirectiveContext *ctx) {
    if (ctx->IMMEDIATE().size() >= 2) {
        this->ctx.ptxMajorVersion = extractIntFromToken(ctx->IMMEDIATE(0)->getSymbol());
        this->ctx.ptxMinorVersion = extractIntFromToken(ctx->IMMEDIATE(1)->getSymbol());
        PTX_DEBUG("PTX version: %d.%d", this->ctx.ptxMajorVersion, this->ctx.ptxMinorVersion);
    }
    return nullptr;
}

std::any PtxVisitor::visitTargetDirective(PtxParser::TargetDirectiveContext *ctx) {
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

std::any PtxVisitor::visitAddressSizeDirective(PtxParser::AddressSizeDirectiveContext *ctx) {
    if (ctx->IMMEDIATE()) {
        this->ctx.ptxAddressSize = extractIntFromToken(ctx->IMMEDIATE()->getSymbol());
        PTX_DEBUG("Address size: %d", this->ctx.ptxAddressSize);
    }
    return nullptr;
}

std::any PtxVisitor::visitVariableDecl(PtxParser::VariableDeclContext *ctx) {
    StatementContext stmtCtx;
    stmtCtx.instructionText = ctx->getText();
    
    // 确定变量类型
    if (ctx->storageClass()->REG()) {
        stmtCtx.type = S_REG;
        DeclarationInstr decl;
        decl.kind = DeclarationInstr::Kind::REG;
        decl.name = ctx->ID()->getText();
        
        // 提取数据类型限定符
        if (ctx->typeSpecifier()) {
            auto qualifiers = extractQualifiersFromContext(ctx->typeSpecifier());
            if (!qualifiers.empty()) {
                decl.dataType = qualifiers[0]; // 取第一个作为数据类型
            }
        }
        
        // 对齐
        if (ctx->alignClause()) {
            decl.alignment = extractIntFromToken(ctx->alignClause()->IMMEDIATE()->getSymbol());
        }
        
        // 数组大小
        if (ctx->arraySize()) {
            auto sizes = ctx->arraySize()->IMMEDIATE();
            if (!sizes.empty()) {
                decl.array_size = extractIntFromToken(sizes[0]->getSymbol());
            }
        } else {
            decl.array_size = 1;
        }
        
        stmtCtx.data = decl;
    }
    else if (ctx->storageClass()->SHARED()) {
        stmtCtx.type = S_SHARED;
        DeclarationInstr decl;
        decl.kind = DeclarationInstr::Kind::SHARED;
        decl.name = ctx->ID()->getText();
        
        // 对齐
        if (ctx->alignClause()) {
            decl.alignment = extractIntFromToken(ctx->alignClause()->IMMEDIATE()->getSymbol());
        }
        
        // 数据类型
        if (ctx->typeSpecifier()) {
            auto qualifiers = extractQualifiersFromContext(ctx->typeSpecifier());
            if (!qualifiers.empty()) {
                decl.dataType = qualifiers[0];
            }
        }
        
        // 数组大小
        if (ctx->arraySize()) {
            auto sizes = ctx->arraySize()->IMMEDIATE();
            if (!sizes.empty()) {
                decl.array_size = extractIntFromToken(sizes[0]->getSymbol());
            }
        } else {
            decl.array_size = 1;
        }
        
        stmtCtx.data = decl;
    }
    else if (ctx->storageClass()->CONST()) {
        stmtCtx.type = S_CONST;
        DeclarationInstr decl;
        decl.kind = DeclarationInstr::Kind::CONST;
        decl.name = ctx->ID()->getText();
        
        // 对齐
        if (ctx->alignClause()) {
            decl.alignment = extractIntFromToken(ctx->alignClause()->IMMEDIATE()->getSymbol());
        }
        
        // 数据类型
        if (ctx->typeSpecifier()) {
            auto qualifiers = extractQualifiersFromContext(ctx->typeSpecifier());
            if (!qualifiers.empty()) {
                decl.dataType = qualifiers[0];
            }
        }
        
        // 数组大小
        if (ctx->arraySize()) {
            auto sizes = ctx->arraySize()->IMMEDIATE();
            if (!sizes.empty()) {
                decl.array_size = extractIntFromToken(sizes[0]->getSymbol());
            }
        } else {
            decl.array_size = 1;
        }
        
        stmtCtx.data = decl;
    }
    else if (ctx->storageClass()->GLOBAL()) {
        stmtCtx.type = S_GLOBAL;
        DeclarationInstr decl;
        decl.kind = DeclarationInstr::Kind::GLOBAL;
        decl.name = ctx->ID()->getText();
        
        // 对齐
        if (ctx->alignClause()) {
            decl.alignment = extractIntFromToken(ctx->alignClause()->IMMEDIATE()->getSymbol());
        } else {
            decl.alignment = 1;
        }
        
        // 数据类型
        if (ctx->typeSpecifier()) {
            auto qualifiers = extractQualifiersFromContext(ctx->typeSpecifier());
            if (!qualifiers.empty()) {
                decl.dataType = qualifiers[0];
            }
        }
        
        // 数组大小
        if (ctx->arraySize()) {
            auto sizes = ctx->arraySize()->IMMEDIATE();
            if (!sizes.empty()) {
                decl.array_size = extractIntFromToken(sizes[0]->getSymbol());
            }
        } else {
            decl.array_size = 1;
        }
        
        stmtCtx.data = decl;
    }
    else if (ctx->storageClass()->LOCAL()) {
        stmtCtx.type = S_LOCAL;
        DeclarationInstr decl;
        decl.kind = DeclarationInstr::Kind::LOCAL;
        decl.name = ctx->ID()->getText();
        
        // 对齐
        if (ctx->alignClause()) {
            decl.alignment = extractIntFromToken(ctx->alignClause()->IMMEDIATE()->getSymbol());
        }
        
        // 数据类型
        if (ctx->typeSpecifier()) {
            auto qualifiers = extractQualifiersFromContext(ctx->typeSpecifier());
            if (!qualifiers.empty()) {
                decl.dataType = qualifiers[0];
            }
        }
        
        // 数组大小
        if (ctx->arraySize()) {
            auto sizes = ctx->arraySize()->IMMEDIATE();
            if (!sizes.empty()) {
                decl.array_size = extractIntFromToken(sizes[0]->getSymbol());
            }
        } else {
            decl.array_size = 1;
        }
        
        stmtCtx.data = decl;
    }
    else if (ctx->storageClass()->PARAM()) {
        stmtCtx.type = S_PARAM;
        DeclarationInstr decl;
        decl.kind = DeclarationInstr::Kind::PARAM;
        decl.name = ctx->ID()->getText();
        
        // 数据类型
        if (ctx->typeSpecifier()) {
            auto qualifiers = extractQualifiersFromContext(ctx->typeSpecifier());
            if (!qualifiers.empty()) {
                decl.dataType = qualifiers[0];
            }
        }
        
        // 数组大小
        if (ctx->arraySize()) {
            auto sizes = ctx->arraySize()->IMMEDIATE();
            if (!sizes.empty()) {
                decl.array_size = extractIntFromToken(sizes[0]->getSymbol());
            }
        } else {
            decl.array_size = 1;
        }
        
        stmtCtx.data = decl;
    }
    
    // 添加到适当的上下文
    if (currentKernel) {
        currentKernel->kernelStatements.push_back(stmtCtx);
    } else {
        this->ctx.ptxStatements.push_back(stmtCtx);
    }
    
    return nullptr;
}

std::any PtxVisitor::visitFunctionDecl(PtxParser::FunctionDeclContext *ctx) {
    // 创建新的kernel上下文
    currentKernel = new KernelContext();
    
    // 函数名
    currentKernel->kernelName = ctx->functionHeader()->ID()->getText();
    
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
    
    // 处理参数
    if (ctx->functionHeader()->paramList()) {
        auto paramList = ctx->functionHeader()->paramList();
        for (auto paramCtx : paramList->paramDecl()) {
            ParamContext param;
            param.paramName = paramCtx->ID()->getText();
            
            // 提取参数类型
            if (paramCtx->typeSpecifier()) {
                auto qualifiers = extractQualifiersFromContext(paramCtx->typeSpecifier());
                param.paramTypes = qualifiers;
            }
            
            currentKernel->kernelParams.push_back(param);
        }
    }
    
    // 处理函数属性
    for (auto attrCtx : ctx->functionHeader()->functionAttribute()) {
        processFunctionAttributes(attrCtx);
    }
    
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

std::any PtxVisitor::visitAbiPreserveDirective(PtxParser::AbiPreserveDirectiveContext *ctx) {
    // ABI保留指令
    AbiDirective abiDir;
    std::string regName = ctx->ID()->getText();
    
    // 提取寄存器编号
    if (regName.find("r") == 0) {
        try {
            abiDir.regNumber = std::stoi(regName.substr(1));
        } catch (...) {
            abiDir.regNumber = 0;
        }
    }
    
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

std::any PtxVisitor::visitInstructionList(PtxParser::InstructionListContext *ctx) {
    // 访问所有指令
    for (auto instr : ctx->instruction()) {
        visit(instr);
    }
    return nullptr;
}

std::any PtxVisitor::visitInstruction(PtxParser::InstructionContext *ctx) {
    // 根据指令类型分发到具体的访问器
    // 这里使用宏来减少重复代码
    
#define X(openum, opname, opstr, opcount, struct_kind)                         \
    if (ctx->opstr##Inst()) {                                                  \
        return visit##opstr##Inst(ctx->opstr##Inst());                         \
    }
    
#include "ptx_ir/ptx_op.def"
#undef X
    
    return nullptr;
}

// ============================================================================
// Instruction Visitors (Generated using macros)
// ============================================================================

// 定义通用指令访问器的宏
#define VISITOR_GENERIC_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        instr.operands.push_back(oc);                                          \
    }                                                                          \
    instr.operandNum = std::min((int)operands.size(), opcount);                \
                                                                               \
    stmtCtx.data = instr;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
                                                                               \
    return nullptr;                                                            \
}

#define VISITOR_CALL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
#define VISITOR_OPERAND_REG(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PredicatePrefix pred;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    pred.qualifiers = qualifiers;                                              \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ReduxSyncInstr redux;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    redux.qualifiers = qualifiers;                                             \
    redux.operation = "";                                                      \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        redux.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = redux;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_MBARRIER_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    MbarrierInstr mbarrier;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    mbarrier.qualifiers = qualifiers;                                          \
    mbarrier.operation = "";                                                   \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        mbarrier.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = mbarrier;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_VOTE_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    VoteInstr vote;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    vote.qualifiers = qualifiers;                                              \
    vote.mode = "";                                                            \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        vote.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = vote;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SHFL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ShflInstr shfl;                                                            \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    shfl.qualifiers = qualifiers;                                              \
    shfl.mode = "";                                                            \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        shfl.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = shfl;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TEXTURE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TextureInstr tex;                                                          \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tex.qualifiers = qualifiers;                                               \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        tex.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = tex;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_SURFACE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    SurfaceInstr surf;                                                         \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    surf.qualifiers = qualifiers;                                              \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        surf.operands.push_back(oc);                                           \
    }                                                                          \
    stmtCtx.data = surf;                                                       \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_REDUCTION_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    ReductionInstr red;                                                        \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    red.qualifiers = qualifiers;                                               \
    red.operation = "";                                                        \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        red.operands.push_back(oc);                                            \
    }                                                                          \
    stmtCtx.data = red;                                                        \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_PREFETCH_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    PrefetchInstr prefetch;                                                    \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    prefetch.qualifiers = qualifiers;                                          \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        prefetch.operands.push_back(oc);                                       \
    }                                                                          \
    stmtCtx.data = prefetch;                                                   \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_CP_ASYNC_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    CpAsyncInstr cpAsync;                                                      \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    cpAsync.qualifiers = qualifiers;                                           \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        cpAsync.operands.push_back(oc);                                        \
    }                                                                          \
    stmtCtx.data = cpAsync;                                                    \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ASYNC_STORE(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AsyncStoreInstr asyncStore;                                                \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncStore.qualifiers = qualifiers;                                        \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncStore.operands.push_back(oc);                                     \
    }                                                                          \
    stmtCtx.data = asyncStore;                                                 \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ASYNC_REDUCE(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    AsyncReduceInstr asyncReduce;                                              \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    asyncReduce.qualifiers = qualifiers;                                       \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        asyncReduce.operands.push_back(oc);                                    \
    }                                                                          \
    stmtCtx.data = asyncReduce;                                                \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TCGEN_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TcgenInstr tcgen;                                                          \
    tcgen.opName = #opstr;                                                     \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tcgen.qualifiers = qualifiers;                                             \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        tcgen.operands.push_back(oc);                                          \
    }                                                                          \
    stmtCtx.data = tcgen;                                                      \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_TENSORMAP_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
    if (!currentKernel) return nullptr;                                        \
    StatementContext stmtCtx;                                                  \
    stmtCtx.instructionText = ctx->getText();                                  \
    stmtCtx.type = S_##opname;                                                 \
    TensormapInstr tensormap;                                                  \
    auto qualifiers = extractQualifiersFromContext(ctx);                       \
    tensormap.qualifiers = qualifiers;                                         \
    auto operands = ctx->operand();                                            \
    for (int i = 0; i < std::min((int)operands.size(), opcount); ++i) {        \
        auto oc = createOperandFromContext(operands[i]);                       \
        tensormap.operands.push_back(oc);                                      \
    }                                                                          \
    stmtCtx.data = tensormap;                                                  \
    currentKernel->kernelStatements.push_back(stmtCtx);                        \
    return nullptr;                                                            \
}

#define VISITOR_ABI_DIRECTIVE(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(PtxParser::opstr##InstContext *ctx) {  \
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

std::any PtxVisitor::visitOperand(PtxParser::OperandContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitSpecialRegister(PtxParser::SpecialRegisterContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitRegister(PtxParser::RegisterContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitImmediate(PtxParser::ImmediateContext *ctx) {
    return nullptr;
}

std::any PtxVisitor::visitAddress(PtxParser::AddressContext *ctx) {
    return nullptr;
}
