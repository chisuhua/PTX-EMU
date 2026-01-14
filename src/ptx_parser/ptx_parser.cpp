#include "ptx_parser/ptx_parser.h"
#include "ptx_ir/ptx_types.h"
#include "utils/logger.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// 定义通用的日志宏
#define PTX_ERROR(fmt, ...) PTX_ERROR_EMU(fmt, ##__VA_ARGS__)
#define PTX_DEBUG(fmt, ...) PTX_DEBUG_EMU(fmt, ##__VA_ARGS__)

// 实现extract_ptx_with_cuobjdump函数
std::string extract_ptx_with_cuobjdump(const std::string &executable_path) {
    // 创建临时文件存储PTX列表
    char ptx_list_cmd[1024];
    snprintf(ptx_list_cmd, 1024,
             "cuobjdump -lptx %s | cut -d : -f 2 | awk '{$1=$1}1' > "
             "__ptx_list_temp__",
             executable_path.c_str());

    if (system(ptx_list_cmd) != 0) {
        PTX_ERROR("Failed to execute: %s", ptx_list_cmd);
        return "";
    }

    // 读取PTX文件列表
    std::ifstream infile("__ptx_list_temp__");
    std::string ptx_file;
    std::string ptx_code;
    char cmd[1024];

    while (std::getline(infile, ptx_file)) {
        PTX_DEBUG("Extracting PTX file: %s", ptx_file.c_str());

        snprintf(cmd, 1024, "cuobjdump -xptx %s %s >/dev/null",
                 ptx_file.c_str(), executable_path.c_str());
        if (system(cmd) != 0) {
            PTX_ERROR("Failed to extract PTX: %s", cmd);
            continue;
        }

        std::ifstream if_ptx(ptx_file);
        std::ostringstream of_ptx;
        char ch;
        while (if_ptx.get(ch)) {
            of_ptx.put(ch);
        }
        // FIXME cubin have multiple ptx
        // ptx_code += of_ptx.str(); // 累加所有PTX代码
        ptx_code = of_ptx.str(); // 累加所有PTX代码

        // 清理临时PTX文件
        snprintf(cmd, 1024, "rm %s", ptx_file.c_str());
        system(cmd);
    }

    // 清理临时文件列表
    system("rm __ptx_list_temp__");

    return ptx_code;
}

void PtxListener::test_semantic() {
    PtxContext &ptx = ptxContext;
    std::printf(".version %d.%d\n", ptx.ptxMajorVersion, ptx.ptxMinorVersion);
    std::printf(".target sm_%d\n", ptx.ptxTarget);
    std::printf(".address_size %d\n", ptx.ptxAddressSize);
    std::printf("number of kernel %zu\n", ptx.ptxKernels.size());
    for (int i = 0; i < ptx.ptxKernels.size(); i++) {
        KernelContext &kernel = ptx.ptxKernels[i];
        if (kernel.ifEntryKernel) {
            std::printf(".entry ");
        }
        if (kernel.ifVisibleKernel) {
            std::printf(".visible ");
        }
        std::printf("%s\n", kernel.kernelName.c_str());
        std::printf("number of param %zu\n", kernel.kernelParams.size());
        for (int i = 0; i < kernel.kernelParams.size(); i++) {
            ParamContext &param = kernel.kernelParams[i];
            std::printf("%s: ", param.paramName.c_str());
            if (param.paramAlign != 0) {
                std::printf("align %d ", param.paramAlign);
            }
            std::printf("%s ", Q2s(param.paramType).c_str());
            if (param.paramNum != 0) {
                std::printf("arraySize %d ", param.paramNum);
            }
            std::printf("\n");
        }
        std::printf("number of statements %zu\n",
                    kernel.kernelStatements.size());
        for (int i = 0; i < kernel.kernelStatements.size(); i++) {
            StatementContext stat = kernel.kernelStatements[i];
            std::printf("%s %p\n", S2s(stat.statementType).c_str(),
                        stat.statement);
        }
    }
}

void PtxListener::fetchOperand(OperandContext &oc) {
    assert(op.size());
    oc.operand = op.front()->operand;
    oc.operandType = op.front()->operandType;
    op.pop();
}

void PtxListener::fetchOperand(std::vector<OperandContext> &oc) {
    assert(op.size());
    oc.emplace_back(
        OperandContext{op.front()->operandType, op.front()->operand});
    op.pop();
}

void PtxListener::enterAst(ptxParser::AstContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitAst(ptxParser::AstContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterVersionDes(ptxParser::VersionDesContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitVersionDes(ptxParser::VersionDesContext *ctx) {
    auto digits = ctx->DIGITS();
    ptxContext.ptxMajorVersion = stoi(digits[0]->getText());
    ptxContext.ptxMinorVersion = stoi(digits[1]->getText());
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterTargetDes(ptxParser::TargetDesContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitTargetDes(ptxParser::TargetDesContext *ctx) {
    /* assume target always be 'sm_xx' */
    auto id = ctx->ID();
    auto str = id->getText();
    assert(str.length() == 5);
    ptxContext.ptxTarget = stoi(id->getText().substr(3, 2));
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterAddressDes(ptxParser::AddressDesContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitAddressDes(ptxParser::AddressDesContext *ctx) {
    ptxContext.ptxAddressSize = stoi(ctx->DIGITS()->getText());
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterPerformanceTuning(
    ptxParser::PerformanceTuningContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitPerformanceTuning(
    ptxParser::PerformanceTuningContext *ctx) {
    /* init val */
    kernelContext->maxntid.x = 0;
    kernelContext->maxntid.y = 0;
    kernelContext->maxntid.z = 0;

    kernelContext->minnctapersm = 0;

    /* extrac val */
    if (ctx->MAXNTID()) {
        kernelContext->maxntid.x = stoi(ctx->DIGITS(0)->getText());
        kernelContext->maxntid.y = stoi(ctx->DIGITS(1)->getText());
        kernelContext->maxntid.z = stoi(ctx->DIGITS(2)->getText());
    } else if (ctx->MINNCTAPERSM()) {
        kernelContext->minnctapersm = stoi(ctx->DIGITS(0)->getText());
    } else
        assert(0 && "performancetuning not recognized!\n");

#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterKernels(ptxParser::KernelsContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitKernels(ptxParser::KernelsContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterKernel(ptxParser::KernelContext *ctx) {
    kernelContext = new KernelContext();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitKernel(ptxParser::KernelContext *ctx) {
    /* ID */
    kernelContext->kernelName = ctx->ID()->getText();

    /* entry */
    if (ctx->ENTRY()) {
        kernelContext->ifEntryKernel = true;
    } else {
        kernelContext->ifEntryKernel = false;
    }

    /* visible */
    if (ctx->VISIBLE()) {
        kernelContext->ifVisibleKernel = true;
    } else {
        kernelContext->ifVisibleKernel = false;
    }

    /* end of parsing kernel */
    ptxContext.ptxKernels.push_back(*kernelContext);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
    kernelContext = nullptr;
}

void PtxListener::enterQualifier(ptxParser::QualifierContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitQualifier(ptxParser::QualifierContext *ctx) {
// Use a macro to generate the if-else chain for all qualifiers
#define X(enum_val, enum_name, str_val)                                        \
    else if (ctx->enum_name()) {                                               \
        qualifier.push(Qualifier::enum_val);                                   \
    }

    if (0) { // Initial condition to start the if-else chain
        // No action for the initial false condition
    }
#include "ptx_ir/ptx_qualifier.def"
#undef X
    else {
        assert(0 && "some qualifier not recognized!\n");
    }

#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterParams(ptxParser::ParamsContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitParams(ptxParser::ParamsContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterParam(ptxParser::ParamContext *ctx) {
    paramContext = new ParamContext();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitParam(ptxParser::ParamContext *ctx) {
    int digit_idx = 0;

    /* ID */
    paramContext->paramName = ctx->ID()->getText();

    /* align */
    if (ctx->ALIGN()) {
        digit_idx++;
        paramContext->paramAlign = stoi(ctx->DIGITS(0)->getText());
    } else {
        paramContext->paramAlign = 0;
    }

    /* paramNum */
    if (ctx->LeftBracket()) {
        paramContext->paramNum = stoi(ctx->DIGITS(digit_idx)->getText());
    } else {
        paramContext->paramNum = 1;
    }

    /* qualifier */
    paramContext->paramType = qualifier.front();
    qualifier.pop();

    /* end of parsing param */
    kernelContext->kernelParams.push_back(*paramContext);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
    delete paramContext;
}

void PtxListener::enterCompoundStatement(
    ptxParser::CompoundStatementContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitCompoundStatement(
    ptxParser::CompoundStatementContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterStatements(ptxParser::StatementsContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitStatements(ptxParser::StatementsContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterStatement(ptxParser::StatementContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitStatement(ptxParser::StatementContext *ctx) {
    assert(op.size() == 0);
    // 获取当前语句的完整文本
    statementContext.instructionText = ctx->getText();
    statementContext.statementType = statementType;
    statementContext.statement = statement;
    if (kernelContext)
        kernelContext->kernelStatements.push_back(statementContext);
    else
        ptxContext.ptxStatements.push_back(statementContext); // const decl
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterRegStatement(ptxParser::RegStatementContext *ctx) {
    statement = new StatementContext::REG();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitRegStatement(ptxParser::RegStatementContext *ctx) {
    auto st = (StatementContext::REG *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->regDataType.push_back(qualifier.front());
        qualifier.pop();
    }

    /* reg */
    assert(op.size());
    assert(op.front()->operandType == O_REG);
    auto reg = *(OperandContext::REG *)op.front()->operand;
    st->regName = reg.regName;
    op.pop();

    /* digits */
    if (ctx->DIGITS()) {
        st->regNum = stoi(ctx->DIGITS()->getText());
    } else {
        st->regNum = 1;
    }

    /* end */
    statementType = S_REG;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterSharedStatement(ptxParser::SharedStatementContext *ctx) {
    statement = new StatementContext::SHARED();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitSharedStatement(ptxParser::SharedStatementContext *ctx) {
    auto st = (StatementContext::SHARED *)statement;

    /* align */
    st->align = stoi(ctx->DIGITS(0)->getText());

    /* qualifier */
    while (qualifier.size()) {
        st->dataType.push_back(qualifier.front());
        qualifier.pop();
    }

    /* ID */
    st->name = ctx->ID()->getText();

    /* size */
    if (ctx->DIGITS(1)) {
        st->size = stoi(ctx->DIGITS(1)->getText());
    } else {
        st->size = 1;
    }

    /* end */
    statementType = S_SHARED;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterConstStatement(ptxParser::ConstStatementContext *ctx) {
    statement = new StatementContext::CONST();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitConstStatement(ptxParser::ConstStatementContext *ctx) {
    auto st = (StatementContext::CONST *)statement;

    /* align */
    st->constAlign = stoi(ctx->DIGITS(0)->getText());

    /* qualifier */
    while (qualifier.size()) {
        st->constDataType.push_back(qualifier.front());
        qualifier.pop();
    }

    /* ID */
    st->constName = ctx->ID()->getText();

    /* size */
    if (ctx->DIGITS(1))
        st->constSize = stoi(ctx->DIGITS(1)->getText());
    else
        st->constSize = 1;

    /* end */
    statementType = S_CONST;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterLocalStatement(ptxParser::LocalStatementContext *ctx) {
    statement = new StatementContext::LOCAL();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitLocalStatement(ptxParser::LocalStatementContext *ctx) {
    auto st = (StatementContext::LOCAL *)statement;

    /* align */
    st->align = stoi(ctx->DIGITS(0)->getText());

    /* qualifier */
    while (qualifier.size()) {
        st->dataType.push_back(qualifier.front());
        qualifier.pop();
    }

    /* ID */
    st->name = ctx->ID()->getText();

    /* size */
    if (ctx->DIGITS(1))
        st->size = stoi(ctx->DIGITS(1)->getText());
    else
        st->size = 1;

    /* end */
    statementType = S_LOCAL;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterDollorStatement(ptxParser::DollorStatementContext *ctx) {
    statement = new StatementContext::DOLLOR();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitDollorStatement(ptxParser::DollorStatementContext *ctx) {
    auto st = (StatementContext::DOLLOR *)statement;

    /* ID */
    st->dollorName = ctx->ID()->getText();

    /* end */
    statementType = S_DOLLOR;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterAtStatement(ptxParser::AtStatementContext *ctx) {
    statement = new StatementContext::AT();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitAtStatement(ptxParser::AtStatementContext *ctx) {
    auto st = (StatementContext::AT *)statement;

    /* reg */
    fetchOperand(st->atPred);

    /* ID */
    st->atLabelName = ctx->ID()->getText();

    /* end */
    statementType = S_AT;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterPragmaStatement(ptxParser::PragmaStatementContext *ctx) {
    statement = new StatementContext::PRAGMA();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitPragmaStatement(ptxParser::PragmaStatementContext *ctx) {
    auto st = (StatementContext::PRAGMA *)statement;

    /* prama string */
    st->pragmaString = ctx->STRING()->getText();

    /* end */
    statementType = S_PRAGMA;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterRetStatement(ptxParser::RetStatementContext *ctx) {
    statement = new StatementContext::RET();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitRetStatement(ptxParser::RetStatementContext *ctx) {
    auto st = (StatementContext::RET *)statement;
    /* end */
    statementType = S_RET;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterBarStatement(ptxParser::BarStatementContext *ctx) {
    statement = new StatementContext::BAR();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitBarStatement(ptxParser::BarStatementContext *ctx) {
    auto st = (StatementContext::BAR *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->qualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* DIGITS */
    st->barId = stoi(ctx->DIGITS()->getText());

    /* end */
    statementType = S_BAR;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterBraStatement(ptxParser::BraStatementContext *ctx) {
    statement = new StatementContext::BRA();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitBraStatement(ptxParser::BraStatementContext *ctx) {
    auto st = (StatementContext::BRA *)statement;

    /* qualifier */
    if (qualifier.size()) {
        st->qualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* ID */
    st->braTarget = ctx->ID()->getText();

    /* end */
    statementType = S_BRA;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterWmmaStatement(ptxParser::WmmaStatementContext *ctx) {
    statement = new StatementContext::WMMA();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitWmmaStatement(ptxParser::WmmaStatementContext *ctx) {
    auto st = (StatementContext::WMMA *)statement;

    /* wmmatype & op */
    if (ctx->LOAD()) {
        st->wmmaType = WMMA_LOAD;
        for (int i = 0; i < 3; i++) {
            fetchOperand(st->operands);
        }
    } else if (ctx->STORE()) {
        st->wmmaType = WMMA_STORE;
        for (int i = 0; i < 3; i++) {
            fetchOperand(st->operands);
        }
    } else if (ctx->WMMA()) {
        st->wmmaType = WMMA_MMA;
        for (int i = 0; i < 4; i++) {
            fetchOperand(st->operands);
        }
    }

    /* qualifier */
    while (qualifier.size()) {
        st->qualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* end */
    statementType = S_WMMA;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterAtomStatement(ptxParser::AtomStatementContext *ctx) {
    statement = new StatementContext::ATOM();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitAtomStatement(ptxParser::AtomStatementContext *ctx) {
    auto st = (StatementContext::ATOM *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->qualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 or op4 */
    if (ctx->operandFour()) {
        for (int i = 0; i < 4; i++) {
            fetchOperand(st->operands[i]);
        }
        st->operandNum = 4;
    } else if (ctx->operandThree()) {
        for (int i = 0; i < 3; i++) {
            fetchOperand(st->operands[i]);
        }
        st->operandNum = 3;
    } else
        assert(0);

    /* end */
    statementType = S_ATOM;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterReg(ptxParser::RegContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitReg(ptxParser::RegContext *ctx) {
    OperandContext *o = new OperandContext();
    OperandContext::REG *r = new OperandContext::REG();
    // 获取完整的寄存器名称，包括可能的点号部分
    std::string regName = "";
    auto ids = ctx->ID();
    if (!ids.empty()) {
        regName = ids[0]->getText();
        if (ids.size() > 1) {
            // 添加DOT和第二个ID部分
            regName += "." + ids[1]->getText();
        }
    }
    extractREG(regName, r->regIdx, r->regName);
    o->operand = r;
    o->operandType = O_REG;
    op.push(o);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterVector(ptxParser::VectorContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitVector(ptxParser::VectorContext *ctx) {
    OperandContext *o = new OperandContext();
    OperandContext::VEC *v = new OperandContext::VEC();

    for (int i = 0; i < ctx->regi().size(); i++) {
        OperandContext oc;
        oc.operandType = O_REG;
        oc.operand = new OperandContext::REG();
        auto r = (OperandContext::REG *)oc.operand;

        // 获取完整的寄存器名称，包括可能的点号部分
        std::string regName = "";
        auto ids = ctx->regi(i)->ID();
        if (!ids.empty()) {
            regName = ids[0]->getText();
            if (ids.size() > 1) {
                // 添加DOT和第二个ID部分
                regName += "." + ids[1]->getText();
            }
        }
        extractREG(regName, r->regIdx, r->regName);
        v->vec.push_back(oc);
    }
    o->operand = v;
    o->operandType = O_VEC;
    op.push(o);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterFetchAddress(ptxParser::FetchAddressContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitFetchAddress(ptxParser::FetchAddressContext *ctx) {
    OperandContext *o = new OperandContext();
    OperandContext::FA *fa = new OperandContext::FA();

    /* base */
    if (ctx->ID()) {
        fa->ID = ctx->ID()->getText();
        fa->reg = nullptr;
    } else if (ctx->regi()) {
        // assume base not require regMinorName
        fa->reg = new OperandContext();
        OperandContext::REG *r = new OperandContext::REG();

        // 获取完整的寄存器名称，包括可能的点号部分
        std::string regName = "";
        auto ids = ctx->regi()->ID();
        if (!ids.empty()) {
            regName = ids[0]->getText();
            if (ids.size() > 1) {
                // 添加DOT和第二个ID部分
                regName += "." + ids[1]->getText();
            }
        }
        extractREG(regName, r->regIdx, r->regName);
        fa->reg->operandType = O_REG;
        fa->reg->operand = r;
    } else
        assert(0);

    /* offset */
    if (ctx->DIGITS()) {
        fa->offsetVal = ctx->DIGITS()->getText();
    } else {
        fa->offsetVal = "0";
    }

    o->operand = fa;
    o->operandType = O_FA;
    op.push(o);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterImm(ptxParser::ImmContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitImm(ptxParser::ImmContext *ctx) {
    OperandContext *o = new OperandContext();
    OperandContext::IMM *imm = new OperandContext::IMM();

    imm->immVal = ctx->DIGITS()->getText();
    o->operand = imm;
    o->operandType = O_IMM;
    op.push(o);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterVar(ptxParser::VarContext *ctx) {
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitVar(ptxParser::VarContext *ctx) {
    OperandContext *o = new OperandContext();
    OperandContext::VAR *var = new OperandContext::VAR();

    var->varName = ctx->ID()->getText();
    o->operand = var;
    o->operandType = O_VAR;
    op.push(o);
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
#define STATEMENT_OPERAND_REG(opstr, opname, opcount)

#define STATEMENT_OPERAND_CONST(opstr, opname, opcount)

#define STATEMENT_OPERAND_MEMORY(opstr, opname, opcount)

#define STATEMENT_SIMPLE_NAME(opstr, opname, opcount)

#define STATEMENT_SIMPLE_STRING(opstr, opname, opcount)

#define STATEMENT_VOID_INSTR(opstr, opname, opcount)

#define STATEMENT_PREDICATE_PREFIX(opstr, opname, opcount)

#define STATEMENT_BRANCH(opstr, opname, opcount)

#define STATEMENT_GENERIC_INSTR(opstr, opname, opcount)                        \
    void PtxListener::enter##opstr##Statement(                                 \
        ptxParser::opstr##StatementContext *ctx) {                             \
        statement = new StatementContext::opname();                            \
        LOG_FUNC();                                                            \
    }                                                                          \
                                                                               \
    void PtxListener::exit##opstr##Statement(                                  \
        ptxParser::opstr##StatementContext *ctx) {                             \
        auto st = static_cast<StatementContext::opname *>(statement);          \
                                                                               \
        /* qualifier */                                                        \
        while (!qualifier.empty()) {                                           \
            st->qualifier.push_back(qualifier.front());                        \
            qualifier.pop();                                                   \
        }                                                                      \
                                                                               \
        /* op2 */                                                              \
        for (int i = 0; i < opcount; ++i) {                                    \
            fetchOperand(st->operands);                                        \
        }                                                                      \
                                                                               \
        /* end */                                                              \
        statementType = S_##opname;                                            \
        LOG_FUNC();                                                            \
    }

#define STATEMENT_ATOM_INSTR(op_str, op_name, opcount)

#define STATEMENT_WMMA_INSTR(op_str, op_name, opcount)

#define STATEMENT_BARRIER(op_str, op_name, opcount)

#define X(openum, opname, opstr, opcount, struct_kind)                         \
    STATEMENT_##struct_kind(opstr, opname, opcount)
#include "ptx_ir/ptx_op.def"
#undef X