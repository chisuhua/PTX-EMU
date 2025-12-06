#include "ptx_parser/ptx_parser.h"
#include "ptx_ir/ptx_types.h"
#include <cassert>
#include <cstdio>
#include <queue>
#include <string>
#include <vector>

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
    if (ctx->U64()) {
        qualifier.push(Qualifier::Q_U64);
    } else if (ctx->U32()) {
        qualifier.push(Qualifier::Q_U32);
    } else if (ctx->U16()) {
        qualifier.push(Qualifier::Q_U16);
    } else if (ctx->U8()) {
        qualifier.push(Qualifier::Q_U8);
    } else if (ctx->PRED()) {
        qualifier.push(Qualifier::Q_PRED);
    } else if (ctx->B8()) {
        qualifier.push(Qualifier::Q_B8);
    } else if (ctx->B16()) {
        qualifier.push(Qualifier::Q_B16);
    } else if (ctx->B32()) {
        qualifier.push(Qualifier::Q_B32);
    } else if (ctx->B64()) {
        qualifier.push(Qualifier::Q_B64);
    } else if (ctx->F8()) {
        qualifier.push(Qualifier::Q_F8);
    } else if (ctx->F16()) {
        qualifier.push(Qualifier::Q_F16);
    } else if (ctx->F32()) {
        qualifier.push(Qualifier::Q_F32);
    } else if (ctx->F64()) {
        qualifier.push(Qualifier::Q_F64);
    } else if (ctx->S8()) {
        qualifier.push(Qualifier::Q_S8);
    } else if (ctx->S16()) {
        qualifier.push(Qualifier::Q_S16);
    } else if (ctx->S32()) {
        qualifier.push(Qualifier::Q_S32);
    } else if (ctx->S64()) {
        qualifier.push(Qualifier::Q_S64);
    } else if (ctx->V2()) {
        qualifier.push(Qualifier::Q_V2);
    } else if (ctx->V4()) {
        qualifier.push(Qualifier::Q_V4);
    } else if (ctx->CONST()) {
        qualifier.push(Qualifier::Q_CONST);
    } else if (ctx->PARAM()) {
        qualifier.push(Qualifier::Q_PARAM);
    } else if (ctx->GLOBAL()) {
        qualifier.push(Qualifier::Q_GLOBAL);
    } else if (ctx->LOCAL()) {
        qualifier.push(Qualifier::Q_LOCAL);
    } else if (ctx->SHARED()) {
        qualifier.push(Qualifier::Q_SHARED);
    } else if (ctx->GT()) {
        qualifier.push(Qualifier::Q_GT);
    } else if (ctx->GE()) {
        qualifier.push(Qualifier::Q_GE);
    } else if (ctx->EQ()) {
        qualifier.push(Qualifier::Q_EQ);
    } else if (ctx->NE()) {
        qualifier.push(Qualifier::Q_NE);
    } else if (ctx->LT()) {
        qualifier.push(Qualifier::Q_LT);
    } else if (ctx->TO()) {
        qualifier.push(Qualifier::Q_TO);
    } else if (ctx->WIDE()) {
        qualifier.push(Qualifier::Q_WIDE);
    } else if (ctx->SYNC()) {
        qualifier.push(Qualifier::Q_SYNC);
    } else if (ctx->LO()) {
        qualifier.push(Qualifier::Q_LO);
    } else if (ctx->HI()) {
        qualifier.push(Qualifier::Q_HI);
    } else if (ctx->UNI()) {
        qualifier.push(Qualifier::Q_UNI);
    } else if (ctx->RN()) {
        qualifier.push(Qualifier::Q_RN);
    } else if (ctx->A()) {
        qualifier.push(Qualifier::Q_A);
    } else if (ctx->B()) {
        qualifier.push(Qualifier::Q_B);
    } else if (ctx->D()) {
        qualifier.push(Qualifier::Q_D);
    } else if (ctx->ROW()) {
        qualifier.push(Qualifier::Q_ROW);
    } else if (ctx->ALIGNED()) {
        qualifier.push(Qualifier::Q_ALIGNED);
    } else if (ctx->M8N8K4()) {
        qualifier.push(Qualifier::Q_M8N8K4);
    } else if (ctx->M16N16K16()) {
        qualifier.push(Qualifier::Q_M16N16K16);
    } else if (ctx->NEU()) {
        qualifier.push(Qualifier::Q_NEU);
    } else if (ctx->NC()) {
        qualifier.push(Qualifier::Q_NC);
    } else if (ctx->FTZ()) {
        qualifier.push(Qualifier::Q_FTZ);
    } else if (ctx->APPROX()) {
        qualifier.push(Qualifier::Q_APPROX);
    } else if (ctx->LTU()) {
        qualifier.push(Qualifier::Q_LTU);
    } else if (ctx->LE()) {
        qualifier.push(Qualifier::Q_LE);
    } else if (ctx->GTU()) {
        qualifier.push(Qualifier::Q_GTU);
    } else if (ctx->LEU()) {
        qualifier.push(Qualifier::Q_LEU);
    } else if (ctx->DOTADD()) {
        qualifier.push(Qualifier::Q_DOTADD);
    } else if (ctx->GEU()) {
        qualifier.push(Qualifier::Q_GEU);
    } else if (ctx->RZI()) {
        qualifier.push(Qualifier::Q_RZI);
    } else if (ctx->DOTOR()) {
        qualifier.push(Qualifier::Q_DOTOR);
    } else if (ctx->SAT()) {
        qualifier.push(Qualifier::Q_SAT);
    } else
        assert(0 && "some qualifier not recognized!\n");

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
    st->sharedAlign = stoi(ctx->DIGITS(0)->getText());

    /* qualifier */
    while (qualifier.size()) {
        st->sharedDataType.push_back(qualifier.front());
        qualifier.pop();
    }

    /* ID */
    st->sharedName = ctx->ID()->getText();

    /* size */
    if (ctx->DIGITS(1)) {
        st->sharedSize = stoi(ctx->DIGITS(1)->getText());
    } else {
        st->sharedSize = 1;
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
    st->localAlign = stoi(ctx->DIGITS(0)->getText());

    /* qualifier */
    while (qualifier.size()) {
        st->localDataType.push_back(qualifier.front());
        qualifier.pop();
    }

    /* ID */
    st->localName = ctx->ID()->getText();

    /* size */
    if (ctx->DIGITS(1))
        st->localSize = stoi(ctx->DIGITS(1)->getText());
    else
        st->localSize = 1;

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
        st->braQualifier.push_back(qualifier.front());
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
        st->braQualifier.push_back(qualifier.front());
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

void PtxListener::enterRcpStatement(ptxParser::RcpStatementContext *ctx) {
    statement = new StatementContext::RCP();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitRcpStatement(ptxParser::RcpStatementContext *ctx) {
    auto st = (StatementContext::RCP *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->rcpQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->rcpOp[i]);
    }

    /* end */
    statementType = S_RCP;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterLdStatement(ptxParser::LdStatementContext *ctx) {
    statement = new StatementContext::LD();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitLdStatement(ptxParser::LdStatementContext *ctx) {
    auto st = (StatementContext::LD *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->ldQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->ldOp[i]);
    }

    /* end */
    statementType = S_LD;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterMovStatement(ptxParser::MovStatementContext *ctx) {
    statement = new StatementContext::MOV();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitMovStatement(ptxParser::MovStatementContext *ctx) {
    auto st = (StatementContext::MOV *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->movQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->movOp[i]);
    }

    /* end */
    statementType = S_MOV;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterSetpStatement(ptxParser::SetpStatementContext *ctx) {
    statement = new StatementContext::SETP();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitSetpStatement(ptxParser::SetpStatementContext *ctx) {
    auto st = (StatementContext::SETP *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->setpQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->setpOp[i]);
    }

    /* end */
    statementType = S_SETP;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterCvtaStatement(ptxParser::CvtaStatementContext *ctx) {
    statement = new StatementContext::CVTA();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitCvtaStatement(ptxParser::CvtaStatementContext *ctx) {
    auto st = (StatementContext::CVTA *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->cvtaQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->cvtaOp[i]);
    }

    /* end */
    statementType = S_CVTA;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterCvtStatement(ptxParser::CvtStatementContext *ctx) {
    statement = new StatementContext::CVT();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitCvtStatement(ptxParser::CvtStatementContext *ctx) {
    auto st = (StatementContext::CVT *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->cvtQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->cvtOp[i]);
    }

    /* end */
    statementType = S_CVT;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterMulStatement(ptxParser::MulStatementContext *ctx) {
    statement = new StatementContext::MUL();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitMulStatement(ptxParser::MulStatementContext *ctx) {
    auto st = (StatementContext::MUL *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->mulQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->mulOp[i]);
    }

    /* end */
    statementType = S_MUL;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterDivStatement(ptxParser::DivStatementContext *ctx) {
    statement = new StatementContext::DIV();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitDivStatement(ptxParser::DivStatementContext *ctx) {
    auto st = (StatementContext::DIV *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->divQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->divOp[i]);
    }

    /* end */
    statementType = S_DIV;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterSubStatement(ptxParser::SubStatementContext *ctx) {
    statement = new StatementContext::SUB();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitSubStatement(ptxParser::SubStatementContext *ctx) {
    auto st = (StatementContext::SUB *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->subQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->subOp[i]);
    }

    /* end */
    statementType = S_SUB;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterAddStatement(ptxParser::AddStatementContext *ctx) {
    statement = new StatementContext::ADD();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitAddStatement(ptxParser::AddStatementContext *ctx) {
    auto st = (StatementContext::ADD *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->addQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->addOp[i]);
    }

    /* end */
    statementType = S_ADD;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterShlStatement(ptxParser::ShlStatementContext *ctx) {
    statement = new StatementContext::SHL();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitShlStatement(ptxParser::ShlStatementContext *ctx) {
    auto st = (StatementContext::SHL *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->shlQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->shlOp[i]);
    }

    /* end */
    statementType = S_SHL;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterShrStatement(ptxParser::ShrStatementContext *ctx) {
    statement = new StatementContext::SHR();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitShrStatement(ptxParser::ShrStatementContext *ctx) {
    auto st = (StatementContext::SHR *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->shrQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->shrOp[i]);
    }

    /* end */
    statementType = S_SHR;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterMaxStatement(ptxParser::MaxStatementContext *ctx) {
    statement = new StatementContext::MAX();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitMaxStatement(ptxParser::MaxStatementContext *ctx) {
    auto st = (StatementContext::MAX *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->maxQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->maxOp[i]);
    }

    /* end */
    statementType = S_MAX;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterMinStatement(ptxParser::MinStatementContext *ctx) {
    statement = new StatementContext::MIN();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitMinStatement(ptxParser::MinStatementContext *ctx) {
    auto st = (StatementContext::MIN *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->minQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->minOp[i]);
    }

    /* end */
    statementType = S_MIN;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterAndStatement(ptxParser::AndStatementContext *ctx) {
    statement = new StatementContext::AND();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitAndStatement(ptxParser::AndStatementContext *ctx) {
    auto st = (StatementContext::AND *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->andQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->andOp[i]);
    }

    /* end */
    statementType = S_AND;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterOrStatement(ptxParser::OrStatementContext *ctx) {
    statement = new StatementContext::OR();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitOrStatement(ptxParser::OrStatementContext *ctx) {
    auto st = (StatementContext::OR *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->orQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->orOp[i]);
    }

    /* end */
    statementType = S_OR;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterStStatement(ptxParser::StStatementContext *ctx) {
    statement = new StatementContext::ST();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitStStatement(ptxParser::StStatementContext *ctx) {
    auto st = (StatementContext::ST *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->stQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->stOp[i]);
    }

    /* end */
    statementType = S_ST;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterSelpStatement(ptxParser::SelpStatementContext *ctx) {
    statement = new StatementContext::SELP();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitSelpStatement(ptxParser::SelpStatementContext *ctx) {
    auto st = (StatementContext::SELP *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->selpQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op4 */
    for (int i = 0; i < 4; i++) {
        fetchOperand(st->selpOp[i]);
    }

    /* end */
    statementType = S_SELP;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterMadStatement(ptxParser::MadStatementContext *ctx) {
    statement = new StatementContext::MAD();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitMadStatement(ptxParser::MadStatementContext *ctx) {
    auto st = (StatementContext::MAD *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->madQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op4 */
    for (int i = 0; i < 4; i++) {
        fetchOperand(st->madOp[i]);
    }

    /* end */
    statementType = S_MAD;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterFmaStatement(ptxParser::FmaStatementContext *ctx) {
    statement = new StatementContext::FMA();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitFmaStatement(ptxParser::FmaStatementContext *ctx) {
    auto st = (StatementContext::FMA *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->fmaQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op4 */
    for (int i = 0; i < 4; i++) {
        fetchOperand(st->fmaOp[i]);
    }

    /* end */
    statementType = S_FMA;
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
            fetchOperand(st->wmmaOp[i]);
        }
    } else if (ctx->STORE()) {
        st->wmmaType = WMMA_STORE;
        for (int i = 0; i < 3; i++) {
            fetchOperand(st->wmmaOp[i]);
        }
    } else if (ctx->WMMA()) {
        st->wmmaType = WMMA_MMA;
        for (int i = 0; i < 4; i++) {
            fetchOperand(st->wmmaOp[i]);
        }
    }

    /* qualifier */
    while (qualifier.size()) {
        st->wmmaQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* end */
    statementType = S_WMMA;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterNegStatement(ptxParser::NegStatementContext *ctx) {
    statement = new StatementContext::NEG();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitNegStatement(ptxParser::NegStatementContext *ctx) {
    auto st = (StatementContext::NEG *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->negQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->negOp[i]);
    }

    /* end */
    statementType = S_NEG;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterNotStatement(ptxParser::NotStatementContext *ctx) {
    statement = new StatementContext::NOT();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitNotStatement(ptxParser::NotStatementContext *ctx) {
    auto st = (StatementContext::NOT *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->notQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->notOp[i]);
    }

    /* end */
    statementType = S_NOT;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterSqrtStatement(ptxParser::SqrtStatementContext *ctx) {
    statement = new StatementContext::SQRT();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitSqrtStatement(ptxParser::SqrtStatementContext *ctx) {
    auto st = (StatementContext::SQRT *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->sqrtQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->sqrtOp[i]);
    }

    /* end */
    statementType = S_SQRT;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterCosStatement(ptxParser::CosStatementContext *ctx) {
    statement = new StatementContext::COS();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitCosStatement(ptxParser::CosStatementContext *ctx) {
    auto st = (StatementContext::COS *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->cosQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->cosOp[i]);
    }

    /* end */
    statementType = S_COS;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterLg2Statement(ptxParser::Lg2StatementContext *ctx) {
    statement = new StatementContext::LG2();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitLg2Statement(ptxParser::Lg2StatementContext *ctx) {
    auto st = (StatementContext::LG2 *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->lg2Qualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->lg2Op[i]);
    }

    /* end */
    statementType = S_LG2;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterEx2Statement(ptxParser::Ex2StatementContext *ctx) {
    statement = new StatementContext::EX2();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitEx2Statement(ptxParser::Ex2StatementContext *ctx) {
    auto st = (StatementContext::EX2 *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->ex2Qualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->ex2Op[i]);
    }

    /* end */
    statementType = S_EX2;
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
        st->atomQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 or op4 */
    if (ctx->operandFour()) {
        for (int i = 0; i < 4; i++) {
            fetchOperand(st->atomOp[i]);
        }
        st->operandNum = 4;
    } else if (ctx->operandThree()) {
        for (int i = 0; i < 3; i++) {
            fetchOperand(st->atomOp[i]);
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

void PtxListener::enterXorStatement(ptxParser::XorStatementContext *ctx) {
    statement = new StatementContext::XOR();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitXorStatement(ptxParser::XorStatementContext *ctx) {
    auto st = (StatementContext::XOR *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->xorQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op3 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->xorOp[i]);
    }

    /* end */
    statementType = S_XOR;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterAbsStatement(ptxParser::AbsStatementContext *ctx) {
    statement = new StatementContext::ABS();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitAbsStatement(ptxParser::AbsStatementContext *ctx) {
    auto st = (StatementContext::ABS *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->absQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->absOp[i]);
    }

    /* end */
    statementType = S_ABS;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterSinStatement(ptxParser::SinStatementContext *ctx) {
    statement = new StatementContext::SIN();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitSinStatement(ptxParser::SinStatementContext *ctx) {
    auto st = (StatementContext::SIN *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->sinQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->sinOp[i]);
    }

    /* end */
    statementType = S_SIN;
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterRsqrtStatement(ptxParser::RsqrtStatementContext *ctx) {
    statement = new StatementContext::RSQRT();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::exitRsqrtStatement(ptxParser::RsqrtStatementContext *ctx) {
    auto st = (StatementContext::RSQRT *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->rsqrtQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 2; i++) {
        fetchOperand(st->rsqrtOp[i]);
    }

    /* end */
    statementType = S_RSQRT;

#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}

void PtxListener::enterRemStatement(ptxParser::RemStatementContext *ctx) {
    statement = new StatementContext::REM();
#ifdef LOG
    std::cout << __func__ << std::endl;
#endif
}
void PtxListener::exitRemStatement(ptxParser::RemStatementContext *ctx) {
    auto st = (StatementContext::REM *)statement;

    /* qualifier */
    while (qualifier.size()) {
        st->remQualifier.push_back(qualifier.front());
        qualifier.pop();
    }

    /* op2 */
    for (int i = 0; i < 3; i++) {
        fetchOperand(st->remOp[i]);
    }

    /* end */
    statementType = S_REM;
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
    extractREG(ctx->ID()->getText(), r->regIdx, r->regName);
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
        extractREG(ctx->regi(i)->ID()->getText(), r->regIdx, r->regName);
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
        extractREG(ctx->regi()->ID()->getText(), r->regIdx, r->regName);
        fa->reg->operandType = O_REG;
        fa->reg->operand = r;
    } else
        assert(0);

    /* offset */
    if (ctx->DIGITS()) {
        fa->offsetVal = ctx->DIGITS()->getText();
    } else {
        fa->offsetVal = "";
    }

    /* end */
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
