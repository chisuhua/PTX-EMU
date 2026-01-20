#ifndef PTX_PARSER_H
#define PTX_PARSER_H

#include <cassert>
#include <cstdio>
#include <queue>

#include "ptxParser.h"
#include "ptxParserBaseListener.h"
#include "ptx_ir/ptx_context.h"

using namespace ptxparser;
using namespace antlr4;

// Forward declaration of PtxListener class
class PtxListener : public ptxParserBaseListener {
public:
    PtxContext ptxContext;
    KernelContext *kernelContext = nullptr;
    ParamContext *paramContext = nullptr;
    std::queue<Qualifier> qualifier;
    StatementType statementType;
    StatementContext statementContext;
    void *statement = nullptr;
    std::queue<OperandContext *> op;

    /* helper function */
    void test_semantic();
    void fetchOperand(OperandContext &oc);
    void fetchOperand(std::vector<OperandContext> &oc);

    /* listener function */
    virtual void enterAst(ptxParser::AstContext *ctx) override;
    virtual void exitAst(ptxParser::AstContext *ctx) override;
    virtual void enterVersionDes(ptxParser::VersionDesContext *ctx) override;
    virtual void exitVersionDes(ptxParser::VersionDesContext *ctx) override;
    virtual void enterTargetDes(ptxParser::TargetDesContext *ctx) override;
    virtual void exitTargetDes(ptxParser::TargetDesContext *ctx) override;
    virtual void enterAddressDes(ptxParser::AddressDesContext *ctx) override;
    virtual void exitAddressDes(ptxParser::AddressDesContext *ctx) override;
    virtual void
    enterPerformanceTuning(ptxParser::PerformanceTuningContext *ctx) override;
    virtual void
    exitPerformanceTuning(ptxParser::PerformanceTuningContext *ctx) override;
    virtual void enterKernels(ptxParser::KernelsContext *ctx) override;
    virtual void exitKernels(ptxParser::KernelsContext *ctx) override;
    virtual void enterKernel(ptxParser::KernelContext *ctx) override;
    virtual void exitKernel(ptxParser::KernelContext *ctx) override;
    virtual void enterQualifier(ptxParser::QualifierContext *ctx) override;
    virtual void exitQualifier(ptxParser::QualifierContext *ctx) override;
    virtual void enterParams(ptxParser::ParamsContext *ctx) override;
    virtual void exitParams(ptxParser::ParamsContext *ctx) override;
    virtual void enterParam(ptxParser::ParamContext *ctx) override;
    virtual void exitParam(ptxParser::ParamContext *ctx) override;
    virtual void
    enterCompoundStatement(ptxParser::CompoundStatementContext *ctx) override;
    virtual void
    exitCompoundStatement(ptxParser::CompoundStatementContext *ctx) override;
    virtual void enterStatements(ptxParser::StatementsContext *ctx) override;
    virtual void exitStatements(ptxParser::StatementsContext *ctx) override;
    virtual void enterStatement(ptxParser::StatementContext *ctx) override;
    virtual void exitStatement(ptxParser::StatementContext *ctx) override;
    virtual void
    enterRegStatement(ptxParser::RegStatementContext *ctx) override;
    virtual void exitRegStatement(ptxParser::RegStatementContext *ctx) override;
    virtual void
    enterSharedStatement(ptxParser::SharedStatementContext *ctx) override;
    virtual void
    exitSharedStatement(ptxParser::SharedStatementContext *ctx) override;
    virtual void
    enterConstStatement(ptxParser::ConstStatementContext *ctx) override;
    virtual void
    exitConstStatement(ptxParser::ConstStatementContext *ctx) override;
    virtual void
    enterLocalStatement(ptxParser::LocalStatementContext *ctx) override;
    virtual void
    exitLocalStatement(ptxParser::LocalStatementContext *ctx) override;
    virtual void
    enterGlobalStatement(ptxParser::GlobalStatementContext *ctx) override;
    virtual void
    exitGlobalStatement(ptxParser::GlobalStatementContext *ctx) override;
    virtual void
    enterDollorStatement(ptxParser::DollorStatementContext *ctx) override;
    virtual void
    exitDollorStatement(ptxParser::DollorStatementContext *ctx) override;
    virtual void enterAtStatement(ptxParser::AtStatementContext *ctx) override;
    virtual void exitAtStatement(ptxParser::AtStatementContext *ctx) override;
    virtual void
    enterPragmaStatement(ptxParser::PragmaStatementContext *ctx) override;
    virtual void
    exitPragmaStatement(ptxParser::PragmaStatementContext *ctx) override;
    virtual void
    enterRetStatement(ptxParser::RetStatementContext *ctx) override;
    virtual void exitRetStatement(ptxParser::RetStatementContext *ctx) override;
    virtual void
    enterBarStatement(ptxParser::BarStatementContext *ctx) override;
    virtual void exitBarStatement(ptxParser::BarStatementContext *ctx) override;
    virtual void
    enterBraStatement(ptxParser::BraStatementContext *ctx) override;
    virtual void exitBraStatement(ptxParser::BraStatementContext *ctx) override;

    virtual void
    enterWmmaStatement(ptxParser::WmmaStatementContext *ctx) override;
    virtual void
    exitWmmaStatement(ptxParser::WmmaStatementContext *ctx) override;

    virtual void enterReg(ptxParser::RegContext *ctx) override;
    virtual void exitReg(ptxParser::RegContext *ctx) override;
    virtual void enterVector(ptxParser::VectorContext *ctx) override;
    virtual void exitVector(ptxParser::VectorContext *ctx) override;
    virtual void
    enterFetchAddress(ptxParser::FetchAddressContext *ctx) override;
    virtual void exitFetchAddress(ptxParser::FetchAddressContext *ctx) override;
    virtual void enterImm(ptxParser::ImmContext *ctx) override;
    virtual void exitImm(ptxParser::ImmContext *ctx) override;
    virtual void enterVar(ptxParser::VarContext *ctx) override;
    virtual void exitVar(ptxParser::VarContext *ctx) override;

#define STATEMENT_DECL_OPERAND_REG(opstr, opname, opcount)
#define STATEMENT_DECL_OPERAND_CONST(opstr, opname, opcount)
#define STATEMENT_DECL_OPERAND_MEMORY(opstr, opname, opcount)
#define STATEMENT_DECL_SIMPLE_NAME(opstr, opname, opcount)
#define STATEMENT_DECL_SIMPLE_STRING(opstr, opname, opcount)
#define STATEMENT_DECL_VOID_INSTR(opstr, opname, opcount)
#define STATEMENT_DECL_BRANCH(opstr, opname, opcount)
#define STATEMENT_DECL_GENERIC_INSTR(opstr, opname, opcount)                   \
    virtual void enter##opstr##Statement(                                      \
        ptxParser::opstr##StatementContext *ctx) override;                     \
    virtual void exit##opstr##Statement(                                       \
        ptxParser::opstr##StatementContext *ctx) override;

#define STATEMENT_DECL_ATOM_INSTR(opstr, op_name, opcount)                     \
    virtual void enter##opstr##Statement(                                      \
        ptxParser::opstr##StatementContext *ctx);                              \
    virtual void exit##opstr##Statement(                                       \
        ptxParser::opstr##StatementContext *ctx);

#define STATEMENT_DECL_PREDICATE_PREFIX(opstr, opname, opcount)
#define STATEMENT_DECL_WMMA_INSTR(opstr, op_name, opcount)
#define STATEMENT_DECL_BARRIER(opstr, op_name, opcount)

#define X(openum, opname, opstr, opcount, struct_kind)                         \
    STATEMENT_DECL_##struct_kind(opstr, opname, opcount)
#include "ptx_ir/ptx_op.def"
#undef X
};

#endif