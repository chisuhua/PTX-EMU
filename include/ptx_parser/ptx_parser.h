#ifndef PTX_PARSER_H
#define PTX_PARSER_H

#include <cassert>
#include <cstdio>
#include <queue>

#include "ptxParser.h"
#include "ptxParserBaseListener.h"
#include "ptx_ir/kernel_context.h"

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
    enterRcpStatement(ptxParser::RcpStatementContext *ctx) override;
    virtual void exitRcpStatement(ptxParser::RcpStatementContext *ctx) override;
    virtual void enterLdStatement(ptxParser::LdStatementContext *ctx) override;
    virtual void exitLdStatement(ptxParser::LdStatementContext *ctx) override;
    virtual void
    enterMovStatement(ptxParser::MovStatementContext *ctx) override;
    virtual void exitMovStatement(ptxParser::MovStatementContext *ctx) override;
    virtual void
    enterSetpStatement(ptxParser::SetpStatementContext *ctx) override;
    virtual void
    exitSetpStatement(ptxParser::SetpStatementContext *ctx) override;
    virtual void
    enterCvtaStatement(ptxParser::CvtaStatementContext *ctx) override;
    virtual void
    exitCvtaStatement(ptxParser::CvtaStatementContext *ctx) override;
    virtual void
    enterCvtStatement(ptxParser::CvtStatementContext *ctx) override;
    virtual void exitCvtStatement(ptxParser::CvtStatementContext *ctx) override;
    virtual void
    enterMulStatement(ptxParser::MulStatementContext *ctx) override;
    virtual void exitMulStatement(ptxParser::MulStatementContext *ctx) override;
    virtual void
    enterDivStatement(ptxParser::DivStatementContext *ctx) override;
    virtual void exitDivStatement(ptxParser::DivStatementContext *ctx) override;
    virtual void
    enterSubStatement(ptxParser::SubStatementContext *ctx) override;
    virtual void exitSubStatement(ptxParser::SubStatementContext *ctx) override;
    virtual void
    enterAddStatement(ptxParser::AddStatementContext *ctx) override;
    virtual void exitAddStatement(ptxParser::AddStatementContext *ctx) override;
    virtual void
    enterShlStatement(ptxParser::ShlStatementContext *ctx) override;
    virtual void exitShlStatement(ptxParser::ShlStatementContext *ctx) override;
    virtual void
    enterShrStatement(ptxParser::ShrStatementContext *ctx) override;
    virtual void exitShrStatement(ptxParser::ShrStatementContext *ctx) override;
    virtual void
    enterMaxStatement(ptxParser::MaxStatementContext *ctx) override;
    virtual void exitMaxStatement(ptxParser::MaxStatementContext *ctx) override;
    virtual void
    enterMinStatement(ptxParser::MinStatementContext *ctx) override;
    virtual void exitMinStatement(ptxParser::MinStatementContext *ctx) override;
    virtual void
    enterAndStatement(ptxParser::AndStatementContext *ctx) override;
    virtual void exitAndStatement(ptxParser::AndStatementContext *ctx) override;
    virtual void enterOrStatement(ptxParser::OrStatementContext *ctx) override;
    virtual void exitOrStatement(ptxParser::OrStatementContext *ctx) override;
    virtual void enterStStatement(ptxParser::StStatementContext *ctx) override;
    virtual void exitStStatement(ptxParser::StStatementContext *ctx) override;
    virtual void
    enterSelpStatement(ptxParser::SelpStatementContext *ctx) override;
    virtual void
    exitSelpStatement(ptxParser::SelpStatementContext *ctx) override;
    virtual void
    enterMadStatement(ptxParser::MadStatementContext *ctx) override;
    virtual void exitMadStatement(ptxParser::MadStatementContext *ctx) override;
    virtual void
    enterFmaStatement(ptxParser::FmaStatementContext *ctx) override;
    virtual void exitFmaStatement(ptxParser::FmaStatementContext *ctx) override;
    virtual void
    enterWmmaStatement(ptxParser::WmmaStatementContext *ctx) override;
    virtual void
    exitWmmaStatement(ptxParser::WmmaStatementContext *ctx) override;
    virtual void
    enterNegStatement(ptxParser::NegStatementContext *ctx) override;
    virtual void exitNegStatement(ptxParser::NegStatementContext *ctx) override;
    virtual void
    enterNotStatement(ptxParser::NotStatementContext *ctx) override;
    virtual void exitNotStatement(ptxParser::NotStatementContext *ctx) override;
    virtual void
    enterSqrtStatement(ptxParser::SqrtStatementContext *ctx) override;
    virtual void
    exitSqrtStatement(ptxParser::SqrtStatementContext *ctx) override;
    virtual void
    enterCosStatement(ptxParser::CosStatementContext *ctx) override;
    virtual void exitCosStatement(ptxParser::CosStatementContext *ctx) override;
    virtual void
    enterLg2Statement(ptxParser::Lg2StatementContext *ctx) override;
    virtual void exitLg2Statement(ptxParser::Lg2StatementContext *ctx) override;
    virtual void
    enterEx2Statement(ptxParser::Ex2StatementContext *ctx) override;
    virtual void exitEx2Statement(ptxParser::Ex2StatementContext *ctx) override;
    virtual void
    enterAtomStatement(ptxParser::AtomStatementContext *ctx) override;
    virtual void
    exitAtomStatement(ptxParser::AtomStatementContext *ctx) override;
    virtual void
    enterXorStatement(ptxParser::XorStatementContext *ctx) override;
    virtual void exitXorStatement(ptxParser::XorStatementContext *ctx) override;
    virtual void
    enterAbsStatement(ptxParser::AbsStatementContext *ctx) override;
    virtual void exitAbsStatement(ptxParser::AbsStatementContext *ctx) override;
    virtual void
    enterSinStatement(ptxParser::SinStatementContext *ctx) override;
    virtual void exitSinStatement(ptxParser::SinStatementContext *ctx) override;
    virtual void
    enterRsqrtStatement(ptxParser::RsqrtStatementContext *ctx) override;
    virtual void
    exitRsqrtStatement(ptxParser::RsqrtStatementContext *ctx) override;
    virtual void
    enterRemStatement(ptxParser::RemStatementContext *ctx) override;
    virtual void exitRemStatement(ptxParser::RemStatementContext *ctx) override;
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
};

#endif