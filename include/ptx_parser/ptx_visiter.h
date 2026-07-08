#ifndef PTX_VISITOR_H
#define PTX_VISITOR_H

#include "ptxParser.h"
#include "ptxParserBaseVisitor.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/operand_context.h"
#include <any>
#include <vector>

// 移除using namespace，使用完全限定名
class PtxVisitor : public ptxparser::ptxParserBaseVisitor {
public:
    explicit PtxVisitor(PtxContext &context) : ctx(context) {}

    // Top-level entry point
    std::any visitPtxFile(ptxparser::ptxParser::PtxFileContext *ctx) override;
    
    // Declaration visitors
    std::any visitDeclaration(ptxparser::ptxParser::DeclarationContext *ctx) override;
    std::any visitVersionDirective(ptxparser::ptxParser::VersionDirectiveContext *ctx) override;
    std::any visitTargetDirective(ptxparser::ptxParser::TargetDirectiveContext *ctx) override;
    std::any visitAddressSizeDirective(ptxparser::ptxParser::AddressSizeDirectiveContext *ctx) override;
    std::any visitVariableDecl(ptxparser::ptxParser::VariableDeclContext *ctx) override;
    std::any visitFunctionDecl(ptxparser::ptxParser::FunctionDeclContext *ctx) override;
    std::any visitAbiPreserveDirective(ptxparser::ptxParser::AbiPreserveDirectiveContext *ctx) override;
    
    // Function body visitors
    std::any visitInstructionList(ptxparser::ptxParser::InstructionListContext *ctx) override;
    std::any visitInstruction(ptxparser::ptxParser::InstructionContext *ctx) override;
    
    // Instruction category visitors
    // call.uni instruction visitor
    std::any visitCallUniInst(ptxparser::ptxParser::CallUniInstContext *ctx) override;
// These generate declarations for visitor methods based on struct_kind
// Some struct_kinds like structure (operands/declarations) don't need visitor methods
#define  VISITOR_DECL_ABI_DIRECTIVE(opstr)
#define  VISITOR_DECL_OPERAND_REG(opstr)
#define  VISITOR_DECL_OPERAND_CONST(opstr)
#define  VISITOR_DECL_OPERAND_MEMORY(opstr)
#define  VISITOR_DECL_SIMPLE_NAME(opstr)
#define  VISITOR_DECL_SIMPLE_STRING(opstr)
#define  VISITOR_DECL_VOID_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_PREDICATE_PREFIX(opstr)
#define  VISITOR_DECL_BRANCH(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_ATOM_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_WMMA_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_BARRIER(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_CALL_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_LABEL_INSTR(opstr)
#define  VISITOR_DECL_MEMBAR_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_MBARRIER_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_FENCE_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_REDUX_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_VOTE_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_SHFL_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_TEXTURE_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_SURFACE_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_REDUCTION_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_PREFETCH_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_ASYNC_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_ASYNC_STORE(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_ASYNC_REDUCE(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#define  VISITOR_DECL_TCGEN_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
// Tcgen05: 11 S_TCGEN05_* X-Macro entries all dispatch through a single
// visitTcgen05Inst (grammar has 1 tcgen05Inst rule). Make this a no-op
// for the X-Macro expansion; the real visitTcgen05Inst is declared below.
#define  VISITOR_TCGEN05_INSTR(opstr)  /* no-op: see visitTcgen05Inst below */
#define  VISITOR_DECL_TCGEN05_INSTR(opstr)  /* no-op: see visitTcgen05Inst below */
#define  VISITOR_DECL_TENSORMAP_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;

#define  VISITOR_DECL_GENERIC_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;

#define  VISITOR_DECL_CP_ASYNC_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;

#define X(enum_val, type_name, opstr, op_count, struct_kind, instr_kind) \
    VISITOR_DECL_##struct_kind(opstr)

// Stage 2b: NEW - Warp-level synchronization visitor declaration
#define  VISITOR_DECL_WARP_BARRIER(opstr) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#include "ptx_ir/ptx_op.def"
#undef X

    // Operand visitors
    std::any visitOperand(ptxparser::ptxParser::OperandContext *ctx) override;

    // tcgen05 instruction visitor (Blackwell sm_100+, ADR-0016). Hand-maintained:
    // the grammar has a single tcgen05Inst rule, so the per-instruction X-Macro
    // expansion in ptx_visitor.cpp is bypassed. S_TCGEN05_* in ptx_types.h.
    std::any visitTcgen05Inst(ptxparser::ptxParser::Tcgen05InstContext *ctx) override;
    std::any visitSpecialRegister(ptxparser::ptxParser::SpecialRegisterContext *ctx) override;
    std::any visitRegister(ptxparser::ptxParser::RegisterContext *ctx) override;
    std::any visitImmediate(ptxparser::ptxParser::ImmediateContext *ctx) override;
    std::any visitAddress(ptxparser::ptxParser::AddressContext *ctx) override;

private:
    PtxContext &ctx;
    KernelContext *currentKernel = nullptr;
    std::vector<Qualifier> currentQualifiers;
    
    // Helper methods
    Qualifier tokenToQualifier(antlr4::Token *token);
    std::vector<Qualifier> extractQualifiersFromContext(antlr4::ParserRuleContext *ctx);
    OperandContext createOperandFromContext(ptxparser::ptxParser::OperandContext *ctx);
    int extractIntFromToken(antlr4::Token *token);
    std::string extractStringFromToken(antlr4::Token *token);
};

#endif // PTX_VISITOR_H
