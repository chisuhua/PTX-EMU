#ifndef PTX_VISITOR_H
#define PTX_VISITOR_H

// #include "ptxParser.h"
#include "ptxParserBaseVisitor.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/operand_context.h"

using namespace ptxparser;

class PtxVisitor : public ptxParserBaseVisitor {
  std::vector<int> v;
public:
    explicit PtxVisitor(PtxContext &context) : ctx(context) {}

    // Top-level entry point
    std::any visitPtxFile(ptxParser::PtxFileContext *ctx) override;
    
    // Declaration visitors
    std::any visitDeclaration(ptxParser::DeclarationContext *ctx) override;
    std::any visitVersionDirective(ptxParser::VersionDirectiveContext *ctx) override;
    std::any visitTargetDirective(ptxParser::TargetDirectiveContext *ctx) override;
    std::any visitAddressSizeDirective(ptxParser::AddressSizeDirectiveContext *ctx) override;
    std::any visitVariableDecl(ptxParser::VariableDeclContext *ctx) override;
    std::any visitFunctionDecl(ptxParser::FunctionDeclContext *ctx) override;
    std::any visitAbiPreserveDirective(ptxParser::AbiPreserveDirectiveContext *ctx) override;
    
    // Function body visitors
    std::any visitInstructionList(ptxParser::InstructionListContext *ctx) override;
    std::any visitInstruction(ptxParser::InstructionContext *ctx) override;
    
    // Instruction category visitors
#define  VISIT_ABI_DIRECTIVE(opstr)
#define  VISIT_OPERAND_REG(opstr)
#define  VISIT_OPERAND_CONST(opstr)
#define  VISIT_OPERAND_MEMORY(opstr)
#define  VISIT_SIMPLE_NAME(opstr)
#define  VISIT_SIMPLE_NAME(opstr)
#define  VISIT_SIMPLE_STRING(opstr)
#define  VISIT_VOID_INSTR(opstr)
#define  VISIT_PREDICATE_PREFIX(opstr)
#define  VISIT_BRANCH(opstr)
#define  VISIT_ATOM_INSTR(opstr)
#define  VISIT_WMMA_INSTR(opstr)
#define  VISIT_BARRIER(opstr)
#define  VISIT_CALL_INSTR(opstr)
#define  VISIT_LABEL_INSTR(opstr)
#define  VISIT_MEMBAR_INSTR(opstr)
#define  VISIT_MBARRIER_INSTR(opstr)
#define  VISIT_FENCE_INSTR(opstr)
#define  VISIT_REDUX_INSTR(opstr)
#define  VISIT_VOTE_INSTR(opstr)
#define  VISIT_SHFL_INSTR(opstr)
#define  VISIT_TEXTURE_INSTR(opstr)
#define  VISIT_SURFACE_INSTR(opstr)
#define  VISIT_REDUCTION_INSTR(opstr)
#define  VISIT_PREFETCH_INSTR(opstr)
#define  VISIT_ASYNC_INSTR(opstr)
#define  VISIT_ASYNC_STORE(opstr)
#define  VISIT_ASYNC_REDUCE(opstr)
#define  VISIT_TCGEN_INSTR(opstr)
#define  VISIT_TENSORMAP_INSTR(opstr)

#define  VISIT_GENERIC_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) override;

#define  VISIT_CP_ASYNC_INSTR(opstr) \
    std::any visit##opstr##Inst(ptxParser::opstr##InstContext *ctx) override;

#define X(openum, opname, opstr, opcount, struct_kind) \
    VISIT_##struct_kind(opstr)
#include "ptx_ir/ptx_op.def"
#undef X

    // Operand visitors
    std::any visitOperand(ptxParser::OperandContext *ctx) override;
    std::any visitSpecialRegister(ptxParser::SpecialRegisterContext *ctx) override;
    std::any visitRegister(ptxParser::RegisterContext *ctx) override;
    std::any visitImmediate(ptxParser::ImmediateContext *ctx) override;
    std::any visitAddress(ptxParser::AddressContext *ctx) override;

private:
    PtxContext &ctx;
    KernelContext *currentKernel = nullptr;
    std::vector<Qualifier> currentQualifiers;
    
    // Helper methods
    Qualifier tokenToQualifier(antlr4::Token *token);
    std::vector<Qualifier> extractQualifiersFromContext(antlr4::ParserRuleContext *ctx);
    OperandContext createOperandFromContext(ptxParser::OperandContext *ctx);
    void processFunctionAttributes(ptxParser::FunctionAttributeContext *ctx);
    size_t calculateTypeSize(const std::vector<Qualifier> &types);
    int extractIntFromToken(antlr4::Token *token);
    std::string extractStringFromToken(antlr4::Token *token);
};

#endif // PTX_VISITOR_H
