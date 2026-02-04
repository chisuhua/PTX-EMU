#ifndef PTX_VISITOR_H
#define PTX_VISITOR_H

#include "ptxParser.h"
#include "ptxParserBaseVisitor.h"
#include "ptx_ir/ptx_context.h"
#include <memory>
#include <vector>

class PtxVisitor : public PtxParserBaseVisitor {
public:
    explicit PtxVisitor(PtxContext &context) : ctx(context) {}

    // Top-level entry point
    std::any visitPtxFile(PtxParser::PtxFileContext *ctx) override;
    
    // Declaration visitors
    std::any visitDeclaration(PtxParser::DeclarationContext *ctx) override;
    std::any visitVersionDirective(PtxParser::VersionDirectiveContext *ctx) override;
    std::any visitTargetDirective(PtxParser::TargetDirectiveContext *ctx) override;
    std::any visitAddressSizeDirective(PtxParser::AddressSizeDirectiveContext *ctx) override;
    std::any visitVariableDecl(PtxParser::VariableDeclContext *ctx) override;
    std::any visitFunctionDecl(PtxParser::FunctionDeclContext *ctx) override;
    std::any visitAbiPreserveDirective(PtxParser::AbiPreserveDirectiveContext *ctx) override;
    
    // Function body visitors
    std::any visitInstructionList(PtxParser::InstructionListContext *ctx) override;
    std::any visitInstruction(PtxParser::InstructionContext *ctx) override;
    
    // Instruction category visitors
#define X(stmt_type, op_kind, op_name, op_count, struct_kind) \
    std::any visit##op_kind##Inst(PtxParser::op_kind##InstContext *ctx) override;
#include "ptx_op.def"
#undef X

    // Operand visitors
    std::any visitOperand(PtxParser::OperandContext *ctx) override;
    std::any visitSpecialRegister(PtxParser::SpecialRegisterContext *ctx) override;
    std::any visitRegister(PtxParser::RegisterContext *ctx) override;
    std::any visitImmediate(PtxParser::ImmediateContext *ctx) override;
    std::any visitAddress(PtxParser::AddressContext *ctx) override;

private:
    PtxContext &ctx;
    KernelContext *currentKernel = nullptr;
    std::vector<Qualifier> currentQualifiers;
    
    // Helper methods
    Qualifier tokenToQualifier(antlr4::Token *token);
    std::vector<Qualifier> extractQualifiersFromContext(antlr4::ParserRuleContext *ctx);
    OperandContext createOperandFromContext(PtxParser::OperandContext *ctx);
    void processFunctionAttributes(PtxParser::FunctionAttributeContext *ctx);
    size_t calculateTypeSize(const std::vector<Qualifier> &types);
    int extractIntFromToken(antlr4::Token *token);
    std::string extractStringFromToken(antlr4::Token *token);
};

#endif // PTX_VISITOR_H