#ifndef PTX_VISITOR_CATEGORIES_H
#define PTX_VISITOR_CATEGORIES_H

// 定义所有指令类别的宏
#define VISITOR_GENERIC_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_ATOM_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_CALL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_WMMA_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_BRANCH(opstr, opname, opcount)                                 \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_BARRIER(opstr, opname, opcount)                                \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_OPERAND_REG(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_OPERAND_CONST(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_OPERAND_MEMORY(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_SIMPLE_NAME(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_SIMPLE_STRING(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_VOID_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_PREDICATE_PREFIX(opstr, opname, opcount)                       \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_MEMBAR_INSTR(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_FENCE_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_REDUX_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_MBARRIER_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_VOTE_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_SHFL_INSTR(opstr, opname, opcount)                             \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_TEXTURE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_SURFACE_INSTR(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_REDUCTION_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_PREFETCH_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_CP_ASYNC_INSTR(opstr, opname, opcount)                         \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_ASYNC_STORE(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_ASYNC_REDUCE(opstr, opname, opcount)                           \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_TCGEN_INSTR(opstr, opname, opcount)                            \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_TENSORMAP_INSTR(opstr, opname, opcount)                        \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#define VISITOR_ABI_DIRECTIVE(opstr, opname, opcount)                          \
std::any PtxVisitor::visit##opstr##Inst(ptxParser::opstr##InstContext *ctx);

#endif // PTX_VISITOR_CATEGORIES_H
