#include "memory/hardware_memory_manager.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include "ptx_ir/ptx_types.h"
#include <cstring>
#include <cstdint>

#include "memory/hardware_memory_manager.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"
#include <cmath>

void AtomHandler::processAtomicOperation(ThreadContext *context, void **operands,
                                 const std::vector<Qualifier> &qualifiers,
                                 const std::vector<char> *operand_is_immediate) {
    (void)context;
    (void)operand_is_immediate;
    // TODO: 实现原子操作逻辑
    // 需要修复 ptx_visitor_atom.cpp 中 ANTLR visitor 对 addressExpr 的收集
    // 当前 getRuleContexts<OperandContext>() 漏掉了 atom 中间的位置表达式
    // 语法: operand[0]=dst COMMA addressExpr[middle] COMMA operand[1]=src
}
