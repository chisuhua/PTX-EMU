// ============================================================================
// 包含各个类别的指令访问器实现（dispatch 聚合区）
// ============================================================================

// Tcgen05: 11 S_TCGEN05_* enums share a single visitTcgen05Inst handler
// (grammar has 1 tcgen05Inst rule). X-Macro expansion is a no-op here;
// the dispatch from visitTcgen05Inst to instr.op_kind happens inside
// PtxVisitor::visitTcgen05Inst (ptx_visitor_wmma.cpp:38-).
#define  VISITOR_TCGEN05_INSTR(openum, opstr, opname, opcount)  /* no-op */

// 包含通用指令实现
#include "ptx_visitor_generic.cpp"

// 包含原子指令实现
#include "ptx_visitor_atom.cpp"

// 包含调用指令实现
#include "ptx_visitor_call.cpp"

// 包含 Blackwell tcgen05 visitor 实现 (ADR-0016)
#include "ptx_visitor_tcgen05.cpp"

// 包含分支指令实现
#include "ptx_visitor_branch.cpp"

// 包含屏障指令实现
#include "ptx_visitor_barrier.cpp"

// 包含简单指令实现
#include "ptx_visitor_simple.cpp"

// 包含特殊指令实现
#include "ptx_visitor_special.cpp"

// 包含Warp相关指令实现
#include "ptx_visitor_warp.cpp"

// 包含内存相关指令实现
#include "ptx_visitor_memory.cpp"

// 包含 ABI 指令实现
#include "ptx_visitor_abi.cpp"

#define X(openum, opstr, opname, opcount, struct_kind, instr_kind)                         \
    VISITOR_##struct_kind(openum, opstr, opname, opcount);

#include "ptx_ir/ptx_op.def"
#undef X