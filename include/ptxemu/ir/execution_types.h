#ifndef PTXEMU_IR_EXECUTION_TYPES_H
#define PTXEMU_IR_EXECUTION_TYPES_H

// Phase 1 (HSK-8 ack 738b412c): extract public InstructionState enum into
// ptxemu::ir namespace. EXE_STATE, BAR_TYPE, Dim3, CTAId remain internal
// to ptxsim (PTX-EMU implementation detail, not part of public device API).

namespace ptxemu {
namespace ir {

enum class InstructionState {
    READY,    // 准备执行新指令
    PREPARE,  // 准备阶段
    EXECUTE,  // 执行阶段
    COMMIT    // 提交阶段
};

}  // namespace ir
}  // namespace ptxemu

#endif  // PTXEMU_IR_EXECUTION_TYPES_H