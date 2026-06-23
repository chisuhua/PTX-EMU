// =============================================================================
// arithmetic_conversion.cpp — CvtHandler entry point (T2-6 Sub-task 3)
//
// After Sub-task 3, the entire 1063-line switch logic is delegated to
// ConversionStrategy via cvt_strategy.cpp::GeneralCvtStrategy (same behavior,
// same byte-level results). Sub-task 4 will split the GeneralCvtStrategy
// into 5 specific strategies (FloatToFloat/IntToFloat/FloatToInt/IntToInt/
// Rounding); Sub-task 6 will delete this file once the strategies are
// wired and verified.
// =============================================================================

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/instructions/cvt/cvt_strategy.h"
#include "ptxsim/thread_context.h"

void CvtHandler::processOperation(
    ThreadContext * /*context*/, void **operands,
    const std::vector<Qualifier> &qualifiers,
    const std::vector<char> * /*operand_is_immediate*/) {
    void *dst = operands[0];
    void *src = operands[1];

    // 1. 抽取 Qualifier 到强类型 CvtContext
    auto ctx = ptxsim::cvt_strategy::build_context(qualifiers);

    // 2. 选择 strategy (Sub-task 3: 唯一 GeneralCvtStrategy)
    const auto &strategy = ptxsim::cvt_strategy::select_strategy(ctx);

    // 3. 执行转换
    strategy.convert(dst, src, ctx);
}
