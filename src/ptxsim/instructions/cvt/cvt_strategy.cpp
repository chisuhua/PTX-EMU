// cvt_strategy.cpp
// =============================================================================
// CVT 策略模式 — dispatcher 实现 (commit fc3c352/9837d44/d6123e0 完成部署)
//
// 状态:
//   - build_context():  从 ptxemu::ir::Qualifier 列表构造强类型 CvtContext
//   - select_strategy(): 返回 4 个具体 Strategy 实例之一（按 dst/src 类型）
//   - CvtHandler::processOperation(): 顶层入口，调用 select_strategy + convert
//
// 4 个活 Strategy 类（已由 archive Sub-task 3-4 实施）：
//   - FloatToFloatStrategy  → cvt_float_to_float.cpp
//   - FloatToIntStrategy    → cvt_float_to_int.cpp    (含 .sat/5 rounding/.ftz)
//   - IntToFloatStrategy    → cvt_int_to_float.cpp
//   - IntToIntStrategy      → cvt_int_to_int.cpp
//
// 2026-07: 过渡类 GeneralCvtStrategy (~920 行) 由 fix-cvt-strategy-actual-split
//          移除（pure deletion，无行为变更）。详见 ADR-0015 + debt-audit P0-C1。
// =============================================================================

#include "ptxsim/instructions/cvt/cvt_strategy.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/instructions/cvt/cvt_float_to_float.h"
#include "ptxsim/instructions/cvt/cvt_float_to_int.h"
#include "ptxsim/instructions/cvt/cvt_helpers.h"
#include "ptxsim/instructions/cvt/cvt_int_to_float.h"
#include "ptxsim/instructions/cvt/cvt_int_to_int.h"
#include "ptxsim/ptx_exceptions.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "ptxsim/utils/type_utils.h"

namespace ptxsim {
namespace cvt_strategy {

// ---------------------------------------------------------------------------
// build_context: 从 ptxemu::ir::Qualifier 列表提取 CvtContext
//
// 来源: arithmetic_conversion.cpp:17-90 (旧 processOperation 第一段)
// 抽取后: arithmetic_conversion.cpp::processOperation 直接构造 ctx，
//        避免每次重新解析 ptxemu::ir::Qualifier。
// ---------------------------------------------------------------------------
CvtContext build_context(const std::vector<ptxemu::ir::Qualifier> &qualifiers) {
    CvtContext ctx;

    std::vector<ptxemu::ir::Qualifier> dst_qualifiers, src_qualifiers;
    splitDstSrcQualifiers(qualifiers, dst_qualifiers, src_qualifiers);

    ctx.dst_bytes = getBytes(dst_qualifiers);
    ctx.src_bytes = getBytes(src_qualifiers);
    ctx.dst_is_float = TypeUtils::is_float_type(dst_qualifiers);
    ctx.src_is_float = TypeUtils::is_float_type(src_qualifiers);

    // f16 (half) 强制 2 字节 + 视为 float
    for (const auto &q : dst_qualifiers) {
        if (q == ptxemu::ir::Qualifier::Q_F16) {
            ctx.dst_is_half = true;
            ctx.dst_is_float = true;
            ctx.dst_bytes = 2;
            break;
        }
    }
    for (const auto &q : src_qualifiers) {
        if (q == ptxemu::ir::Qualifier::Q_F16) {
            ctx.src_is_half = true;
            ctx.src_is_float = true;
            ctx.src_bytes = 2;
            break;
        }
    }

    // bytes 默认值兜底 (与原逻辑保持一致)
    if (ctx.dst_bytes == 0) {
        ctx.dst_bytes = getBytes(qualifiers);
    }
    if (ctx.src_bytes == 0 && !ctx.src_is_half) {
        ctx.src_bytes = getBytes(qualifiers);
    }
    if (ctx.src_bytes == 0 && !ctx.src_is_half) {
        ctx.src_bytes = 1;
    }

    // 修饰符
    ctx.has_sat = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_SAT);
    ctx.has_rn = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RN);
    ctx.has_rni = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RNI);
    ctx.has_rz = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RZ);
    ctx.has_rzi = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RZI);
    ctx.has_rm = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RM);
    ctx.has_rmi = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RMI);
    ctx.has_rp = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RP);
    ctx.has_rpi = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RPI);
    ctx.has_rna = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RNA);
    ctx.has_rs = QvecHasQ(qualifiers, ptxemu::ir::Qualifier::Q_RS);

    // 符号性
    ctx.dst_is_signed = TypeUtils::is_signed_type(dst_qualifiers);
    ctx.src_is_signed = TypeUtils::is_signed_type(src_qualifiers);

    return ctx;
}


const ConversionStrategy &select_strategy(const CvtContext &ctx) {
    static const FloatToFloatStrategy f2f;
    static const FloatToIntStrategy f2i;
    static const IntToFloatStrategy i2f;
    static const IntToIntStrategy i2i;

    if (ctx.dst_is_float) {
        return ctx.src_is_float ? static_cast<const ConversionStrategy &>(f2f)
                                : static_cast<const ConversionStrategy &>(i2f);
    }
    return ctx.src_is_float ? static_cast<const ConversionStrategy &>(f2i)
                            : static_cast<const ConversionStrategy &>(i2i);
}

} // namespace cvt_strategy
} // namespace ptxsim

void CvtHandler::processOperation(
    ThreadContext * /*context*/, void **operands,
    const std::vector<ptxemu::ir::Qualifier> &qualifiers,
    const std::vector<char> * /*operand_is_immediate*/) {
    void *dst = operands[0];
    void *src = operands[1];

    auto ctx = ptxsim::cvt_strategy::build_context(qualifiers);
    const auto &strategy = ptxsim::cvt_strategy::select_strategy(ctx);
    strategy.convert(dst, src, ctx);
}
