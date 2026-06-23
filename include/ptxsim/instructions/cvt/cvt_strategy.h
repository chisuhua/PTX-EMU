// cvt_strategy.h
// =============================================================================
// CVT 策略模式骨架 (T2-6 Sub-task 3)
//
// 目的:
//   - 把 arithmetic_conversion.cpp:11-90 的 Qualifier 解析抽离到
//     build_context()，返回强类型 CvtContext。
//   - 引入抽象 ConversionStrategy 基类 + select_strategy() 工厂。
//   - Sub-task 3 仅搭骨架：select_strategy() 返回单一"通用策略"，
//     行为完全等同原 switch 逻辑（zero behavior change）。
//   - Sub-task 4 将通用策略拆为 5 个具体策略 (FloatToFloat /
//     IntToFloat / FloatToInt / IntToInt + Rounding helpers)。
//
// 设计选择: Composition over Inheritance
//   CvtHandler 仍由 X-Macro 单一注册，但内部持有 strategy 引用。
//   原因: instruction_factory.cpp:14-17 实例化 new CvtHandler()，
//   多 Handler 会破坏 X-Macro 注册机制 (master plan §关键约束 DO NOT 拆
//   CvtHandler 类)。
//
// X-Macro constraint: ConversionStrategy 是抽象基类，转换为运行时多态。
// =============================================================================

#ifndef PTXSIM_INSTRUCTIONS_CVT_CVT_STRATEGY_H
#define PTXSIM_INSTRUCTIONS_CVT_CVT_STRATEGY_H

#include <cstdint>
#include <vector>

#include "ptx_ir/ptx_types.h"

namespace ptxsim {
namespace cvt_strategy {

// ---------------------------------------------------------------------------
// CvtContext: 强类型化的 CVT 操作描述 (替代 30+ Qualifier::Q_xxx 检查)
//
// 来源: arithmetic_conversion.cpp:11-90 (processOperation 第一段)
// 抽取原因: 让策略代码不直接接触 std::vector<Qualifier>，类型安全 +
//          测试简单（直接构造字段，不需 build PTX statement）。
// ---------------------------------------------------------------------------
struct CvtContext {
    // 类型尺寸 (bytes): 1 / 2 / 4 / 8
    int dst_bytes = 0;
    int src_bytes = 0;

    // 类型分类
    bool dst_is_float = false; // f16/f32/f64 (PTX 视为 float 类)
    bool src_is_float = false;
    bool dst_is_half = false; // f16 (强制 2 字节)
    bool src_is_half = false;
    bool dst_is_signed = false;
    bool src_is_signed = false;

    // 修饰符 (.sat / 5 个 .rn* 浮点舍入 / 4 个 .rn* 整数舍入 / .rna / .rs)
    bool has_sat = false;
    bool has_rn = false;
    bool has_rni = false;
    bool has_rz = false;
    bool has_rzi = false;
    bool has_rm = false;
    bool has_rmi = false;
    bool has_rp = false;
    bool has_rpi = false;
    bool has_rna = false;
    bool has_rs = false;
};

// 从 Qualifier 列表构造 CvtContext。
// 输入: dst_dtype, src_dtype, 可选 .sat / .rn* / .rni* / .rna / .rs
// 等价于 arithmetic_conversion.cpp:11-90 的所有 Qualifier 解析逻辑。
CvtContext build_context(const std::vector<Qualifier> &qualifiers);

// ---------------------------------------------------------------------------
// ConversionStrategy: 抽象策略基类 (运行时多态)
//
// Sub-task 3 仅一个实现 (GeneralCvtStrategy 在 .cpp 中)，保留 Sub-task 4
// 拆分的扩展点。
// ---------------------------------------------------------------------------
class ConversionStrategy {
public:
    virtual ~ConversionStrategy() = default;

    // 执行 CVT 转换: 读 src，写 dst。
    // src/dst 是 ThreadContext 提供的寄存器指针，宽度由 ctx.dst_bytes/
    // ctx.src_bytes 决定。strategy 不应接触 Qualifier 列表 — 一切由
    // CvtContext 决定。
    virtual void convert(void *dst, void *src, const CvtContext &ctx) const = 0;

    // 策略名 (调试用)
    virtual const char *name() const = 0;
};

// select_strategy(): 根据 CvtContext 选择具体策略。
// 返回的指针生命周期由调用者管理 (Sub-task 3 中是静态单例，Sub-task 4
// 会保持同样约定以避免分配开销)。
const ConversionStrategy &select_strategy(const CvtContext &ctx);

} // namespace cvt_strategy
} // namespace ptxsim

#endif // PTXSIM_INSTRUCTIONS_CVT_CVT_STRATEGY_H
