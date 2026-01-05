#ifndef CONDITION_CODE_REGISTER_H
#define CONDITION_CODE_REGISTER_H

#include <array>
#include <cstddef>
#include <cstdint>

// 条件码寄存器类，用于存储PTX中的条件码状态
class ConditionCodeRegister {
public:
    // 条件码寄存器的数量，根据PTX规范
    static constexpr size_t CC_REG_COUNT = 8;

    // 条件码寄存器索引定义
    static constexpr size_t CARRY_INDEX = 0;
    static constexpr size_t OVERFLOW_INDEX = 1;
    static constexpr size_t ZERO_INDEX = 2;
    static constexpr size_t SIGN_INDEX = 3;

    ConditionCodeRegister();

    // 设置指定索引的条件码寄存器值
    void set_cc_reg(size_t index, bool value);

    // 获取指定索引的条件码寄存器值
    bool get_cc_reg(size_t index) const;

    // 专门的getter方法
    bool get_carry() const { return get_cc_reg(CARRY_INDEX); }
    bool get_overflow() const { return get_cc_reg(OVERFLOW_INDEX); }
    bool get_zero() const { return get_cc_reg(ZERO_INDEX); }
    bool get_sign() const { return get_cc_reg(SIGN_INDEX); }

    // 批量设置条件码寄存器
    void set_cc_regs(const std::array<bool, CC_REG_COUNT> &values);

    // 获取所有条件码寄存器的状态
    const std::array<bool, CC_REG_COUNT> &get_cc_regs() const;

    // 重置所有条件码寄存器为默认值（false）
    void reset();

private:
    // 存储条件码寄存器状态的数组
    std::array<bool, CC_REG_COUNT> cc_regs;
};

#endif // CONDITION_CODE_REGISTER_H