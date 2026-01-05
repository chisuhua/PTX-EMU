#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include <cassert>
#include <iostream>

// 简单的测试函数来验证条件码寄存器功能
void test_cc_register() {
    std::cout << "Testing Condition Code Register functionality..."
              << std::endl;

    // 创建一个ThreadContext实例
    ThreadContext context;

    // 测试1: ADDC指令的进位功能
    std::cout << "Test 1: ADDC with carry flag" << std::endl;

    // 设置初始条件码寄存器状态 - 使用专门的setter方法
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true); // 设置进位标志
    new_cc_reg.set_cc_reg(ConditionCodeRegister::ZERO_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::SIGN_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::OVERFLOW_INDEX, false);
    context.set_condition_codes(new_cc_reg);

    // 模拟执行ADDC指令 (255 + 1 + 1 = 257, 超过uint8_t范围，应该产生进位)
    uint8_t src1 = 255;
    uint8_t src2 = 1;
    uint8_t dst = 0;

    void *operands[3] = {&dst, &src1, &src2};

    // 创建带CC修饰符的qualifiers
    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U8);
    qualifiers.push_back(Qualifier::Q_CC); // 添加.cc修饰符

    ADDC addc_handler;
    addc_handler.process_operation(&context, operands, qualifiers);

    std::cout << "Result: " << (int)dst << std::endl;
    std::cout << "Carry flag after ADDC: " << context.get_condition_codes().get_carry() << std::endl;
    std::cout << "Zero flag after ADDC: " << context.get_condition_codes().get_zero() << std::endl;
    std::cout << "Sign flag after ADDC: " << context.get_condition_codes().get_sign() << std::endl;
    std::cout << "Overflow flag after ADDC: " << context.get_condition_codes().get_overflow()
              << std::endl;

    // 验证结果
    assert(dst == 1);                     // 255 + 1 + 1 = 257, 低8位是1
    assert(context.get_condition_codes().get_carry() == true); // 应该设置进位标志

    std::cout << "Test 1 passed!" << std::endl;

    // 测试2: SUBC指令的借位功能
    std::cout << "\nTest 2: SUBC with borrow flag" << std::endl;

    // 重置条件码寄存器
    new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true); // 作为借位标志
    context.set_condition_codes(new_cc_reg);

    uint32_t src3 = 10;
    uint32_t src4 = 15;
    uint32_t dst2 = 0;

    void *operands2[3] = {&dst2, &src3, &src4};

    // 创建带CC修饰符的qualifiers
    std::vector<Qualifier> qualifiers2;
    qualifiers2.push_back(Qualifier::Q_U32);
    qualifiers2.push_back(Qualifier::Q_CC); // 添加.cc修饰符

    SUBC subc_handler;
    subc_handler.process_operation(&context, operands2, qualifiers2);

    std::cout << "Result: " << dst2 << std::endl;
    std::cout << "Carry flag after SUBC: " << context.get_condition_codes().get_carry() << std::endl;

    // 验证结果: 10 - 15 - 1 = -6, 由于是无符号数，结果是很大的正数
    // 且由于被减数小于减数+借位，应该设置借位标志（即carry标志）
    assert(context.get_condition_codes().get_carry() == true); // 10 < 15+1，应设置借位标志

    std::cout << "Test 2 passed!" << std::endl;

    // 测试3: 不带.cc修饰符的指令不更新条件码寄存器
    std::cout
        << "\nTest 3: ADDC without .cc modifier should not update CC register"
        << std::endl;

    // 设置初始状态
    new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true); // 设置为true
    context.set_condition_codes(new_cc_reg);
    bool initial_carry = context.get_condition_codes().get_carry();

    uint32_t src5 = 10;
    uint32_t src6 = 20;
    uint32_t dst3 = 0;

    void *operands3[3] = {&dst3, &src5, &src6};

    // 创建不带CC修饰符的qualifiers
    std::vector<Qualifier> qualifiers3;
    qualifiers3.push_back(Qualifier::Q_U32); // 不添加.cc修饰符

    // 记录执行前的条件码寄存器状态
    auto old_cc = context.get_condition_codes();

    addc_handler.process_operation(&context, operands3, qualifiers3);

    // 检查条件码寄存器是否保持不变
    assert(context.get_condition_codes().get_carry() == old_cc.get_carry());
    assert(context.get_condition_codes().get_zero() == old_cc.get_zero());
    assert(context.get_condition_codes().get_sign() == old_cc.get_sign());
    assert(context.get_condition_codes().get_overflow() == old_cc.get_overflow());

    std::cout << "Test 3 passed!" << std::endl;

    std::cout << "\nAll tests PASSED!" << std::endl;
}

int main() {
    test_cc_register();
    return 0;
}