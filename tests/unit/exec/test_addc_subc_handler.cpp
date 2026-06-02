// test_addc_subc_handler.cpp
// 直接测试模拟器的 AddcHandler 和 SubcHandler 指令处理器
// 绕过 NVCC 编译，直接调用指令处理器的 processOperation 方法

#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include <cassert>
#include <iostream>
#include <cstdint>
#include <vector>

void test_addc_u32_with_carry() {
    std::cout << "Test: ADDC.u32 with carry" << std::endl;

    ThreadContext context;

    // 设置初始条件码寄存器状态 - 进位标志为 true
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::ZERO_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::SIGN_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::OVERFLOW_INDEX, false);
    context.set_condition_codes(new_cc_reg);

    // 模拟执行: 0xFFFFFFFF + 1 + carry(1) = 0x100000001, 低32位 = 1, 进位 = 1
    uint32_t src1 = 0xFFFFFFFF;
    uint32_t src2 = 1;
    uint32_t dst = 0;

    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U32);
    qualifiers.push_back(Qualifier::Q_CC); // 添加.cc修饰符

    AddcHandler addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    std::cout << "  Result: 0x" << std::hex << dst << std::dec << std::endl;
    std::cout << "  Carry after: " << context.get_condition_codes().get_carry() << std::endl;
    std::cout << "  Zero after: " << context.get_condition_codes().get_zero() << std::endl;

    // 验证结果
    // 0xFFFFFFFF + 1 = 0x100000000, 低32位 = 0, 进位 = 1
    // 再加 carry(1) = 0x100000001, 低32位 = 1, 进位 = 1
    assert(dst == 1);
    assert(context.get_condition_codes().get_carry() == true);

    std::cout << "  PASSED" << std::endl;
}

void test_addc_u32_without_carry() {
    std::cout << "Test: ADDC.u32 without carry" << std::endl;

    ThreadContext context;

    // 设置进位为 false
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, false);
    context.set_condition_codes(new_cc_reg);

    // 模拟执行: 100 + 200 = 300
    uint32_t src1 = 100;
    uint32_t src2 = 200;
    uint32_t dst = 0;

    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U32);
    qualifiers.push_back(Qualifier::Q_CC);

    AddcHandler addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    std::cout << "  Result: " << dst << std::endl;
    std::cout << "  Carry after: " << context.get_condition_codes().get_carry() << std::endl;

    assert(dst == 300);
    assert(context.get_condition_codes().get_carry() == false);

    std::cout << "  PASSED" << std::endl;
}

void test_subc_u32_with_borrow() {
    std::cout << "Test: SUBC.u32 with borrow" << std::endl;

    ThreadContext context;

    // 设置借位为 true (carry 标志在 SUBC 中作为 borrow 使用)
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    context.set_condition_codes(new_cc_reg);

    // 模拟执行: 0 - 1 - borrow(1) = -2 = 0xFFFFFFFE
    uint32_t src1 = 0;
    uint32_t src2 = 1;
    uint32_t dst = 0;

    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U32);
    qualifiers.push_back(Qualifier::Q_CC);

    SubcHandler subc_handler;
    subc_handler.processOperation(&context, operands, qualifiers);

    std::cout << "  Result: 0x" << std::hex << dst << std::dec << std::endl;
    std::cout << "  Carry (borrow) after: " << context.get_condition_codes().get_carry() << std::endl;

    // 0 - 1 - 1 = -2 (作为无符号数是 0xFFFFFFFE)
    assert(dst == 0xFFFFFFFE);
    assert(context.get_condition_codes().get_carry() == true);

    std::cout << "  PASSED" << std::endl;
}

void test_subc_u32_without_borrow() {
    std::cout << "Test: SUBC.u32 without borrow" << std::endl;

    ThreadContext context;

    // 设置借位为 false
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, false);
    context.set_condition_codes(new_cc_reg);

    // 模拟执行: 200 - 100 = 100
    uint32_t src1 = 200;
    uint32_t src2 = 100;
    uint32_t dst = 0;

    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U32);
    qualifiers.push_back(Qualifier::Q_CC);

    SubcHandler subc_handler;
    subc_handler.processOperation(&context, operands, qualifiers);

    std::cout << "  Result: " << dst << std::endl;
    std::cout << "  Carry (borrow) after: " << context.get_condition_codes().get_carry() << std::endl;

    assert(dst == 100);
    assert(context.get_condition_codes().get_carry() == false);

    std::cout << "  PASSED" << std::endl;
}

void test_addc_u8() {
    std::cout << "Test: ADDC.u8" << std::endl;

    ThreadContext context;

    // 设置进位为 true
    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    context.set_condition_codes(new_cc_reg);

    // 模拟执行: 255 + 1 + carry(1) = 257, 低8位 = 1, 进位 = 1
    uint8_t src1 = 255;
    uint8_t src2 = 1;
    uint8_t dst = 0;

    void *operands[3] = {&dst, &src1, &src2};

    std::vector<Qualifier> qualifiers;
    qualifiers.push_back(Qualifier::Q_U8);
    qualifiers.push_back(Qualifier::Q_CC);

    AddcHandler addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    std::cout << "  Result: " << (int)dst << std::endl;
    std::cout << "  Carry after: " << context.get_condition_codes().get_carry() << std::endl;

    assert(dst == 1);
    assert(context.get_condition_codes().get_carry() == true);

    std::cout << "  PASSED" << std::endl;
}

int main() {
    std::cout << "=== Testing Addc/Subc Handlers ===" << std::endl;

    try {
        test_addc_u32_without_carry();
        test_addc_u32_with_carry();
        test_addc_u8();
        test_subc_u32_without_borrow();
        test_subc_u32_with_borrow();

        std::cout << "\n=== All tests PASSED! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}
