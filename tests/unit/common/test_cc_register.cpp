#include "catch_amalgamated.hpp"
#include "ptx_ir/ptx_types.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include <cstdint>
#include <vector>

using namespace ptxsim;

TEST_CASE("CC register: ADDC with carry flag sets carry=true", "[cc][addc]") {
    ThreadContext context;

    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::ZERO_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::SIGN_INDEX, false);
    new_cc_reg.set_cc_reg(ConditionCodeRegister::OVERFLOW_INDEX, false);
    context.set_condition_codes(new_cc_reg);

    uint8_t src1 = 255;
    uint8_t src2 = 1;
    uint8_t dst = 0;
    void *operands[3] = {&dst, &src1, &src2};

    std::vector<ptxemu::ir::Qualifier> qualifiers;
    qualifiers.push_back(ptxemu::ir::Qualifier::Q_U8);
    qualifiers.push_back(ptxemu::ir::Qualifier::Q_CC);

    AddcHandler addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    REQUIRE(dst == 1);
    REQUIRE(context.get_condition_codes().get_carry() == true);
}

TEST_CASE("CC register: SUBC with borrow flag sets carry=true", "[cc][subc]") {
    ThreadContext context;

    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    context.set_condition_codes(new_cc_reg);

    uint32_t src1 = 10;
    uint32_t src2 = 15;
    uint32_t dst = 0;
    void *operands[3] = {&dst, &src1, &src2};

    std::vector<ptxemu::ir::Qualifier> qualifiers;
    qualifiers.push_back(ptxemu::ir::Qualifier::Q_U32);
    qualifiers.push_back(ptxemu::ir::Qualifier::Q_CC);

    SubcHandler subc_handler;
    subc_handler.processOperation(&context, operands, qualifiers);

    REQUIRE(context.get_condition_codes().get_carry() == true);
}

TEST_CASE("CC register: ADDC without .cc does not update CC", "[cc][addc]") {
    ThreadContext context;

    ConditionCodeRegister new_cc_reg = context.get_condition_codes();
    new_cc_reg.set_cc_reg(ConditionCodeRegister::CARRY_INDEX, true);
    context.set_condition_codes(new_cc_reg);
    auto old_cc = context.get_condition_codes();

    uint32_t src1 = 10;
    uint32_t src2 = 20;
    uint32_t dst = 0;
    void *operands[3] = {&dst, &src1, &src2};

    std::vector<ptxemu::ir::Qualifier> qualifiers;
    qualifiers.push_back(ptxemu::ir::Qualifier::Q_U32);

    AddcHandler addc_handler;
    addc_handler.processOperation(&context, operands, qualifiers);

    REQUIRE(context.get_condition_codes().get_carry() == old_cc.get_carry());
    REQUIRE(context.get_condition_codes().get_zero() == old_cc.get_zero());
    REQUIRE(context.get_condition_codes().get_sign() == old_cc.get_sign());
    REQUIRE(context.get_condition_codes().get_overflow() == old_cc.get_overflow());
}
