#include "register/condition_code_register.h"
#include <cstddef>
#include <stdexcept>

ConditionCodeRegister::ConditionCodeRegister() { reset(); }

void ConditionCodeRegister::set_cc_reg(size_t index, bool value) {
    if (index >= CC_REG_COUNT) {
        throw std::out_of_range("Condition code register index out of range");
    }
    cc_regs[index] = value;
}

bool ConditionCodeRegister::get_cc_reg(size_t index) const {
    if (index >= CC_REG_COUNT) {
        throw std::out_of_range("Condition code register index out of range");
    }
    return cc_regs[index];
}

void ConditionCodeRegister::set_cc_regs(
    const std::array<bool, CC_REG_COUNT> &values) {
    cc_regs = values;
}

const std::array<bool, ConditionCodeRegister::CC_REG_COUNT> &
ConditionCodeRegister::get_cc_regs() const {
    return cc_regs;
}

void ConditionCodeRegister::reset() {
    for (auto &reg : cc_regs) {
        reg = false;
    }
}