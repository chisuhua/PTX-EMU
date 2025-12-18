#include "ptxsim/instruction_factory.h"
// 包含所有指令处理器的头文件
#include "ptxsim/instruction_handlers/arithmetic_handler.h"
#include "ptxsim/instruction_handlers/bitwise_handler.h"
#include "ptxsim/instruction_handlers/control_handler.h"
#include "ptxsim/instruction_handlers/math_handler.h"
#include "ptxsim/instruction_handlers/memory_handler.h"
#include "ptxsim/instruction_handlers/misc_handler.h"
#include "ptxsim/instruction_handlers/remaining_handler.h"
#include "ptxsim/instruction_handlers/special_handler.h"
#include "ptxsim/instruction_handlers/structure_handler.h"
#include <functional>
#include <iostream>
#include <unordered_map>

std::unordered_map<StatementType, std::function<InstructionHandler *()>>
    InstructionFactory::handler_map;

InstructionHandler *InstructionFactory::create_handler(StatementType type) {
    auto it = handler_map.find(type);
    if (it != handler_map.end()) {
        return it->second();
    }
    std::cerr << "No handler registered for statement type: "
              << static_cast<int>(type) << std::endl;
    return nullptr;
}

void InstructionFactory::register_handler(
    StatementType type, std::function<InstructionHandler *()> creator) {
    handler_map[type] = creator;
}

void InstructionFactory::initialize() {
    // 注册算术运算指令处理器
    REGISTER_HANDLER(StatementType::S_ADD, ADD);
    REGISTER_HANDLER(StatementType::S_SUB, SUB);
    REGISTER_HANDLER(StatementType::S_MUL, MUL);
    REGISTER_HANDLER(StatementType::S_MUL24, MUL24);
    REGISTER_HANDLER(StatementType::S_DIV, DIV);
    REGISTER_HANDLER(StatementType::S_REM, REM);
    REGISTER_HANDLER(StatementType::S_MIN, MIN);
    REGISTER_HANDLER(StatementType::S_MAX, MAX);
    REGISTER_HANDLER(StatementType::S_NEG, NEG);
    REGISTER_HANDLER(StatementType::S_ABS, ABS);
    REGISTER_HANDLER(StatementType::S_MAD, MAD);
    REGISTER_HANDLER(StatementType::S_FMA, FMA);

    // 注册位运算指令处理器
    REGISTER_HANDLER(StatementType::S_AND, AND);
    REGISTER_HANDLER(StatementType::S_OR, OR);
    REGISTER_HANDLER(StatementType::S_XOR, XOR);
    REGISTER_HANDLER(StatementType::S_NOT, NOT);
    REGISTER_HANDLER(StatementType::S_SHL, SHL);
    REGISTER_HANDLER(StatementType::S_SHR, SHR);

    // 注册内存操作指令处理器
    REGISTER_HANDLER(StatementType::S_LD, LD);
    REGISTER_HANDLER(StatementType::S_ST, ST);
    REGISTER_HANDLER(StatementType::S_MOV, MOV);
    REGISTER_HANDLER(StatementType::S_CVT, CVT);
    REGISTER_HANDLER(StatementType::S_CVTA, CVTA);

    // 注册控制流指令处理器
    REGISTER_HANDLER(StatementType::S_BRA, BRA);
    REGISTER_HANDLER(StatementType::S_RET, RET);

    // 注册同步指令处理器
    REGISTER_HANDLER(StatementType::S_BAR, BAR);

    REGISTER_HANDLER(StatementType::S_SETP, SETP);
    REGISTER_HANDLER(StatementType::S_SELP, SELP);

    // 注册数学函数指令处理器
    REGISTER_HANDLER(StatementType::S_SIN, SIN);
    REGISTER_HANDLER(StatementType::S_COS, COS);
    REGISTER_HANDLER(StatementType::S_LG2, LG2);
    REGISTER_HANDLER(StatementType::S_EX2, EX2);
    REGISTER_HANDLER(StatementType::S_RCP, RCP);
    REGISTER_HANDLER(StatementType::S_RSQRT, RSQRT);
    REGISTER_HANDLER(StatementType::S_SQRT, SQRT);

    // 注册特殊指令处理器
    REGISTER_HANDLER(StatementType::S_ATOM, ATOM);
    REGISTER_HANDLER(StatementType::S_PRAGMA, PRAGMA);
    REGISTER_HANDLER(StatementType::S_AT, AT);

    // 注册其他指令处理器
    REGISTER_HANDLER(StatementType::S_WMMA, WMMA);

    // 注册基础结构指令处理器
    REGISTER_HANDLER(StatementType::S_REG, REG);
    REGISTER_HANDLER(StatementType::S_SHARED, SHARED);
    REGISTER_HANDLER(StatementType::S_LOCAL, LOCAL);
    REGISTER_HANDLER(StatementType::S_DOLLOR, DOLLOR);
}