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
    REGISTER_HANDLER(StatementType::S_ADD, AddHandler);
    REGISTER_HANDLER(StatementType::S_SUB, SubHandler);
    REGISTER_HANDLER(StatementType::S_MUL, MulHandler);
    REGISTER_HANDLER(StatementType::S_DIV, DivHandler);
    REGISTER_HANDLER(StatementType::S_REM, RemHandler);
    REGISTER_HANDLER(StatementType::S_MIN, MinHandler);
    REGISTER_HANDLER(StatementType::S_MAX, MaxHandler);
    REGISTER_HANDLER(StatementType::S_NEG, NegHandler);
    REGISTER_HANDLER(StatementType::S_ABS, AbsHandler);
    REGISTER_HANDLER(StatementType::S_MAD, MadHandler);
    REGISTER_HANDLER(StatementType::S_FMA, FmaHandler);

    // 注册位运算指令处理器
    REGISTER_HANDLER(StatementType::S_AND, AndHandler);
    REGISTER_HANDLER(StatementType::S_OR, OrHandler);
    REGISTER_HANDLER(StatementType::S_XOR, XorHandler);
    REGISTER_HANDLER(StatementType::S_NOT, NotHandler);
    REGISTER_HANDLER(StatementType::S_SHL, ShlHandler);
    REGISTER_HANDLER(StatementType::S_SHR, ShrHandler);

    // 注册内存操作指令处理器
    REGISTER_HANDLER(StatementType::S_LD, LdHandler);
    REGISTER_HANDLER(StatementType::S_ST, StHandler);
    REGISTER_HANDLER(StatementType::S_MOV, MovHandler);
    REGISTER_HANDLER(StatementType::S_CVT, CvtHandler);
    REGISTER_HANDLER(StatementType::S_CVTA, CvtaHandler);

    // 注册控制流指令处理器
    REGISTER_HANDLER(StatementType::S_BRA, BraHandler);
    REGISTER_HANDLER(StatementType::S_RET, RetHandler);
    REGISTER_HANDLER(StatementType::S_BAR, BarHandler);
    REGISTER_HANDLER(StatementType::S_SETP, SetpHandler);
    REGISTER_HANDLER(StatementType::S_SELP, SelpHandler);

    // 注册数学函数指令处理器
    REGISTER_HANDLER(StatementType::S_SIN, SinHandler);
    REGISTER_HANDLER(StatementType::S_COS, CosHandler);
    REGISTER_HANDLER(StatementType::S_LG2, Lg2Handler);
    REGISTER_HANDLER(StatementType::S_EX2, Ex2Handler);
    REGISTER_HANDLER(StatementType::S_RCP, RcpHandler);
    REGISTER_HANDLER(StatementType::S_RSQRT, RsqrtHandler);
    REGISTER_HANDLER(StatementType::S_SQRT, SqrtHandler);

    // 注册特殊指令处理器
    REGISTER_HANDLER(StatementType::S_ATOM, AtomHandler);
    REGISTER_HANDLER(StatementType::S_PRAGMA, PragmaHandler);
    REGISTER_HANDLER(StatementType::S_AT, AtHandler);

    // 注册其他指令处理器
    REGISTER_HANDLER(StatementType::S_WMMA, WmmaHandler);

    // 注册基础结构指令处理器
    REGISTER_HANDLER(StatementType::S_REG, RegHandler);
    REGISTER_HANDLER(StatementType::S_SHARED, SharedHandler);
    REGISTER_HANDLER(StatementType::S_LOCAL, LocalHandler);
    REGISTER_HANDLER(StatementType::S_DOLLOR, DollorHandler);
}