#include "ptxsim/instruction_factory.h"
// 包含所有指令处理器的头文件
#include "ptxsim/instruction_handlers/arithmetic_handler.h"
#include "ptxsim/instruction_handlers/bit_manipulate.h"
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
#define X(enum_val, type_name, str)                                            \
    REGISTER_HANDLER(StatementType::enum_val, type_name);
#include "ptx_ir/ptx_op.def"
#undef X
}