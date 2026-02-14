#include "ptxsim/instruction_factory.h"
#include "ptxsim/instruction_handlers.h"

std::unordered_map<StatementType, InstructionHandler *>
    InstructionFactory::handler_map;
bool InstructionFactory::initialized = false;

void InstructionFactory::initialize() {
    if (initialized)
        return;

    // 为每种指令类型注册对应的 Handler 实例
    // 注意：Handler 类名为 type_name##Handler
#define X(enum_val, op_name, opstr, op_count, struct_kind, instr_kind) \
    handler_map[enum_val] = new opstr##Handler();
#include "ptx_ir/ptx_op.def"
#undef X

    initialized = true;
}

InstructionHandler *InstructionFactory::get_handler(StatementType type) {
    auto it = handler_map.find(type);
    if (it != handler_map.end()) {
        return it->second;
    }
    return nullptr;
}

// Clean up all allocated handlers
void InstructionFactory::cleanup() {
    for (auto& pair : handler_map) {
        delete pair.second;
    }
    handler_map.clear();
    initialized = false;
}
