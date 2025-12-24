#include "ptxsim/instruction_factory.h"
#include "ptxsim/instruction_handlers_decl.h"

std::unordered_map<StatementType, InstructionHandler *>
    InstructionFactory::handler_map;
bool InstructionFactory::initialized = false;

void InstructionFactory::initialize() {
    if (initialized)
        return;

// 为每种指令类型注册处理器
#define X(enum_val, type_name, str, op_count, struct_kind)                     \
    handler_map[enum_val] = new type_name();
#include "ptx_ir/ptx_op.def"
#undef X

    initialized = true;
}

InstructionHandler *InstructionFactory::create_handler(StatementType type) {
    auto it = handler_map.find(type);
    if (it != handler_map.end()) {
        return it->second;
    }
    return nullptr;
}