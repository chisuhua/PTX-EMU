#ifndef INSTRUCTION_FACTORY_H
#define INSTRUCTION_FACTORY_H

#include "ptx_ir/statement_context.h"
#include <unordered_map>
#include <functional>
#include <iostream>

// Forward declaration
class ThreadContext;

class InstructionHandler {
public:
    virtual ~InstructionHandler() = default;
    virtual void execute(ThreadContext* context, StatementContext& stmt) = 0;
};

// 定义注册宏，方便注册指令处理器
#define REGISTER_HANDLER(type, handler_class) \
    InstructionFactory::register_handler(type, []() -> InstructionHandler* { \
        return new handler_class(); \
    })

class InstructionFactory {
public:
    static InstructionHandler* create_handler(StatementType type);
    
    // 注册指令处理器
    static void register_handler(StatementType type, 
                               std::function<InstructionHandler*()> creator);
                               
    // 初始化函数，用于注册所有指令处理器
    static void initialize();
    
private:
    static std::unordered_map<StatementType, 
                            std::function<InstructionHandler*()>> handler_map;
};


#endif // INSTRUCTION_FACTORY_H