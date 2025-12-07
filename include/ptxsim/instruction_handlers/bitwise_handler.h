#ifndef BITWISE_HANDLER_H
#define BITWISE_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/thread_context.h"
#include "ptx_ir/ptx_types.h"
#include <vector>

class BitwiseHandler : public InstructionHandler {
public:
    virtual void execute(ThreadContext* context, StatementContext& stmt) = 0;
    
protected:
    virtual void process_operation(ThreadContext* context, 
                                  void* dst, void* src1, void* src2,
                                  std::vector<Qualifier>& qualifiers) = 0;
};

// AND指令处理器
class AndHandler : public BitwiseHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

// OR指令处理器
class OrHandler : public BitwiseHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

// XOR指令处理器
class XorHandler : public BitwiseHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

// SHL指令处理器
class ShlHandler : public BitwiseHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

// SHR指令处理器
class ShrHandler : public BitwiseHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

#endif // BITWISE_HANDLER_H