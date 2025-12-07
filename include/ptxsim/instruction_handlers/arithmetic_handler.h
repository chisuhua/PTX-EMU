#ifndef ARITHMETIC_HANDLER_H
#define ARITHMETIC_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/thread_context.h"
#include <vector>

class ArithmeticHandler : public InstructionHandler {
public:
    virtual void execute(ThreadContext* context, StatementContext& stmt) = 0;
    
protected:
    virtual void process_operation(ThreadContext* context, 
                                  void* dst, void* src1, void* src2,
                                  std::vector<Qualifier>& qualifiers) = 0;
};

// 具体指令处理器
class AddHandler : public ArithmeticHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

class SubHandler : public ArithmeticHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

class MulHandler : public ArithmeticHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

class DivHandler : public ArithmeticHandler {
public:
    void execute(ThreadContext* context, StatementContext& stmt) override;
    
protected:
    void process_operation(ThreadContext* context, 
                          void* dst, void* src1, void* src2,
                          std::vector<Qualifier>& qualifiers) override;
};

#endif // ARITHMETIC_HANDLER_H