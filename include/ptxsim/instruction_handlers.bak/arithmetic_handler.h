#ifndef ARITHMETIC_HANDLER_H
#define ARITHMETIC_HANDLER_H

#include "ptxsim/instruction_factory.h"
#include "ptxsim/ptx_debug.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/type_utils.h"
#include <vector>

class ArithmeticHandler : public InstructionHandler {
public:
    virtual void execute(ThreadContext *context, StatementContext &stmt) = 0;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) = 0;

    // 为需要4个操作数的指令（如MAD/FMA）提供额外的虚函数
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2, void *src3,
                                   std::vector<Qualifier> &qualifiers) {
        // 默认实现为空，具体的MAD/FMA处理器会重写此函数
    }
};

class ADD : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2,
                           std::vector<Qualifier> &qualifiers) override;
};

class SUB : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2,
                           std::vector<Qualifier> &qualifiers) override;
};

class MUL : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2,
                           std::vector<Qualifier> &qualifiers) override;
};

class MUL24 : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2,
                           std::vector<Qualifier> &qualifiers) override;
};

class DIV : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2,
                           std::vector<Qualifier> &qualifiers) override;
};

// FMA和MAD指令处理器
class MAD : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {};

    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2, void *src3,
                           std::vector<Qualifier> &qualifiers) override;
};

class MAD24 : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {};

    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2, void *src3,
                           std::vector<Qualifier> &qualifiers) override;
};

class FMA : public ArithmeticHandler {
public:
    void execute(ThreadContext *context, StatementContext &stmt) override;

protected:
    virtual void process_operation(ThreadContext *context, void *dst,
                                   void *src1, void *src2,
                                   std::vector<Qualifier> &qualifiers) {};
    void process_operation(ThreadContext *context, void *dst, void *src1,
                           void *src2, void *src3,
                           std::vector<Qualifier> &qualifiers) override;
};

#endif // ARITHMETIC_HANDLER_H