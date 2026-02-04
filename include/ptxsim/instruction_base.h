// instruction_base.h
#ifndef INSTRUCTION_HANDLE_H
#define INSTRUCTION_HANDLE_H

#include <memory>
#include <vector>

// Forward declaration
class ThreadContext;

// Include headers in order of dependency
#include "../ptx_ir/ptx_types.h"        // For Qualifier (no dependencies)
#include "../ptx_ir/operand_context.h"  // For OperandContext (depends on ptx_types.h)
#include "../ptx_ir/statement_context.h" // Depends on operand_context.h and ptx_types.h
#include "ptxsim/execution_types.h"     // For CpAsyncInstr and other instruction types

// 基础指令处理器接口
class InstructionHandler {
public:
    virtual ~InstructionHandler() = default;
    virtual void ExecPipe(ThreadContext *context, StatementContext &stmt) = 0;
};

// Base classes for different instruction categories
class DeclarationHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
};

class SimpleHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
};

class VoidHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
};

class BranchHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
    virtual void executeBranch(ThreadContext *context, const BranchInstr &instr) {
        // Default implementation does nothing
        (void)context;
        (void)instr;
    }
};

class BarrierHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
    virtual void executeBarrier(ThreadContext *context, const BarrierInstr &instr) {
        // Default implementation does nothing
        (void)context;
        (void)instr;
    }
};

class CallHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
    virtual void executeCall(ThreadContext *context, const CallInstr &instr) {
        // Default implementation does nothing
        (void)context;
        (void)instr;
    }
};

// Generic instruction handler with prepare/operate/commit pipeline
class PipelineHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
    
protected:
    virtual bool prepareOperands(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool executeOperation(ThreadContext *context, StatementContext &stmt) = 0;
    virtual bool commitResults(ThreadContext *context, StatementContext &stmt) = 0;
    
    // Helper methods for operand management
    bool acquireAllOperands(ThreadContext *context, std::vector<OperandContext> &operands, 
                           const std::vector<Qualifier> &qualifiers, int opCount);
    void releaseAllOperands(std::vector<OperandContext> &operands, int opCount);
};

// Specific pipeline handler types
class GenericPipelineHandler : public PipelineHandler {
protected:
    bool prepareOperands(ThreadContext *context, StatementContext &stmt) override;
    bool executeOperation(ThreadContext *context, StatementContext &stmt) override;
    bool commitResults(ThreadContext *context, StatementContext &stmt) override;
    
    virtual void processOperation(ThreadContext *context, void **operands, 
                                const std::vector<Qualifier> &qualifiers) {
        // Default implementation does nothing
        (void)context;
        (void)operands;
        (void)qualifiers;
    }
};

class AtomicPipelineHandler : public PipelineHandler {
protected:
    bool prepareOperands(ThreadContext *context, StatementContext &stmt) override;
    bool executeOperation(ThreadContext *context, StatementContext &stmt) override;
    bool commitResults(ThreadContext *context, StatementContext &stmt) override;
    
    virtual void processAtomicOperation(ThreadContext *context, void **operands, 
                                      const std::vector<Qualifier> &qualifiers) {
        // Default implementation does nothing
        (void)context;
        (void)operands;
        (void)qualifiers;
    }
};

class WmmaPipelineHandler : public PipelineHandler {
protected:
    bool prepareOperands(ThreadContext *context, StatementContext &stmt) override;
    bool executeOperation(ThreadContext *context, StatementContext &stmt) override;
    bool commitResults(ThreadContext *context, StatementContext &stmt) override;
    
    virtual void processWmmaOperation(ThreadContext *context, void **operands, 
                                    const std::vector<Qualifier> &qualifiers) {
        // Default implementation does nothing
        (void)context;
        (void)operands;
        (void)qualifiers;
    }
};

// Async memory copy instruction handler (e.g., cp.async)
class AsyncCopyHandler : public InstructionHandler {
public:
    void ExecPipe(ThreadContext *context, StatementContext &stmt) override;
protected:
    virtual void executeAsyncCopy(ThreadContext *context, const CpAsyncInstr &instr) {
        // Default implementation does nothing
        (void)context;
        (void)instr;
    }
};

#endif
