#include "ptxsim/instruction_handlers/structure_handler.h"
#include "ptxsim/utils/type_utils.h"
#include "ptxsim/interpreter.h"
#include "ptxsim/thread_context.h"
#include <iostream>
#include <cstring>
#include <cstdint>

void RegHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::REG*)stmt.statement;
    for (int i = 0; i < ss->regNum; i++) {
        PtxInterpreter::Reg *r = new PtxInterpreter::Reg();
        r->byteNum = TypeUtils::get_bytes(ss->regDataType);
        r->elementNum = 1;
        // 存储完整的寄存器名称，包括索引部分
        r->name = ss->regName + std::to_string(i);
        r->regType = ss->regDataType.back();
        r->addr = new char[r->byteNum];
        memset(r->addr, 0, r->byteNum);
        context->name2Reg[r->name] = r;
        std::cout << "Registered register: " << r->name << std::endl;
    }
}

void SharedHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::SHARED*)stmt.statement;
    PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();
    s->byteNum = TypeUtils::get_bytes(ss->sharedDataType) * ss->sharedSize;
    s->elementNum = ss->sharedSize;
    s->name = ss->sharedName;
    s->symType = ss->sharedDataType.back();
    s->val = (uint64_t)(new char[s->byteNum]);
    memset((void *)s->val, 0, s->byteNum);
    (*context->name2Share)[s->name] = s;
    context->name2Sym[s->name] = s;
}

void LocalHandler::execute(ThreadContext* context, StatementContext& stmt) {
    auto ss = (StatementContext::LOCAL*)stmt.statement;
    PtxInterpreter::Symtable *s = new PtxInterpreter::Symtable();
    s->byteNum = TypeUtils::get_bytes(ss->localDataType) * ss->localSize;
    s->elementNum = ss->localSize;
    s->name = ss->localName;
    s->symType = ss->localDataType.back();
    s->val = (uint64_t)(new char[s->byteNum]);
    memset((void *)s->val, 0, s->byteNum);
    context->name2Sym[s->name] = s;
}

void DollorHandler::execute(ThreadContext* context, StatementContext& stmt) {
    // Labels are already set up before execution starts
    // Nothing to do here during execution
    auto ss = (StatementContext::DOLLOR*)stmt.statement;
    // 可以添加一些调试输出或记录标签信息
}