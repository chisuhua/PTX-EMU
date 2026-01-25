#include "ptx_interpreter.h"
#include "cudart/cuda_driver.h" // 使用CudaDriver头文件
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/sm_context.h"
#include "utils/logger.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>

// 不再需要在这里声明g_gpu_context，已在头文件中声明

PtxInterpreter::PtxInterpreter()
    : ptxContext(nullptr), kernelContext(nullptr), kernelArgs(nullptr),
      param_space(nullptr) {
    // 不再创建 GPUContext
}

void PtxInterpreter::launchPtxInterpreter(PtxContext &ptx, std::string &kernel,
                                          void **args, Dim3 &gridDim,
                                          Dim3 &blockDim) {
    // 初始化指令工厂，注册所有指令处理器
    InstructionFactory::initialize();

    // 使用传入的ptx引用，而不是尝试访问可能已失效的引用
    this->ptxContext = &ptx;
    this->gridDim = gridDim;
    this->blockDim = blockDim;
    this->kernelArgs = args;
    this->param_space = nullptr; // 初始化param_space

    // 根据kernel名称获取kernelContext
    for (auto &e : ptx.ptxKernels) {
        if (e.kernelName == kernel) {
            this->kernelContext = &e;
            break;
        }
    }

    std::map<std::string, Symtable *> name2Sym;
    std::map<std::string, int> label2pc;

    funcInterpreter(name2Sym, label2pc, ptx, kernel, args, gridDim, blockDim);

    // 内核执行结束后，不再立即释放参数空间，而是通过回调机制在任务完成后释放
}

void PtxInterpreter::funcInterpreter(
    std::map<std::string, Symtable *> &name2Sym,
    std::map<std::string, int> &label2pc, PtxContext &ptx, std::string &kernel,
    void **args, Dim3 &gridDim, Dim3 &blockDim) {
    // Setup symbols
    setupConstantSymbols(name2Sym);
    setupKernelArguments(name2Sym);
    setupLabels(label2pc);

    // 构建KernelLaunchRequest并提交到全局GPUContext
    if (g_gpu_context) {
        // 只传递name2Sym和label2pc的所有权，statements由ptxContext持有
        auto name2sym_ptr =
            std::make_shared<std::map<std::string, Symtable *>>(name2Sym);
        auto label2pc_ptr =
            std::make_shared<std::map<std::string, int>>(label2pc);

        // 预先计算总的本地内存需求
        size_t total_local_memory_needed = 0;
        size_t local_mem_per_thread = 0;
        
        // 遍历语句查找本地内存声明，计算每个线程需要的本地内存大小
        for (const auto &stmt : kernelContext->kernelStatements) {
            if (stmt.statementType == S_LOCAL) {
                auto localStmt = (StatementContext::LOCAL *)stmt.statement;
                size_t element_size = Q2bytes(localStmt->dataType[0]);
                size_t var_size = element_size * localStmt->size;
                local_mem_per_thread += var_size;
            }
        }

        // 计算总的本地内存需求 (每个CTA的线程总数 * 每线程本地内存)
        int total_threads = gridDim.x * gridDim.y * gridDim.z * 
                           blockDim.x * blockDim.y * blockDim.z;
        total_local_memory_needed = total_threads * local_mem_per_thread;

        // 如果需要本地内存，则预先分配
        void *local_memory_base = nullptr;
        if (total_local_memory_needed > 0) {
            local_memory_base = CudaDriver::instance().malloc(total_local_memory_needed);
            if (!local_memory_base) {
                PTX_ERROR_EMU("Failed to allocate local memory of size %zu bytes", 
                             total_local_memory_needed);
            }
        }

        // 创建完成回调，用于在任务完成后释放参数空间和本地内存
        auto param_space_ptr = this->param_space; // 捕获当前param_space指针
        auto local_memory_ptr = local_memory_base; // 捕获本地内存指针
        auto local_mem_size = total_local_memory_needed; // 捕获本地内存大小
        auto completion_callback = [param_space_ptr, local_memory_ptr, local_mem_size]() {
            if (param_space_ptr) {
                PTX_DEBUG_EMU("Freeing PARAM space at %p", param_space_ptr);
                CudaDriver::instance().free(param_space_ptr);
            }
            
            if (local_memory_ptr && local_mem_size > 0) {
                PTX_DEBUG_EMU("Freeing LOCAL memory at %p, size %zu", 
                             local_memory_ptr, local_mem_size);
                CudaDriver::instance().free(local_memory_ptr);
            }
        };

        // 构建请求，statements由ptxContext持有，不转移所有权
        KernelLaunchRequest request(
            args, gridDim, blockDim,
            &kernelContext
                 ->kernelStatements, // 直接引用kernelContext中的statements
            name2sym_ptr, label2pc_ptr, 0, completion_callback);

        // 设置本地内存信息到请求中
        request.set_local_memory_info(local_memory_base, local_mem_per_thread);

        // 提交请求
        g_gpu_context->submit_kernel_request(std::move(request));
    }
}

void PtxInterpreter::setupConstantSymbols(
    std::map<std::string, Symtable *> &name2Sym) {
    if (!ptxContext) {
        PTX_DEBUG_EMU("ptxContext is null in setupConstantSymbols");
        return;
    }

    for (auto e : ptxContext->ptxStatements) {
        if (e.statementType != S_CONST)
            continue;

        Symtable *s = new Symtable();
        auto st = (StatementContext::CONST *)e.statement;
        if (!st) {
            delete s;
            continue;
        }

        assert(st->constDataType.size() == 1);
        s->name = st->constName;
        s->symType = st->constDataType.back();
        s->elementNum = st->constSize;
        s->byteNum = Q2bytes(st->constDataType.back());
        s->val = constName2addr[s->name];
        if (!s->val) {
            delete s;
            continue;
        }
        name2Sym[s->name] = s;
    }
}

void PtxInterpreter::setupKernelArguments(
    std::map<std::string, Symtable *> &name2Sym) {
    PTX_DEBUG_EMU("Setting up %zu kernel arguments",
                  kernelContext->kernelParams.size());

    // 计算参数总大小
    size_t total_param_size = 0;
    for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
        auto e = kernelContext->kernelParams[i];
        total_param_size +=
            Q2bytes(e.paramTypes[0]) * (e.paramNum ? e.paramNum : 1);
    }

    // 申请PARAM空间，使用 CudaDriver 提供的 malloc_param 函数
    if (total_param_size > 0) {
        this->param_space = CudaDriver::instance().malloc(total_param_size);
        if (this->param_space == nullptr) {
            PTX_DEBUG_EMU("Failed to allocate PARAM space of size %zu",
                          total_param_size);
            return; // 或者抛出异常
        }
        memset(this->param_space, 0, total_param_size);
        PTX_DEBUG_EMU("Allocated PARAM space of size %zu at %p",
                      total_param_size, this->param_space);
    } else {
        this->param_space = nullptr;
        PTX_DEBUG_EMU("No PARAM space needed, total_param_size is 0");
    }

    // 遍历参数，将值填入PARAM空间，并在符号表中记录地址
    size_t offset = 0;
    for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
        auto e = kernelContext->kernelParams[i];
        Symtable *s = new Symtable();
        s->name = e.paramName;
        s->elementNum = e.paramNum;
        s->symType = e.paramTypes[0];
        s->byteNum = Q2bytes(e.paramTypes[0]);

        // 计算当前参数大小
        size_t param_size = s->byteNum * (e.paramNum ? e.paramNum : 1);

        // 检查是否需要分配空间
        if (this->param_space != nullptr) {
            // 将参数值拷贝到PARAM空间
            memcpy((char *)this->param_space + offset, kernelArgs[i],
                   param_size);
            s->val = (uint64_t)((char *)this->param_space + offset);
        } else {
            s->val = (uint64_t)kernelArgs[i];
        }

        name2Sym[s->name] = s;
        offset += param_size;
        PTX_DEBUG_EMU(
            "Added kernel argument to name2Sym: name=%s, "
            "symbol_table_entry = %p, stored_value = 0x%llx,"
            "first_8_bytes_of_data = 0x%llx, param_size=%d, param_bytes=%d ",
            s->name.c_str(), s, s->val, *(uint64_t *)(s->val), param_size,
            s->byteNum);
    }
}

void PtxInterpreter::setupLabels(std::map<std::string, int> &label2pc) {
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        auto e = kernelContext->kernelStatements[i];
        if (e.statementType == S_DOLLOR) {
            auto s = (StatementContext::DOLLOR *)e.statement;
            label2pc[s->dollorName] = i;
        }
    }
}

void PtxInterpreter::set_ptx_context(const PtxContext &ptx) {
    // 存储ptxContext的副本而不是引用，以避免悬垂引用问题
    this->owned_ptx_context = std::make_unique<PtxContext>(ptx);
    this->ptxContext = this->owned_ptx_context.get();
}

PtxContext &PtxInterpreter::get_ptx_context() { return *this->ptxContext; }