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
        int total_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x *
                            blockDim.y * blockDim.z;
        total_local_memory_needed = total_threads * local_mem_per_thread;

        // 如果需要本地内存，则预先分配
        void *local_memory_base = nullptr;
        if (total_local_memory_needed > 0) {
            local_memory_base =
                CudaDriver::instance().malloc(total_local_memory_needed);
            if (!local_memory_base) {
                PTX_ERROR_EMU(
                    "Failed to allocate local memory of size %zu bytes",
                    total_local_memory_needed);
            }
        }

        // 收集所有的S_PARAM符号，计算总大小并分配空间
        size_t total_param_size = 0;
        std::vector<std::pair<std::string, StatementContext::PARAM *>>
            param_symbols;

        for (const auto &stmt : kernelContext->kernelStatements) {
            if (stmt.statementType == S_PARAM) {
                auto paramStmt = (StatementContext::PARAM *)stmt.statement;
                if (!paramStmt)
                    continue;

                // 计算参数大小
                if (!paramStmt->dataType.empty()) {
                    size_t param_size = Q2bytes(paramStmt->dataType[0]);
                    // 考虑对齐，向上取整到8字节边界
                    if (param_size % 8 != 0) {
                        param_size = ((param_size / 8) + 1) * 8;
                    }
                    total_param_size += param_size;

                    // 记录参数符号信息
                    param_symbols.push_back({paramStmt->name, paramStmt});
                }
            }
        }

        // 为所有参数符号申请空间
        void *param_base_addr = nullptr;
        if (total_param_size > 0) {
            param_base_addr = CudaDriver::instance().malloc(total_param_size);
            if (param_base_addr == nullptr) {
                PTX_DEBUG_EMU("Failed to allocate PARAM space of size %zu",
                              total_param_size);
            } else {
                memset(param_base_addr, 0, total_param_size);
                PTX_DEBUG_EMU("Allocated PARAM space of size %zu at %p",
                              total_param_size, param_base_addr);
            }
        }

        // 根据偏移设置每个参数符号
        size_t current_param_offset = 0;
        for (const auto &param_info : param_symbols) {
            auto paramStmt = param_info.second;
            std::string param_name = param_info.first;

            if (!paramStmt->dataType.empty()) {
                size_t param_size = Q2bytes(paramStmt->dataType[0]);
                // 考虑对齐，向上取整到8字节边界
                if (param_size % 8 != 0) {
                    param_size = ((param_size / 8) + 1) * 8;
                }

                // 创建Symtable对象
                Symtable *s = new Symtable();
                s->name = param_name;
                s->symType = paramStmt->dataType[0];
                s->elementNum = 1; // 默认为1，可根据需要调整
                s->byteNum = Q2bytes(paramStmt->dataType[0]);

                // 设置参数在param空间中的地址
                if (param_base_addr != nullptr) {
                    s->val = (uint64_t)((char *)param_base_addr +
                                        current_param_offset);
                } else {
                    s->val = 0; // 如果param空间分配失败，设为0
                }

                // 添加到符号表（如果已有同名符号，替换它）
                if (name2Sym.find(s->name) != name2Sym.end()) {
                    // 删除旧的Symtable对象以避免内存泄漏
                    delete name2Sym[s->name];
                }
                name2Sym[s->name] = s;

                PTX_DEBUG_EMU("Added param symbol: name=%s, addr=%p, size=%zu, "
                              "offset=%zu",
                              s->name.c_str(), (void *)s->val, s->byteNum,
                              current_param_offset);

                // 更新偏移
                current_param_offset += param_size;
            }
        }

        // 收集所有的S_GLOBAL符号，计算总大小并分配空间
        size_t total_global_size = 0;
        std::vector<std::pair<std::string, StatementContext::GLOBAL *>>
            global_symbols;

        // 遍历ptxStatements来查找全局符号（因为它们不在kernel内部）
        for (const auto &stmt : ptx.ptxStatements) {
            if (stmt.statementType == S_GLOBAL) {
                auto globalStmt = (StatementContext::GLOBAL *)stmt.statement;
                if (!globalStmt)
                    continue;

                // 计算全局变量大小
                if (!globalStmt->dataType.empty()) {
                    size_t element_size = Q2bytes(globalStmt->dataType[0]);
                    size_t var_size = element_size * globalStmt->size;

                    // 考虑对齐，向上取整到8字节边界
                    if (var_size % 8 != 0) {
                        var_size = ((var_size / 8) + 1) * 8;
                    }
                    total_global_size += var_size;

                    // 记录全局符号信息
                    global_symbols.push_back({globalStmt->name, globalStmt});
                }
            }
        }

        // 为所有全局符号申请空间
        void *global_base_addr = nullptr;
        if (total_global_size > 0) {
            global_base_addr = CudaDriver::instance().malloc(total_global_size);
            if (global_base_addr == nullptr) {
                PTX_DEBUG_EMU("Failed to allocate GLOBAL space of size %zu",
                              total_global_size);
            } else {
                memset(global_base_addr, 0, total_global_size);
                PTX_DEBUG_EMU("Allocated GLOBAL space of size %zu at %p",
                              total_global_size, global_base_addr);
            }
        }

        // 根据偏移设置每个全局符号，并初始化其值
        size_t current_global_offset = 0;
        for (const auto &global_info : global_symbols) {
            auto globalStmt = global_info.second;
            std::string global_name = global_info.first;

            if (!globalStmt->dataType.empty()) {
                size_t element_size = Q2bytes(globalStmt->dataType[0]);
                size_t var_size = element_size * globalStmt->size;

                // 考虑对齐，向上取整到8字节边界
                if (var_size % 8 != 0) {
                    var_size = ((var_size / 8) + 1) * 8;
                }

                // 创建Symtable对象
                Symtable *s = new Symtable();
                s->name = global_name;
                s->symType = globalStmt->dataType[0];
                s->elementNum = globalStmt->size; // 数组大小
                s->byteNum = Q2bytes(globalStmt->dataType[0]);

                // 设置全局变量在全局空间中的地址
                if (global_base_addr != nullptr) {
                    s->val = (uint64_t)((char *)global_base_addr +
                                        current_global_offset);
                } else {
                    s->val = 0; // 如果全局空间分配失败，设为0
                }

                // 添加到符号表（如果已有同名符号，替换它）
                if (name2Sym.find(s->name) != name2Sym.end()) {
                    // 删除旧的Symtable对象以避免内存泄漏
                    delete name2Sym[s->name];
                }
                name2Sym[s->name] = s;

                PTX_DEBUG_EMU("Added global symbol: name=%s, addr=%p, "
                              "size=%zu, offset=%zu",
                              s->name.c_str(), (void *)s->val, s->byteNum,
                              current_global_offset);

                // 初始化全局变量的值（如果有的话）
                if (!globalStmt->initValues.empty()) {
                    void *dest_addr = (void *)((char *)global_base_addr +
                                               current_global_offset);
                    size_t element_size = Q2bytes(globalStmt->dataType[0]);

                    // 根据数据类型初始化值
                    for (size_t i = 0; i < globalStmt->initValues.size() &&
                                       i < globalStmt->size;
                         ++i) {
                        switch (globalStmt->dataType[0]) {
                        case Qualifier::Q_B8:
                        case Qualifier::Q_U8:
                        case Qualifier::Q_S8: {
                            char *target = (char *)dest_addr + i * element_size;
                            *target =
                                static_cast<char>(globalStmt->initValues[i]);
                            break;
                        }
                        case Qualifier::Q_B16:
                        case Qualifier::Q_U16:
                        case Qualifier::Q_S16:
                        case Qualifier::Q_F16: {
                            short *target =
                                (short *)((char *)dest_addr + i * element_size);
                            *target =
                                static_cast<short>(globalStmt->initValues[i]);
                            break;
                        }
                        case Qualifier::Q_B32:
                        case Qualifier::Q_U32:
                        case Qualifier::Q_S32:
                        case Qualifier::Q_F32: {
                            int *target =
                                (int *)((char *)dest_addr + i * element_size);
                            *target =
                                static_cast<int>(globalStmt->initValues[i]);
                            break;
                        }
                        case Qualifier::Q_B64:
                        case Qualifier::Q_U64:
                        case Qualifier::Q_S64:
                        case Qualifier::Q_F64: {
                            long long *target =
                                (long long *)((char *)dest_addr +
                                              i * element_size);
                            *target = static_cast<long long>(
                                globalStmt->initValues[i]);
                            break;
                        }
                        default: {
                            // 默认按int处理
                            int *target =
                                (int *)((char *)dest_addr + i * element_size);
                            *target =
                                static_cast<int>(globalStmt->initValues[i]);
                            break;
                        }
                        }
                    }

                    PTX_DEBUG_EMU(
                        "Initialized global symbol: name=%s with %zu values",
                        global_name.c_str(), globalStmt->initValues.size());
                }

                // 更新偏移
                current_global_offset += var_size;
            }
        }

        // 创建完成回调，用于在任务完成后释放参数空间和本地内存
        auto param_space_ptr = param_base_addr;    // 捕获param空间指针
        auto global_space_ptr = global_base_addr;  // 捕获global空间指针
        auto local_memory_ptr = local_memory_base; // 捕获本地内存指针
        auto local_mem_size = total_local_memory_needed; // 捕获本地内存大小
        auto completion_callback = [param_space_ptr, global_space_ptr,
                                    local_memory_ptr, local_mem_size]() {
            if (param_space_ptr) {
                PTX_DEBUG_EMU("Freeing PARAM space at %p", param_space_ptr);
                CudaDriver::instance().free(param_space_ptr);
            }

            if (global_space_ptr) {
                PTX_DEBUG_EMU("Freeing GLOBAL space at %p", global_space_ptr);
                CudaDriver::instance().free(global_space_ptr);
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