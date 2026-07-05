#include "ptx_interpreter.h"
#include "cudart/cuda_driver.h" // 使用CudaDriver头文件
#include "ptx_ir/kernel_context.h"
#include "ptx_ir/statement_context.h"
#include "ptxsim/cta_context.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/instruction_factory.h"
#include "ptxsim/sm_context.h"
#include "utils/logger.h"
#include "ptx_parser/cfg_builder.h"  // SIMT v2.0 CFG analysis
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
                                          Dim3 &blockDim, size_t sharedMem) {
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

    // 安全检查：如果未找到kernel，打印调试信息并退出
    if (!this->kernelContext) {
        PTX_ERROR_EMU("Kernel not found: %s. Available kernels:", kernel.c_str());
        for (auto &e : ptx.ptxKernels) {
            PTX_ERROR_EMU("  - %s", e.kernelName.c_str());
        }
        return;
    }

    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;

    funcInterpreter(name2Sym, label2pc, ptx, kernel, args, gridDim, blockDim, sharedMem);

    // 内核执行结束后，不再立即释放参数空间，而是通过回调机制在任务完成后释放
}

void PtxInterpreter::funcInterpreter(
    std::map<std::string, std::unique_ptr<Symtable>> &name2Sym,
    std::map<std::string, int> &label2pc, PtxContext &ptx, std::string &kernel,
    void **args, Dim3 &gridDim, Dim3 &blockDim, size_t sharedMem) {
    // Setup symbols
    setupConstantSymbols(name2Sym);
    setupKernelArguments(name2Sym);

    // 将ptxStatements中的S_SHARED全局声明合并到kernelStatements
    // 必须在 setupLabels (CFG pass) 之前完成 — 否则插入会偏移后续 PC，
    // 使 CFG pass 已写入 operands[1] 的 reconvergence_pc 指向错误的指令。
    // 用 already_inserted 保护，避免多次 launch 时重复插入。
    {
        bool already_inserted = false;
        for (const auto &stmt : kernelContext->kernelStatements) {
            if (stmt.type == S_SHARED) {
                already_inserted = true;
                break;
            }
        }
        if (!already_inserted) {
            for (const auto &stmt : ptx.ptxStatements) {
                if (stmt.type == S_SHARED) {
                    kernelContext->kernelStatements.insert(
                        kernelContext->kernelStatements.begin(), stmt);
                }
            }
        }
    }

    setupLabels(label2pc);

    // Override barrier participation masks with launch-time blockDim.
    // PTX sources typically lack .reqntid directives, so the parser defaults to
    // 0xFFFFFFFF. The runtime blockDim is the authoritative thread count.
    {
        int total_threads = blockDim.x * blockDim.y * blockDim.z;
        total_threads = std::min(total_threads, 32);
        uint32_t mask = (total_threads >= 32) ? 0xFFFFFFFFu : ((1u << total_threads) - 1);

        for (auto &stmt : kernelContext->kernelStatements) {
            if (stmt.type == S_BAR_WARP_SYNC) {
                auto &barrier = std::get<BarWarpSyncInstr>(stmt.data);
                if (barrier.operands.size() >= 2) {
                    barrier.operands[0] = OperandContext{ImmOperand{std::to_string(mask)}};
                    barrier.operands[0].invalidatePhyAddr();
                } else if (barrier.operands.size() == 1) {
                    barrier.operands[0] = OperandContext{ImmOperand{std::to_string(mask)}};
                    barrier.operands[0].invalidatePhyAddr();
                }
            }
        }
    }

    // 构建KernelLaunchRequest并提交到全局GPUContext
    if (g_gpu_context) {
        // 预先计算总的本地内存需求
        size_t total_local_memory_needed = 0;
        size_t local_mem_per_thread = 0;

        // 遍历语句查找本地内存声明，计算每个线程需要的本地内存大小
        // BUGFIX (历史): 用 `size` 已设置则用之（总字节数），否则
        //   element_size * array_size（兼容 .b8 arr[N] 形式）。
        for (const auto &stmt : kernelContext->kernelStatements) {
            if (stmt.type == S_LOCAL) {
                const auto &localDecl =
                    std::get<DeclarationInstr>(stmt.data);
                size_t element_size = Q2bytes(localDecl.dataType);
                size_t var_size;
                if (localDecl.size) {
                    var_size = *localDecl.size;
                } else {
                    var_size =
                        element_size *
                        (localDecl.array_size > 0 ? localDecl.array_size : 1);
                }
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
        std::vector<std::pair<std::string, const DeclarationInstr *>>
            param_symbols;

        for (const auto &stmt : kernelContext->kernelStatements) {
            if (stmt.type == S_PARAM) {
                const auto &paramDecl =
                    std::get<DeclarationInstr>(stmt.data);

                size_t param_size = Q2bytes(paramDecl.dataType);
                // 考虑对齐，向上取整到8字节边界
                if (param_size % 8 != 0) {
                    param_size = ((param_size / 8) + 1) * 8;
                }
                total_param_size += param_size;

                // 记录参数符号信息
                param_symbols.push_back({paramDecl.name, &paramDecl});
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
            auto paramDecl = param_info.second;
            std::string param_name = param_info.first;

            size_t param_size = Q2bytes(paramDecl->dataType);
            // 考虑对齐，向上取整到8字节边界
            if (param_size % 8 != 0) {
                param_size = ((param_size / 8) + 1) * 8;
            }

            // 创建Symtable对象（unique_ptr 自动管理生命周期）
            auto s = std::make_unique<Symtable>();
            s->name = param_name;
            s->symType = paramDecl->dataType;
            s->elementNum = 1; // 默认为1，可根据需要调整
            s->byteNum = Q2bytes(paramDecl->dataType);

            // 设置参数在param空间中的地址
            if (param_base_addr != nullptr) {
                s->val = (uint64_t)((char *)param_base_addr +
                                    current_param_offset);
            } else {
                s->val = 0; // 如果param空间分配失败，设为0
            }

            // 添加到符号表（unique_ptr 替换旧值时自动释放旧 Symtable）
            // Capture logging values before std::move(s) invalidates s
            const std::string log_name = s->name;
            const uint64_t log_val = s->val;
            const size_t log_byteNum = s->byteNum;
            name2Sym[s->name] = std::move(s);

            PTX_DEBUG_EMU("Added param symbol: name=%s, addr=%p, size=%zu, "
                          "offset=%zu",
                          log_name.c_str(), (void *)log_val, log_byteNum,
                          current_param_offset);

            // 更新偏移
            current_param_offset += param_size;
        }

        // 收集所有的S_GLOBAL符号，计算总大小并分配空间
        size_t total_global_size = 0;
        std::vector<std::pair<std::string, const DeclarationInstr *>>
            global_symbols;

        // 遍历ptxStatements来查找全局符号（因为它们不在kernel内部）
        for (const auto &stmt : ptx.ptxStatements) {
            if (stmt.type == S_GLOBAL) {
                const auto &globalDecl =
                    std::get<DeclarationInstr>(stmt.data);

                // 计算全局变量大小
                size_t element_size = Q2bytes(globalDecl.dataType);
                size_t var_size = element_size *
                                  (globalDecl.size ? *globalDecl.size : 1);

                // 考虑对齐，向上取整到8字节边界
                if (var_size % 8 != 0) {
                    var_size = ((var_size / 8) + 1) * 8;
                }
                total_global_size += var_size;

                // 记录全局符号信息
                global_symbols.push_back({globalDecl.name, &globalDecl});
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
            auto globalDecl = global_info.second;
            std::string global_name = global_info.first;

            // element_size is per-element bytes (e.g., 1 for .b8, 4 for .b32)
            size_t element_size = Q2bytes(globalDecl->dataType);
            // Use array_size for both allocation and element count for consistency
            size_t array_size = globalDecl->array_size > 0 ? globalDecl->array_size : 1;
            size_t var_size = element_size * array_size;

            // 考虑对齐，向上取整到8字节边界
            if (var_size % 8 != 0) {
                var_size = ((var_size / 8) + 1) * 8;
            }

            // 创建Symtable对象（unique_ptr 自动管理生命周期）
            auto s = std::make_unique<Symtable>();
            s->name = global_name;
            s->symType = globalDecl->dataType;
            s->elementNum = array_size;
            s->byteNum = Q2bytes(globalDecl->dataType);

            // 设置全局变量在全局空间中的地址
            if (global_base_addr != nullptr) {
                s->val = (uint64_t)((char *)global_base_addr +
                                    current_global_offset);
            } else {
                s->val = 0; // 如果全局空间分配失败，设为0
            }

            // 添加到符号表（unique_ptr 替换旧值时自动释放旧 Symtable）
            // Capture logging + loop values before std::move(s) invalidates s
            const std::string log_name = s->name;
            const uint64_t log_val = s->val;
            const size_t log_byteNum = s->byteNum;
            const size_t log_elementNum = s->elementNum;
            name2Sym[s->name] = std::move(s);

            PTX_DEBUG_EMU("Added global symbol: name=%s, addr=%p, "
                          "size=%zu, offset=%zu",
                          log_name.c_str(), (void *)log_val, log_byteNum,
                          current_global_offset);

            // 初始化全局变量的值（如果有的话）
            if (!globalDecl->initValues.empty()) {
                void *dest_addr = (void *)((char *)global_base_addr +
                                           current_global_offset);

                for (size_t i = 0; i < globalDecl->initValues.size() &&
                                   i < log_elementNum;
                     ++i) {
                    switch (globalDecl->dataType) {
                    case Qualifier::Q_B8:
                    case Qualifier::Q_U8:
                    case Qualifier::Q_S8: {
                        char *target = (char *)dest_addr + i * element_size;
                        *target = static_cast<char>(globalDecl->initValues[i]);
                        break;
                    }
                    case Qualifier::Q_B16:
                    case Qualifier::Q_U16:
                    case Qualifier::Q_S16:
                    case Qualifier::Q_F16: {
                        short *target = (short *)((char *)dest_addr + i * element_size);
                        *target = static_cast<short>(globalDecl->initValues[i]);
                        break;
                    }
                    case Qualifier::Q_B32:
                    case Qualifier::Q_U32:
                    case Qualifier::Q_S32:
                    case Qualifier::Q_F32: {
                        int *target = (int *)((char *)dest_addr + i * element_size);
                        *target = static_cast<int>(globalDecl->initValues[i]);
                        break;
                    }
                    case Qualifier::Q_B64:
                    case Qualifier::Q_U64:
                    case Qualifier::Q_S64:
                    case Qualifier::Q_F64: {
                        long long *target = (long long *)((char *)dest_addr + i * element_size);
                        *target = static_cast<long long>(globalDecl->initValues[i]);
                        break;
                    }
                    default: {
                        int *target = (int *)((char *)dest_addr + i * element_size);
                        *target = static_cast<int>(globalDecl->initValues[i]);
                        break;
                    }
                    }
                }

                PTX_DEBUG_EMU(
                    "Initialized global symbol: name=%s with %zu values",
                    global_name.c_str(), globalDecl->initValues.size());
            }

            // 更新偏移
            current_global_offset += var_size;
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

        // 在所有符号（params, globals等）都添加到name2Sym之后，再创建共享指针
        // 这样可以确保KernelLaunchRequest获得的是包含完整符号信息的 map
        // 注意：name2Sym 的所有权已转移到 name2sym_ptr（map 含 unique_ptr 不可拷贝）
        auto name2sym_ptr =
            std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>(
                std::move(name2Sym));
        auto label2pc_ptr =
            std::make_shared<std::map<std::string, int>>(label2pc);

        // 构建请求，statements由ptxContext持有，不转移所有权
        KernelLaunchRequest request(
            args, gridDim, blockDim,
            &kernelContext
                 ->kernelStatements, // 直接引用kernelContext中的statements
            name2sym_ptr, label2pc_ptr, 0, completion_callback);

        // 设置本地内存信息到请求中
        request.set_local_memory_info(local_memory_base, local_mem_per_thread);

        // 设置动态共享内存大小
        request.shared_mem_size = sharedMem;

        // 提交请求
        g_gpu_context->submit_kernel_request(std::move(request));
    }
}

void PtxInterpreter::setupConstantSymbols(
    std::map<std::string, std::unique_ptr<Symtable>> &name2Sym) {
    if (!ptxContext) {
        PTX_DEBUG_EMU("ptxContext is null in setupConstantSymbols");
        return;
    }

    for (const auto &e : ptxContext->ptxStatements) {
        if (e.type == S_CONST || e.type == S_GLOBAL) {
            auto s = std::make_unique<Symtable>();
            const auto &decl = std::get<DeclarationInstr>(e.data);

            s->name = decl.name;
            s->symType = decl.dataType;
            s->elementNum = decl.size ? *decl.size : 1;
            s->byteNum = Q2bytes(decl.dataType);
            s->val = constName2addr[s->name];
            if (!s->val) {
                continue;  // unique_ptr 自动释放
            }
            name2Sym[s->name] = std::move(s);
        } else if (e.type == S_SHARED) {
            // 处理全局S_SHARED声明（如.extern __shared__）
            // 这些符号的地址将由CTAContext在运行时设置（动态共享内存）
            auto s = std::make_unique<Symtable>();
            const auto &decl = std::get<DeclarationInstr>(e.data);

            s->name = decl.name;
            s->symType = decl.dataType;
            s->elementNum = decl.array_size;  // 对于extern shared为0
            s->byteNum = Q2bytes(decl.dataType) * (decl.array_size > 0 ? decl.array_size : 1);
            s->val = 0;  // 动态共享内存地址在CTAContext中设置
            name2Sym[s->name] = std::move(s);
        }
    }
}

void PtxInterpreter::setupKernelArguments(
    std::map<std::string, std::unique_ptr<Symtable>> &name2Sym) {
    PTX_DEBUG_EMU("Setting up %zu kernel arguments",
                  kernelContext->kernelParams.size());

    const size_t pointerBytes =
        (this->ptxContext != nullptr && this->ptxContext->ptxAddressSize == 32)
            ? 4
            : 8;

    auto get_param_bytes = [pointerBytes](const ParamContext &p) -> size_t {
        if (p.byteSize > 0) {
            return p.byteSize;
        }
        if (!p.paramTypes.empty()) {
            bool hasPtrType = false;
            for (auto q : p.paramTypes) {
                if (q == Qualifier::Q_PTR) {
                    hasPtrType = true;
                    continue;
                }
                int b = Q2bytes(q);
                if (b > 0) {
                    return static_cast<size_t>(b);
                }
            }
            if (hasPtrType) {
                return pointerBytes;
            }
        }
        if (p.isPtr) {
            return pointerBytes;
        }
        return 0;
    };

    auto get_param_type = [](const ParamContext &p) -> Qualifier {
        if (!p.paramTypes.empty()) {
            return p.paramTypes[0];
        }
        return Qualifier::Q_U64;
    };

    // 计算参数总大小
    size_t total_param_size = 0;
    for (int i = 0; i < kernelContext->kernelParams.size(); i++) {
        auto e = kernelContext->kernelParams[i];
        size_t paramBytes = get_param_bytes(e);
        if (paramBytes == 0) {
            PTX_ERROR_EMU(
                "Cannot infer kernel parameter byte size: index=%d name=%s",
                i, e.paramName.c_str());
            this->param_space = nullptr;
            return;
        }
        total_param_size += paramBytes * (e.paramNum ? e.paramNum : 1);
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
        auto s = std::make_unique<Symtable>();
        s->name = e.paramName;
        s->elementNum = e.paramNum;
        s->symType = get_param_type(e);
        size_t param_bytes = get_param_bytes(e);
        if (param_bytes == 0) {
            PTX_ERROR_EMU(
                "Cannot infer kernel parameter byte size during mapping: "
                "index=%d name=%s",
                i, e.paramName.c_str());
            return;  // unique_ptr 在 s 析构时自动释放
        }
        s->byteNum = static_cast<int>(param_bytes);

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

        // Capture values before std::move(s) invalidates the unique_ptr
        const std::string log_name = s->name;
        const uint64_t log_val = s->val;
        const size_t log_byteNum = s->byteNum;
        const Symtable* log_s_ptr = s.get();
        name2Sym[s->name] = std::move(s);
        offset += param_size;
        uint64_t first8Bytes = 0;
        if (param_size > 0 && log_val != 0) {
            size_t previewSize = std::min(param_size, sizeof(first8Bytes));
            memcpy(&first8Bytes, reinterpret_cast<void *>(log_val), previewSize);
        }

        PTX_DEBUG_EMU(
            "Added kernel argument to name2Sym: name=%s, "
            "symbol_table_entry = %p, stored_value = 0x%llx,"
            "first_8_bytes_of_data = 0x%llx, param_size=%d, param_bytes=%d ",
            log_name.c_str(), log_s_ptr, log_val, first8Bytes, param_size,
            log_byteNum);
    }

    PTX_DEBUG_EMU("setupKernelArguments completed: symbol_count=%zu",
                  name2Sym.size());
}

void PtxInterpreter::setupLabels(std::map<std::string, int> &label2pc) {
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        const auto &e = kernelContext->kernelStatements[i];
        // Register label declarations (both S_LABEL and legacy S_DOLLOR formats)
        if (e.type == S_LABEL) {
            const auto &s = std::get<LabelInstr>(e.data);
            PTX_INFO_EMU("Registering label: '%s' at PC=%d", s.labelName.c_str(), i);
            label2pc[s.labelName] = i;
            if (!s.labelName.empty() && s.labelName[0] == '$') {
                label2pc[s.labelName.substr(1)] = i;
                PTX_INFO_EMU("  (also as: '%s' at PC=%d)", s.labelName.substr(1).c_str(), i);
            }
        } else if (e.type == S_DOLLOR) {
            const auto &s = std::get<DollarNameInstr>(e.data);
            label2pc[s.name] = i;
            PTX_INFO_EMU("Registering label: '%s' at PC=%d", s.name.c_str(), i);
            if (!s.name.empty() && s.name[0] == '$') {
                label2pc[s.name.substr(1)] = i;
                PTX_INFO_EMU("  (also as: '%s' at PC=%d)", s.name.substr(1).c_str(), i);
            }
        }
    }
    PTX_INFO_EMU("Total labels registered: %zu", label2pc.size());
    
    // SIMT v2.0: CFG analysis for reconvergence PC
    PTX_INFO_EMU("Running CFG analysis...");
    try {
        ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(
            kernelContext->kernelStatements, label2pc);
        ptx::cfg::PostDominatorMap postDoms = 
            ptx::cfg::CFGBuilder::computePostDominators(cfg);
        
        int updated_branches = 0;
        int updated_barriers = 0;
        int fallback_branches = 0;
        int fallback_barriers = 0;
        for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
            const auto &stmt = kernelContext->kernelStatements[i];
            
            // Get post-dominator for current PC
            auto it = postDoms.find(i);
            int reconvergence_pc = -1;
            if (it != postDoms.end() && it->second >= 0) {
                reconvergence_pc = it->second;
            }
            
            if (stmt.type == S_BRA) {
                auto &branch = std::get<BranchInstr>(
                    kernelContext->kernelStatements[i].data);
                int old_reconvergence = branch.reconvergence_pc;
                if (reconvergence_pc >= 0) {
                    branch.reconvergence_pc = reconvergence_pc;
                    updated_branches++;
                    PTX_DEBUG_EMU("CFG[PC=%d]: S_BRA updated - old_reconvergence_pc=%d, new_reconvergence_pc=%d",
                                 i, old_reconvergence, reconvergence_pc);
                } else {
                    // Fallback: use next instruction as reconvergence point
                    branch.reconvergence_pc = i + 1;
                    fallback_branches++;
                    PTX_DEBUG_EMU("CFG[PC=%d]: S_BRA FALLBACK - old_reconvergence_pc=%d, new_reconvergence_pc=%d (no post-dominator)",
                                 i, old_reconvergence, i + 1);
                    PTX_WARN_EMU("Branch at PC=%d: no valid post-dominator, using fallback pc=%d", i, i + 1);
                }
            }
            else if (stmt.type == S_BAR_WARP_SYNC) {
                auto &barrier = std::get<BarWarpSyncInstr>(
                    kernelContext->kernelStatements[i].data);
                if (barrier.operands.size() >= 2) {
                    int barrier_reconvergence = i + 1;
                    barrier.operands[1] = OperandContext{ImmOperand{std::to_string(barrier_reconvergence)}};
                    updated_barriers++;
                    PTX_INFO_EMU("CFG[PC=%d]: S_BAR_WARP_SYNC updated - new_reconvergence_pc=%d",
                                i, barrier_reconvergence);
                }
            }
            else if (stmt.type == S_BAR) {
                auto &barrier = std::get<BarrierInstr>(
                    kernelContext->kernelStatements[i].data);
                int old_reconvergence = barrier.reconvergence_pc;
                if (reconvergence_pc >= 0) {
                    barrier.reconvergence_pc = reconvergence_pc;
                    updated_barriers++;
                    PTX_DEBUG_EMU("CFG[PC=%d]: S_BAR updated - old_reconvergence_pc=%d, new_reconvergence_pc=%d",
                                 i, old_reconvergence, reconvergence_pc);
                } else {
                    barrier.reconvergence_pc = i + 1;
                    fallback_barriers++;
                    PTX_DEBUG_EMU("CFG[PC=%d]: S_BAR FALLBACK - old_reconvergence_pc=%d, new_reconvergence_pc=%d (no post-dominator)",
                                 i, old_reconvergence, i + 1);
                }
            }
        }
        
        PTX_INFO_EMU("CFG analysis complete: updated %d branches (%d fallback), %d barriers (%d fallback)",
                     updated_branches, fallback_branches, updated_barriers, fallback_barriers);
    } catch (const std::exception& e) {
        PTX_ERROR_EMU("CFG analysis failed: %s", e.what());
    }
}

void PtxInterpreter::set_ptx_context(const PtxContext &ptx) {
    // 存储ptxContext的副本而不是引用，以避免悬垂引用问题
    this->owned_ptx_context = std::make_unique<PtxContext>(ptx);
    this->ptxContext = this->owned_ptx_context.get();
}

PtxContext &PtxInterpreter::get_ptx_context() { return *this->ptxContext; }