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

// g_gpu_context 定义搬迁自 cudart_sim.cpp:92 per ADR-0021 v1.1 amendment
std::unique_ptr<GPUContext> g_gpu_context;

extern "C" size_t get_gpu_clock_from_context() {
    if (g_gpu_context) {
        return g_gpu_context->get_clock();
    }
    return 0;
}

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

KernelLaunchRequest PtxInterpreter::prepareKernelLaunchRequest(
    PtxContext &ptx, const std::string &kernel, void **args, Dim3 &gridDim,
    Dim3 &blockDim, size_t sharedMem) {

    // 确保 InstructionFactory 已初始化（bridge 路径不走 launchPtxInterpreter，
    // 其调用的 initialize() 是唯一的初始化入口；此处添加幂等调用以覆盖 bridge 路径）
    InstructionFactory::initialize();

    // 根据 kernel 名称获取 kernelContext（同 launchPtxInterpreter）
    this->ptxContext = &ptx;
    this->gridDim = gridDim;
    this->blockDim = blockDim;
    this->kernelArgs = args;
    this->param_space = nullptr;
    this->kernelContext = nullptr;
    for (auto &e : ptx.ptxKernels) {
        if (e.kernelName == kernel) {
            this->kernelContext = &e;
            break;
        }
    }
    if (!this->kernelContext) {
        PTX_ERROR_EMU("Kernel not found: %s. Available kernels:", kernel.c_str());
        for (auto &e : ptx.ptxKernels) {
            PTX_ERROR_EMU("  - %s", e.kernelName.c_str());
        }
        // Return a default-constructed request (caller must check on_complete==null)
        return KernelLaunchRequest();
    }

    std::map<std::string, std::unique_ptr<Symtable>> name2Sym;
    std::map<std::string, int> label2pc;

    // Setup symbols
    setupConstantSymbols(name2Sym);
    setupKernelArguments(name2Sym);

    // 将ptxStatements中的S_SHARED全局声明合并到kernelStatements
    // 必须在 setupLabels (CFG pass) 之前完成
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

    // Override barrier participation masks with launch-time blockDim
    {
        int total_threads = blockDim.x * blockDim.y * blockDim.z;
        total_threads = std::min(total_threads, 32);
        uint32_t mask = (total_threads >= 32) ? 0xFFFFFFFFu : ((1u << total_threads) - 1);

        for (auto &stmt : kernelContext->kernelStatements) {
            if (stmt.type == S_BAR_WARP_SYNC) {
                auto &barrier = std::get<BarWarpSyncInstr>(stmt.data);
                // Phase 0.3d: invalidatePhyAddr removed (operand_phy_addr
                // field gone). Reassigning OperandContext already defaults
                // to nullptr phy_addr; cache is cleared on next acquire.
                (void)mask;
                if (barrier.operands.size() >= 2) {
                    barrier.operands[0] = ptxemu::ir::OperandContext{ImmOperand{std::to_string(mask)}};
                } else if (barrier.operands.size() == 1) {
                    barrier.operands[0] = ptxemu::ir::OperandContext{ImmOperand{std::to_string(mask)}};
                }
            }
        }
    }

    if (!g_gpu_context) {
        return KernelLaunchRequest();
    }

    // ── 本地内存分配 ──
    size_t total_local_memory_needed = 0;
    size_t local_mem_per_thread = 0;
    for (const auto &stmt : kernelContext->kernelStatements) {
        if (stmt.type == S_LOCAL) {
            const auto &localDecl = std::get<DeclarationInstr>(stmt.data);
            size_t element_size = Q2bytes(localDecl.dataType);
            size_t var_size;
            if (localDecl.size) {
                var_size = *localDecl.size;
            } else {
                var_size = element_size *
                           (localDecl.array_size > 0 ? localDecl.array_size : 1);
            }
            local_mem_per_thread += var_size;
        }
    }

    int total_threads = gridDim.x * gridDim.y * gridDim.z *
                        blockDim.x * blockDim.y * blockDim.z;
    total_local_memory_needed = total_threads * local_mem_per_thread;

    void *local_memory_base = nullptr;
    if (total_local_memory_needed > 0) {
        local_memory_base = CudaDriver::instance().malloc(total_local_memory_needed);
        if (!local_memory_base) {
            PTX_ERROR_EMU("Failed to allocate local memory of size %zu bytes",
                          total_local_memory_needed);
        }
    }

    // ── PARAM 空间分配 ──
    size_t total_param_size = 0;
    std::vector<std::pair<std::string, const DeclarationInstr *>> param_symbols;
    for (const auto &stmt : kernelContext->kernelStatements) {
        if (stmt.type == S_PARAM) {
            const auto &paramDecl = std::get<DeclarationInstr>(stmt.data);
            size_t param_size = Q2bytes(paramDecl.dataType);
            if (param_size % 8 != 0)
                param_size = ((param_size / 8) + 1) * 8;
            total_param_size += param_size;
            param_symbols.push_back({paramDecl.name, &paramDecl});
        }
    }

    void *param_base_addr = nullptr;
    if (total_param_size > 0) {
        param_base_addr = CudaDriver::instance().malloc(total_param_size);
        if (param_base_addr) {
            memset(param_base_addr, 0, total_param_size);
            PTX_DEBUG_EMU("Allocated PARAM space of size %zu at %p",
                          total_param_size, param_base_addr);
        }
    }

    size_t current_param_offset = 0;
    for (const auto &param_info : param_symbols) {
        auto paramDecl = param_info.second;
        size_t param_size = Q2bytes(paramDecl->dataType);
        if (param_size % 8 != 0)
            param_size = ((param_size / 8) + 1) * 8;
        auto s = std::make_unique<Symtable>();
        s->name = param_info.first;
        s->symType = paramDecl->dataType;
        s->elementNum = 1;
        s->byteNum = Q2bytes(paramDecl->dataType);
        s->val = param_base_addr
                     ? (uint64_t)((char *)param_base_addr + current_param_offset)
                     : 0;
        name2Sym[s->name] = std::move(s);
        current_param_offset += param_size;
    }

    // ── GLOBAL 空间分配 ──
    size_t total_global_size = 0;
    std::vector<std::pair<std::string, const DeclarationInstr *>> global_symbols;
    for (const auto &stmt : ptx.ptxStatements) {
        if (stmt.type == S_GLOBAL) {
            const auto &globalDecl = std::get<DeclarationInstr>(stmt.data);
            size_t element_size = Q2bytes(globalDecl.dataType);
            size_t var_size = element_size * (globalDecl.size ? *globalDecl.size : 1);
            if (var_size % 8 != 0)
                var_size = ((var_size / 8) + 1) * 8;
            total_global_size += var_size;
            global_symbols.push_back({globalDecl.name, &globalDecl});
        }
    }

    void *global_base_addr = nullptr;
    if (total_global_size > 0) {
        global_base_addr = CudaDriver::instance().malloc(total_global_size);
        if (global_base_addr) {
            memset(global_base_addr, 0, total_global_size);
            PTX_DEBUG_EMU("Allocated GLOBAL space of size %zu at %p",
                          total_global_size, global_base_addr);
        }
    }

    size_t current_global_offset = 0;
    for (const auto &global_info : global_symbols) {
        auto globalDecl = global_info.second;
        size_t element_size = Q2bytes(globalDecl->dataType);
        size_t array_size = globalDecl->array_size > 0 ? globalDecl->array_size : 1;
        size_t var_size = element_size * array_size;
        if (var_size % 8 != 0)
            var_size = ((var_size / 8) + 1) * 8;
        auto s = std::make_unique<Symtable>();
        s->name = global_info.first;
        s->symType = globalDecl->dataType;
        s->elementNum = array_size;
        s->byteNum = Q2bytes(globalDecl->dataType);
        s->val = global_base_addr
                     ? (uint64_t)((char *)global_base_addr + current_global_offset)
                     : 0;

        // 初始化全局变量值
        if (!globalDecl->initValues.empty() && global_base_addr) {
            void *dest_addr = (void *)((char *)global_base_addr + current_global_offset);
            for (size_t i = 0; i < globalDecl->initValues.size() && i < array_size; ++i) {
                char *target = (char *)dest_addr + i * element_size;
                switch (globalDecl->dataType) {
                case ptxemu::ir::Qualifier::Q_B8:  case ptxemu::ir::Qualifier::Q_U8:  case ptxemu::ir::Qualifier::Q_S8:
                    *target = static_cast<char>(globalDecl->initValues[i]);  break;
                case ptxemu::ir::Qualifier::Q_B16: case ptxemu::ir::Qualifier::Q_U16: case ptxemu::ir::Qualifier::Q_S16:
                case ptxemu::ir::Qualifier::Q_F16:
                    *(short *)target = static_cast<short>(globalDecl->initValues[i]); break;
                case ptxemu::ir::Qualifier::Q_B32: case ptxemu::ir::Qualifier::Q_U32: case ptxemu::ir::Qualifier::Q_S32:
                case ptxemu::ir::Qualifier::Q_F32:
                    *(int *)target = static_cast<int>(globalDecl->initValues[i]); break;
                case ptxemu::ir::Qualifier::Q_B64: case ptxemu::ir::Qualifier::Q_U64: case ptxemu::ir::Qualifier::Q_S64:
                case ptxemu::ir::Qualifier::Q_F64:
                    *(long long *)target = static_cast<long long>(globalDecl->initValues[i]); break;
                default:
                    *(int *)target = static_cast<int>(globalDecl->initValues[i]); break;
                }
            }
        }
        name2Sym[s->name] = std::move(s);
        current_global_offset += var_size;
    }

    // ── 完成回调（内存释放）──
    auto param_space_ptr = param_base_addr;
    auto global_space_ptr = global_base_addr;
    auto local_memory_ptr = local_memory_base;
    auto local_mem_size = total_local_memory_needed;
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

    auto name2sym_ptr =
        std::make_shared<std::map<std::string, std::unique_ptr<Symtable>>>(
            std::move(name2Sym));
    auto label2pc_ptr =
        std::make_shared<std::map<std::string, int>>(label2pc);

    KernelLaunchRequest request(
        args, gridDim, blockDim,
        &kernelContext->kernelStatements,
        name2sym_ptr, label2pc_ptr, 0, completion_callback);
    request.set_local_memory_info(local_memory_base, local_mem_per_thread);
    request.shared_mem_size = sharedMem;

    return request;
}

void PtxInterpreter::funcInterpreter(
    std::map<std::string, std::unique_ptr<Symtable>> &name2Sym,
    std::map<std::string, int> &label2pc, PtxContext &ptx, std::string &kernel,
    void **args, Dim3 &gridDim, Dim3 &blockDim, size_t sharedMem) {

    // Delegate all IR setup, symbol resolution, label mapping, CFG analysis,
    // and memory allocation to prepareKernelLaunchRequest. The output
    // name2Sym/label2pc references are intentionally NOT populated — callers
    // (launchPtxInterpreter) do not use them after this function returns.
    auto request = prepareKernelLaunchRequest(ptx, kernel, args, gridDim,
                                              blockDim, sharedMem);
    if (g_gpu_context) {
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
                if (q == ptxemu::ir::Qualifier::Q_PTR) {
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

    auto get_param_type = [](const ParamContext &p) -> ptxemu::ir::Qualifier {
        if (!p.paramTypes.empty()) {
            return p.paramTypes[0];
        }
        return ptxemu::ir::Qualifier::Q_U64;
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
                    barrier.operands[1] = ptxemu::ir::OperandContext{ImmOperand{std::to_string(barrier_reconvergence)}};
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