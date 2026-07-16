/**
 * @author gtyinstinct
 * generate fake libcudart.so to replace origin libcudart.so
 */

#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptx_parser/ptx_visiter.h"
#include "cudart/cuda_driver.h"       // 替换为新的驱动内存管理器
#include "cudart/cudart_intrinsics.h" // 添加缺失的CUDA类型定义
#include "cudart/ptx_interpreter.h"

using namespace antlr4;
using namespace ptxparser;
#include "inipp/inipp.h"
#include "memory/simple_memory.h"
#include "ptx_interpreter.h"
// #include "ptx_parser/ptx_grammar.h" // 添加解析器相关的头文件
// #include "ptx_parser/ptx_parser.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/ptx_config.h"
#include "ptxsim/ptx_exceptions.h" // 添加DebugConfig所需的头文件
#include "utils/cubin_utils.h" // 添加cuobjdump工具函数
#include "utils/logger.h"

#include <stdexcept> // std::runtime_error for fatal initialization errors
#include <string>
#include <fstream>
#include <filesystem>
#include <cstdio>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

// ============================================================================
// SingletonGuard (D-PTX-2): 检测 4 个全局单例的重复初始化
// ============================================================================
// F12b-LD 文档 §10.1 明确指 PTX-EMU 单例在多实例仿真中导致静默状态损坏。
// SingletonGuard 在 __cudaRegisterFatBinary 入口检测重复调用，FATAL abort。
// ============================================================================
class SingletonGuard {
public:
    static SingletonGuard& instance() {
        static SingletonGuard guard;
        return guard;
    }

    // 返回 true 如果已经初始化过（重复调用）
    bool check_and_mark() {
        if (initialized_) {
            return true;  // 重复初始化
        }
        initialized_ = true;
        return false;  // 首次初始化
    }

    void reset() {
        initialized_ = false;
    }

private:
    SingletonGuard() = default;
    ~SingletonGuard() = default;
    SingletonGuard(const SingletonGuard&) = delete;
    SingletonGuard& operator=(const SingletonGuard&) = delete;

    bool initialized_ = false;
};

// 新增缺失的全局变量和数据结构
std::map<uint64_t, std::string> func2name;
std::map<uint64_t, cudaKernel_t> func2kernel;
std::map<cudaKernel_t, const char *> kernel2func;
dim3 _gridDim, _blockDim;
size_t _sharedMem;

// 全局GPUContext和PtxInterpreter实例
std::unique_ptr<GPUContext> g_gpu_context;
std::unique_ptr<PtxInterpreter> g_ptx_interpreter;

// ============================================================================
// CppTLM Bridge 全局指针 (D-PTX-1)
// ============================================================================
// 默认 nullptr，加载 libcpptlm_cudart.so 后赋值。
// nullptr 时所有操作走原有同步路径（字节级相同）。
// ============================================================================
#include "cudart/cpptlm_bridge.h"
CppTLMBridge* g_cpptlm_bridge = nullptr;

// ============================================================================
// cpptlm_attach_bridge / cpptlm_detach_bridge ABI entry points (B1)
// ============================================================================
// Per ADR-0021 (D-PTX-1): CppTLM's libcpptlm_cudart.so calls these on
// load/unload to install/uninstall the bridge pointer. Both are idempotent:
//   - attach: overwrite is allowed (last-call-wins); nullptr bridges call to
//     detach semantics per cpptlm_bridge.h:160 documentation contract.
//   - detach: safe to call when already nullptr (no-op).
//
// Metis second-pass review B1: declarations in cpptlm_bridge.h:161,168 were
// symbols without definitions, causing link errors. Implementations live
// here (same TU as g_cpptlm_bridge per D-PTX-1) to ensure the global pointer
// is mutated only through these ABI entry points.
// ============================================================================
extern "C" PTXEMU_BRIDGE_API void cpptlm_attach_bridge(CppTLMBridge* bridge) {
    PTX_DEBUG_EMU("cpptlm_attach_bridge: bridge=%p (was %p)",
                  (void*)bridge, (void*)g_cpptlm_bridge);
    // nullptr bridge ≡ detach (per cpptlm_bridge.h:160 contract).
    g_cpptlm_bridge = bridge;
}

extern "C" PTXEMU_BRIDGE_API void cpptlm_detach_bridge() {
    PTX_DEBUG_EMU("cpptlm_detach_bridge (was %p)", (void*)g_cpptlm_bridge);
    g_cpptlm_bridge = nullptr;
}

// ============================================================================
// 异步 kernel 注册表 (D-PTX-1 + Task #2)
// ============================================================================
// PendingKernel: 记录已提交但未完成的 kernel
// g_pending_kernels: kernel_id → PendingKernel 映射
// g_active_streams: 活跃 stream ID 集合（含默认 stream 0）
// ============================================================================
struct PendingKernel {
    uint64_t kernel_id;
    std::string kernel_name;
    uint64_t stream_id;
    Dim3 grid_dim;
    Dim3 block_dim;
    size_t shared_mem;
    std::vector<std::vector<uint8_t>> args_copy;  // deep-copy 的参数
    bool completed = false;
};

static std::atomic<uint64_t> next_kernel_id{1};
static std::unordered_map<uint64_t, PendingKernel> g_pending_kernels;
static std::unordered_set<uint64_t> g_active_streams{0};  // 默认包含 stream 0
static std::mutex g_pending_kernels_mutex;

static uint64_t generate_kernel_id() {
    return next_kernel_id.fetch_add(1);
}

static size_t count_kernel_args(void** args) {
    if (!args) return 0;
    size_t count = 0;
    while (args[count] != nullptr) ++count;
    return count;
}

// 配置文件路径
// PTX_EMU_CONFIG 环境变量可覆盖默认 config.ini（用于按场景切换 trace/日志级别）
static std::string get_config_file_path() {
 const char *env = std::getenv("PTX_EMU_CONFIG");
 if (env != nullptr && env[0] != '\0') {
  return std::string(env);
 }
 return std::string("config.ini");
}

// 初始化调试环境和GPUContext
void initialize_environment() {
    // 解析配置文件一次，然后分别设置各个组件
    inipp::Ini<char> ini;
    std::string config_path = get_config_file_path();
 std::ifstream is(config_path);
    if (!is.is_open()) {
        // CWD 下未找到时，尝试 configs/ 子目录（mini/perf 等 ini 都放在那里）
        std::string alt_path = "configs/" + config_path;
        is.clear();
        is.open(alt_path);
        if (is.is_open()) {
            config_path = alt_path;
        }
    }
    if (is.is_open()) {
        ini.parse(is);

        // 设置日志配置
        auto logger_section = ini.sections["logger"];
        ptxsim::LoggerConfig::get().load_from_ini_section(logger_section);

        // 设置调试器配置
        auto debugger_section = ini.sections["debugger"];
        ptxsim::DebugConfig::get().load_from_ini_section(debugger_section);

        // 从INI配置文件中读取GPU配置文件路径
        // 环境变量 PTX_EMU_GPU_CONFIG 可覆盖配置文件设置
        std::string gpu_config_filename;
        auto gpu_section = ini.sections["gpu"];
        inipp::get_value(gpu_section, "gpu_config_file", gpu_config_filename);

        // 检查环境变量覆盖
        const char* env_config = std::getenv("PTX_EMU_GPU_CONFIG");
        if (env_config != nullptr && strlen(env_config) > 0) {
            gpu_config_filename = env_config;
            PTX_INFO_EMU("GPU config overridden by PTX_EMU_GPU_CONFIG=%s", gpu_config_filename.c_str());
        }
        if (!gpu_config_filename.empty()) {
            // 创建GPUContext并直接加载JSON配置
            g_gpu_context =
                std::make_unique<GPUContext>("configs/" + gpu_config_filename);
        } else {
            // 如果INI文件中没有指定GPU配置文件或加载失败，使用默认配置
            g_gpu_context = std::make_unique<GPUContext>();
        }
        g_gpu_context->init();
        g_ptx_interpreter = std::make_unique<PtxInterpreter>();

        PTX_INFO_EMU("Configuration loaded from %s", config_path.c_str());
    } else {
        PTX_INFO_EMU("No configuration file found, using default settings");
        // 设置默认的日志级别
        ptxsim::LoggerConfig::get().set_global_level(ptxsim::log_level::info);

        // 使用默认GPU配置
        g_gpu_context = std::make_unique<GPUContext>();
        g_gpu_context->init();
        g_ptx_interpreter = std::make_unique<PtxInterpreter>();
    }
}

#ifdef __cplusplus
extern "C" {
#endif

size_t get_gpu_clock_from_context() {
    if (g_gpu_context) {
        return g_gpu_context->get_clock();
    }
    return 0;
}

void **__cudaRegisterFatBinary(void **fatCubinHandle, void *fat_bin,
                               unsigned long long fat_bin_size,
                               unsigned int version) {
    // SingletonGuard (D-PTX-2): 检测重复初始化，防止多实例仿真静默状态损坏
    if (SingletonGuard::instance().check_and_mark()) {
        std::cerr << "[FATAL] __cudaRegisterFatBinary called multiple times. "
                  << "PTX-EMU does not support multi-instance simulation. "
                  << "See ADR-0021 D-PTX-2 for details." << std::endl;
        std::abort();
    }

    // 初始化调试环境
    static bool debug_initialized = false;
    if (!debug_initialized) {
        initialize_environment();
        debug_initialized = true;
    }

    PTX_DEBUG_EMU("Called __cudaRegisterFatBinary(%p, %p, %llu, %u)",
                  fatCubinHandle, fat_bin, fat_bin_size, version);

    // 1. 获取当前进程路径
    char self_exe_path[1025] = "";
    long size = readlink("/proc/self/exe", self_exe_path, 1024);
    if (size == -1) {
        PTX_ERROR_CUDART("Could not read /proc/self/exe");
        // P1-3: throw instead of exit(1) so callers can catch fatal init errors.
        throw std::runtime_error("cudart: could not read /proc/self/exe");
    }
    self_exe_path[size] = '\0';

    // 2. 从当前进程提取PTX代码
    std::string ptx_code = extract_ptx_with_cuobjdump(self_exe_path);

    if (ptx_code.empty()) {
        std::cerr << "Error: Could not extract PTX code" << std::endl;
        return nullptr;
    }

    // 3. 预处理PTX代码（规范化多行entry params和inline param blocks）
    {
        char input_path[] = "/tmp/ptxemu_input_XXXXXX.ptx";
        char output_path[] = "/tmp/ptxemu_output_XXXXXX.ptx";
        int fd = mkstemps(input_path, 4);
        if (fd == -1) {
            std::cerr << "Error: Could not create temp input file" << std::endl;
            return nullptr;
        }
        close(fd);

        std::ofstream infile(input_path);
        infile << ptx_code;
        infile.close();

        // 查找 PTX_EMU_PATH：优先环境变量，否则从可执行文件路径推导
        const char *ptx_emu_path = getenv("PTX_EMU_PATH");
        std::string resolved_path;

        if (ptx_emu_path && strlen(ptx_emu_path) > 0) {
            resolved_path = ptx_emu_path;
        } else {
            // 可执行文件位于 <project_root>/build/bin/，向上两级得到项目根目录
            std::string exe_dir = std::string(self_exe_path);
            size_t last_slash = exe_dir.find_last_of("/");
            if (last_slash != std::string::npos) {
                std::string bin_dir = exe_dir.substr(0, last_slash);
                size_t prev_slash = bin_dir.find_last_of("/");
                if (prev_slash != std::string::npos) {
                    std::string build_dir = bin_dir.substr(0, prev_slash);
                    prev_slash = build_dir.find_last_of("/");
                    if (prev_slash != std::string::npos) {
                        resolved_path = build_dir.substr(0, prev_slash);
                    }
                }
            }
            if (resolved_path.empty()) {
                resolved_path = ".";
            }
        }

        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "python3 %s/tests/ptx/ptx_preprocess.py %s %s 2>/dev/null",
                 resolved_path.c_str(), input_path, output_path);

        int ret = system(cmd);
        if (ret != 0) {
            std::cerr << "Warning: PTX preprocessing failed, using original" << std::endl;
        } else {
            std::ifstream outfile(output_path);
            std::stringstream ss;
            ss << outfile.rdbuf();
            ptx_code = ss.str();
            outfile.close();
        }

        std::remove(input_path);
        std::remove(output_path);
    }

    // 使用g_gpu_context的get_device_memory函数获取SimpleMemory实例
    // SimpleMemory *simple_mem = g_gpu_context->get_device_memory();
    //
    // // 设置CudaDriver使用的SimpleMemory实例
    // CudaDriver::instance().set_simple_memory(simple_mem);

    // 3. 解析PTX代码
    ANTLRInputStream input(ptx_code);
    ptxLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    tokens.fill();
    ptxParser parser(&tokens);

    // 创建PtxContext和PtxVisitor
    PtxContext ptxContext;
    PtxVisitor visitor(ptxContext);

    // 访问解析树
    visitor.visit(parser.ptxFile());

    // 4. 初始化PtxInterpreter - 使用拷贝避免悬垂引用
    g_ptx_interpreter->set_ptx_context(ptxContext);

    // 5. 返回虚拟句柄
    static int dummy_handle = 0;
    *fatCubinHandle = &dummy_handle;
    return fatCubinHandle;
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            cudaKernel_t deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
    PTX_DEBUG_EMU("Called __cudaRegisterFunction(%p, %s, %p, %s)",
                  fatCubinHandle, hostFun, deviceFun, deviceName);

    func2name[(uint64_t)hostFun] = *(new std::string(deviceName));
    func2kernel[(uint64_t)hostFun] = (cudaKernel_t)deviceFun;
    kernel2func[deviceFun] = hostFun;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
    PTX_DEBUG_EMU("Called __cudaRegisterFatBinaryEnd(%p)", fatCubinHandle);
    // 目前不需要做任何事情
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    PTX_DEBUG_EMU("Called cuModuleLoad(%p, %s)", module, fname);

    // 在仿真环境中，我们不实际加载模块
    // 直接返回成功
    *module = reinterpret_cast<CUmodule>(0x12345678);
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
    PTX_DEBUG_EMU("Called cuModuleGetFunction(%p, %p, %s)", hfunc, hmod, name);

    // 在仿真环境中，我们将函数名存储在句柄中
    *hfunc = reinterpret_cast<CUfunction>(const_cast<char *>(name));
    return CUDA_SUCCESS;
}

// 补充缺失的 __cudaPushCallConfiguration 函数
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream) {
    PTX_DEBUG_EMU("Called __cudaPushCallConfiguration(grid=(%d,%d,%d), "
                  "block=(%d,%d,%d), sharedMem=%zu, stream=%p)",
                  gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                  blockDim.z, sharedMem, stream);

    _gridDim = gridDim;
    _blockDim = blockDim;
    _sharedMem = sharedMem;
    return 0;
}

// 补充缺失的 __cudaPopCallConfiguration 函数
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
    PTX_DEBUG_EMU("Called __cudaPopCallConfiguration(%p, %p, %p, %p)", gridDim,
                  blockDim, sharedMem, stream);

    *gridDim = _gridDim;
    *blockDim = _blockDim;
    *sharedMem = _sharedMem;
    return cudaSuccess;
}

// 补充缺失的 __cudaGetKernel 函数
cudaError_t __cudaGetKernel(cudaKernel_t *kernelPtr, const void *funcAddr) {
    PTX_DEBUG_EMU("Called __cudaGetKernel(%p, %p)", kernelPtr, funcAddr);

    *kernelPtr = func2kernel[(uint64_t)funcAddr];
    return cudaSuccess;
}

// 补充缺失的 cudaLaunchKernel 函数
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaLaunchKernel(func=%p, grid=(%d,%d,%d), "
                  "block=(%d,%d,%d), args=%p, sharedMem=%zu, stream=%p)",
                  func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                  blockDim.z, args, sharedMem, stream);

    // 添加参数内容打印的日志，增强安全性
    if (args) {
        // 打印参数数组地址
        PTX_DEBUG_EMU("cudaLaunchKernel args array address: %p", args);

        // int i = 0;
        // if (args[i]) {
        //     PTX_DEBUG_EMU(
        //         "cudaLaunchKernel argument[%d]: address=%p, value=0x%lx", i,
        //         args[i], *(uint64_t *)args[i]);
        // }
    }

    PTX_DEBUG_EMU("deviceFunName %s", func2name[(uint64_t)func].c_str());
    PTX_DEBUG_EMU("gridDim(%d,%d,%d)", gridDim.x, gridDim.y, gridDim.z);
    PTX_DEBUG_EMU("blockDim(%d,%d,%d)", blockDim.x, blockDim.y, blockDim.z);

    Dim3 gridDim3(gridDim.x, gridDim.y, gridDim.z);
    Dim3 blockDim3(blockDim.x, blockDim.y, blockDim.z);

    // 验证共享内存大小是否超出硬件限制
    const size_t MAX_SHARED_MEM = 49152; // 48KB 默认限制
    if (sharedMem > MAX_SHARED_MEM) {
        std::cerr << "Warning: cudaLaunchKernel: sharedMem " << sharedMem 
                  << " exceeds limit " << MAX_SHARED_MEM << ", truncating" << std::endl;
        sharedMem = MAX_SHARED_MEM;
    }

    // ========================================================================
    // Bridge 异步路径 (D-PTX-1 + Task #2)
    // ========================================================================
    // 当 g_cpptlm_bridge != nullptr 时，走异步提交路径：
    // 1. 生成唯一 kernel_id
    // 2. deep-copy kernel args（bridge 调用后 host 端 args 可能失效）
    // 3. 调用 bridge->submit_kernel() 异步提交
    // 4. 注册到 g_pending_kernels 等待 poll
    // 5. 立即返回 cudaSuccess
    // ========================================================================
    if (g_cpptlm_bridge) {
        uint64_t kernel_id = generate_kernel_id();
        uint64_t stream_id = reinterpret_cast<uintptr_t>(stream);

        // deep-copy kernel args
        std::vector<std::vector<uint8_t>> args_copy;
        if (args) {
            size_t arg_count = count_kernel_args(args);
            args_copy.reserve(arg_count);
            for (size_t i = 0; i < arg_count; ++i) {
                if (args[i]) {
                    // 假设每个参数最大 8 字节（指针或基本类型）
                    std::vector<uint8_t> arg_data(8);
                    std::memcpy(arg_data.data(), args[i], 8);
                    args_copy.push_back(std::move(arg_data));
                }
            }
        }

        // 准备 bridge 调用参数
        std::vector<const void*> bridge_args;
        bridge_args.reserve(args_copy.size());
        for (const auto& arg : args_copy) {
            bridge_args.push_back(arg.data());
        }

        // 调用 bridge 异步提交
        const char* kernel_name = func2name[(uint64_t)func].c_str();
        int submit_result = g_cpptlm_bridge->submit_kernel(
            kernel_id, kernel_name,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            bridge_args.data(), bridge_args.size(),
            sharedMem, stream_id);

        if (submit_result != 0) {
            std::cerr << "Error: CppTLM bridge submit_kernel failed with code " 
                      << submit_result << std::endl;
            return (cudaError_t)submit_result;
        }

        // 注册到 pending_kernels
        {
            std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
            PendingKernel pk;
            pk.kernel_id = kernel_id;
            pk.kernel_name = kernel_name;
            pk.stream_id = stream_id;
            pk.grid_dim = gridDim3;
            pk.block_dim = blockDim3;
            pk.shared_mem = sharedMem;
            pk.args_copy = std::move(args_copy);
            pk.completed = false;
            g_pending_kernels[kernel_id] = std::move(pk);
        }

        // 确保 stream 在 active_streams 中
        g_active_streams.insert(stream_id);

        PTX_DEBUG_EMU("cudaLaunchKernel: async submit kernel_id=%lu to CppTLM bridge", kernel_id);
        return cudaSuccess;
    }

    // ========================================================================
    // 原有同步路径（bridge == nullptr 时字节级相同）
    // ========================================================================
    // 调用 PtxInterpreter 的 launch 函数，传递 sharedMem 参数
    try {
        g_ptx_interpreter->launchPtxInterpreter(
            g_ptx_interpreter->get_ptx_context(), func2name[(uint64_t)func], args,
            gridDim3, blockDim3, sharedMem);

        // 等待kernel执行完成
        g_gpu_context->wait_for_completion();
    } catch (const PtxEmuException& e) {
        std::cerr << "PTX execution error: " << e.what()
                  << " [code=" << e.get_error_code_name() << "]" << std::endl;
        return (cudaError_t)999;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error during kernel execution: " << e.what() << std::endl;
        return (cudaError_t)999;
    }

    return cudaSuccess;
}

// 补充缺失的 __cudaLaunchKernel 函数
cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem,
                               cudaStream_t stream) {
    return cudaLaunchKernel(kernel2func[kernel], gridDim, blockDim, args,
                            sharedMem, stream);
}

// 补充缺失的 __cudaRegisterVar 函数
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       int size, int constant, int global) {
    PTX_DEBUG_EMU("Called __cudaRegisterVar(%p, %p, %p, %s, %d, %d, %d, %d)",
                  fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size,
                  constant, global);

    std::string s(deviceName);
    g_ptx_interpreter->constName2addr[s] = (uint64_t)hostVar;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
    PTX_DEBUG_EMU("Called cudaMemcpy(%p, %p, %zu, %d)", dst, src, count, kind);
    PTX_DEBUG_CUDART(
        "cudaMemcpy ENTRY: dst=%p src=%p count=%zu kind=%d (host<->device "
        "routing routed through CudaDriver global pool)",
        dst, src, count, static_cast<int>(kind));

    if (!dst || !src || count == 0) {
        PTX_WARN_CUDART("cudaMemcpy REJECT: invalid args (dst=%p src=%p count=%zu)",
                        dst, src, count);
        return cudaErrorInvalidValue;
    }

    // 获取CudaDriver的全局内存池地址
    uint64_t global_pool = (uint64_t)CudaDriver::instance().get_global_pool();
    uint64_t global_size = (uint64_t)CudaDriver::instance().get_global_size();
    if (!global_pool) {
        PTX_ERROR_CUDART(
            "cudaMemcpy REJECT: CudaDriver global pool not initialized");
        return cudaErrorInitializationError;
    }
    PTX_DEBUG_CUDART("cudaMemcpy POOL: base=0x%lx size=0x%lx", global_pool,
                     global_size);

    // 根据复制类型执行内存复制
    switch (kind) {
    case cudaMemcpyHostToHost: {
        PTX_DEBUG_CUDART("cudaMemcpy BRANCH HostToHost: count=%zu", count);
        std::memcpy(dst, src, count);
        break;
    }
    case cudaMemcpyHostToDevice: {
        PTX_DEBUG_CUDART("cudaMemcpy BRANCH HostToDevice: count=%zu", count);
        // dst是设备指针（即偏移量），src是主机指针
        uint64_t device_offset = reinterpret_cast<uint64_t>(dst);
        if (device_offset >= global_pool) {
            device_offset -= global_pool;
        }
        if (device_offset >= global_size) {
            PTX_WARN_CUDART(
                "cudaMemcpy REJECT: H2D offset 0x%lx out of range (pool size=0x%lx)",
                device_offset, global_size);
            return cudaErrorInvalidValue;
        }

        PTX_DEBUG_CUDART(
            "cudaMemcpy H2D COPY: pool+0x%lx <- host %p, %zu bytes",
            device_offset, src, count);
        std::memcpy((uint8_t *)(global_pool + device_offset), src, count);
        break;
    }
    case cudaMemcpyDeviceToHost: {
        PTX_DEBUG_CUDART("cudaMemcpy BRANCH DeviceToHost: count=%zu", count);
        // src是设备指针（即偏移量），dst是主机指针
        uint64_t device_offset = reinterpret_cast<uint64_t>(src);
        if (device_offset >= global_pool) {
            device_offset -= global_pool;
        }
        if (device_offset >= global_size) {
            PTX_WARN_CUDART(
                "cudaMemcpy REJECT: D2H offset 0x%lx out of range (pool size=0x%lx)",
                device_offset, global_size);
            return cudaErrorInvalidValue;
        }

        PTX_DEBUG_CUDART(
            "cudaMemcpy D2H COPY: host %p <- pool+0x%lx, %zu bytes",
            dst, device_offset, count);
        std::memcpy(dst, (uint8_t *)(global_pool + device_offset), count);
        break;
    }
    case cudaMemcpyDeviceToDevice: {
        PTX_DEBUG_CUDART("cudaMemcpy BRANCH DeviceToDevice: count=%zu", count);
        // 设备到设备的复制
        uint64_t src_device_offset = reinterpret_cast<uint64_t>(src);
        uint64_t dst_device_offset = reinterpret_cast<uint64_t>(dst);

        if (src_device_offset >= global_pool) {
            src_device_offset -= global_pool;
        }
        if (dst_device_offset >= global_pool) {
            dst_device_offset -= global_pool;
        }

        if ((src_device_offset >= global_size) ||
            (dst_device_offset >= global_size)) {
            PTX_WARN_CUDART(
                "cudaMemcpy REJECT: D2D out of range src=0x%lx dst=0x%lx (pool size=0x%lx)",
                src_device_offset, dst_device_offset, global_size);
            return cudaErrorInvalidValue;
        }

        PTX_DEBUG_CUDART(
            "cudaMemcpy D2D COPY: pool+0x%lx <- pool+0x%lx, %zu bytes",
            dst_device_offset, src_device_offset, count);
        std::memcpy((uint8_t *)(global_pool + dst_device_offset),
                    (uint8_t *)(global_pool + src_device_offset), count);
        break;
    }
    default:
        PTX_WARN_CUDART("cudaMemcpy REJECT: unsupported kind=%d",
                        static_cast<int>(kind));
        return cudaErrorInvalidValue;
    }

    PTX_DEBUG_CUDART("cudaMemcpy OK: %zu bytes transferred (kind=%d)", count,
                     static_cast<int>(kind));
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaMemcpyAsync(%p, %p, %zu, %d, %p)", dst, src,
                  count, kind, stream);
    PTX_DEBUG_CUDART(
        "cudaMemcpyAsync ENTRY: dst=%p src=%p count=%zu kind=%d stream=%p "
        "(emulator downgrades to sync: stream parameter is ignored)",
        dst, src, count, static_cast<int>(kind), stream);

    // 异步复制在仿真器中与同步复制相同
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    PTX_DEBUG_EMU("Called cudaMemset(%p, %d, %zu)", devPtr, value, count);
    if (!devPtr) return cudaErrorInvalidValue;

    uint8_t *global_pool = CudaDriver::instance().get_global_pool();
    uint64_t global_size = CudaDriver::instance().get_global_size();
    if (!global_pool) return cudaErrorInitializationError;

    uint64_t device_offset = reinterpret_cast<uint64_t>(devPtr);
    if (device_offset >= (uint64_t)global_pool) {
        device_offset -= (uint64_t)global_pool;
    }
    if (device_offset >= global_size) {
        return cudaErrorInvalidValue;
    }

    std::memset(global_pool + device_offset, value, count);
    return cudaSuccess;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    PTX_DEBUG_EMU("Called cudaMalloc(%p, %zu)", devPtr, size);

    if (!devPtr) {
        return cudaErrorInvalidValue;
    }

    // 使用 CudaDriver 分配内存
    *devPtr = CudaDriver::instance().malloc(size);
    if (!*devPtr) {
        return cudaErrorMemoryAllocation;
    }

    return cudaSuccess;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size) {
    PTX_DEBUG_EMU("Called cudaMallocManaged(%p, %zu)", devPtr, size);

    if (!devPtr) {
        return cudaErrorInvalidValue;
    }

    // 使用 CudaDriver 分配托管内存
    *devPtr = CudaDriver::instance().malloc_managed(size);
    if (!*devPtr) {
        return cudaErrorMemoryAllocation;
    }

    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
    PTX_DEBUG_EMU("Called cudaFree(%p)", devPtr);

    // 使用 CudaDriver 释放内存
    auto ret = CudaDriver::instance().free(devPtr);
    if (ret != Success) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr) {
    PTX_DEBUG_EMU("Called cudaFreeHost(%p)", ptr);

    // Host内存由系统管理，无需特殊处理
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    PTX_DEBUG_EMU("Called cudaMallocHost(%p, %zu)", ptr, size);

    if (!ptr) {
        return cudaErrorInvalidValue;
    }

    // Host内存由系统分配
    *ptr = std::malloc(size);
    if (!*ptr) {
        return cudaErrorMemoryAllocation;
    }

    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() {
    PTX_DEBUG_EMU("Called cudaDeviceSynchronize()");

    // ========================================================================
    // Bridge 路径 (D-PTX-1 + Task #3): 遍历所有 active_streams
    // ========================================================================
    if (g_cpptlm_bridge) {
        std::vector<uint64_t> completed_ids;

        {
            std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
            for (const auto& [id, pk] : g_pending_kernels) {
                if (!pk.completed) {
                    uint64_t remaining = g_cpptlm_bridge->poll_kernel(id);
                    if (remaining == 0) {
                        completed_ids.push_back(id);
                    }
                }
            }
        }

        // 循环外统一 erase（迭代器安全）
        if (!completed_ids.empty()) {
            std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
            for (uint64_t id : completed_ids) {
                g_pending_kernels.erase(id);
            }
        }

        PTX_DEBUG_EMU("cudaDeviceSynchronize: completed %zu kernels", completed_ids.size());
        return cudaSuccess;
    }

    // nullptr fallback: 同步是立即完成的
    return cudaSuccess;
}

cudaError_t cudaPeekAtLastError() {
    PTX_DEBUG_EMU("Called cudaPeekAtLastError()");

    // 在仿真器中，通常没有错误
    return cudaSuccess;
}

cudaError_t cudaGetLastError() {
    PTX_DEBUG_EMU("Called cudaGetLastError()");

    // 在仿真器中，通常没有错误
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    PTX_DEBUG_EMU("Called cudaSetDevice(%d)", device);

    // 在仿真器中，只支持一个设备
    if (device != 0) {
        return cudaErrorInvalidDevice;
    }

    return cudaSuccess;
}

cudaError_t cudaDeviceReset() {
    PTX_DEBUG_EMU("Called cudaDeviceReset()");

    // 重置全局GPU上下文
    g_gpu_context.reset();

    return cudaSuccess;
}

cudaError_t cudaFuncSetCacheConfig(const char *func,
                                   cudaFuncCache cacheConfig) {
    PTX_DEBUG_EMU("Called cudaFuncSetCacheConfig(%p, %d)", func, cacheConfig);

    // 在仿真器中，缓存配置不起作用
    return cudaSuccess;
}

cudaError_t cudaFuncSetSharedMemConfig(const char *func,
                                       cudaSharedMemConfig config) {
    PTX_DEBUG_EMU("Called cudaFuncSetSharedMemConfig(%p, %d)", func, config);

    // 在仿真器中，共享内存配置不起作用
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    PTX_DEBUG_EMU("Called cudaStreamCreate(%p)", stream);

    if (!stream) {
        return cudaErrorInvalidValue;
    }

    // ========================================================================
    // Bridge 路径 (D-PTX-1 + Task #3): 生成唯一 64-bit stream_id
    // ========================================================================
    uint64_t stream_id = generate_kernel_id();  // 复用 kernel_id 生成器
    g_active_streams.insert(stream_id);
    *stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id));

    PTX_DEBUG_EMU("cudaStreamCreate: assigned stream_id=%lu", stream_id);
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaStreamDestroy(%p)", stream);

    // Per CUDA spec: cudaStreamDestroy must be preceded by
    // cudaStreamSynchronize to ensure all work completes. We track
    // streams via g_active_streams (see cudaStreamCreate insert).
    //
    // B3 (Metis second-pass review): previously called
    //   `delete reinterpret_cast<int *>(stream)` — UB because stream is a
    //   uint64_t encoded as void* (never heap-allocated; see cudaStreamCreate
    //   at line ~905). The fake runtime tracks streams in g_active_streams,
    //   so destruction = erase the ID. Reuses g_pending_kernels_mutex per
    //   state-modification-audit (lessons-learned §2): create/destroy
    //   mutate the same set and must be symmetric under the same lock.
    if (stream) {  // non-default stream (default stream is nullptr/0)
        uint64_t stream_id = reinterpret_cast<uintptr_t>(stream);
        {
            std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
            g_active_streams.erase(stream_id);
        }
        PTX_DEBUG_EMU("cudaStreamDestroy: removed stream_id=%lu", stream_id);
    }
    // Default stream (nullptr/0): no-op per CUDA spec.
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaStreamSynchronize(%p)", stream);

    uint64_t stream_id = reinterpret_cast<uintptr_t>(stream);

    // ========================================================================
    // Bridge 路径 (D-PTX-1 + Task #3): 按 stream_id 过滤 + poll_kernel
    // ========================================================================
    // 迭代器失效修复：先收集 completed_ids，循环外统一 erase
    // （避免 range-for 中 unordered_map::erase 触发 UB）
    // ========================================================================
    if (g_cpptlm_bridge) {
        std::vector<uint64_t> completed_ids;

        {
            std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
            for (const auto& [id, pk] : g_pending_kernels) {
                if (pk.stream_id == stream_id && !pk.completed) {
                    uint64_t remaining = g_cpptlm_bridge->poll_kernel(id);
                    if (remaining == 0) {
                        completed_ids.push_back(id);
                    }
                }
            }
        }

        // 循环外统一 erase（迭代器安全）
        if (!completed_ids.empty()) {
            std::lock_guard<std::mutex> lock(g_pending_kernels_mutex);
            for (uint64_t id : completed_ids) {
                g_pending_kernels.erase(id);
            }
        }

        PTX_DEBUG_EMU("cudaStreamSynchronize: stream_id=%lu, completed %zu kernels",
                      stream_id, completed_ids.size());
        return cudaSuccess;
    }

    // nullptr fallback: 同步是立即完成的
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
    PTX_DEBUG_EMU("Called cudaEventCreate(%p)", event);

    if (!event) {
        return cudaErrorInvalidValue;
    }

    // 在仿真器中，事件只是一个占位符
    *event = reinterpret_cast<cudaEvent_t>(new int(0));
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    PTX_DEBUG_EMU("Called cudaEventDestroy(%p)", event);

    if (event) {
        delete reinterpret_cast<int *>(event);
    }

    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaEventRecord(%p, %p)", event, stream);

    // 在仿真器中，事件记录立即完成
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    PTX_DEBUG_EMU("Called cudaEventSynchronize(%p)", event);

    // 在仿真器中，同步是立即完成的
    return cudaSuccess;
}

float cudaEventElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    PTX_DEBUG_EMU("Called cudaEventElapsedTime(%p, %p)", start, end);

    // 在仿真器中，我们不测量实际时间
    // 返回一个虚拟值
    return 1.0f; // 1毫秒
}

// 补充缺失的 cudaGetDeviceCount 函数
cudaError_t cudaGetDeviceCount(int *count) {
    PTX_DEBUG_EMU("Called cudaGetDeviceCount(%p)", count);

    if (!count) {
        return cudaErrorInvalidValue;
    }

    *count = 1;
    return cudaSuccess;
}

// 补充缺失的 cudaGetDeviceProperties 函数
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    PTX_DEBUG_EMU("Called cudaGetDeviceProperties(%p, %d)", prop, device);

    if (!prop) {
        return cudaErrorInvalidValue;
    }

    if (device != 0) {
        return cudaErrorInvalidDevice;
    }

    // 初始化设备属性
    memset(prop, 0, sizeof(cudaDeviceProp));
    snprintf(prop->name, sizeof(prop->name), "PTX-EMU Virtual Device");
    prop->major = 8;
    prop->minor = 0;
    prop->totalGlobalMem = 1ULL << 32; // 4GB
    prop->sharedMemPerBlock = 49152;   // 48KB
    prop->regsPerBlock = 65536;
    prop->warpSize = 32;
    prop->memPitch = 2147483647;
    prop->maxThreadsPerBlock = 1024;
    prop->maxThreadsDim[0] = 1024;
    prop->maxThreadsDim[1] = 1024;
    prop->maxThreadsDim[2] = 64;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    // prop->clockRate = 1000000; // 1GHz // 已在较新版本中移除
    prop->totalConstMem = 65536;
    prop->textureAlignment = 512;
    // prop->deviceOverlap = 1; // 已在较新版本中移除
    prop->multiProcessorCount = 80; // 假设80个SM
    // prop->kernelExecTimeoutEnabled = 0; // 已在较新版本中移除
    prop->integrated = 0;
    prop->canMapHostMemory = 1;
    // prop->computeMode = 0; // 已在较新版本中移除
    prop->maxTexture1D = 65536;
    prop->maxTexture1DMipmap = 65536;
    // prop->maxTexture1DLinear = 134217728;  // 已在较新版本中移除
    prop->maxTexture2D[0] = 65536;
    prop->maxTexture2D[1] = 65536;
    prop->maxTexture2DMipmap[0] = 65536;
    prop->maxTexture2DMipmap[1] = 65536;
    // prop->maxTexture2DLinear[0] = 134217728; // 已在较新版本中移除
    // prop->maxTexture2DLinear[1] = 65536; // 已在较新版本中移除
    // prop->maxTexture2DLinear[2] = 2048; // 已在较新版本中移除
    prop->maxTexture3D[0] = 16384;
    prop->maxTexture3D[1] = 16384;
    prop->maxTexture3D[2] = 16384;
    prop->maxTexture3DAlt[0] = 16384;
    prop->maxTexture3DAlt[1] = 16384;
    prop->maxTexture3DAlt[2] = 16384;
    prop->maxTextureCubemap = 65536;
    prop->maxTexture1DLayered[0] = 65536;
    prop->maxTexture1DLayered[1] = 2048;
    prop->maxTexture2DLayered[0] = 65536;
    prop->maxTexture2DLayered[1] = 65536;
    prop->maxTexture2DLayered[2] = 2048;
    prop->maxTextureCubemapLayered[0] = 65536;
    prop->maxTextureCubemapLayered[1] = 2048;
    prop->maxSurface1D = 65536;
    prop->maxSurface2D[0] = 65536;
    prop->maxSurface2D[1] = 65536;
    prop->maxSurface3D[0] = 16384;
    prop->maxSurface3D[1] = 16384;
    prop->maxSurface3D[2] = 16384;
    prop->maxSurface1DLayered[0] = 65536;
    prop->maxSurface1DLayered[1] = 2048;
    prop->maxSurface2DLayered[0] = 65536;
    prop->maxSurface2DLayered[1] = 65536;
    prop->maxSurface2DLayered[2] = 2048;
    prop->maxSurfaceCubemap = 65536;
    prop->maxSurfaceCubemapLayered[0] = 65536;
    prop->maxSurfaceCubemapLayered[1] = 2048;
    prop->surfaceAlignment = 512;
    prop->concurrentKernels = 16;
    prop->ECCEnabled = 0;
    prop->pciBusID = 0;
    prop->pciDeviceID = 0;
    prop->tccDriver = 1;
    prop->asyncEngineCount = 2;
    prop->unifiedAddressing = 1;
    // prop->memoryClockRate = 1000000; // 1GHz // 已在较新版本中移除
    prop->memoryBusWidth = 320;
    prop->l2CacheSize = 4194304; // 4MB
    prop->persistingL2CacheMaxSize = 0;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->streamPrioritiesSupported = 0;
    prop->globalL1CacheSupported = 1;
    prop->localL1CacheSupported = 1;
    prop->sharedMemPerMultiprocessor = 163840; // 160KB
    prop->regsPerMultiprocessor = 65536;
    prop->managedMemory = 1;
    prop->isMultiGpuBoard = 0;
    prop->multiGpuBoardGroupID = 0;
    prop->hostNativeAtomicSupported = 0;
    // prop->singleToDoublePrecisionPerfRatio = 32; // 已在较新版本中移除
    prop->pageableMemoryAccess = 0;
    prop->concurrentManagedAccess = 0;
    prop->computePreemptionSupported = 0;
    prop->canUseHostPointerForRegisteredMem = 0;
    prop->cooperativeLaunch = 1;
    // prop->cooperativeMultiDeviceLaunch = 1; // 已在较新版本中移除
    prop->sharedMemPerBlockOptin = 49152;
    prop->pageableMemoryAccessUsesHostPageTables = 0;
    prop->directManagedMemAccessFromHost = 0;
    prop->maxBlocksPerMultiProcessor = 32;
    prop->accessPolicyMaxWindowSize = 1024;
    prop->reservedSharedMemPerBlock = 0;

    return cudaSuccess;
}

// 补充缺失的 cudaMemcpyToSymbol 函数
cudaError_t cudaMemcpyToSymbol(void *symbol, void *src, size_t count,
                               size_t offset, cudaMemcpyKind kind) {
    PTX_DEBUG_EMU("Called cudaMemcpyToSymbol(%p, %p, %zu, %zu, %d)", symbol,
                  src, count, offset, kind);

    if (!symbol || !src) {
        return cudaErrorInvalidValue;
    }

    // 获取CudaDriver的全局内存池地址
    uint8_t *global_pool = CudaDriver::instance().get_global_pool();
    if (!global_pool) {
        return cudaErrorInitializationError;
    }

    // 将数据复制到符号地址（加上偏移量）
    uint64_t symbol_offset = reinterpret_cast<uint64_t>(symbol) + offset;
    if (symbol_offset >= CudaDriver::instance().get_global_size()) {
        return cudaErrorInvalidValue;
    }

    std::memcpy(global_pool + symbol_offset, src, count);
    return cudaSuccess;
}

// 补充缺失的 cudaGetDevice 函数
cudaError_t cudaGetDevice(int *device) {
    PTX_DEBUG_EMU("Called cudaGetDevice(%p)", device);

    if (!device) {
        return cudaErrorInvalidValue;
    }

    *device = 0; // 仿真器只有一个设备
    return cudaSuccess;
}

// 补充缺失的 __cudaInitModule 函数
char __cudaInitModule(void **fatCubinHandle) {
    PTX_DEBUG_EMU("Called __cudaInitModule(%p)", fatCubinHandle);

    return 1; // 返回成功标识
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    PTX_DEBUG_EMU("Called __cudaUnregisterFatBinary(%p)", fatCubinHandle);

    // 清理PtxInterpreter
    g_ptx_interpreter.reset();

    // 重置全局GPU上下文
    g_gpu_context.reset();
}

#ifdef __cplusplus
}
#endif
