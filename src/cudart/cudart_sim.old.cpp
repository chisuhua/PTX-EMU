/**
 * @author gtyinstinct
 * generate fake libcudart.so to replace origin libcudart.so
 */

#include "inipp/inipp.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <driver_types.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <unistd.h>

#include "antlr4-runtime.h"
#include "memory/memory_manager.h"
#include "memory/simple_memory.h"
#include "ptxLexer.h"
#include "ptxParser.h"
#include "ptxParserBaseListener.h"
#include "ptx_interpreter.h"
#include "ptx_parser/ptx_parser.h"
#include "ptxsim/gpu_context.h"
#include "ptxsim/ptx_debug.h"
#include "utils/logger.h"

#define __my_func__ __func__

using namespace ptxparser;
using namespace antlr4;

std::string ptx_buffer;
std::map<uint64_t, std::string> func2name;
std::map<uint64_t, cudaKernel_t> func2kernel;
std::map<cudaKernel_t, const char *> kernel2func;
dim3 _gridDim, _blockDim;
size_t _sharedMem;
PtxListener ptxListener;
PtxInterpreter ptxInterpreter;

// 全局GPUContext实例
std::unique_ptr<GPUContext> g_gpu_context;

// std::map<uint64_t, bool> memAlloc;
// 全局初始化（在 main 或首次调用时）
static std::unique_ptr<SimpleMemory>
    g_simple_memory; // 使用 unique_ptr 延迟构造

static void ensure_memory_manager_initialized() {
    static bool initialized = false;
    if (!initialized) {
        // 1. 获取 MemoryManager 单例
        MemoryManager &mgr = MemoryManager::instance();

        // 2. 构造 SimpleMemory（复用 MemoryManager 的内存池）
        g_simple_memory = std::make_unique<SimpleMemory>(
            mgr.get_global_pool(), MemoryManager::GLOBAL_SIZE,
            mgr.get_shared_pool(), MemoryManager::SHARED_SIZE);

        // 3. 注入 MemoryInterface
        mgr.set_memory_interface(g_simple_memory.get());

        // =============================================================================
        // Gem5 接入占位（未来只需取消注释以下代码，并注释 SimpleMemory 部分）
        // =============================================================================
        // static std::unique_ptr<Gem5MemoryService> g_gem5_memory;
        // g_gem5_memory = std::make_unique<Gem5MemoryService>();
        // mgr.set_memory_interface(g_gem5_memory.get());
        // =============================================================================
        initialized = true;
    }
}

// 配置文件路径
static const char *CONFIG_FILE = "config.ini";

#define LOGEMU 1

// 从INI配置中加载日志配置
void load_logger_config(const inipp::Ini<char>::Section &logger_section) {
    auto &logger_config = ptxsim::LoggerConfig::get();

    std::string level_str;
    inipp::get_value(logger_section, "global_level", level_str);
    if (!level_str.empty()) {
        logger_config.set_global_level(
            logger_config.string_to_log_level(level_str));
    }

    std::string target_str;
    inipp::get_value(logger_section, "target", target_str);
    if (!target_str.empty()) {
        logger_config.set_target_from_string(target_str);
    }

    std::string logfile;
    inipp::get_value(logger_section, "logfile", logfile);
    if (!logfile.empty()) {
        logger_config.set_logfile(logfile);
    }

    std::string async_str;
    inipp::get_value(logger_section, "async", async_str);
    if (!async_str.empty()) {
        bool async = (async_str == "true" || async_str == "1");
        logger_config.enable_async_logging(async);
    }

    std::string colorize_str;
    inipp::get_value(logger_section, "colorize", colorize_str);
    if (!colorize_str.empty()) {
        bool colorize = (colorize_str == "true" || colorize_str == "1");
        logger_config.set_use_color_output(colorize);
    }

    // 读取组件级别配置
    for (const auto &pair : logger_section) {
        if (pair.first.substr(0, 9) == "component") {
            std::string component = pair.first.substr(10); // skip "component."
            if (!component.empty()) {
                ptxsim::log_level level =
                    logger_config.string_to_log_level(pair.second);
                logger_config.set_component_level(component, level);
            }
        }
    }
}

// 从INI配置中加载调试器配置
void load_debugger_config(const inipp::Ini<char>::Section &debugger_section) {
    auto &debugger_config = ptxsim::DebugConfig::get();

    std::string trace_instr;
    inipp::get_value(debugger_section, "trace_instruction", trace_instr);
    if (!trace_instr.empty()) {
        bool trace = (trace_instr == "true" || trace_instr == "1");
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::MEMORY, trace);
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::ARITHMETIC, trace);
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::CONTROL, trace);
        debugger_config.enable_instruction_trace(ptxsim::InstructionType::LOGIC,
                                                 trace);
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::CONVERT, trace);
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::SPECIAL, trace);
        debugger_config.enable_instruction_trace(ptxsim::InstructionType::OTHER,
                                                 trace);
    }

    // 按类型设置指令跟踪
    std::string trace_memory;
    inipp::get_value(debugger_section, "trace_instruction_type.memory",
                     trace_memory);
    if (!trace_memory.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::MEMORY,
            (trace_memory == "true" || trace_memory == "1"));
    }

    std::string trace_arithmetic;
    inipp::get_value(debugger_section, "trace_instruction_type.arithmetic",
                     trace_arithmetic);
    if (!trace_arithmetic.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::ARITHMETIC,
            (trace_arithmetic == "true" || trace_arithmetic == "1"));
    }

    std::string trace_control;
    inipp::get_value(debugger_section, "trace_instruction_type.control",
                     trace_control);
    if (!trace_control.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::CONTROL,
            (trace_control == "true" || trace_control == "1"));
    }

    std::string trace_logic;
    inipp::get_value(debugger_section, "trace_instruction_type.logic",
                     trace_logic);
    if (!trace_logic.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::LOGIC,
            (trace_logic == "true" || trace_logic == "1"));
    }

    std::string trace_convert;
    inipp::get_value(debugger_section, "trace_instruction_type.convert",
                     trace_convert);
    if (!trace_convert.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::CONVERT,
            (trace_convert == "true" || trace_convert == "1"));
    }

    std::string trace_special;
    inipp::get_value(debugger_section, "trace_instruction_type.special",
                     trace_special);
    if (!trace_special.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::SPECIAL,
            (trace_special == "true" || trace_special == "1"));
    }

    std::string trace_other;
    inipp::get_value(debugger_section, "trace_instruction_type.other",
                     trace_other);
    if (!trace_other.empty()) {
        debugger_config.enable_instruction_trace(
            ptxsim::InstructionType::OTHER,
            (trace_other == "true" || trace_other == "1"));
    }

    // 设置内存和寄存器跟踪
    std::string trace_mem;
    inipp::get_value(debugger_section, "trace_memory", trace_mem);
    if (!trace_mem.empty()) {
        debugger_config.enable_memory_trace(
            (trace_mem == "true" || trace_mem == "1"));
    }

    std::string trace_reg;
    inipp::get_value(debugger_section, "trace_registers", trace_reg);
    if (!trace_reg.empty()) {
        debugger_config.enable_register_trace(
            (trace_reg == "true" || trace_reg == "1"));
    }
}

// 初始化调试环境和GPUContext
void initialize_environment() {
    // 解析配置文件一次，然后分别设置各个组件
    inipp::Ini<char> ini;
    std::ifstream is(CONFIG_FILE);
    if (is.is_open()) {
        ini.parse(is);

        // 设置日志配置
        auto logger_section = ini.sections["logger"];
        load_logger_config(logger_section);

        // 设置调试器配置
        auto debugger_section = ini.sections["debugger"];
        load_debugger_config(debugger_section);

        PTX_INFO_EMU("Configuration loaded from %s", CONFIG_FILE);
    } else {
        PTX_INFO_EMU("No configuration file found, using default settings");
        // 设置默认的日志级别
        ptxsim::LoggerConfig::get().set_global_level(ptxsim::log_level::info);
    }

    // 初始化全局GPUContext
    static bool gpu_initialized = false;
    if (!gpu_initialized) {
        // 从INI配置文件中读取GPU配置文件路径
        std::string gpu_config_filename;
        auto gpu_section = ini.sections["gpu"];
        inipp::get_value(gpu_section, "gpu_config_file", gpu_config_filename);
        if (!gpu_config_filename.empty()) {
            // 创建GPUContext并直接加载JSON配置
            g_gpu_context =
                std::make_unique<GPUContext>("configs/" + gpu_config_filename);
        } else {
            // 如果INI文件中没有指定GPU配置文件或加载失败，使用默认配置
            g_gpu_context = std::make_unique<GPUContext>();
        }
        g_gpu_context->init();
        gpu_initialized = true;
    }
}

extern "C" {

void **__cudaRegisterFatBinary(void *fatCubin) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif

    // 初始化调试环境
    static bool debug_initialized = false;
    if (!debug_initialized) {
        initialize_environment();
        debug_initialized = true;
    }

    static bool if_executed = 0;

    if (!if_executed) {
        if_executed = 1;
        // get program abspath
        char self_exe_path[1025] = "";
        long size = readlink("/proc/self/exe", self_exe_path, 1024);
        assert(size != -1);
        self_exe_path[size] = '\0';
#ifdef LOGEMU
        printf("EMU: self exe links to %s\n", self_exe_path);
#endif

        // get ptx file name embedded in binary
        char cmd[1024] = "";
        snprintf(cmd, 1024,
                 "cuobjdump -lptx %s | cut -d : -f 2 | awk '{$1=$1}1' > %s",
                 self_exe_path, "__ptx_list__");
        if (system(cmd) != 0) {
#ifdef LOGEMU
            printf("EMU: fail to execute %s\n", cmd);
#endif
            exit(0);
        }

        // get ptx embedded in binary
        std::ifstream infile("__ptx_list__");
        std::string ptx_file;
        while (std::getline(infile, ptx_file)) {
#ifdef LOGEMU
            printf("EMU: extract PTX file %s \n", ptx_file.c_str());
#endif
            snprintf(cmd, 1024, "cuobjdump -xptx %s %s >/dev/null",
                     ptx_file.c_str(), self_exe_path);
            if (system(cmd) != 0) {
#ifdef LOGEMU
                printf("EMU: fail to execute %s\n", cmd);
#endif
                exit(0);
            }
            std::ifstream if_ptx(ptx_file);
            std::ostringstream of_ptx;
            char ch;
            while (of_ptx && if_ptx.get(ch))
                of_ptx.put(ch);
            ptx_buffer = of_ptx.str();
        }

        // clean intermediate file
        snprintf(cmd, 1024, "rm __ptx_list__ %s", ptx_file.c_str());
        system(cmd);

        // launch antlr4 parse
        ANTLRInputStream input(ptx_buffer);
        ptxLexer lexer(&input);
        CommonTokenStream tokens(&lexer);
        tokens.fill();
        ptxParser parser(&tokens);
        parser.addParseListener(&ptxListener);
        tree::ParseTree *tree = parser.ast();
    }
#ifdef LOGEMU
    printf("EMU: call_end %s\n", __my_func__);
    ptxListener.test_semantic();
#endif
    return nullptr;
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            cudaKernel_t deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("EMU: hostFun %p\n", hostFun);
    printf("EMU: deviceFun %p\n", deviceFun);
    printf("EMU: deviceFunName %s\n", deviceName);
#endif
    func2name[(uint64_t)hostFun] = *(new std::string(deviceName));
    func2kernel[(uint64_t)hostFun] = (cudaKernel_t)deviceFun;
    kernel2func[deviceFun] = hostFun;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    ptxListener.test_semantic();
#endif
}

cudaError_t cudaMalloc(void **p, size_t s) {
    // *p = malloc(s);
    // memAlloc[(uint64_t)p] = 1;
    ensure_memory_manager_initialized();
    *p = MemoryManager::instance().malloc(s);
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("EMU: malloc %p\n", *p);
#endif
    return (*p) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       enum cudaMemcpyKind kind) {
    memcpy(dst, src, count);
#ifdef LOGEMU
    printf("EMU: memcpy dst:%p src:%p\n", dst, src);
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event, unsigned int flags) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

unsigned __cudaPushCallConfiguration(
    dim3 gridDim, dim3 blockDim, size_t sharedMem = 0,
    struct CUstream_st *stream = 0 // temporily ignore stream
) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("EMU: gridDim(%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("EMU: blockDim(%d,%d,%d)\n", blockDim.x, blockDim.y, blockDim.z);
#endif
    _gridDim = gridDim;
    _blockDim = blockDim;
    _sharedMem = sharedMem;
    return 0;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t end) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    ensure_memory_manager_initialized();
    auto ret = MemoryManager::instance().free(devPtr);
    if (ret == Success)
        return cudaSuccess;
    else if (ret == ErrorMemoryAllocation)
        return cudaErrorMemoryAllocation;
    else if (ret == ErrorInvalidValue)
        return cudaErrorInvalidValue;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
    *gridDim = _gridDim;
    *blockDim = _blockDim;
    *sharedMem = _sharedMem;
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t __cudaGetKernel(cudaKernel_t *kernelPtr, const void *funcAddr) {
    // PTX-EMU 的实现
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("EMU: kernelPtr=%p funcAddr=%p\n", kernelPtr, funcAddr);
#endif
    *kernelPtr = func2kernel[(uint64_t)funcAddr];
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream // temporily ignore stream
) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("EMU: func %p\n", func);
    printf("EMU: arg %p\n", args);
#endif

    // 添加参数内容打印的日志，增强安全性
    if (args) {
        // 打印参数数组地址
        PTX_DEBUG_EMU("cudaLaunchKernel args array address: %p", args);

        int i = 0;
        PTX_DEBUG_EMU("cudaLaunchKernel argument[%d]: address=%p, value=0x%lx",
                      i, args[i], *(uint64_t *)args[i]);
    }

    printf("EMU: deviceFunName %s\n", func2name[(uint64_t)func].c_str());
    printf("EMU: gridDim(%d,%d,%d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("EMU: blockDim(%d,%d,%d)\n", blockDim.x, blockDim.y, blockDim.z);

    Dim3 gridDim3(gridDim.x, gridDim.y, gridDim.z);
    Dim3 blockDim3(blockDim.x, blockDim.y, blockDim.z);

    // 调用PtxInterpreter的launch函数，提交请求到全局GPUContext
    ptxInterpreter.launchPtxInterpreter(ptxListener.ptxContext,
                                        func2name[(uint64_t)func], args,
                                        gridDim3, blockDim3);

    // 等待kernel执行完成
    g_gpu_context->wait_for_completion();

    return cudaSuccess;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem,
                               cudaStream_t stream // temporily ignore stream
) {
    return cudaLaunchKernel(kernel2func[kernel], gridDim, blockDim, args,
                            sharedMem, stream);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       int size, int constant, int global) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("%p %p %s\n", hostVar, deviceAddress, deviceName);
#endif
    std::string s(deviceName);
    ptxInterpreter.constName2addr[s] = (uint64_t)hostVar;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                              unsigned int flags = cudaMemAttachGlobal) {
    // *devPtr = malloc(size);
    // memAlloc[(uint64_t)devPtr] = 1;
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    ensure_memory_manager_initialized();
    *devPtr = MemoryManager::instance().malloc_managed(size);
    return (*devPtr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaDeviceSynchronize(void) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaGetLastError(void) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream = 0) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaPeekAtLastError(void) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaThreadSynchronize(void) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int *count) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    assert(count);
    *count = 1;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    prop->name[0] = '\0';
    strcpy(prop->name, "PTX-EMU");
    prop->major = 8;
    prop->minor = 0;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

cudaError_t cudaMemcpyToSymbol(void *symbol, void *src, size_t count,
                               size_t offset = 0,
                               cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
    printf("src:%p symbal:%p\n", src, symbol);
#endif
    memcpy((void *)((uint64_t)symbol + offset), src, count);
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

char __cudaInitModule(void **fatCubinHandle) {
#ifdef LOGEMU
    printf("EMU: call %s\n", __my_func__);
#endif
    return cudaSuccess;
}

} // end of extern "C"