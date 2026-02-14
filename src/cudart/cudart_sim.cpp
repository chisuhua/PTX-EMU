/**
 * @author gtyinstinct
 * generate fake libcudart.so to replace origin libcudart.so
 */

#include "antlr4-runtime.h"
#include "ptxLexer.h"
#include "ptxParser.h"
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
#include "ptxsim/ptx_config.h" // 添加DebugConfig所需的头文件
#include "utils/cubin_utils.h" // 添加cuobjdump工具函数
#include "utils/logger.h"
#include <string>

// 添加缺失的宏定义
#define PTX_ERROR_CUDART(fmt, ...)                                             \
    do {                                                                       \
        fprintf(stderr, "[ERROR][CUDART] %s:%d: " fmt "\n", __FILE__,          \
                __LINE__, ##__VA_ARGS__);                                      \
        fflush(stderr);                                                        \
    } while (0)

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

// 新增缺失的全局变量和数据结构
std::map<uint64_t, std::string> func2name;
std::map<uint64_t, cudaKernel_t> func2kernel;
std::map<cudaKernel_t, const char *> kernel2func;
dim3 _gridDim, _blockDim;
size_t _sharedMem;

// 全局GPUContext和PtxInterpreter实例
std::unique_ptr<GPUContext> g_gpu_context;
std::unique_ptr<PtxInterpreter> g_ptx_interpreter;

// 配置文件路径
static const char *CONFIG_FILE = "config.ini";

// 初始化调试环境和GPUContext
void initialize_environment() {
    // 解析配置文件一次，然后分别设置各个组件
    inipp::Ini<char> ini;
    std::ifstream is(CONFIG_FILE);
    if (is.is_open()) {
        ini.parse(is);

        // 设置日志配置
        auto logger_section = ini.sections["logger"];
        ptxsim::LoggerConfig::get().load_from_ini_section(logger_section);

        // 设置调试器配置
        auto debugger_section = ini.sections["debugger"];
        ptxsim::DebugConfig::get().load_from_ini_section(debugger_section);

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
        g_ptx_interpreter = std::make_unique<PtxInterpreter>();

        PTX_INFO_EMU("Configuration loaded from %s", CONFIG_FILE);
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
        exit(1);
    }
    self_exe_path[size] = '\0';

    // 2. 从当前进程提取PTX代码
    std::string ptx_code = extract_ptx_with_cuobjdump(self_exe_path);

    if (ptx_code.empty()) {
        std::cerr << "Error: Could not extract PTX code" << std::endl;
        return nullptr;
    }

    // 使用g_gpu_context的get_device_memory函数获取SimpleMemory实例
    SimpleMemory *simple_mem = g_gpu_context->get_device_memory();

    // 设置CudaDriver使用的SimpleMemory实例
    CudaDriver::instance().set_simple_memory(simple_mem);

    // 3. 解析PTX代码
    // FIXME: This code is incomplete - ptxListener is commented out
    // ANTLRInputStream input(ptx_code);
    // ptxLexer lexer(&input);
    // CommonTokenStream tokens(&lexer);
    // tokens.fill();
    // ptxParser parser(&tokens);

    // 4. 初始化PtxInterpreter - 现在会拷贝ptxContext以避免悬垂引用
    // g_ptx_interpreter->set_ptx_context(ptxListener.ptxContext);

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

    // 调用PtxInterpreter的launch函数
    g_ptx_interpreter->launchPtxInterpreter(
        g_ptx_interpreter->get_ptx_context(), func2name[(uint64_t)func], args,
        gridDim3, blockDim3);

    // 等待kernel执行完成
    g_gpu_context->wait_for_completion();

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

    if (!dst || !src || count == 0) {
        return cudaErrorInvalidValue;
    }

    // 获取CudaDriver的全局内存池地址
    uint64_t global_pool = (uint64_t)CudaDriver::instance().get_global_pool();
    uint64_t global_size = (uint64_t)CudaDriver::instance().get_global_size();
    if (!global_pool) {
        return cudaErrorInitializationError;
    }

    // 根据复制类型执行内存复制
    switch (kind) {
    case cudaMemcpyHostToHost: {
        // 从主机内存复制到设备内存
        std::memcpy(dst, src, count);
        break;
    }
    case cudaMemcpyHostToDevice: {
        // 从主机内存复制到设备内存
        // dst是设备指针（即偏移量），src是主机指针
        uint64_t device_offset = reinterpret_cast<uint64_t>(dst);
        if (device_offset >= global_pool) {
            device_offset -= global_pool;
        }
        if (device_offset >= global_size) {
            return cudaErrorInvalidValue;
        }

        std::memcpy((uint8_t *)(global_pool + device_offset), src, count);
        break;
    }
    case cudaMemcpyDeviceToHost: {
        // 从设备内存复制到主机内存
        // src是设备指针（即偏移量），dst是主机指针
        uint64_t device_offset = reinterpret_cast<uint64_t>(src);
        if (device_offset >= global_pool) {
            device_offset -= global_pool;
        }
        if (device_offset >= global_size) {
            return cudaErrorInvalidValue;
        }

        std::memcpy(dst, (uint8_t *)(global_pool + device_offset), count);
        break;
    }
    case cudaMemcpyDeviceToDevice: {
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
            return cudaErrorInvalidValue;
        }

        std::memcpy((uint8_t *)(global_pool + dst_device_offset),
                    (uint8_t *)(global_pool + src_device_offset), count);
        break;
    }
    default:
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaMemcpyAsync(%p, %p, %zu, %d, %p)", dst, src,
                  count, kind, stream);

    // 异步复制在仿真器中与同步复制相同
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    PTX_DEBUG_EMU("Called cudaMemset(%p, %d, %zu)", devPtr, value, count);

    if (!devPtr) {
        return cudaErrorInvalidValue;
    }

    // 获取CudaDriver的全局内存池地址
    uint8_t *global_pool = CudaDriver::instance().get_global_pool();
    if (!global_pool) {
        return cudaErrorInitializationError;
    }

    uint64_t device_offset = reinterpret_cast<uint64_t>(devPtr);
    if (device_offset >= CudaDriver::instance().get_global_size()) {
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

    // 在仿真器中，同步是立即完成的
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

    // 在仿真器中，流只是一个占位符
    *stream = reinterpret_cast<cudaStream_t>(new int(0));
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaStreamDestroy(%p)", stream);

    if (stream) {
        delete reinterpret_cast<int *>(stream);
    }

    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    PTX_DEBUG_EMU("Called cudaStreamSynchronize(%p)", stream);

    // 在仿真器中，同步是立即完成的
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