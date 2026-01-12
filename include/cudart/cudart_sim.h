#ifndef CUDART_SIM_H
#define CUDART_SIM_H

#include <driver_types.h>
#include <cuda_runtime.h>
#include <memory>

// 前向声明
class GPUContext;
class PtxInterpreter;

// 全局GPUContext和PtxInterpreter实例声明
extern std::unique_ptr<GPUContext> g_gpu_context;
extern std::unique_ptr<PtxInterpreter> g_ptx_interpreter;

// CUDA运行时函数模拟声明
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 注册胖二进制文件（fat binary），提取并解析PTX代码
 * 
 * @param fatCubinHandle 句柄指针
 * @param fat_bin fat binary数据
 * @param fat_bin_size fat binary大小
 * @param version 版本号
 * @return void** 注册后的句柄
 */
void **__cudaRegisterFatBinary(void **fatCubinHandle, void *fat_bin,
                               unsigned long long fat_bin_size,
                               unsigned int version);

/**
 * @brief 注册CUDA函数
 * 
 * @param fatCubinHandle fat binary句柄
 * @param hostFun 主机端函数指针
 * @param deviceFun 设备端函数
 * @param deviceName 设备端函数名
 * @param thread_limit 线程限制
 * @param tid 线程ID指针
 * @param bid 块ID指针
 * @param bDim 块维度指针
 * @param gDim 网格维度指针
 * @param wSize 共享内存大小指针
 */
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            cudaKernel_t deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize);

/**
 * @brief 注册全局变量
 * 
 * @param fatCubinHandle fat binary句柄
 * @param hostVar 主机端变量指针
 * @param deviceAddress 设备端地址
 * @param deviceName 设备端变量名
 * @param ext ext标志
 * @param size 变量大小
 * @param constant 是否为常量
 * @param global 是否为全局变量
 */
void __cudaRegisterVar(void **fatCubinHandle, void *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global);

/**
 * @brief 取消注册fat binary
 * 
 * @param fatCubinHandle fat binary句柄
 */
void __cudaUnregisterFatBinary(void **fatCubinHandle);

#ifdef __cplusplus
}
#endif

#endif // CUDART_SIM_H