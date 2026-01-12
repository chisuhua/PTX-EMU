#ifndef CUDART_INTRINSICS_H
#define CUDART_INTRINSICS_H

#include <stddef.h> // For size_t

// 定义CUDA的__host__和__device__宏，如果尚未定义
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 避免与CUDA SDK头文件冲突 - 检查CUDA运行时头文件是否已经定义了这些类型
// 我们使用更严格的保护，确保在CUDA头文件已经定义时不重复定义
#ifndef __CUDA_RUNTIME_H__
#ifndef __DRIVER_TYPES_H__
#ifndef __VECTOR_TYPES_H__
#ifndef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#ifndef __CUDACC_RTC__

// 定义错误类型
typedef enum cudaError_enum {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMissingConfiguration = 2,
    cudaErrorMemoryAllocation = 3,
    cudaErrorInitializationError = 4,
    cudaErrorLaunchFailure = 6,
    cudaErrorPriorLaunchFailure = 7,
    cudaErrorLaunchTimeout = 8,
    cudaErrorLaunchOutOfResources = 9,
    cudaErrorInvalidDeviceFunction = 10,
    cudaErrorInvalidConfiguration = 11,
    cudaErrorInvalidDevice = 12,
    cudaErrorStartupFailure = 13
} cudaError_t;

// 定义dim3结构体
typedef struct dim3 {
    unsigned int x, y, z;
    __host__ __device__ dim3(unsigned int x = 1, unsigned int y = 1,
                             unsigned int z = 1)
        : x(x), y(y), z(z) {}
} dim3;

// 定义向量类型
typedef struct uint3 {
    unsigned int x, y, z;
} uint3;

typedef struct uint4 {
    unsigned int x, y, z, w;
} uint4;

typedef struct char3 {
    signed char x, y, z;
    unsigned char pad; // padding
} char3;

typedef struct char4 {
    signed char x, y, z, w;
} char4;

typedef struct uchar3 {
    unsigned char x, y, z;
    unsigned char pad; // padding
} uchar3;

typedef struct uchar4 {
    unsigned char x, y, z, w;
} uchar4;

typedef struct short3 {
    short x, y, z;
} short3;

typedef struct short4 {
    short x, y, z, w;
} short4;

typedef struct ushort3 {
    unsigned short x, y, z;
} ushort3;

typedef struct ushort4 {
    unsigned short x, y, z, w;
} ushort4;

typedef struct int3 {
    int x, y, z;
} int3;

typedef struct int4 {
    int x, y, z, w;
} int4;

typedef struct long3 {
    long long int x, y, z;
} long3;

typedef struct long4 {
    long long int x, y, z, w;
} long4;

typedef struct ulong3 {
    unsigned long long int x, y, z;
} ulong3;

typedef struct ulong4 {
    unsigned long long int x, y, z, w;
} ulong4;

typedef struct float3 {
    float x, y, z;
} float3;

typedef struct float4 {
    float x, y, z, w;
} float4;

typedef struct double3 {
    double x, y, z;
} double3;

typedef struct double4 {
    double x, y, z, w;
} double4;

// 定义CUDA函数缓存类型
typedef enum {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2
} cudaFuncCache;

// 定义CUDA共享内存配置类型
typedef enum {
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2
} cudaSharedMemConfig;

// 定义内存拷贝类型
typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
} cudaMemcpyKind;

// 定义设备属性类型
typedef enum {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxRegistersPerBlock = 11,
    cudaDevAttrClockRate = 12,
    cudaDevAttrMemoryClockRate = 13,
    cudaDevAttrMemoryBusWidth = 14,
    cudaDevAttrL2CacheSize = 15
} cudaDeviceAttr;

// 定义计算模式类型
typedef enum {
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3
} cudaComputeMode;

// 定义CUDA驱动API类型
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef void *CUevent;

typedef enum { CUDA_SUCCESS = 0 } CUresult;

// 定义cudaDeviceProp结构体
typedef struct __cudaDeviceProp_v1 {
    char name[256];           ///< ASCII string identifying device
    size_t totalGlobalMem;    ///< Global memory available on device in bytes
    size_t sharedMemPerBlock; ///< Shared memory available per block in bytes
    int regsPerBlock;         ///< 32-bit registers available per block
    int warpSize;             ///< Warp size in threads
    size_t memPitch;        ///< Maximum pitch in bytes allowed by memory copies
    int maxThreadsPerBlock; ///< Maximum number of threads per block
    int maxThreadsDim[3];   ///< Maximum size of each dimension of a block
    int maxGridSize[3];     ///< Maximum size of each dimension of a grid
    int clockRate;          ///< Clock frequency in kilohertz
    size_t totalConstMem;   ///< Constant memory available on device in bytes
    int major;              ///< Major compute capability
    int minor;              ///< Minor compute capability
    size_t textureAlignment; ///< Alignment requirement for textures
    int deviceOverlap; ///< Device can concurrently copy memory and execute a
                       ///< kernel
    int multiProcessorCount;      ///< Number of multiprocessors on device
    int kernelExecTimeoutEnabled; ///< Specifies whether there is a run time
                                  ///< limit on kernels
    int integrated;               ///< Device is integrated with host memory
    int canMapHostMemory;         ///< Device can map host memory with
                                  ///< cudaHostAlloc/cudaHostGetDevicePointer
    int computeMode;  ///< Compute mode (See ::cudaComputeMode for details)
    int maxTexture1D; ///< Maximum 1D texture size
    int maxTexture1DMipmap;     ///< Maximum 1D mipmapped texture size
    int maxTexture1DLinear;     ///< Maximum 1D texture size (legacy, removed in
                                ///< newer versions)
    int maxTexture2D[2];        ///< Maximum 2D texture dimensions
    int maxTexture2DMipmap[2];  ///< Maximum 2D mipmapped texture dimensions
    int maxTexture2DLinear[3];  ///< Maximum 2D texture dimensions (linear
                                ///< memory) (legacy, removed in newer versions)
    int maxTexture2DGather[2];  ///< Maximum 2D texture dimensions if texture
                                ///< gather operations have to be performed
                                ///< (legacy, removed in newer versions)
    int maxTexture3D[3];        ///< Maximum 3D texture dimensions
    int maxTexture3DAlt[3];     ///< Alternate maximum 3D texture dimensions
                                ///< (legacy, removed in newer versions)
    int maxTextureCubemap;      ///< Maximum cubemap texture dimensions (legacy,
                                ///< removed in newer versions)
    int maxTexture1DLayered[2]; ///< Maximum 1D layered texture dimensions
                                ///< (legacy, removed in newer versions)
    int maxTexture2DLayered[3]; ///< Maximum 2D layered texture dimensions
                                ///< (legacy, removed in newer versions)
    int maxTextureCubemapLayered[2]; ///< Maximum cubemap layered texture
                                     ///< dimensions (legacy, removed in newer
                                     ///< versions)
    size_t surfaceAlignment;         ///< Alignment requirements for surfaces
    int concurrentKernels; ///< Device can possibly execute multiple kernels
                           ///< concurrently
    int ECCEnabled;        ///< Device has ECC support enabled
    int pciBusID;          ///< PCI bus identifier of the device
    int pciDeviceID; ///< PCI device (also known as slot) identifier of the
                     ///< device
    int tccDriver;   ///< 1 if device is using a TCC driver, 0 if using a WDDM
                     ///< driver (Windows only)
    int asyncEngineCount;  ///< Number of asynchronous engines (Windows only)
                           ///< (legacy, removed in newer versions)
    int unifiedAddressing; ///< Device shares a unified address space with the
                           ///< host (legacy, removed in newer versions)
    int memoryClockRate; ///< Peak memory clock frequency in kilohertz (legacy,
                         ///< removed in newer versions)
    int memoryBusWidth; ///< Global memory bus width in bits (legacy, removed in
                        ///< newer versions)
    int l2CacheSize;    ///< Size of L2 cache in bytes (legacy, removed in newer
                        ///< versions)
    int persistingL2CacheMaxSize;    ///< Device's maximum l2 persisting lines
                                     ///< capacity setting in bytes (legacy,
                                     ///< removed in newer versions)
    int maxThreadsPerMultiProcessor; ///< Maximum resident threads per
                                     ///< multiprocessor (legacy, removed in
                                     ///< newer versions)
    int streamPrioritiesSupported;   ///< Device supports stream priorities
                                     ///< (legacy, removed in newer versions)
    int globalL1CacheSupported;      ///< Device supports caching globals in L1
                                     ///< (legacy, removed in newer versions)
    int localL1CacheSupported;       ///< Device supports caching locals in L1
                                     ///< (legacy, removed in newer versions)
    size_t sharedMemPerMultiprocessor; ///< Shared memory available per
                                       ///< multiprocessor in bytes (legacy,
                                       ///< removed in newer versions)
    int regsPerMultiprocessor;         ///< 32-bit registers available per
                               ///< multiprocessor (legacy, removed in newer
                               ///< versions)
    int managedMemory;   ///< Device supports allocating managed memory on this
                         ///< system (legacy, removed in newer versions)
    int isMultiGpuBoard; ///< Device is on a multi-GPU board (legacy, removed in
                         ///< newer versions)
    int multiGpuBoardGroupID; ///< Unique identifier for a group of devices on
                              ///< the same multi-GPU board (legacy, removed in
                              ///< newer versions)
    int hostNativeAtomicSupported; ///< Link between the device and the host
                                   ///< supports native atomic operations
                                   ///< (legacy, removed in newer versions)
    int singleToDoublePrecisionPerfRatio; ///< Ratio of single precision
                                          ///< performance (FPS) to double
                                          ///< precision performance (legacy,
                                          ///< removed in newer versions)
    int pageableMemoryAccess; ///< Device supports coherently accessing pageable
                              ///< memory without calling
                              ///< cudaHostAlloc/cudaHostGetDevicePointer
                              ///< (legacy, removed in newer versions)
    int concurrentManagedAccess; ///< Device can coherently access managed
                                 ///< memory concurrently with the CPU (legacy,
                                 ///< removed in newer versions)
    int computePreemptionSupported; ///< Device supports Compute Preemption
                                    ///< (legacy, removed in newer versions)
    int canUseHostPointerForRegisteredMem; ///< Device can access host
                                           ///< registered memory at the same
                                           ///< virtual address as the CPU
                                           ///< (legacy, removed in newer
                                           ///< versions)
    int cooperativeLaunch; ///< Device supports launching cooperative kernels
                           ///< via cudaLaunchCooperativeKernel (legacy, removed
                           ///< in newer versions)
    int cooperativeMultiDeviceLaunch; ///< Device can participate in cooperative
                                      ///< kernels launched against multiple
                                      ///< devices (legacy, removed in newer
                                      ///< versions)
    size_t sharedMemPerBlockOptin; ///< Per-block dynamic shared memory usable
                                   ///< by the device (legacy, removed in newer
                                   ///< versions)
    int pageableMemoryAccessUsesHostPageTables; ///< Device accesses pageable
                                                ///< memory via the host's page
                                                ///< tables (legacy, removed in
                                                ///< newer versions)
    int directManagedMemAccessFromHost; ///< Host can directly access managed
                                        ///< memory on the device (legacy,
                                        ///< removed in newer versions)
    int maxBlocksPerMultiProcessor; ///< Maximum number of resident blocks per
                                    ///< multiprocessor (legacy, removed in
                                    ///< newer versions)
    int accessPolicyMaxWindowSize;  ///< The maximum value of
                                    ///< ::cudaAccessPolicyWindow::num_bytes
                                    ///< (legacy, removed in newer versions)
    int reservedSharedMemPerBlock;  ///< The maximum size in bytes of shared
                                    ///< memory per block that is available for
                                    ///< the user-managed shared memory (legacy,
                                    ///< removed in newer versions)
    int maxSurface1D;               ///< Maximum 1D surface size
    int maxSurface2D[2];            ///< Maximum 2D surface dimensions
    int maxSurface3D[3];            ///< Maximum 3D surface dimensions
    int maxSurface1DLayered[2];     ///< Maximum 1D layered surface dimensions
    int maxSurface2DLayered[3];     ///< Maximum 2D layered surface dimensions
    int maxSurfaceCubemap;          ///< Maximum cubemap surface dimensions
    int maxSurfaceCubemapLayered[2]; ///< Maximum cubemap layered surface
                                     ///< dimensions
} cudaDeviceProp;

// 定义一些常用的CUDA类型
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef void *cudaKernel_t;

#endif // __CUDACC_RTC__
#endif // __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif // __VECTOR_TYPES_H__
#endif // __DRIVER_TYPES_H__
#endif // __CUDA_RUNTIME_H__

// 即使在CUDA头文件已定义的情况下，也需要定义回调函数类型
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status,
                                     void *userData);

#ifdef __cplusplus
}
#endif

#endif // CUDART_INTRINSICS_H