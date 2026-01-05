# GPGPU架构模拟设计文档

## 概述

本文档详细描述了PTX-EMU项目中GPGPU硬件架构的模拟设计。该架构包括GPUContext作为顶层容器，管理多个SMContext（流式多处理器），以及相关的硬件抽象模块。

## 架构层次

### 1. GPUContext（顶层GPU模拟）

GPUContext是整个GPU模拟的顶层容器，负责管理整个GPU的硬件资源和状态。

#### 1.1 硬件配置管理

GPUContext通过配置文件管理GPU的硬件参数：

```cpp
struct GPUConfig {
    int num_sms;                    // SM数量
    int max_warps_per_sm;           // 每个SM最大warp数
    int max_threads_per_sm;         // 每个SM最大线程数
    size_t shared_mem_size_per_sm;  // 每个SM共享内存大小
    int registers_per_sm;           // 每个SM寄存器数量
    int max_blocks_per_sm;          // 每个SM最大block数
    int warp_size;                  // warp大小
};
```

#### 1.2 硬件资源管理

- **SM管理**：管理多个SMContext实例，控制GPU级别的执行状态
- **资源分配**：管理全局资源分配，如全局内存、常量内存等
- **执行协调**：协调多个SM的执行，处理跨SM同步和通信

#### 1.3 配置驱动初始化

GPUContext支持从JSON配置文件加载硬件参数，实现不同GPU架构的灵活配置。

### 2. SMContext（流式多处理器）

SMContext模拟NVIDIA GPU中的流式多处理器（Streaming Multiprocessor）。

#### 2.1 硬件资源

- **Warp调度器**：管理warp的执行调度
- **共享内存**：每个SM的共享内存管理
- **寄存器文件**：管理分配给该SM的寄存器资源
- **执行单元**：模拟SM中的CUDA核心、SFU、LD/ST单元等

#### 2.2 资源管理

- **线程块管理**：管理分配到该SM的CTA（Cooperative Thread Array）
- **warp管理**：维护活跃warp列表和调度状态
- **内存管理**：管理共享内存和寄存器资源分配

#### 2.3 执行模型

- **warp级执行**：按warp粒度调度和执行指令
- **SIMT执行**：实现单指令多线程执行模型
- **分支处理**：处理warp内的分支分歧和重新合并

### 3. CTAContext（线程块上下文）

CTAContext管理单个线程块的执行。

#### 3.1 功能

- **线程管理**：管理线程块中的所有线程
- **同步机制**：实现线程块内的同步操作
- **资源分配**：管理线程块的共享内存和寄存器分配

### 4. WarpContext（warp上下文）

WarpContext管理单个warp的执行。

#### 4.1 功能

- **活跃掩码管理**：维护warp中活跃线程的掩码
- **指令分发**：将指令分发给warp中的活跃线程
- **分歧处理**：处理warp内的分支分歧

### 5. ThreadContext（线程上下文）

ThreadContext管理单个CUDA线程的执行状态。

#### 5.1 功能

- **寄存器状态**：维护线程的寄存器状态
- **执行状态**：管理线程的执行状态和PC
- **内存访问**：处理线程的内存访问请求

## 支持的GPU架构

### 1. NVIDIA Ampere架构

Ampere架构是NVIDIA最新的GPU架构之一，具有以下特点：

#### 1.1 硬件参数配置

```json
{
  "gpu_arch": "Ampere",
  "name": "NVIDIA A100",
  "num_sms": 108,
  "max_warps_per_sm": 64,
  "max_threads_per_sm": 2048,
  "shared_mem_size_per_sm": 163840,
  "registers_per_sm": 65536,
  "max_blocks_per_sm": 32,
  "warp_size": 32,
  "memory_bandwidth_gbps": 1555,
  "compute_units": {
    "cuda_cores": 6912,
    "tensor_cores": 432,
    "rt_cores": 54
  },
  "clock_rate_khz": 1410000
}
```

#### 1.2 特性

- **多实例GPU**：支持将GPU划分为多个独立的GPU实例
- **第三代Tensor Core**：支持TF32、BF16、FP16等数据类型
- **MIG（Multi-Instance GPU）**：硬件级隔离
- **改进的内存层次结构**：更大的L2缓存

### 2. NVIDIA Hopper架构

Hopper架构是NVIDIA面向HPC和AI的下一代架构。

#### 2.1 硬件参数配置

```json
{
  "gpu_arch": "Hopper",
  "name": "NVIDIA H100",
  "num_sms": 132,
  "max_warps_per_sm": 72,
  "max_threads_per_sm": 2304,
  "shared_mem_size_per_sm": 245760,
  "registers_per_sm": 65536,
  "max_blocks_per_sm": 32,
  "warp_size": 32,
  "memory_bandwidth_gbps": 3350,
  "compute_units": {
    "cuda_cores": 18432,
    "tensor_cores": 576,
    "rt_cores": 132
  },
  "clock_rate_khz": 1830000,
  "special_features": [
    "Transformer Engine",
    "DPX instructions",
    "FP8 support",
    "HBM3 memory"
  ]
}
```

#### 2.2 特性

- **Transformer Engine**：专门优化Transformer模型的计算引擎
- **第四代Tensor Core**：支持FP8精度，性能翻倍
- **DPX指令**：针对动态编程的专用指令
- **改进的MIG**：更细粒度的GPU实例划分

### 3. NVIDIA Blackwell架构

Blackwell架构是NVIDIA最新的GPU架构，专注于AI和HPC工作负载。

#### 3.1 硬件参数配置

```json
{
  "gpu_arch": "Blackwell",
  "name": "NVIDIA B200",
  "num_sms": 144,
  "max_warps_per_sm": 80,
  "max_threads_per_sm": 2560,
  "shared_mem_size_per_sm": 245760,
  "registers_per_sm": 65536,
  "max_blocks_per_sm": 32,
  "warp_size": 32,
  "memory_bandwidth_gbps": 4800,
  "compute_units": {
    "cuda_cores": 20480,
    "tensor_cores": 640,
    "rt_cores": 144
  },
  "clock_rate_khz": 2000000,
  "special_features": [
    "Transformer Engine v2",
    "FP4 and FP6 support",
    "Transformer Die-to-Die interconnect",
    "Advanced sparsity support"
  ]
}
```

#### 3.2 特性

- **Transformer Engine v2**：进一步优化Transformer模型的计算
- **FP4和FP6支持**：支持极低精度计算以提升AI推理性能
- **Die-to-Die互连**：连接多个芯片实现更大规模GPU
- **高级稀疏性支持**：硬件级稀疏矩阵计算支持

## 硬件模块配置

### 1. Tensor Core模块配置

Tensor Core是NVIDIA GPU中用于加速矩阵运算的专用单元，配置参数包括：

```json
{
  "tensor_cores": {
    "count": 32,
    "supported_types": [
      "FP64",
      "FP32",
      "FP16",
      "BF16",
      "TF32",
      "FP8",
      "INT8",
      "INT4"
    ],
    "matrix_size": [8, 8, 4],
    "max_precision": "FP64",
    "throughput_per_cycle": {
      "fp16": 128,
      "fp32": 64,
      "fp64": 32,
      "int8": 256,
      "int4": 512
    },
    "execution_units": [
      {
        "type": "wmma",
        "name": "wmma.884.f16",
        "operation": "A[8x4] * B[4x8] + C[8x8] = D[8x8]",
        "precision": "FP16"
      }
    ]
  }
}
```

#### 1.1 Tensor Core特性

- **精度支持**：支持多种数据精度的矩阵运算
- **吞吐量**：每个周期可执行的运算数量
- **矩阵尺寸**：支持的矩阵运算尺寸
- **执行单元**：不同类型的操作单元

### 2. 内存子系统配置

内存子系统包括全局内存、L1/L2缓存和共享内存，配置参数包括：

```json
{
  "memory_system": {
    "global_memory": {
      "capacity": 85899345920,
      "bandwidth_gbps": 1555,
      "type": "HBM3",
      "channels": 12
    },
    "l2_cache": {
      "capacity": 41943040,
      "line_size": 32,
      "associativity": 32,
      "bandwidth_gbps": 3000
    },
    "l1_cache": {
      "capacity_per_sm": 131072,
      "line_size": 32,
      "associativity": 16
    },
    "shared_memory": {
      "capacity_per_sm": 163840,
      "banks": 32,
      "bank_width_bits": 32
    },
    "constant_memory": {
      "capacity": 65536
    },
    "texture_memory": {
      "capacity": 134217728,
      "cache_size": 12288
    },
    "memory_hierarchy": {
      "global_to_l2_bandwidth_ratio": 1.0,
      "l2_to_l1_bandwidth_ratio": 4.0,
      "l1_to_sm_bandwidth_ratio": 8.0
    }
  }
}
```

#### 2.1 内存子系统特性

- **全局内存**：容量、带宽、类型和通道数
- **L2缓存**：容量、行大小、关联度和带宽
- **L1缓存**：每个SM的容量、行大小和关联度
- **共享内存**：每个SM的容量、bank数和bank宽度
- **内存层次**：不同层级间的带度比率

### 3. 芯片通信模块配置

芯片通信模块管理GPU内部和外部的通信，配置参数包括：

```json
{
  "interconnect": {
    "nvlink": {
      "version": "3.0",
      "links": 12,
      "bandwidth_gbps": {
        "unidirectional": 50,
        "bidirectional": 100
      },
      "protocols": ["hs", "ls"]
    },
    "pcie": {
      "version": "5.0",
      "lanes": 16,
      "bandwidth_gbps": {
        "unidirectional": 32,
        "bidirectional": 64
      }
    },
    "on_chip_interconnect": {
      "type": "crossbar",
      "bandwidth_gbps": 2000,
      "latency_ns": 1,
      "endpoints": [
        "sm",
        "l2_cache",
        "memory_controllers",
        "nvlink_controllers",
        "pcie_controller"
      ]
    },
    "fabric_interconnect": {
      "name": "GFC",
      "bandwidth_gbps": 5000,
      "latency_ns": 5,
      "max_connections": 256
    }
  }
}
```

#### 3.1 芯片通信模块特性

- **NVLink**：版本、链路数、带宽和协议
- **PCIe**：版本、通道数和带宽
- **片上互连**：类型、带宽、延迟和端点
- **片间互连**：带宽、延迟和最大连接数

## 执行流程

### 1. 初始化阶段

1. **GPUContext初始化**：从配置文件加载GPU硬件参数
2. **SM创建**：根据配置创建指定数量的SMContext实例
3. **资源分配**：为每个SM分配相应的硬件资源

### 2. 执行阶段

1. **Kernel启动**：GPUContext接收kernel启动请求
2. **资源分配**：将CTA分配到可用的SM
3. **warp调度**：SMContext调度warp执行
4. **指令执行**：按SIMT模型执行指令

### 3. 状态管理

- **全局状态**：GPUContext管理整个GPU的执行状态
- **SM状态**：每个SMContext管理自身的执行状态
- **同步机制**：处理跨SM和跨warp的同步

## 内存层次结构

### 1. 全局内存

- **模拟特性**：带宽限制、延迟模型
- **容量**：可配置的内存容量

### 2. L2缓存

- **模拟特性**：缓存命中/未命中、替换策略
- **容量**：根据GPU架构配置

### 3. L1缓存/共享内存

- **可配置**：L1缓存和共享内存的容量分配
- **bank冲突检测**：检测共享内存bank冲突

### 4. 寄存器文件

- **容量限制**：每个SM的寄存器数量限制
- **分配策略**：为线程分配寄存器资源

## 扩展性设计

### 1. 配置驱动

- **JSON配置**：通过JSON文件配置GPU参数
- **架构支持**：轻松添加新GPU架构支持

### 2. 模块化设计

- **独立组件**：各硬件模块独立设计，便于扩展
- **接口抽象**：定义清晰的接口，支持不同实现

### 3. 可插拔组件

- **调度器**：可替换的warp调度策略
- **内存模型**：可替换的内存访问模型
- **执行单元**：可扩展的执行单元类型

## 性能考虑

### 1. 模拟开销

- **批处理**：减少模拟器本身的开销
- **缓存优化**：优化数据结构以提高缓存命中率

### 2. 并行化

- **多线程模拟**：利用多核CPU并行模拟多个SM
- **流水线执行**：实现指令级流水线

### 3. 内存效率

- **紧凑数据结构**：减少内存占用
- **对象池**：重用对象以减少分配开销