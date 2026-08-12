/**
 * @file test_divergence_sync.cpp
 * @brief Warp divergence with barrier sync 的 Catch2 单元测试
 *
 * 测试分歧路径 + __syncthreads() 屏障同步:
 * - Lane 0-15:   计算 sum 0..lane (0, 1, 3, 6, 10, ...)
 * - Lane 16-31:  计算 product 1..(lane-15) (1, 2, 6, 24, 120, ...)
 * - __syncthreads(): warp 级屏障同步
 * - Lane 交换:  每个 lane 读取其他 lane 写入的值
 */
#include "catch_amalgamated.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

// 计算单个 lane 的期望值
static int expected_lane_value(int lane) {
    if (lane < 16) {
        int value = 0;
        for (int i = 0; i <= lane; i++) {
            value += i;
        }
        return value;
    } else {
        int value = 1;
        for (int i = 1; i <= lane - 15; i++) {
            value *= i;
        }
        return value;
    }
}

// ====================================================================
// Kernel: 分歧 + 屏障 + Lane 间数据交换
//
// 行为:
//   Lane 写入: shared_data[lane] = 自己计算的值
//   __syncthreads()
//   Lane 读取: output[tid] = shared_data[32 - lane - 1]
//              (lane 0 读 shared_data[31], lane 31 读 shared_data[0])
// ====================================================================
template<typename T>
__global__ void test_divergence_sync_kernel(T* output) {
    __shared__ T shared_data[32];

    int tid = threadIdx.x;
    int lane = tid % 32;

    T value;
    if (lane < 16) {
        value = 0;
        for (int i = 0; i <= lane; i++) {
            value += i;
        }
    } else {
        value = 1;
        for (int i = 1; i <= lane - 15; i++) {
            value *= i;
        }
    }

    shared_data[lane] = value;
    __syncthreads();

    // Lane 间交换: lane i 读取 shared_data[31-i]
    output[tid] = shared_data[31 - lane];
}

// ====================================================================
// 测试用例
// ====================================================================

TEST_CASE("Divergence: barrier sync with lane data exchange",
          "[divergence][barrier_sync]") {

    int* dev_buf = nullptr;
    cudaError_t err;

    err = cudaMalloc(&dev_buf, 32 * sizeof(int));
    REQUIRE(err == cudaSuccess);

    test_divergence_sync_kernel<int><<<1, 32>>>(dev_buf);

    err = cudaGetLastError();
    REQUIRE(err == cudaSuccess);

    err = cudaDeviceSynchronize();
    REQUIRE(err == cudaSuccess);

    int h_output[32] = {0};
    err = cudaMemcpy(h_output, dev_buf, 32 * sizeof(int),
                     cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);

    cudaFree(dev_buf);

    // 验证: lane i 读取的是 lane (31-i) 计算的值
    for (int lane = 0; lane < 32; lane++) {
        int expected = expected_lane_value(31 - lane);
        INFO("lane " << lane << ": output = " << h_output[lane]
                     << ", expected = " << expected);
    }
    for (int lane = 0; lane < 32; lane++) {
        int expected = expected_lane_value(31 - lane);
        INFO("lane " << lane << ": output = " << h_output[lane]
                     << ", expected = " << expected);
        REQUIRE(h_output[lane] == expected);
    }

    std::cout << "  PASS: lane exchange after barrier sync" << std::endl;
}
