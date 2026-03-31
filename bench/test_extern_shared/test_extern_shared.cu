/**
 * @file test_extern_shared.cu
 * @brief 测试 extern __shared__ 动态共享内存解析
 * @author Agent
 * @date 2026-04-01
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cassert>

// 最小化 extern shared 测试
__global__ void test_extern_shared_kernel(int *output) {
    extern __shared__ int shared_data[];  // 动态共享内存
    int tid = threadIdx.x;
    
    // 写入
    shared_data[tid] = tid * 2;
    
    __syncthreads();
    
    // 读取
    output[tid] = shared_data[tid];
}

bool run_test() {
    printf("test_extern_shared: ");
    const int size = 4;
    int *d_output;
    cudaError_t err;
    
    err = cudaMalloc(&d_output, size * sizeof(int));
    assert(err == cudaSuccess);
    
    // 启动内核，指定动态共享内存大小
    test_extern_shared_kernel<<<1, size, size * sizeof(int)>>>(d_output);
    
    int h_output[size];
    err = cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    cudaFree(d_output);
    
    // 验证结果
    for (int i = 0; i < size; i++) {
        int expected = i * 2;
        if (h_output[i] != expected) {
            printf("FAIL at %d: expected %d, got %d\n", i, expected, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

int main() {
    printf("=== Extern Shared Memory Test ===\n\n");
    bool result = run_test();
    printf("\n===========================\n");
    return result ? 0 : 1;
}
