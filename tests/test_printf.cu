#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_kernel(int arg) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 测试基本printf功能
    printf("Hello from thread %d, arg %d\n", tid, arg);

    // 测试不同类型的数据
    int intVal = tid * 2;
    float floatVal = tid * 3.14f;
    double doubleVal = tid * 2.718281828;

    printf("Thread %d: int=%d, float=%.2f, double=%.3f\n",
           tid, intVal, floatVal, doubleVal);

    // 测试字符串
    printf("String test: %s\n", "Hello World");

    // 测试十六进制和字符
    printf("Hex: %x, Char: %c\n", tid, 'A' + (tid % 26));
}

int main() {
    printf("Starting printf test...\n");

    // Launch kernel with 1 blocks, 1 threads each
    test_kernel<<<1, 1>>>(9);

    // Synchronize to ensure all threads finish printing
    cudaDeviceSynchronize();

    printf("Printf test completed.\n");

    return 0;
}