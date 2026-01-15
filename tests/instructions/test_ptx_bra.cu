#include <cuda.h>
#include <iostream>
#include <cassert>

__global__ void test_bra_kernel(int *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;

    int limit = tid + 1;
    int counter = 0;
    int predicate = 0;

    // 使用内联PTX测试BRA指令
    asm volatile (
        "{                                        \n\t"
        "    mov.s32 %0, 0;                       \n\t"  // 初始化计数器为0
        "$loop_start%=:                            \n\t"
        "    add.s32 %0, %0, 1;                   \n\t"  // 计数器加1
        "    setp.lt.u32 %%p1, %0, %2;            \n\t"  // 检查counter < limit
        "    @%%p1 bra.uni $loop_start%=;          \n\t"  // 如果为真，则跳转回循环开始
        "    selp.s32 %1, 0, 1, %%p1;             \n\t"  // 将p1的值保存到predicate变量中
        "}                                        \n\t"
        : "=&r"(counter), "=&r"(predicate)          // 输出
        : "r"(limit)
        : "memory"                      // 告诉编译器内存和pred寄存器可能被修改
    );

    result[tid] = counter;
}

int main() {
    const int N = 4;
    int *h_data = new int[N];
    int *d_data;

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = 0;
    }

    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(4);
    dim3 grid(1);
    test_bra_kernel<<<grid, block>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (h_data[i] != i + 1) {  // 每个线程应该循环(thread_id + 1)次
            std::cout << "FAIL: at:" << i << " expect:" << (i+1) << " got:" << h_data[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "PASS: BRA instruction test passed!" << std::endl;
    }

    cudaFree(d_data);
    delete[] h_data;

    return passed ? 0 : 1;
}