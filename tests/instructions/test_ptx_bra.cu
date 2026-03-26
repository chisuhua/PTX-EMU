#include <cuda.h>
#include <iostream>

// 测试2：添加早期退出分支
__global__ void test_bra_kernel(int *result, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 早期退出 - 这会被编译为 bra 指令
    if (tid >= n) return;

    // 存储 tid + 1
    result[tid] = tid + 1;
}

int main() {
    const int N = 4;
    int *h_data = new int[N];
    int *d_data;

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_data[i] = -1;
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
        int expected = i + 1;
        if (h_data[i] != expected) {
            std::cout << "FAIL: at:" << i << " expect:" << expected << " got:" << h_data[i] << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "PASS: Branch test with early exit passed!" << std::endl;
    }

    cudaFree(d_data);
    delete[] h_data;

    return passed ? 0 : 1;
}
