// 测试 extern __shared__ 动态共享内存解析
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void test_extern_shared_kernel(int *output) {
    extern __shared__ int shared_data[];
    int tid = threadIdx.x;
    shared_data[tid] = tid * 2;
    __syncthreads();
    output[tid] = shared_data[tid];
}

bool test_extern_shared() {
    printf("test_extern_shared: ");
    const int size = 4;
    int *d_output;
    cudaMalloc(&d_output, size * sizeof(int));
    test_extern_shared_kernel<<<1, size, size * sizeof(int)>>>(d_output);
    int h_output[size];
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    for (int i = 0; i < size; i++) {
        if (h_output[i] != i * 2) {
            printf("FAIL at %d: expected %d, got %d\n", i, i*2, h_output[i]);
            return false;
        }
    }
    printf("PASS\n");
    return true;
}

int main() {
    printf("=== Extern Shared Memory Test ===\n");
    return test_extern_shared() ? 0 : 1;
}
