#include <cstdio>
__shared__ int shared_data[32];

__global__ void test_store() {
    int tid = threadIdx.x;
    shared_data[tid] = tid * tid + 1;
    __syncthreads();
    
    if (tid == 0) {
        int sum = 0;
        for (int i = 0; i < 32; i++) {
            sum += shared_data[i];
        }
        printf("Sum: %d\n", sum);
    }
}

int main() {
    test_store<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
