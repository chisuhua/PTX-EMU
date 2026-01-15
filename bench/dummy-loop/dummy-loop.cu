#include <stdio.h>
#include <stdlib.h>

#define SIZE 64

template<typename T>
__global__ void dummy_d(T *a) {
    int i = threadIdx.x;
    
    // 使用 for 循环初始化值，每次迭代将值增加 i
    a[i] = 0;
    for(int j = 0; j <= i; j++) {
        a[i] += j;
    }
}

int main(int argc, char* argv[]) {
    int N = SIZE;
    int *host_a = (int*)malloc(sizeof(int)*N);
    
    // 初始化数组
    for (int i = 0; i < N; i++) {
        host_a[i] = 0;
    }
    
    int *device_a;
    cudaMalloc((void**)&device_a, sizeof(int)*N);

    cudaMemcpy(device_a, host_a, sizeof(int)*N, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(N);

    dummy_d<<<grid, block>>>(device_a);

    cudaMemcpy(host_a, device_a, sizeof(int)*N, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; i++) {
        // 预期结果是 0+1+2+...+i = i*(i+1)/2
        int expected = i * (i + 1) / 2;
        if (host_a[i] != expected) {
            printf("at:%p expect:%d got:%d\n", &host_a[i], expected, host_a[i]);
            success = false;
            break;
        }
    }

    if (success)
        printf("PASS\n");
    else
        printf("FAIL\n");

    cudaFree(device_a);
    free(host_a);
    return 0;
}