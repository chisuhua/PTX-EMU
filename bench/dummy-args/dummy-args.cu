#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define SIZE 64

template<typename T>
__global__ void dummy_args_d(T *a, 
                             int scalar_int, 
                             float scalar_float, 
                             double scalar_double,
                             bool scalar_bool,
                             int *int_array,
                             float *float_array) {
    int i = threadIdx.x;
    
    // 只让一个线程工作，将所有参数写入数组
        int index = 0;
        // 写入数组初始值
        a[index++] = i;
        
        // 写入各种标量参数值
        a[index++] = scalar_int;
        a[index++] = (int)scalar_float;
        a[index++] = (int)scalar_double;
        a[index++] = scalar_bool ? 1 : 0;
        
        // 写入数组参数的前几个值
        for (int j = 0; j < 5 && j < SIZE/2; j++) {
            a[index++] = int_array[j];
        }
        
        for (int j = 0; j < 5 && j < SIZE/2; j++) {
            a[index++] = (int)float_array[j];
        }
}

int main(int argc, char* argv[]) {
    int N = SIZE;
    int *host_a = (int*)malloc(sizeof(int)*N);
    int *host_int_array = (int*)malloc(sizeof(int)*N/2);
    float *host_float_array = (float*)malloc(sizeof(float)*N/2);
    
    // 初始化数组
    for (int i = 0; i < N; i++) {
        host_a[i] = 0;
    }
    
    for (int i = 0; i < N/2; i++) {
        host_int_array[i] = i + 10;
        host_float_array[i] = (float)(i + 20);
    }
    
    int *device_a;
    int *device_int_array;
    float *device_float_array;
    
    cudaMalloc((void**)&device_a, sizeof(int)*N);
    cudaMalloc((void**)&device_int_array, sizeof(int)*N/2);
    cudaMalloc((void**)&device_float_array, sizeof(float)*N/2);

    cudaMemcpy(device_a, host_a, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_int_array, host_int_array, sizeof(int)*N/2, cudaMemcpyHostToDevice);
    cudaMemcpy(device_float_array, host_float_array, sizeof(float)*N/2, cudaMemcpyHostToDevice);

    dim3 grid(1);
    dim3 block(1);  // 只启动一个线程

    // 调用内核，传递多种类型的参数
    dummy_args_d<<<grid, block>>>(device_a, 
                                  1000,      // scalar int
                                  2000.5f,   // scalar float
                                  3000.7,    // scalar double
                                  true,      // scalar bool
                                  device_int_array,
                                  device_float_array);

    cudaMemcpy(host_a, device_a, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // 验证结果 - 只检查第一个线程写入的值
    bool success = true;
    int expected_values[] = {0, 1000, 2000, 3000, 1, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24};
    int num_expected = sizeof(expected_values) / sizeof(expected_values[0]);
    
    for (int i = 0; i < num_expected; i++) {
        if (host_a[i] != expected_values[i]) {
            printf("at:%d expect:%d got:%d\n", i, expected_values[i], host_a[i]);
            success = false;
        }
    }

    if (success)
        printf("PASS\n");
    else
        printf("FAIL\n");

    cudaFree(device_a);
    cudaFree(device_int_array);
    cudaFree(device_float_array);
    free(host_a);
    free(host_int_array);
    free(host_float_array);
    return 0;
}