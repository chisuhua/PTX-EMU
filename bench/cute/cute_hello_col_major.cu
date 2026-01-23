// cute_col_major_demo.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include <cute/tensor.hpp>
using namespace cute;

using half = __half;

__global__ void test_col_major_view(half const* gptr, float* out) {
    // === 1. 行优先（row-major）布局：标准 C 风格 ===
    auto row_major_layout = make_layout(make_shape(_4{}, _4{}), make_stride(_4{}, _1{}));
    auto row_major_tensor = make_tensor(gptr, row_major_layout);

    // === 2. 列优先（col-major）布局：Fortran 风格 ===
    auto col_major_layout = make_layout(make_shape(_4{}, _4{}), make_stride(_1{}, _4{}));
    auto col_major_tensor = make_tensor(gptr, col_major_layout);

    // === 3. 验证：row_major(0,1) 应等于 col_major(1,0) ===
    half val1 = row_major_tensor(0, 1);   // logical [row=0, col=1]
    half val2 = col_major_tensor(1, 0);   // logical [row=1, col=0] in col-major view

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 将结果写入全局内存，供host端读取
        out[0] = __half2float(val1);  // row_major(0,1) 的值
        out[1] = __half2float(val2);  // col_major(1,0) 的值
    }
}

int main() {
    constexpr int N = 16;
    std::vector<half> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = __float2half(static_cast<float>(i * 10)); // 0, 10, 20, ..., 150
    }

    // 打印原始 4x4 矩阵（行优先）
    std::cout << "Original 4x4 matrix (row-major):\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << __half2float(h_data[i * 4 + j]) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    half* d_data;
    float* d_out;
    cudaMalloc(&d_data, N * sizeof(half));
    cudaMalloc(&d_out, 2 * sizeof(float));  // 为两个结果值分配空间
    cudaMemcpy(d_data, h_data.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    test_col_major_view<<<1, 32>>>(d_data, d_out);
    cudaDeviceSynchronize();

    // 从GPU读取结果
    float h_results[2];
    cudaMemcpy(h_results, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Host端打印和验证结果
    float row_major_val = h_results[0];   // row_major(0,1) 的值
    float col_major_val = h_results[1];   // col_major(1,0) 的值
    
    std::cout << "Results from kernel:\n";
    std::cout << "row_major(0,1) = " << row_major_val << std::endl;
    std::cout << "col_major_view(1,0) = " << col_major_val << std::endl;

    if (row_major_val == col_major_val) {
        std::cout << "✅ Verification PASSED: They are equal!\n";
    } else {
        std::cout << "❌ Verification FAILED!\n";
    }

    cudaFree(d_data);
    cudaFree(d_out);
    return 0;
}