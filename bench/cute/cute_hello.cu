// cute_hello.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Only need this one header for basic CUTE
#include <cute/tensor.hpp>
using namespace cute;

using half = __half;

__global__ void cute_hello_kernel(half const* gptr, float* out) {
    // === Step 1: Define a 4x4 row-major layout ===
    // Shape: (4, 4)
    // Stride: (4, 1) → row-major
    auto layout = make_layout(make_shape(_4{}, _4{}), make_stride(_4{}, _1{}));
    
    // Create a tensor view over global memory
    auto tensor = make_tensor(gptr, layout);

    // === Step 2: Access elements using logical coordinates ===
    // This is equivalent to gptr[i * 4 + j]
    half a00 = tensor(0, 0); // top-left
    half a12 = tensor(1, 2); // row=1, col=2 → index = 1*4+2 = 6

    // === Step 3: Create a transposed VIEW (no data copy!) ===
    // New layout: same shape, but stride (1, 4) → column-major interpretation
    auto transposed_layout = make_layout(make_shape(_4{}, _4{}), make_stride(_1{}, _4{}));
    auto tview = make_tensor(gptr, transposed_layout);

    // Now tview(i,j) == original(j,i)
    half t01 = tview(0, 1); // should equal original(1,0)

    // Print from thread 0
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      // 写入几个值到 global memory，防止优化
        out[0] = __half2float(a00);
        out[1] = __half2float(a12);
        out[2] = __half2float(t01);
    //     printf("Original [0][0] = %.1f\n", __half2float(a00));
    //     printf("Original [1][2] = %.1f\n", __half2float(a12));
    //     printf("Transposed view [0][1] = %.1f (should equal Original[1][0] = %.1f)\n",
    //            __half2float(t01), __half2float(tensor(1, 0)));
    }
}

int main() {
    constexpr int N = 16;
    std::vector<half> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = __float2half(static_cast<float>(i * 1.5f)); // 0.0, 1.5, 3.0, ...
    }

    half* d_data;
    float* d_out;
    cudaMalloc(&d_data, N * sizeof(half));
    cudaMalloc(&d_out, 3 * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    cute_hello_kernel<<<1, 32>>>(d_data, d_out);
    cudaDeviceSynchronize();

    float h_out[3];
    cudaMemcpy(h_out, d_out, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Result: [" << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << "]\n";

    cudaFree(d_data);
    return 0;
}