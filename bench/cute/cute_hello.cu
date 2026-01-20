// cute_hello.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

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
        out[3] = __half2float(tensor(1,0));
    }
}

// Helper: get float bits as uint32_t
uint32_t float_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    return bits;
}

int main() {
    constexpr int N = 16;
    std::vector<half> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = __float2half(static_cast<float>(i * 1.5f)); // 0.0, 1.5, 3.0, ...
    }
    // 🔍 Print the original 4x4 Tile on host (before copying to GPU)
    std::cout << "Original 4x4 Tile (row-major) on host:\n";
    std::cout << "Format: value | half(16b) | float(32b)\n\n";

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            half h = h_data[i * 4 + j];
            float f = __half2float(h);
            unsigned short half_bits = __half_as_ushort(h);
            uint32_t float_bits = float_to_bits(f);

            // Save current flags
            std::ios_base::fmtflags f_flags(std::cout.flags());

            std::cout << std::fixed << std::setprecision(1) << std::setw(6) << f
                      << " | 0x" << std::hex << std::uppercase << std::setfill('0') << std::setw(4) << half_bits
                      << " | 0x" << std::setw(8) << float_bits
                      << "  ";

            // Restore flags (back to dec, etc.)
            std::cout.flags(f_flags);
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    half* d_data;
    float* d_out;
    cudaMalloc(&d_data, N * sizeof(half));
    cudaMalloc(&d_out, 4 * sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(half), cudaMemcpyHostToDevice);

    cute_hello_kernel<<<1, 32>>>(d_data, d_out);
    cudaDeviceSynchronize();

    // Optional: check for kernel errors
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
    // }

    float h_out[4];
    cudaMemcpy(h_out, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i) {
      float f = h_out[i];
      uint32_t float_bits = float_to_bits(f);

      std::cout << std::setw(6) << std::fixed << std::setprecision(1) << f
                      << " | 0x" << std::setw(8) << float_bits << "  ";
      std::cout << "\n";
    }


    cudaFree(d_data);
    return 0;
}