// step2_tiled_copy_fixed.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include <cute/tensor.hpp>
using namespace cute;

using half = __half;

__global__ void tiled_copy_kernel(half const* g_input, half* g_output) {
    // Only use the first 4 threads
    if (threadIdx.x >= 4) return;

    // Global input: 4x4 row-major
    auto gLayout = make_layout(make_shape(_4{}, _4{}), make_stride(_4{}, _1{}));
    auto gTensor = make_tensor(const_cast<half*>(g_input), gLayout);

    // TiledCopy: 4 threads, each loads 1x4
    auto cpy_atom = Copy_Atom<DefaultCopy, half>{};
    auto tiled_copy = make_tiled_copy(
        cpy_atom,
        Layout<_4, _1>{} ,   // Thread layout (4 threads in M)
        Layout<_1, _4>{}     // Value layout per thread (1x4)
    );

    // Get this thread's slice of the global tensor
    auto thr_slice = tiled_copy.get_thread_slice(threadIdx.x);
    auto g_slice = thr_slice.partition_S(gTensor);

    // Register fragment: just a raw array
    half frag[4];

    // Perform copy: GMEM -> Register
    {
        // Use default row-major layout for (1,4) — automatically gives stride (_4{}, _1{})
        auto rLayout = make_layout(make_shape(_1{}, _4{}), make_stride(_4{}, _1{}));
        // auto rLayout = make_layout(make_shape(_1{}, _4{}), make_stride(_1{}, _1{}));
        auto rTensor = make_tensor(&frag[0], rLayout);
        //auto rTensor = make_tensor(&frag[0], Layout<_1, _4>{});
        auto r_slice = thr_slice.partition_D(rTensor);
        copy(tiled_copy, g_slice, r_slice);
    }

    // Write back to global memory
    for (int j = 0; j < 4; ++j) {
        g_output[threadIdx.x * 4 + j] = frag[j];
    }
}

// Host test (same as before)
bool test_tiled_copy() {
    constexpr int total = 16;
    std::vector<half> h_input(total);
    for (int i = 0; i < total; ++i) {
        h_input[i] = __float2half(static_cast<float>(i));
    }

    half *d_input, *d_output;
    cudaMalloc(&d_input,  total * sizeof(half));
    cudaMalloc(&d_output, total * sizeof(half));
    cudaMemcpy(d_input, h_input.data(), total * sizeof(half), cudaMemcpyHostToDevice);

    tiled_copy_kernel<<<1, 4>>>(d_input, d_output);
    cudaDeviceSynchronize();

    std::vector<half> h_output(total);
    cudaMemcpy(h_output.data(), d_output, total * sizeof(half), cudaMemcpyDeviceToHost);

    bool passed = true;
    for (int i = 0; i < total; ++i) {
        float out = __half2float(h_output[i]);
        float exp = __half2float(h_input[i]);
        if (out != exp) {
            std::cerr << "Mismatch at [" << i << "]: " << out << " vs " << exp << "\n";
            passed = false;
        }
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return passed;
}

int main() {
    if (test_tiled_copy()) {
        std::cout << "✅ Step 2: TiledCopy test PASSED!\n";
    } else {
        std::cout << "❌ FAILED!\n";
        return 1;
    }
    return 0;
}