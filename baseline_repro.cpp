// Temporary baseline reproduction launcher for fix-path2d-ptxir-execution-bugs.
// NOT a Phase 5 deliverable. Used in Task 1.3 (Phase 1 baseline) to exercise path_2D.
//
// Build: cd build && g++ -std=c++20 -I${CUDA_PATH}/include -I../include \
//                    ../baseline_repro.cpp -L./lib -lcudart -lptxemu_device \
//                    -Wl,-rpath,./lib -o baseline_repro

#include <cudart/cpptlm_module.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <pathxir_or_embedded_binary>\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "FAIL: cannot open %s\n", argv[1]); return 2; }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});

    std::fprintf(stderr, "[baseline_repro] loading %s (%zu bytes)\n", argv[1], bytes.size());
    auto h = ptxemu_image_load(bytes.data(), bytes.size());
    if (h == 0) { std::fprintf(stderr, "FAIL: ptxemu_image_load returned 0\n"); return 3; }
    std::fprintf(stderr, "[baseline_repro] load OK handle=%llu\n", (unsigned long long)h);

    int v = ptxemu_module_version();
    std::fprintf(stderr, "[baseline_repro] module version=%d\n", v);

    int kc = ptxemu_image_kernel_count(h);
    std::fprintf(stderr, "[baseline_repro] kernel_count=%d\n", kc);
    if (kc < 1) {
        std::fprintf(stderr, "FAIL: kernel_count=%d (no kernels)\n", kc);
        ptxemu_image_unload(h);
        return 4;
    }

    char name[256] = {0};
    int nr = ptxemu_image_kernel_name_at(h, 0, name, sizeof(name));
    std::fprintf(stderr, "[baseline_repro] kernel_name[0]=%d '%s'\n", nr, name);

    // Allocate 3 device buffers (A, B, C) + numElements via fake cudaMalloc
    void* A = nullptr; void* B = nullptr; void* C = nullptr;
    size_t N = 16;
    if (cudaMalloc(&A, N * sizeof(float)) != 0) { std::fprintf(stderr, "FAIL: cudaMalloc A\n"); return 5; }
    if (cudaMalloc(&B, N * sizeof(float)) != 0) { std::fprintf(stderr, "FAIL: cudaMalloc B\n"); return 5; }
    if (cudaMalloc(&C, N * sizeof(float)) != 0) { std::fprintf(stderr, "FAIL: cudaMalloc C\n"); return 5; }

    // Init A=1, B=2
    std::vector<float> a(N, 1.0f), b(N, 2.0f);
    cudaMemcpy(A, a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    void* args[] = {&A, &B, &C, &N};
    int rc = ptxemu_image_execute_named(h, name,
                                         /*grid*/1, 1, 1,
                                         /*block*/N, 1, 1,
                                         /*shm*/0, args, 4);
    std::fprintf(stderr, "[baseline_repro] execute_named rc=%d\n", rc);

    if (rc == 0) {
        std::vector<float> c(N);
        cudaMemcpy(c.data(), C, N * sizeof(float), cudaMemcpyDeviceToHost);
        std::fprintf(stderr, "[baseline_repro] C[0..3]=%.3f %.3f %.3f %.3f (expect 3.0)\n",
                     c[0], c[1], c[2], c[3]);
    }

    ptxemu_image_unload(h);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return rc == 0 ? 0 : 6;
}