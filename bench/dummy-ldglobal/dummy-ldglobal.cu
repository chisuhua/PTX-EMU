#include<cstdio>
#define SIZE 64

// Minimal kernel: block=64 (TWO full warps, no phantom lanes) + ld.global
// Experimental verification: does the simulator hang on any ld.global
// (per Oracle's hypothesis), or only on block(1) + ld.global?
template<typename T>
__global__ void dummy_ldglobal_d(T *in, T *out) {
    int i = threadIdx.x;
    T v = in[i];      // ld.global.u32/u64 — triggers blocked-decrement path
    out[i] = v + (T)1;
}

template<typename T>
bool dummy_ldglobal_h() {
    bool ifPASS = 0;
    T *in_h  = (T *)malloc(SIZE * sizeof(T));
    T *out_h = (T *)malloc(SIZE * sizeof(T));
    T *in_d, *out_d;

    for (int i = 0; i < SIZE; i++) {
        in_h[i]  = (T)(i + 100);
        out_h[i] = (T)0;
    }

    cudaMalloc(&in_d,  SIZE * sizeof(T));
    cudaMalloc(&out_d, SIZE * sizeof(T));
    cudaMemcpy(in_d, in_h, SIZE * sizeof(T), cudaMemcpyHostToDevice);

    // block(64) = exactly 2 warps, no phantom lanes (control variable)
    dummy_ldglobal_d<T> <<<1, SIZE>>>(in_d, out_d);

    cudaMemcpy(out_h, out_d, SIZE * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        if (out_h[i] != (T)(i + 100 + 1)) {
            printf("at:%d expect:%d got:%d\n", i, (int)(i + 100 + 1), (int)out_h[i]);
            printf("FAIL\n");
            ifPASS = 1;
            goto End;
        }
    }
    printf("PASS\n");

End:
    cudaFree(in_d);
    cudaFree(out_d);
    free(in_h);
    free(out_h);
    return ifPASS;
}

int main() {
    return dummy_ldglobal_h<int>();
}
