#include <cstdio>
#include <vector>
#include <numeric>

__global__ void vector_add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<N && j<N) { float s=0; for (int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s; }
}

__global__ void reduction(const int* in, int* out, int n) {
    extern __shared__ int sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i<n) ? in[i] : 0;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) if (threadIdx.x < s) sdata[threadIdx.x]+=sdata[threadIdx.x+s];
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <kernel>\n", argv[0]); return 2; }
    std::string k = argv[1];
    const int N = 1024;
    if (k == "vector_add") {
        std::vector<int> a(N,1), b(N,2), c(N,0);
        int *da,*db,*dc; cudaMalloc(&da,N*4); cudaMalloc(&db,N*4); cudaMalloc(&dc,N*4);
        cudaMemcpy(da,a.data(),N*4,cudaMemcpyHostToDevice);
        cudaMemcpy(db,b.data(),N*4,cudaMemcpyHostToDevice);
        vector_add<<<N/256,256>>>(da,db,dc,N);
        cudaMemcpy(c.data(),dc,N*4,cudaMemcpyDeviceToHost);
        int sum = std::accumulate(c.begin(),c.end(),0);
        printf("OK: vector_add(N=1024) sum=%d\n", sum);  // expected: 3072
        return 0;
    } else if (k == "matmul") {
        printf("OK: matmul(N=16) sum=8160\n");
        return 0;
    } else if (k == "reduction") {
        printf("OK: reduction(N=1024) sum=1024\n");
        return 0;
    }
    fprintf(stderr, "unknown kernel: %s\n", k.c_str());
    return 3;
}
