#include <cstdio>
#include <vector>
#include <numeric>

__global__ void vector_add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    std::vector<int> a(N,1), b(N,2), c(N,0);
    int *da,*db,*dc; cudaMalloc(&da,N*4); cudaMalloc(&db,N*4); cudaMalloc(&dc,N*4);
    cudaMemcpy(da,a.data(),N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),N*4,cudaMemcpyHostToDevice);
    vector_add<<<N/256,256>>>(da,db,dc,N);
    cudaMemcpy(c.data(),dc,N*4,cudaMemcpyDeviceToHost);
    int sum = std::accumulate(c.begin(),c.end(),0);
    printf("OK: vector_add(N=1024) sum=%d\n", sum);  // 3072
    return 0;
}
