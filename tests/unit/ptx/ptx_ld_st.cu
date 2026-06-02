#include "ptx_ld_st.cuh"

// --- Kernels for LD operations ---

__global__ void ld_u8_kernel(uint8_t* input, uint8_t* output) {
    ptx_ld_u8(output, input);
}

__global__ void ld_u16_kernel(uint16_t* input, uint16_t* output) {
    ptx_ld_u16(output, input);
}

__global__ void ld_u32_kernel(uint32_t* input, uint32_t* output) {
    ptx_ld_u32(output, input);
}

__global__ void ld_u64_kernel(uint64_t* input, uint64_t* output) {
    ptx_ld_u64(output, input);
}

__global__ void ld_f32_kernel(float* input, float* output) {
    ptx_ld_f32(output, input);
}

__global__ void ld_f64_kernel(double* input, double* output) {
    ptx_ld_f64(output, input);
}

// --- Kernels for ST operations ---

__global__ void st_u8_kernel(uint8_t* output, uint8_t value) {
    ptx_st_u8(output, value);
}

__global__ void st_u16_kernel(uint16_t* output, uint16_t value) {
    ptx_st_u16(output, value);
}

__global__ void st_u32_kernel(uint32_t* output, uint32_t value) {
    ptx_st_u32(output, value);
}

__global__ void st_u64_kernel(uint64_t* output, uint64_t value) {
    ptx_st_u64(output, value);
}

__global__ void st_f32_kernel(float* output, float value) {
    ptx_st_f32(output, value);
}

__global__ void st_f64_kernel(double* output, double value) {
    ptx_st_f64(output, value);
}

// --- Test functions for LD operations ---

void test_ptx_ld_u8(uint8_t value, uint8_t* result) {
    uint8_t* d_data;
    cudaMalloc(&d_data, sizeof(uint8_t));
    cudaMemcpy(d_data, &value, sizeof(uint8_t), cudaMemcpyHostToDevice);

    uint8_t* d_result;
    cudaMalloc(&d_result, sizeof(uint8_t));

    ld_u8_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
}

void test_ptx_ld_u16(uint16_t value, uint16_t* result) {
    uint16_t* d_data;
    cudaMalloc(&d_data, sizeof(uint16_t));
    cudaMemcpy(d_data, &value, sizeof(uint16_t), cudaMemcpyHostToDevice);

    uint16_t* d_result;
    cudaMalloc(&d_result, sizeof(uint16_t));

    ld_u16_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
}

void test_ptx_ld_u32(uint32_t value, uint32_t* result) {
    uint32_t* d_data;
    cudaMalloc(&d_data, sizeof(uint32_t));
    cudaMemcpy(d_data, &value, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t* d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));

    ld_u32_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
}

void test_ptx_ld_u64(uint64_t value, uint64_t* result) {
    uint64_t* d_data;
    cudaMalloc(&d_data, sizeof(uint64_t));
    cudaMemcpy(d_data, &value, sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t* d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));

    ld_u64_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
}

void test_ptx_ld_f32(float value, float* result) {
    float* d_data;
    cudaMalloc(&d_data, sizeof(float));
    cudaMemcpy(d_data, &value, sizeof(float), cudaMemcpyHostToDevice);

    float* d_result;
    cudaMalloc(&d_result, sizeof(float));

    ld_f32_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
}

void test_ptx_ld_f64(double value, double* result) {
    double* d_data;
    cudaMalloc(&d_data, sizeof(double));
    cudaMemcpy(d_data, &value, sizeof(double), cudaMemcpyHostToDevice);

    double* d_result;
    cudaMalloc(&d_result, sizeof(double));

    ld_f64_kernel<<<1, 1>>>(d_data, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
}

// --- Test functions for ST operations ---

void test_ptx_st_u8(uint8_t* addr, uint8_t value, uint8_t* result) {
    uint8_t* d_data;
    cudaMalloc(&d_data, sizeof(uint8_t));

    st_u8_kernel<<<1, 1>>>(d_data, value);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_data, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void test_ptx_st_u16(uint16_t* addr, uint16_t value, uint16_t* result) {
    uint16_t* d_data;
    cudaMalloc(&d_data, sizeof(uint16_t));

    st_u16_kernel<<<1, 1>>>(d_data, value);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_data, sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void test_ptx_st_u32(uint32_t* addr, uint32_t value, uint32_t* result) {
    uint32_t* d_data;
    cudaMalloc(&d_data, sizeof(uint32_t));

    st_u32_kernel<<<1, 1>>>(d_data, value);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_data, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void test_ptx_st_u64(uint64_t* addr, uint64_t value, uint64_t* result) {
    uint64_t* d_data;
    cudaMalloc(&d_data, sizeof(uint64_t));

    st_u64_kernel<<<1, 1>>>(d_data, value);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_data, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void test_ptx_st_f32(float* addr, float value, float* result) {
    float* d_data;
    cudaMalloc(&d_data, sizeof(float));

    st_f32_kernel<<<1, 1>>>(d_data, value);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

void test_ptx_st_f64(double* addr, double value, double* result) {
    double* d_data;
    cudaMalloc(&d_data, sizeof(double));

    st_f64_kernel<<<1, 1>>>(d_data, value);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_data, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

// --- Helper functions for shared memory tests ---

__global__ void ld_shared_kernel(uint32_t* input, uint32_t* output) {
    __shared__ uint32_t shared_data[32];
    
    int tid = threadIdx.x;
    shared_data[tid] = input[tid];
    __syncthreads();
    
    *output = shared_data[0];
}

__global__ void st_shared_kernel(uint32_t* output) {
    __shared__ uint32_t shared_data[32];
    
    int tid = threadIdx.x;
    if(tid == 0) {
        shared_data[0] = 42;
    }
    __syncthreads();
    
    *output = shared_data[0];
}

void test_ptx_shared_load_store(uint32_t* result) {
    uint32_t h_input[32];
    for(int i = 0; i < 32; i++) {
        h_input[i] = i + 1;
    }
    
    uint32_t *d_input, *d_result;
    cudaMalloc(&d_input, 32 * sizeof(uint32_t));
    cudaMalloc(&d_result, sizeof(uint32_t));
    
    cudaMemcpy(d_input, h_input, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    ld_shared_kernel<<<1, 32>>>(d_input, d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_result);
}

void test_ptx_shared_store_only(uint32_t* result) {
    uint32_t *d_result;
    cudaMalloc(&d_result, sizeof(uint32_t));
    
    st_shared_kernel<<<1, 32>>>(d_result);
    cudaDeviceSynchronize();
    
    cudaMemcpy(result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    cudaFree(d_result);
}