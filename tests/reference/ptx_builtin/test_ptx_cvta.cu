#include "catch_amalgamated.hpp"
#include "ptx_cvta.cuh"
#include <cstdint>

// CVTA (Convert Address) 指令测试
// CVTA指令用于在不同的地址空间之间转换地址

TEST_CASE("PTX: cvta.to.global.u64", "[ptx][cvta][address_conversion][global]") {
    uint64_t input_address = 0x1000;
    uint64_t result = 0;
    
    // 分配设备内存
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMemcpy(d_result, &result, sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 启动kernel
    kernel_test_ptx_cvta_to_global_u64<<<1, 1>>>(input_address, d_result);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    // 验证转换后的地址与输入地址相同
    REQUIRE(result == input_address);
}

TEST_CASE("PTX: cvta.to.local.u64", "[ptx][cvta][address_conversion][local]") {
    uint64_t input_address = 0x2000;
    uint64_t result = 0;
    
    // 分配设备内存
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMemcpy(d_result, &result, sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 启动kernel
    kernel_test_ptx_cvta_to_local_u64<<<1, 1>>>(input_address, d_result);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    // 验证转换后的地址与输入地址相同
    REQUIRE(result == input_address);
}

TEST_CASE("PTX: cvta.to.param.u64", "[ptx][cvta][address_conversion][param]") {
    uint64_t input_address = 0x3000;
    uint64_t result = 0;
    
    // 分配设备内存
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMemcpy(d_result, &result, sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 启动kernel
    kernel_test_ptx_cvta_to_param_u64<<<1, 1>>>(input_address, d_result);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    // 验证转换后的地址与输入地址相同
    REQUIRE(result == input_address);
}

// TEST_CASE("PTX: cvta.to.shared.u32", "[ptx][cvta][address_conversion][shared]") {
//     uint32_t input_address = 0x100;
//     uint32_t result = 0;
    
//     // 分配设备内存
//     uint32_t *d_result;
//     cudaMalloc(&d_result, sizeof(uint32_t));
//     cudaMemcpy(d_result, &result, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
//     // 启动kernel
//     kernel_test_ptx_cvta_to_shared_u32<<<1, 1>>>(input_address, d_result);
//     cudaDeviceSynchronize();
    
//     // 复制结果回主机
//     cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//     cudaFree(d_result);
    
//     // 验证转换后的地址与输入地址相同
//     REQUIRE(result == input_address);
// }

// TEST_CASE("PTX: cvta.to.global.u32", "[ptx][cvta][address_conversion][global][u32]") {
//     uint32_t input_address = 0x4000;
//     uint32_t result = 0;
    
//     // 分配设备内存
//     uint32_t *d_result;
//     cudaMalloc(&d_result, sizeof(uint32_t));
//     cudaMemcpy(d_result, &result, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
//     // 启动kernel
//     kernel_test_ptx_cvta_to_global_u32<<<1, 1>>>(input_address, d_result);
//     cudaDeviceSynchronize();
    
//     // 复制结果回主机
//     cudaMemcpy(&result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//     cudaFree(d_result);
    
//     // 验证转换后的地址与输入地址相同
//     REQUIRE(result == input_address);
// }

TEST_CASE("PTX: cvta.const.u64", "[ptx][cvta][address_conversion][const]") {
    uint64_t input_address = 0x5000;
    uint64_t result = 0;
    
    // 分配设备内存
    uint64_t *d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));
    cudaMemcpy(d_result, &result, sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // 启动kernel
    kernel_test_ptx_cvta_const_u64<<<1, 1>>>(input_address, d_result);
    cudaDeviceSynchronize();
    
    // 复制结果回主机
    cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    // 验证转换后的地址与输入地址相同
    REQUIRE(result == input_address);
}