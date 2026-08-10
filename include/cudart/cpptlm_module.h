#ifndef CPPTLM_MODULE_H
#define CPPTLM_MODULE_H

#include <cstddef>
#include <cstdint>

#define CPPTLM_MODULE_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size);

int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size);

int ptxemu_image_execute(uint64_t handle,
                         uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t block_x, uint32_t block_y, uint32_t block_z,
                         size_t shared_mem_bytes,
                         void** kernel_args, size_t args_count);

int ptxemu_image_unload(uint64_t handle);

int ptxemu_module_version(void);

#ifdef __cplusplus
}
#endif

#endif
