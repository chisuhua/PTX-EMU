#ifndef CPPTLM_MODULE_H
#define CPPTLM_MODULE_H

#include <cstddef>
#include <cstdint>

// VERSION 2 (Phase C4): adds 3 multi-kernel enumeration APIs
// - ptxemu_image_kernel_count(handle)
// - ptxemu_image_kernel_name_at(handle, idx, buf, buf_size)
// - ptxemu_image_execute_named(handle, name, ...)
// Consumer must check ptxemu_module_version() >= 2 before calling these.
#define CPPTLM_MODULE_VERSION 2

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

// === Multi-kernel API (requires CPPTLM_MODULE_VERSION >= 2) ===

// Returns the number of kernels in the loaded module, or -1 if handle invalid.
int ptxemu_image_kernel_count(uint64_t handle);

// Writes the kernel name at index `idx` into `buf` (NUL-terminated on success).
// Returns the required length (excluding NUL), or -1 if:
//   - handle invalid
//   - idx out of range
//   - buf_size == 0 (caller should re-call with sufficient buffer)
// If buf_size < required length, truncates to buf_size-1 bytes and NUL-terminates.
int ptxemu_image_kernel_name_at(uint64_t handle, uint32_t idx,
                                char* buf, size_t buf_size);

// Like ptxemu_image_execute but selects the kernel by name instead of kernels[0].
// Returns -1 if handle invalid or kernel_name not found.
int ptxemu_image_execute_named(uint64_t handle, const char* kernel_name,
                               uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                               uint32_t block_x, uint32_t block_y, uint32_t block_z,
                               size_t shared_mem_bytes,
                               void** kernel_args, size_t args_count);

#ifdef __cplusplus
}
#endif

#endif  // CPPTLM_MODULE_H
