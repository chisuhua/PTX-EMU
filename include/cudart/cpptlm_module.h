#ifndef CPPTLM_MODULE_H
#define CPPTLM_MODULE_H

#include <cstddef>
#include <cstdint>

// VERSION 3 (Phase R-a): adds ptxemu_mem_register for memory domain coordination
// - ptxemu_mem_register(base, size)  tells PTX-EMU that UsrLinuxEmu's HAL heap
//   is accessible at [base, base+size) for ld.global/st.global
// - Consumer must check ptxemu_module_version() >= 3 before calling.
// - Still includes v1 (image_load/execute/unload) and v2 (multi-kernel APIs).
#define CPPTLM_MODULE_VERSION 3

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

// Memory Domain API (requires CPPTLM_MODULE_VERSION >= 3)
// Registers an external memory region that PTX-EMU can dereference.
// Without registration, ld.global/st.global to addresses in this range
// will fail (InvalidMemoryAccessException) or silently read wrong data
// (audit Defect 3).
// @param base Base address of region (must be page-aligned)
// @param size Size in bytes (must be page-aligned)
// @return 0=success, -EINVAL on bad params, -ENOMEM on overflow
int ptxemu_mem_register(uint64_t base, size_t size);

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
