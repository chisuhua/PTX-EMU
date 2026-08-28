#include "cudart/cpptlm_module.h"

#include "cudart/cuda_driver.h"
#include "cudart/ptx_context_adapter.h"
#include "cudart/ptx_interpreter.h"
#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptxir_format.h"
#include "utils/logger.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cudart {

// PtxEmuImageExecutor 7 SINGLE-GPU-INSTANCE assumptions (per ADR-0029 §D6
// + ptx-lessons-learned §10):
//   #1 g_gpu_context: global unique — all images share one simulated GPU
//       (defined in ptx_interpreter.cpp, extern-declared in ptx_interpreter.h)
//   #2 CudaDriver::instance(): singleton — all images share one memory pool
//       (CudaDriver::instance().malloc/free used in PtxContext lifecycle)
//   #3 g_cpptlm_bridge: standalone mode (nullptr) — CppTLM orthogonal
//       (separate library; image executor path does not interact)
//   #4 g_image_executor: this singleton — process-global pointer
//       (static PtxEmuImageExecutor* g_image_executor = &instance();)
//   #5 exec_mu_: mutex — serializes concurrent same-handle launches (D3 fix)
//   #6 PtxInterpreter: stateful non-reentrant — fresh instance per launch
//       (PtxInterpreter interpreter; declared in execute() local scope)
//   #7 No SingletonGuard coupling — image executor path is orthogonal to
//       legacy LD_PRELOAD __cudaRegisterFatBinary registration
//       (g_image_executor constructed independently; __cudaRegisterFatBinary
//        SingletonGuard applies only to the legacy path)
class PtxEmuImageExecutor {
public:
    static PtxEmuImageExecutor& instance() {
        static PtxEmuImageExecutor inst;
        return inst;
    }

    PtxEmuImageExecutor(const PtxEmuImageExecutor&) = delete;
    PtxEmuImageExecutor& operator=(const PtxEmuImageExecutor&) = delete;

    uint64_t load_image(const uint8_t* bytes, size_t size) {
        if (bytes == nullptr || size == 0) return 0;

        bool is_standalone_ptxir = (size >= 4 &&
            std::memcmp(bytes, "PTXI", 4) == 0);
        bool is_embedded = PTXIRLoader::hasEmbeddedPTXIR(bytes, size);

        if (!is_standalone_ptxir && !is_embedded) {
            PTX_DEBUG_EMU("image_load: rejected (no PTXIR/Embedded magic), size=%zu", size);
            return 0;
        }

        uint64_t handle = next_handle_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mu_);
            images_[handle] = std::vector<uint8_t>(bytes, bytes + size);
        }
        PTX_DEBUG_EMU("image_load: handle=%llu size=%zu",
                       (unsigned long long)handle, size);
        return handle;
    }

    int get_kernel_name(uint64_t handle, char* buf, size_t buf_size) {
        if (buf_size == 0) return -EINVAL;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = images_.find(handle);
        if (it == images_.end()) return -EINVAL;
        auto bytes_copy = it->second;
        auto manifest = read_manifest_from_ptxir_section(bytes_copy.data(), bytes_copy.size());
        // v2 multi-kernel: select first entry from kernels vector
        // v1 backward-compat: if kernels empty, fall back to kernel_name
        const std::string& name = manifest.kernels.empty()
            ? manifest.kernel_name
            : manifest.kernels[0].name;
        if (name.empty()) return -EINVAL;
        size_t copy_len = std::min(name.size(), buf_size - 1);
        std::memcpy(buf, name.data(), copy_len);
        buf[copy_len] = '\0';
        return 0;
    }

    int execute(uint64_t handle,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                size_t shared_mem_bytes,
                void** kernel_args, size_t args_count) {
        (void)args_count;
        // Plan task 3.4: explicit missing-context branch — must not report
        // successful execution when the shared GPU context is unavailable.
        if (g_gpu_context == nullptr) return -EINVAL;
        std::lock_guard<std::mutex> exec_lock(exec_mu_);
        std::vector<uint8_t> bytes_copy;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = images_.find(handle);
            if (it == images_.end()) return -EINVAL;
            bytes_copy = it->second;
        }

        std::vector<ptxemu::ir::StatementContext> stmts;
        try {
            stmts = PTXIRLoader::deserializeForCubin(bytes_copy.data(), bytes_copy.size());
        } catch (...) {
            return -EINVAL;
        }
        if (stmts.empty()) return -EINVAL;

        auto manifest = read_manifest_from_ptxir_section(bytes_copy.data(), bytes_copy.size());
        if (manifest.kernels.empty()) {
            return -EINVAL;
        }
        EmbeddedKernelManifest em;
        em.kernelName = manifest.kernels[0].name;
        em.ptxAddressSize = manifest.ptx_address_size;
        em.params = manifest.params;

        auto ctx = PtxContextAdapter::fromEmbedded(std::move(stmts), em);

        PtxInterpreter interpreter;
        std::string kernel_name = manifest.kernel_name;

        Dim3 grid_dim(grid_x, grid_y, grid_z);
        Dim3 block_dim(block_x, block_y, block_z);

        interpreter.launchPtxInterpreter(ctx, kernel_name, kernel_args,
                                         grid_dim, block_dim, shared_mem_bytes);
        // Plan task 3.3: synchronous completion — drive exe_once() until the
        // kernel marks itself complete. The call stays inside exec_mu_ so
        // concurrent same-handle launches remain serialized.
        g_gpu_context->wait_for_completion();
        return 0;
    }

    int unload(uint64_t handle) {
        if (!exec_mu_.try_lock()) return -EBUSY;
        exec_mu_.unlock();

        std::lock_guard<std::mutex> lock(mu_);
        auto it = images_.find(handle);
        if (it == images_.end()) return -EINVAL;
        images_.erase(it);
        return 0;
    }

    int version() const { return CPPTLM_MODULE_VERSION; }

    int kernel_count(uint64_t handle) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = images_.find(handle);
        if (it == images_.end()) return -1;
        auto manifest = read_manifest_from_ptxir_section(it->second.data(), it->second.size());
        return static_cast<int>(manifest.kernels.size());
    }

    int kernel_name_at(uint64_t handle, uint32_t idx, char* buf, size_t buf_size) {
        if (buf_size == 0) return -1;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = images_.find(handle);
        if (it == images_.end()) return -1;
        auto manifest = read_manifest_from_ptxir_section(it->second.data(), it->second.size());
        if (idx >= manifest.kernels.size()) return -1;
        const std::string& name = manifest.kernels[idx].name;
        size_t copy_len = std::min(name.size(), buf_size - 1);
        std::memcpy(buf, name.data(), copy_len);
        buf[copy_len] = '\0';
        return static_cast<int>(name.size());
    }

    int execute_named(uint64_t handle, const char* kernel_name,
                      uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                      uint32_t block_x, uint32_t block_y, uint32_t block_z,
                      size_t shared_mem_bytes,
                      void** kernel_args, size_t args_count) {
        (void)args_count;
        if (kernel_name == nullptr) return -EINVAL;
        std::lock_guard<std::mutex> exec_lock(exec_mu_);
        std::vector<uint8_t> bytes_copy;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = images_.find(handle);
            if (it == images_.end()) return -1;
            bytes_copy = it->second;
        }
        // Plan task 3.4: explicit missing-context branch (must not report
        // success). Placed AFTER the handle lookup so that a stale handle
        // still returns -1 (pre-existing contract for invalid handle).
        if (g_gpu_context == nullptr) return -EINVAL;

        std::vector<ptxemu::ir::StatementContext> stmts;
        try {
            stmts = PTXIRLoader::deserializeForCubin(bytes_copy.data(), bytes_copy.size());
        } catch (...) {
            return -EINVAL;
        }
        if (stmts.empty()) return -EINVAL;

        auto manifest = read_manifest_from_ptxir_section(bytes_copy.data(), bytes_copy.size());
        auto kernel_it = std::find_if(manifest.kernels.begin(), manifest.kernels.end(),
            [&](const KernelEntry& ke) { return ke.name == kernel_name; });
        if (kernel_it == manifest.kernels.end()) return -1;

        EmbeddedKernelManifest em;
        em.kernelName = kernel_it->name;
        em.ptxAddressSize = manifest.ptx_address_size;
        em.params = manifest.params;

        auto ctx = PtxContextAdapter::fromEmbedded(std::move(stmts), em);

        PtxInterpreter interpreter;
        std::string kn = kernel_name;

        Dim3 grid_dim(grid_x, grid_y, grid_z);
        Dim3 block_dim(block_x, block_y, block_z);

        interpreter.launchPtxInterpreter(ctx, kn, kernel_args,
                                        grid_dim, block_dim, shared_mem_bytes);
        // Plan task 3.3: synchronous completion (see execute() for rationale).
        g_gpu_context->wait_for_completion();
        return 0;
    }

private:
    PtxEmuImageExecutor() = default;

    std::mutex mu_;
    std::mutex exec_mu_;
    std::unordered_map<uint64_t, std::vector<uint8_t>> images_;
    std::atomic<uint64_t> next_handle_{1};
};

static PtxEmuImageExecutor* g_image_executor = &PtxEmuImageExecutor::instance();

extern "C" uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size) {
    return g_image_executor->load_image(image_bytes, image_size);
}

extern "C" int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size) {
    return g_image_executor->get_kernel_name(handle, buf, buf_size);
}

extern "C" int ptxemu_image_execute(uint64_t handle,
                                     uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                     uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                     size_t shared_mem_bytes,
                                     void** kernel_args, size_t args_count) {
    return g_image_executor->execute(handle, grid_x, grid_y, grid_z,
                                      block_x, block_y, block_z,
                                      shared_mem_bytes, kernel_args, args_count);
}

extern "C" int ptxemu_image_unload(uint64_t handle) {
    return g_image_executor->unload(handle);
}

extern "C" int ptxemu_module_version(void) {
    return g_image_executor->version();
}

extern "C" int ptxemu_image_kernel_count(uint64_t handle) {
    return g_image_executor->kernel_count(handle);
}

extern "C" int ptxemu_image_kernel_name_at(uint64_t handle, uint32_t idx,
                                           char* buf, size_t buf_size) {
    return g_image_executor->kernel_name_at(handle, idx, buf, buf_size);
}

extern "C" int ptxemu_image_execute_named(uint64_t handle, const char* kernel_name,
                                           uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                           uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                           size_t shared_mem_bytes,
                                           void** kernel_args, size_t args_count) {
    return g_image_executor->execute_named(handle, kernel_name,
                                          grid_x, grid_y, grid_z,
                                          block_x, block_y, block_z,
                                          shared_mem_bytes,
                                          kernel_args, args_count);
}

}  // namespace cudart
