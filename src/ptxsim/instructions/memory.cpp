#include "memory/hardware_memory_manager.h"
#include "ptx_ir/instruction_latency_table.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/warp_state.h"
#include "ptxsim/utils/qualifier_utils.h"
#include <iostream>

void LdHandler::processOperation(ThreadContext *context, void *op[2],
                           const std::vector<Qualifier> &qualifier,
                           const std::vector<char> *operand_is_immediate) {
  void *dst = op[0];
  void *host_ptr = op[1];

  PTX_INFO_EMU("LD: dst=%p host_ptr=%p", dst, host_ptr);

  if (!dst || !host_ptr) {
    std::cerr << "Error: Null pointer in LD instruction" << std::endl;
    return;
  }

  MemorySpace space = getAddressSpace(qualifier);
  size_t data_size = getBytes(qualifier);

  if (!QvecHasQ(qualifier, Qualifier::Q_V2) &&
      !QvecHasQ(qualifier, Qualifier::Q_V4)) {
    // Block active threads for the post-load latency only on global
    // loads — shared/local/const are 1-cycle on this simulator and
    // must not be marked blocked. See regression commit 2b9d803.
    WarpContext *warp_ctx = context->warp_context_;
    if (warp_ctx != nullptr && space == MemorySpace::GLOBAL) {
      auto latency = ptxsim::getLatency(S_LD);
      if (latency.cycles > 0) {
        auto &ws = warp_ctx->get_warp_state();
        for (auto &thread : ws.threads) {
          if (thread.is_active && !thread.is_exited) {
            thread.is_blocked = true;
            thread.blocked_cycles_remaining = latency.cycles;
          }
        }
      }
    }
    HardwareMemoryManager::instance().access(host_ptr, dst, data_size,
                                             false, space);
    return;
  }

  size_t step = getBytes(qualifier);
  // BUGFIX: read VEC element addresses directly from op[0] (the VEC
  // destination), no longer from the global vecOp_phy_addrs FIFO. The
  // per-ThreadContext stack in acquire_operand (case VEC) hands us a
  // void** that points to the array of element addresses; we cast and
  // iterate. This decouples V2/V4 LD/ST from the FIFO, so non-V2/V4
  // handlers (e.g. mov.b64 with a vector source) can no longer leave
  // stale entries that subsequent V4 LD/ST pop in their place.
  void **vecAddrs = static_cast<void **>(dst);

  size_t vec_size = 0;
  if (QvecHasQ(qualifier, Qualifier::Q_V2)) {
    vec_size = 2;
  } else if (QvecHasQ(qualifier, Qualifier::Q_V4)) {
    vec_size = 4;
  }
  for (size_t i = 0; i < vec_size; ++i) {
    void *element_dst = vecAddrs[i];
    uint64_t element_host_ptr = reinterpret_cast<uint64_t>(host_ptr) + i * step;

    HardwareMemoryManager::instance().access(
        reinterpret_cast<void *>(element_host_ptr), element_dst, data_size,
        false, space);
  }
}

void StHandler::processOperation(ThreadContext *context, void *op[2],
                           const std::vector<Qualifier> &qualifiers,
                           const std::vector<char> *operand_is_immediate) {
  void *host_ptr = op[0]; // ← 目标地址：cudaMalloc 返回的主机指针
  void *src = op[1];      // ← 源数据：寄存器或立即数地址

  // 空指针检查
  if (!host_ptr || !src) {
    std::cerr << "Error: Null pointer in ST instruction" << std::endl;
    return;
  }

  MemorySpace space = getAddressSpace(qualifiers);
  size_t data_size = getBytes(qualifiers);
  uint64_t src_val = 0;
  memcpy(&src_val, src, data_size);

  // ========================
  // 1. 标量 ST（无向量）
  // ========================
  if (!QvecHasQ(qualifiers, Qualifier::Q_V2) &&
      !QvecHasQ(qualifiers, Qualifier::Q_V4)) {
    // 对于 PARAM 空间，需要更新符号表条目中的值
    if (space == MemorySpace::PARAM) {
        // For st.param [param0], %rd4:
        // - host_ptr is the GPU address where we should write (param0's slot address)
        // - src_val is the value to write
        uint64_t *gpu_addr = (uint64_t *)host_ptr;
        *gpu_addr = src_val;
    } else {
        // 根据地址空间选择内存访问方式
        // 对于其他内存空间，使用HardwareMemoryManager访问
        HardwareMemoryManager::instance().access(host_ptr, src, data_size,
                                                 /*is_write=*/true, space);
    }
    return;
  }

  // ========================
  // 2. 向量 ST（V2/V4）
  // ========================
  size_t step = getBytes(qualifiers); // 元素步长
  // BUGFIX: see LdHandler V2/V4 path — read VEC element addresses
  // directly from op[1] (the VEC source) instead of the global FIFO.
  void **vecAddrs = static_cast<void **>(src);

  size_t vec_size = 0;
  if (QvecHasQ(qualifiers, Qualifier::Q_V2)) {
    vec_size = 2;
  } else if (QvecHasQ(qualifiers, Qualifier::Q_V4)) {
    vec_size = 4;
  }

  // 逐元素写入
  for (size_t i = 0; i < vec_size; ++i) {
    void *element_src = vecAddrs[i];
    uint64_t element_host_ptr = reinterpret_cast<uint64_t>(host_ptr) + i * step;

    // 对于其他内存空间，使用HardwareMemoryManager访问
    HardwareMemoryManager::instance().access(
        reinterpret_cast<void *>(element_host_ptr), element_src, data_size,
        /*is_write=*/true, space);
  }
}
