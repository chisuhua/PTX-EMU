#include "memory/hardware_memory_manager.h" // 添加HardwareMemoryManager头文件
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include <iostream>

void LdHandler::processOperation(ThreadContext *context, void *op[2],
                           const std::vector<Qualifier> &qualifier) {
  void *dst = op[0];
  void *host_ptr = op[1]; // ← 这是 cudaMalloc 返回的主机指针

  // 空指针检查
  if (!dst || !host_ptr) {
    std::cerr << "Error: Null pointer in LD instruction" << std::endl;
    return;
  }

  // 获取地址空间和数据大小
  MemorySpace space = getAddressSpace(qualifier);
  size_t data_size = getBytes(qualifier);

  // ========================
  // 1. 标量 LD（无向量）
  // ========================
  if (!QvecHasQ(qualifier, Qualifier::Q_V2) &&
      !QvecHasQ(qualifier, Qualifier::Q_V4)) {
    HardwareMemoryManager::instance().access(host_ptr, dst, data_size,
                                             /*is_write=*/false, space);
    return;
  }

  // ========================
  // 2. 向量 LD（V2/V4）
  // ========================
  size_t step = getBytes(qualifier); // 元素步长
  auto vecAddr = context->vecOp_phy_addrs.front();
  context->vecOp_phy_addrs.pop();

  size_t vec_size = 0;
  if (QvecHasQ(qualifier, Qualifier::Q_V2)) {
    vec_size = 2;
    assert(vecAddr.size() == 2);
  } else if (QvecHasQ(qualifier, Qualifier::Q_V4)) {
    vec_size = 4;
    assert(vecAddr.size() == 4);
  }

  // 逐元素读取
  for (size_t i = 0; i < vec_size; ++i) {
    void *element_dst = vecAddr[i];
    uint64_t element_host_ptr = reinterpret_cast<uint64_t>(host_ptr) + i * step;

    // 对于其他内存空间，使用HardwareMemoryManager访问
    HardwareMemoryManager::instance().access(
        reinterpret_cast<void *>(element_host_ptr), element_dst, data_size,
        /*is_write=*/false, space);
  }
}

void StHandler::processOperation(ThreadContext *context, void *op[2],
                           const std::vector<Qualifier> &qualifiers) {
  void *host_ptr = op[0]; // ← 目标地址：cudaMalloc 返回的主机指针
  void *src = op[1];      // ← 源数据：寄存器或立即数地址

  // 空指针检查
  if (!host_ptr || !src) {
    std::cerr << "Error: Null pointer in ST instruction" << std::endl;
    return;
  }

  // 获取地址空间和数据大小
  MemorySpace space = getAddressSpace(qualifiers);
  size_t data_size = getBytes(qualifiers);

  // ========================
  // 1. 标量 ST（无向量）
  // ========================
  if (!QvecHasQ(qualifiers, Qualifier::Q_V2) &&
      !QvecHasQ(qualifiers, Qualifier::Q_V4)) {
    // 根据地址空间选择内存访问方式
    // 对于其他内存空间，使用HardwareMemoryManager访问
    HardwareMemoryManager::instance().access(host_ptr, src, data_size,
                                             /*is_write=*/true, space);
    return;
  }

  // ========================
  // 2. 向量 ST（V2/V4）
  // ========================
  size_t step = getBytes(qualifiers); // 元素步长
  auto vecAddr = context->vecOp_phy_addrs.front();
  context->vecOp_phy_addrs.pop();

  size_t vec_size = 0;
  if (QvecHasQ(qualifiers, Qualifier::Q_V2)) {
    vec_size = 2;
    assert(vecAddr.size() == 2);
  } else if (QvecHasQ(qualifiers, Qualifier::Q_V4)) {
    vec_size = 4;
    assert(vecAddr.size() == 4);
  }

  // 逐元素写入
  for (size_t i = 0; i < vec_size; ++i) {
    void *element_src = vecAddr[i];
    uint64_t element_host_ptr = reinterpret_cast<uint64_t>(host_ptr) + i * step;

    // 对于其他内存空间，使用HardwareMemoryManager访问
    HardwareMemoryManager::instance().access(
        reinterpret_cast<void *>(element_host_ptr), element_src, data_size,
        /*is_write=*/true, space);
  }
}
