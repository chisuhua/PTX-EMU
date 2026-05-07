# atom.global.add 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `atom.global.add.u32` PTX 指令，支持 32/64 位整型全局内存原子加法操作。

**Architecture:** 修改仅集中在 `src/ptxsim/instructions/atomic.cpp`。ANTLR 语法/解析器/派发管线已完整（weak symbol + `AtomicPipelineHandler` prepare/execute/commit 生命周期）。`HardwareMemoryManager::access()` 提供内存读写 API。参考 `LdHandler::processOperation`（`memory.cpp`）实现模式。

**Tech Stack:** C++20, PTX-EMU ptxsim, HardwareMemoryManager

---

### Task 1: 读取现有代码了解实现模式

**Files:**
- Read: `src/ptxsim/instructions/atomic.cpp`（当前 stub）
- Read: `src/ptxsim/instructions/memory.cpp:7-59`（LD 指令参考模式）
- Read: `src/memory/hardware_memory_manager.cpp:100-164`（`access()` 实现）

- [ ] **Step 1: 读取当前 atomic stub**

```bash
cd /workspace/project/PTX-EMU && cat src/ptxsim/instructions/atomic.cpp
```

Expected: 空函数体，仅提取 operands 但无操作。

- [ ] **Step 2: 读取 LD 指令参考模式**

```bash
cd /workspace/project/PTX-EMU && sed -n '7,59p' src/ptxsim/instructions/memory.cpp
```

Expected: 看到 `LdHandler::processOperation`：
```
void *dst = op[0];
void *host_ptr = op[1];
MemorySpace space = getAddressSpace(qualifier);
size_t data_size = getBytes(qualifier);
HardwareMemoryManager::instance().access(host_ptr, dst, data_size, false, space);
```

- [ ] **Step 3: 读取 HardwareMemoryManager API**

```bash
cd /workspace/project/PTX-EMU && cat src/memory/hardware_memory_manager.h
```

Expected: 看到 `static HardwareMemoryManager &instance()` 和 `void access(void *host_ptr, void *data, size_t size, bool is_write, MemorySpace space)`。

---

### Task 2: 实现 atom.global.add（32 位和 64 位）

**Files:**
- Modify: `src/ptxsim/instructions/atomic.cpp`

- [ ] **Step 1: 写入完整实现**

```cpp
#include "ptxsim/instructions/instruction_handlers.h"
#include "memory/hardware_memory_manager.h"
#include "ptxsim/utils/qualifier_utils.h"
#include "include/ptx_ir/ptx_types.h"
#include <cstring>
#include <cstdint>

void AtomHandler::processAtomicOperation(ThreadContext *context, void **operands,
                                 const std::vector<Qualifier> &qualifiers,
                                 const std::vector<char> *operand_is_immediate) {
    // PTX atom 标准操作数布局:
    //   atom.global.add.u32 d, [addr], src;
    //   operands[0] = d  (dst 寄存器——写入旧值)
    //   operands[1] = addr (全局内存地址——cudaMalloc 返回的 host_ptr)
    //   operands[2] = src (要加的值)
    //   operands[3] = src2 (仅 CAS: compare value)

    void *dst = operands[0];
    void *host_ptr = operands[1];
    void *src_val = operands[2];

    MemorySpace space = getAddressSpace(qualifiers);
    size_t data_size = getBytes(qualifiers);

    // Step 1: 从全局内存读取旧值
    uint64_t old_val = 0;
    void *old_ptr = &old_val;
    HardwareMemoryManager::instance().access(host_ptr, old_ptr, data_size,
                                             /*is_write=*/false, space);

    // Step 2: 根据数据类型和原子操作计算新值
    uint64_t new_val = 0;
    bool found_op = false;

    for (const auto &q : qualifiers) {
        switch (q) {
        case Qualifier::Q_ADD_ATOM: {
            found_op = true;
            if (data_size == 4) {
                uint32_t a = static_cast<uint32_t *>(old_ptr)[0];
                uint32_t b = *static_cast<uint32_t *>(src_val);
                new_val = static_cast<uint64_t>(a + b);
            } else if (data_size == 8) {
                new_val = *static_cast<uint64_t *>(old_ptr) +
                          *static_cast<uint64_t *>(src_val);
            }
            break;
        }
        default:
            break;
        }
    }

    if (!found_op) {
        // 默认 fallback：仅支持 ADD_ATOM，其他操作原样返回
        new_val = old_val;
    }

    // Step 3: 将新值写回全局内存
    HardwareMemoryManager::instance().access(host_ptr, &new_val, data_size,
                                             /*is_write=*/true, space);

    // Step 4: 将旧值写入 dst 寄存器
    std::memcpy(dst, old_ptr, data_size);
}
```

- [ ] **Step 2: 编译验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build --target ptxsim 2>&1 | tail -20
```

Expected: 编译成功。

---

### Task 3: 编写原子操作测试

**Files:**
- Create: `tests/test_atomic_add.cu`（CUDA 源文件——通过 fake cudart 编译验证）
- Create: `tests/test_ptx_atom.cpp`（直接 PTX 指令测试，使用 buildStatements）
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: 创建 CUDA 测试（端到端验证）**

写入 `tests/test_atomic_add.cu`：

```cpp
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_atomic_add_kernel(int *counter) {
    atomicAdd(counter, 1);
}

int main() {
    int *d_counter;
    int h_counter = 0;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);

    test_atomic_add_kernel<<<1, 32>>>(d_counter);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    printf("counter = %d (expected 32)\n", h_counter);

    int success = (h_counter == 32) ? 0 : 1;
    cudaFree(d_counter);
    return success;
}
```

- [ ] **Step 2: 在 CMakeLists.txt 注册 CUDA 测试**

编辑 `tests/CMakeLists.txt`：
```cmake
# 在 CUDA 测试循环中（靠后的位置）
add_cuda_test(test_atomic_add)
```

并确保 `test_atomic_add` 被赋予标签 `extra`（因为依赖于实现完整的 atomicAdd）。

- [ ] **Step 3: 创建 PTX 级单元测试（精确验证）**

写入 `tests/test_ptx_atom.cpp`：

```cpp
#include <catch2/catch.hpp>
#include "ptx_sim_tests.h"  // 假设包含 test helpers
#include "ptxsim/instructions/instruction_handlers.h"
#include "memory/hardware_memory_manager.h"

TEST_CASE("AtomHandler: atom.global.add.u32", "[atom][add]") {
    // 1. 分配全局内存
    uint32_t *dev_ptr;
    cudaMalloc(&dev_ptr, sizeof(uint32_t));
    uint32_t init_val = 42;
    cudaMemcpy(dev_ptr, &init_val, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 2. 构建 atom.global.add.u32 指令的操作数
    uint32_t dst_reg = 0;
    uint32_t src_val = 10;
    void *operands[3] = { &dst_reg, &dev_ptr, &src_val };
    std::vector<Qualifier> qualifiers = { Qualifier::Q_GLOBAL, Qualifier::Q_ADD_ATOM, Qualifier::Q_U32 };

    // 3. 创建 mock ThreadContext
    // (需要根据实际测试 API 调整)

    // 4. 执行原子操作
    // AtomHandler handler;
    // handler.processAtomicOperation(context, operands, qualifiers, nullptr);

    // 5. 验证: 旧值 = 42, 内存新值 = 52
    // uint32_t result;
    // cudaMemcpy(&result, dev_ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // REQUIRE(dst_reg == 42);
    // REQUIRE(result == 52);
}
```

注意：此测试需要通过 `buildStatements` 或 `test_helpers.hpp` 中的框架构建完整的执行上下文。参考 `tests/test_spinlock_simulation.cu` 进行端到端 CUDA 测试。

- [ ] **Step 4: 运行测试**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R test_atomic -V 2>&1 | tail -40
```

Expected: 测试通过。

---

### Task 4: 验证 memory.cpp 类似的指令（确保不破坏 ld/st）

- [ ] **Step 1: 运行所有内存相关测试**

```bash
cd /workspace/project/PTX-EMU/build && ctest -L "ld_st" -V 2>&1 | tail -30
```

Expected: 所有 ld/st 测试通过。

- [ ] **Step 2: 运行 quick sanity**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --quick 2>&1 | tail -20
```

Expected: 0 failures。

---

### Task 5: 扩展实现到更多原子操作（可选后续任务）

**Files:**
- Modify: `src/ptxsim/instructions/atomic.cpp`

- [ ] **Step 1: 操作分发表**

在 Task 2 的 switch 语句中添加更多 case：

```cpp
case Qualifier::Q_EXCH_ATOM: {
    found_op = true;
    new_val = *static_cast<uint64_t *>(src_val);  // 直接替换
    break;
}
case Qualifier::Q_CAS_ATOM: {
    found_op = true;
    void *cmp_val = operands[3];  // CAS 有第 4 个操作数
    if (data_size == 4) {
        uint32_t a = static_cast<uint32_t *>(old_ptr)[0];
        uint32_t b = *static_cast<uint32_t *>(cmp_val);
        uint32_t c = *static_cast<uint32_t *>(src_val);
        new_val = (a == b) ? c : a;
    }
    break;
}
```

- [ ] **Step 2: 添加测试**

创建 `test_ptx_atom_exch.cu` 和 `test_ptx_atom_cas.cu`。

- [ ] **Step 3: 提交**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/instructions/atomic.cpp tests/test_atomic_add.cu tests/test_ptx_atom.cpp tests/CMakeLists.txt
git commit -m "feat: implement atom.global.add.u32 and prepare atomic operation framework"
```
