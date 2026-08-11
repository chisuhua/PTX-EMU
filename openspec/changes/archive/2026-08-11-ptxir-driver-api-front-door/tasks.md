# Tasks: ptxir-driver-api-front-door

> **TDD 5-step discipline** (per `test-driven-development` skill):
> 1. Write failing test → 2. Verify fail → 3. Implement → 4. Verify pass → 5. Commit
> **Phase 12.3.A 任务映射** (per `roadmap.md:122-146`)

## Phase 1: ModuleRegistry 基础设施

### Task 1.1: 单元测试 - ModuleRecord/FunctionRecord/ModuleRegistry 接口 (TDD Red)
- **MUST**: 写 `tests/unit/cudart/test_module_registry.cpp` 的 stub 版本（接口契约）
- **MUST NOT**: 不实现 module_registry.h/cpp
- **验证**: `cmake --build build && ctest -R test_module_registry` FAIL（接口未定义）
- **关联**: Phase 12.3.A1 + Oracle C1 (lock order 文档化)

### Task 1.2: 实现 ModuleRecord/FunctionRecord/ModuleRegistry
- **MUST**: 新建 `include/cudart/module_registry.h` + `src/cudart/module_registry.cpp`
- **MUST**: `ModuleRegistry::insert/lookup/remove` 全部 `std::mutex` 保护
- **MUST**: 文档化 lock order：`ModuleRegistry::mutex` → per-`PtxContext` lock（never reverse）
- **MUST**: image bytes deep copy to `ModuleRecord` private storage
- **MUST NOT**: 复用 `cuda_driver.h`（内存分配器职责不混合）
- **验证**: Task 1.1 测试 PASS

### Task 1.3: 提交 Commit 1
- **MUST**: 单独立 commit `feat(cudart): add ModuleRegistry with mutex-protected handles`
- **MUST NOT**: 与其他 Driver API 入口混在同一 commit

## Phase 2: cuModuleLoadData + cuModuleGetFunction + image classifier

### Task 2.1: 单元测试 - 6 类 image classifier (TDD Red, Oracle C2)
- **MUST**: 写 `tests/unit/cudart/test_image_classifier.cpp` 6 个 scenario（PTX text / standalone PTXIR / exec-tail / cubin / fatbin / Tile IR）
- **验证**: FAIL（classifier 未实现）

### Task 2.2: 实现 6 类 image classifier
- **MUST**: 新建 `src/cudart/image_classifier.cpp`（纯函数，无副作用）
- **MUST**: 不读 `/proc/self/exe` / 不调 `cuobjdump` / 不读 `PTXIR_MODE`
- **验证**: Task 2.1 测试 PASS

### Task 2.3: 单元测试 - cuModuleLoadData 契约 (TDD Red)
- **MUST**: 写 `tests/unit/cudart/test_cuda_driver_api.cpp` 验证：
  - 接受 standalone PTXIR bytes 返回 `CUmodule` handle
  - image bytes deep copy 后 caller-owned pointer 可安全释放
  - 未知 image class 返回 `CUDA_ERROR_INVALID_IMAGE`
- **验证**: FAIL

### Task 2.4: 实现 cuModuleLoadData + 替换 cuModuleGetFunction stub
- **MUST**: `src/cudart/cudart_sim.cpp` 新增 `cuModuleLoadData` 入口
- **MUST**: eager parse + image bytes deep copy
- **MUST**: 替换 `cuModuleGetFunction` stub at line 513 为真版本
- **MUST**: 复用 `PTXIRLoader::deserializeForCubin()`（grep 验证单点）
- **验证**: Task 2.3 测试 PASS + `nm -D build/lib/libcudart.so | grep "T.*cuModuleLoadData"` 输出存在

### Task 2.5: 提交 Commit 2
- **MUST**: 单独立 commit `feat(cudart): add cuModuleLoadData + 6-class image classifier`

## Phase 3: cuLaunchKernel(CUfunction) + cuModuleUnload + error mapping

### Task 3.1: 单元测试 - 7 类 error mapping (TDD Red)
- **MUST**: 写 `tests/unit/cudart/test_error_mapping.cpp` 7 个 scenario
- **验证**: FAIL

### Task 3.2: 实现 cuLaunchKernel(CUfunction) Driver API 版本
- **MUST**: `src/cudart/cudart_sim.cpp` 新增 `cuLaunchKernel(CUfunction, ...)` 入口
- **MUST**: 复用现有 `cudaLaunchKernel` 主路径
- **MUST**: per-launch fresh `PtxContext`（不缓存 `kernelStatements`）
- **验证**: 集成测试 PASS

### Task 3.3: 实现 cuModuleUnload + in-flight busy 边界
- **MUST**: `src/cudart/cudart_sim.cpp` 新增 `cuModuleUnload` 入口
- **MUST**: in-flight 时返回 `CUDA_ERROR_INVALID_HANDLE`（busy）
- **MUST**: 释放 `ModuleRecord` + 失效 child `CUfunction` handles
- **验证**: stale handle 测试 PASS

### Task 3.4: 实现 7 类 error mapping
- **MUST**: 在 `cudart_sim.cpp` 统一 error code 映射表
- **验证**: Task 3.1 测试 PASS

### Task 3.5: 提交 Commit 3
- **MUST**: 单独立 commit `feat(cudart): add cuLaunchKernel(CUfunction) + cuModuleUnload + 7-class error mapping`

## Phase 4: 回归测试 (D3 mutation bug + ABI 稳定性)

### Task 4.1: 集成测试 - D3 mutation bug 复检 (Oracle C3)
- **MUST**: 新建 `tests/integration/test_in_memory_mutation.cpp`：
  - (a) 同 bytes 两次 deserialize→byte-identical
  - (b) 顺序 launch 1000 次不同 blockDim→输出确定无累积
  - (c) image bytes hash 经 N 次 launch 不变（SHA-256 比对）
- **验证**: 全部 PASS

### Task 4.2: 集成测试 - ABI 稳定性回归 (Oracle C7)
- **MUST**: 验证 `git diff cpptlm_bridge.h` 为空
- **MUST**: 验证 `CPPTLMBRIDGE_VERSION` 保持 2
- **MUST**: 验证 SONAME 不变
- **验证**: 测试 PASS + nm 验证 5 ABI 入口符号不变

### Task 4.3: 提交 Commit 4
- **MUST**: 单独立 commit `test(cudart): add D3 mutation + ABI stability regression gates`

## Phase 5: Integration + acceptance gates

### Task 5.1: 集成测试 - 端到端 cuModuleLoadData → cuLaunchKernel → cuModuleUnload
- **MUST**: 新建 `tests/integration/test_cuda_driver_api.cpp`
- **MUST**: 覆盖 A8b scenario（`PTXIR_MODE=off` 不影响 in-memory path）
- **MUST**: 覆盖 legacy + in-memory 同进程共存
- **验证**: 集成测试 PASS

### Task 5.2: acceptance gate - nm -D verify 4 个新 T 符号
- **MUST**: `nm -D build/lib/libcudart.so | grep -E "cu(ModuleLoadData|ModuleGetFunction|ModuleUnload|LaunchKernel)" | grep " T "` 输出 4 行
- **验证**: grep 通过

### Task 5.3: 提交 Commit 5
- **MUST**: 单独立 commit `test(cudart): add end-to-end Driver API integration tests + nm verification`

## Phase 6: Docs + archive readiness

### Task 6.1: openspec validate 通过
- **MUST**: `openspec validate ptxir-driver-api-front-door` 无错误
- **MUST**: 3 个 spec.md 文件 delta 解析正确

### Task 6.2: ADR-0029 D7 5 byte-identical gates 复检
- **MUST**: `tests/integration/test_phase0_byte_identical_gates.cpp` 全部 PASS

### Task 6.3: 提交最终 commit
- **MUST**: `docs(changelog): phase 12.3.A Driver API front door ship`

---

## 依赖与阻塞

- **本 change 不阻塞其他**: 与 `multi-kernel-manifest-adr-0028`（Phase 12.4）**有反向依赖**——本 change 完成后 12.4 才能启动（保持 `deserializeForCubin` 签名稳定）
- **`hal-extension-ptxemu-usrlinu-emu-taskrunner`**（Phase 13）独立，无阻塞关系

## 风险与回退

- 任一 Phase commit 失败可独立 revert（per `ptx-lessons-learned` §3）
- 测试覆盖率不足时停止推进