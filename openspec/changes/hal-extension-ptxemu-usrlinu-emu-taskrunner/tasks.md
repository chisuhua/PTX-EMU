# Tasks: hal-extension-ptxemu-usrlinu-emu-taskrunner

> **PTX-EMU 仓不拥有跨仓协调责任**（Oracle C3）。本 change 范围限于 PTX-EMU 仓侧 ABI 冻结 + RFC 文档 + 测试扩展。
> **TDD 5-step** discipline applies。

## Phase 1: libptxemu_device.so 5 ABI 字节级审计

### Task 1.1: 集成测试 - 5 ABI nm 输出与 baseline 比对 (TDD Red)
- **MUST**: 写 `tests/integration/test_libptxemu_abi_baseline.cpp`：
  - 运行 `nm -D build/lib/libptxemu_device.so | grep ptxemu_`
  - 输出与 Phase 1 ship baseline（保存在 `tests/integration/baselines/libptxemu_abi_baseline.txt`）byte-identical 比对
- **MUST NOT**: 此时不动 ABI（baseline 应已通过）
- **验证**: FAIL（baseline file 不存在）

### Task 1.2: 生成 baseline 文件
- **MUST**: 运行 nm 命令输出保存到 `tests/integration/baselines/libptxemu_abi_baseline.txt`
- **MUST**: baseline 含 5 个 `T ptxemu_*` 符号
- **验证**: Task 1.1 测试 PASS

### Task 1.3: 提交 Commit 1
- **MUST**: `test(ptxemu): add 5 ABI baseline + byte-identical verification`

## Phase 2: DL-isolated 测试扩展

### Task 2.1: 集成测试 - 扩展 DL-isolated 覆盖 (TDD Red)
- **MUST**: 扩展 `tests/integration/test_cpptlm_module_dlopen.cpp`：
  - 5 ABI 入口全部 `dlsym` 成功
  - 在无 `libcudart.so` 环境下 `dlopen` 成功
  - 调用每个 ABI 入口无 undefined symbol 错误
- **验证**: FAIL（覆盖不全）

### Task 2.2: 扩展 DL-isolated 测试实现
- **MUST**: 在 `tests/integration/test_cpptlm_module_dlopen.cpp` 中增加覆盖
- **验证**: Task 2.1 测试 PASS

### Task 2.3: 提交 Commit 2
- **MUST**: `test(ptxemu): expand DL-isolated coverage for all 5 ABI`

## Phase 3: in-flight unload 边界测试扩展

### Task 3.1: 集成测试 - in-flight unload 边界 (TDD Red)
- **MUST**: 扩展 `tests/integration/test_cpptlm_module_inflight.cpp`：
  - 启动 kernel 后立即调 `ptxemu_image_unload(handle)`
  - 断言返回非 0 错误码
  - 等待 kernel 完成后再调 `ptxemu_image_unload` 应成功
- **验证**: FAIL

### Task 3.2: 扩展 in-flight unload 测试
- **MUST**: 在 `tests/integration/test_cpptlm_module_inflight.cpp` 增加并发场景
- **验证**: Task 3.1 测试 PASS

### Task 3.3: 提交 Commit 3
- **MUST**: `test(ptxemu): expand in-flight unload boundary coverage`

## Phase 4: 跨仓 RFC 文档

### Task 4.1: 编写 rfc-hal-extension.md (Oracle C2)
- **MUST**: 新建 `openspec/changes/hal-extension-ptxemu-usrlinu-emu-taskrunner/rfc-hal-extension.md`
- **MUST**: 引用 ADR-0029 §D8 + TaskRunner ADR-035 R5.1 + UsrLinuxEmu ADR-036
- **MUST**: 含 "PTX-EMU 仓不拥有跨仓协调责任；commit 顺序是 integrator 责任" 显式声明 (Oracle C3)
- **MUST**: 记录跨仓 commit 顺序（UsrLinuxEmu → PTX-EMU 验证 → TaskRunner）
- **验证**: RFC 文件含以上要素

### Task 4.2: 提交 Commit 4
- **MUST**: `docs(ptxemu): add cross-repo HAL extension RFC`

## Phase 5: archive readiness

### Task 5.1: 跨仓污染 grep 验证 (Oracle C1)
- **MUST**: `grep -r "UsrLinuxEmu\|TaskRunner" src/ include/ CMakeLists.txt` 输出为空
- **验证**: 空输出

### Task 5.2: openspec validate 通过
- **MUST**: `openspec validate hal-extension-ptxemu-usrlinu-emu-taskrunner` 无错误

### Task 5.3: 提交最终 commit
- **MUST**: `docs(changelog): phase 13 HAL extension PTX-EMU side ship`

---

## 阻塞关系

- **本 change 阻塞**: 无（PTX-EMU 仓只 freeze，不阻塞下游）
- **本 change 被阻塞**: 无（与 Phase 12.3.A / 12.4 独立）
- **跨仓启动前置**: 跨仓 commit 顺序由 TaskRunner 仓 integrator 决策；本仓侧只需保证 ABI 冻结 + RFC 落地

## 风险与回退

- 任一 Phase commit 失败可独立 revert
- nm 验证失败（ABI 意外变更）必须停止推进 + revert
- grep 验证失败（跨仓污染）必须清理后重试