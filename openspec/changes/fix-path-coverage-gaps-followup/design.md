# fix-path-coverage-gaps-followup Design

## Context

Pre-existing failure on `main` branch (verified via `git stash` exclusion test during `fix-path-coverage-gaps` execution):

```
Test #222: integration_libptxemu_abi_baseline (Failed)
  000000000000ade0 T ptxemu_image_execute_named    <- ACTUAL
  000000000000b850 T ptxemu_image_execute
  ...
  0000000000008060 T ptxemu_image_kernel_name       <- BASELINE
  0000000000008320 T ptxemu_image_unload
```

The test (`tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp`) does:
```cpp
std::string cmd = "nm -D " + lib.string() + " 2>/dev/null | grep ptxemu_ | sort";
// ... REQUIRE(current == baseline);
```

Baseline (`tests/integration/cpptlm/baselines/libptxemu_abi_baseline.txt`):
```
0000000000008060 T ptxemu_image_kernel_name
0000000000008320 T ptxemu_image_unload
0000000000008470 T ptxemu_module_version
00000000000095e0 T ptxemu_image_execute
0000000000009620 T ptxemu_image_load
```

Every build changes the load addresses. The baseline was generated once and never regenerated. Result: the test is **always failing** (verified by running on `main` without my changes).

## Decision

### Decision 1: Strip addresses from comparison (not regenerate baseline)

**选择**: Modify `run_nm()` to use `awk '{print $2, $3}'` — keep only type + name. Sort + uniq for stability.

**理由**:
- Addresses are non-semantic (Linker-determined, change every build)
- ABI stability is determined by symbol presence + type, NOT by address
- Regenerating baseline still leaves the test fragile (next build will have different addresses)

**替代方案**:
- 每次 build 前自动重新生成 baseline → 在 CI 加 pre-test hook，但增加 CI 复杂度
- 用 `nm --defined-only` 过滤 undefined symbols → 不解决 address 问题

### Decision 2: Add regeneration README

**选择**: 在 `baselines/` 目录添加 `README.md`，说明如何重新生成 baseline：
```bash
nm -D build/lib/libptxemu_device.so | awk '{print $2, $3}' | grep ptxemu_ | sort -u > tests/integration/cpptlm/baselines/libptxemu_abi_baseline.txt
```

**理由**: 当 ABI 真正变化时（添加/删除符号），提供明确的可执行 regeneration 命令。

## Out of Scope

- 不修改 `include/cudart/cpptlm_module.h`（ABI 5 symbols 契约保持）
- 不修改 baseline 文件目录
- 不引入 CI hook 自动 regeneration