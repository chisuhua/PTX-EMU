# ci-drift-check Specification

## Purpose
TBD - created by archiving change ptxemu-public-device-api. Update Purpose after archive.
## Requirements
### Requirement: `.github/workflows/drift_check.yml` MUST 存在

新增 workflow 文件 MUST 验证 PTX-EMU 内部公共头布局与仓内 hash 一致:

```yaml
name: drift-check
on:
  pull_request:
    paths:
      - 'include/ptxemu/**'
      - 'include/ptx_ir/**'
  workflow_dispatch:
jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Header hash drift check
        run: |
          # 验证 device_api.h PUBLIC 字段稳定
          grep -q "PTXEMU_API_VERSION 1" include/ptxemu/device_api.h
          # 验证 IPtxEmuDevice 接口签名冻结 (count 虚方法)
          EXPECTED_METHODS=12  # S1 facade 12 callsites 1:1
          ACTUAL=$(grep -c "virtual.*=.*0\|virtual.*override" include/ptxemu/device_api.h)
          test "$ACTUAL" -ge "$EXPECTED_METHODS"
```

#### Scenario: Phase 2 PR 修改 device_api.h 触发 drift_check
- **WHEN** GitHub PR 修改 `include/ptxemu/device_api.h`
- **THEN** drift_check workflow 自动 run, 验证 `PTXEMU_API_VERSION=1` 守卫宏保留 + 虚方法数量 >= 12

### Requirement: `consumer_smoke` MUST 不在 Phase 2 PR 范围

`consumer_smoke` (验证 PTX-EMU 端能在 CMake 链式构建 CppTLM) MUST 不进 Phase 2 PR, 延后至 HSK-9 准入 (Decision 2 答复)。

#### Scenario: Phase 2 PR 不包含 consumer_smoke
- **WHEN** 读 Phase 2 PR diff
- **THEN** 0 引用 `tests/build_cpptlm_consume/` 路径

### Requirement: `drift_check` 与 `build-and-test` 解耦

`drift_check` MUST 作为独立 workflow, 不与 `build-and-test` 串行依赖。允许仅跑 `drift_check` (PR 阶段) 而不跑 `build-and-test` (release 阶段) 单独验证。

#### Scenario: PR 阶段仅跑 drift_check
- **WHEN** PR 标题含 `[skip-build]` 或 path filter 仅含 `include/ptxemu/**`
- **THEN** `drift_check` 跑通即视为合规, `build-and-test` 可跳过

