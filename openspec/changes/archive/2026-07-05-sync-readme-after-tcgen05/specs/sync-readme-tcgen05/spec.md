# Sync README After tcgen05 — Spec

> **Type**: 文档同步（不引入 behavior 变化）
> **Status**: Proposed (2026-07-05)
> **Ref**: archive/2026-07-04-implement-wmma-tensor-core-{tcgen05,phase-0-infra}/

## MODIFIED Requirements

### `project-readme` requirement: README.md MUST reflect current implementation state

#### Scenario: New developer reads README.md "已知限制" section
- **WHEN** developer reads README.md to understand current capabilities
- **THEN** README.md MUST accurately reflect which PTX features are implemented vs stubbed
- **AND** README.md MUST NOT claim WMMA / Tensor Core is a stub (since commit `4151268` Fix #14 implements tcgen05)

#### Scenario: New developer searches for Blackwell tcgen05 documentation
- **WHEN** developer needs to find Blackwell tcgen05 architecture / roadmap docs
- **THEN** README.md "文档导航" section MUST link to:
  - `docs/adr/ADR-0016-blackwell-only-tcgen05.md` (architecture decision)
  - `docs/dev-process/post-tcgen05-roadmap.md` (H5 planning)

#### Scenario: Reader wants current status
- **WHEN** reader sees README.md "状态" line at line 3
- **THEN** line MUST reflect SIMT v2.0 completion + Blackwell tcgen05 completion + H5 planning phase
- **AND NOT** claim "Phase 10 进行中" without qualification

### `project-readme-implemented-features` requirement: README.md MUST enumerate key implemented features

#### Scenario: Reader wants overview of Blackwell tcgen05 implementation
- **WHEN** reader wants to verify all 5 tcgen05 sub-instructions are implemented
- **THEN** README.md MUST contain "已实现功能" section listing:
  - `tcgen05.mma` (fragment arithmetic) — commit `535dd9d` Fix #10
  - `tcgen05.ld` / `tcgen05.st` (TMA + TMEM integration) — commit `35808d6` Fix #12
  - `tcgen05.commit` / `tcgen05.wait` (async flow) — commit `0213ff1` Fix #13
  - Plus 4 supporting infrastructure: TMA descriptors (Fix #5 `ad527f5`), TMEM (Fix #6 `758edb0`), cluster arrive/wait (Fix #7 `e513235`), TcQueue (Fix #8 `c0fa43f`)

## ADDED Requirements

### `project-readme-no-hardcoded-stats` requirement

#### Scenario: README.md contains hardcoded statistics
- **WHEN** developer encounters a hardcoded percentage or version number in README.md
- **THEN** README.md MUST replace with link to auto-generated stat (e.g., `docs/audits/`)
- **AND NOT** include stale hardcoded values that diverge from actual implementation state

### `project-readme-env-adaptive-toolchain` requirement

#### Scenario: Reader wants to know CUDA Toolkit compatibility
- **WHEN** reader wants to know supported CUDA version
- **THEN** README.md MUST describe env.sh auto-detection (`NVCC_PATH=$(which nvcc)`)
- **AND NOT** hardcode a specific CUDA Toolkit version

## REMOVED Requirements

无（纯增量同步，不删除现有章节）

## Compatibility

- **Behavior 兼容**: 是（纯文档）
- **Build 兼容**: 是（无 CMake 改动）
- **Test 兼容**: 是（无测试改动）
- **API 兼容**: 是（无公开 API 改动）
- **Risk**: 🟢 极低
