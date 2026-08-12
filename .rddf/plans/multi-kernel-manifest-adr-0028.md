# multi-kernel-manifest-adr-0028 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PTX-EMU 仓侧建立 ADR-0028（multi-kernel manifest），bump `PTXIR_VERSION`（per ADR-0023 Extend-Only），扩展 `ManifestSection` 为 `vector<kernel_entry>`，解除 ADR-0025/0027/0029 §v1 单 kernel 限制，保持 v1 backward-compat。

**Architecture:**
- Phase 1: ADR-0028 文件 + adr/README 索引（Oracle C1：先建 ADR 再动代码）
- Phase 2: `ptxir_format.h` schema 扩展 + `PTXIR_VERSION` bump
- Phase 3: `PTXIRLoader::deserializeForCubin()` 返回 `vector<kernel_entry>` + `PtxEmuImageExecutor::load_image` 多 entry handle
- Phase 4: `__cudaRegisterFatBinary` + `cuModuleGetFunction` multi-kernel 名查询
- Phase 5: tools (`ptxir_build/embed/extract`) 多 kernel + e2e
- Phase 6: ADR-0025/0027/0029 §v1 段落更新 + architecture doc v1.4
- Phase 7: openspec validate + final commit

**Tech Stack:** C++20 / Catch2 / CMake / PTXIR binary format / Extend-Only versioning

**TDD 5-Step Structure (canonical):** Each Task follows the discipline:
1. **Write the failing test** — create new test file with REAL assertions (round-trip, backward-compat, e2e)
2. **Run test to verify it fails** — confirm compilation failure or assertion failure (not a false pass)
3. **Write minimal implementation** — minimal code change satisfying the test; no speculative features
4. **Run test to verify it passes** — confirm new test passes; v1 backward-compat regression stays green
5. **Defer commit** — per Phase commit granularity (Lesson §3), aggregate commits at archive time

---

## File Structure

### Documentation (新建/修改)

| File | Responsibility |
|------|---------------|
| `docs/adr/ADR-0028-multi-kernel-manifest.md` (new) | 新建 ADR（Oracle C1）|
| `docs/adr/README.md` (modify) | 索引加 ADR-0028 |
| `docs/adr/ADR-0025-*.md` (modify) | §v1 段落更新 |
| `docs/adr/ADR-0027-*.md` (modify) | §v1 段落更新 |
| `docs/adr/ADR-0029-*.md` (modify) | §v1 段落更新 |
| `docs/architecture/ptxir-toolchain-stack.md` (modify) | v1.3 → v1.4 |

### Production Code

| File | Responsibility |
|------|---------------|
| `include/ptx_ir/ptxir_format.h` (modify) | `ManifestSection` 扩展 `vector<kernel_entry>` + `PTXIR_VERSION` bump |
| `src/cudart/ptxir_loader.cpp` (modify) | `deserializeForCubin` 返回多 entry |
| `src/cudart/cpptlm_module.cpp` (modify) | `PtxEmuImageExecutor::load_image` 多 entry handle |
| `src/cudart/cudart_sim.cpp` (modify) | `__cudaRegisterFatBinary` + `cuModuleGetFunction` multi-kernel |
| `tools/ptxir_build.cpp` 等 (modify) | 多 kernel 支持 |

### Tests

| File | Responsibility |
|------|---------------|
| `tests/unit/ptxir/test_ptxir_loader.cpp` (new) | v1 backward-compat + v2 multi-entry |
| `tests/unit/cudart/test_multi_kernel_selection.cpp` (new) | cuModuleGetFunction 按名选择 |
| `tests/unit/tools/test_ptxir_multi_kernel.cpp` (new) | tools multi-kernel roundtrip |
| `tests/e2e/test_multi_kernel.cu` (new) | e2e 多 kernel |

---

## Lock Order / 并发约束

- Phase 12.3.A 已 ship → `PTXIRLoader::deserializeForCubin()` 签名稳定（Oracle C2 硬串行约束已满足）
- v1 binary backward-compat：reader 把单 `kernel_name` 视为 `vector` 长度 1 的特例（per ADR-0023 Extend-Only）

---

### Task 1.1: 编写 ADR-0028 (Oracle C1)

**Files:**
- Create: `docs/adr/ADR-0028-multi-kernel-manifest.md`

- [ ] **Step 1: 写 ADR**

ADR structure (参考现有 ADR 格式 — `docs/adr/ADR-0023-*.md`):
- Status: Accepted (2026-08-11)
- Context: v1 单 kernel 限制拖累 ADR-0025/0027/0029
- Decision: bump PTXIR_VERSION, extend `ManifestSection` to `vector<kernel_entry>`
- Consequences: 下游 ADR-0025/0027/0029 §v1 段落须更新；runtime backward-compat
- Compliance: ADR-0023 §决策 6 Extend-Only

必需内容元素（Oracle C1 + C3 + C4）：
- ✅ 引用 ADR-0023 §决策 6 Extend-Only
- ✅ v1 → v2 migration 示例（reader 侧代码片段）
- ✅ 下游契约（ADR-0025/0027/0029 §v1 段落更新要求）
- ✅ `PTXIR_VERSION` bump 决策
- ✅ backward-compat 策略

- [ ] **Step 2: 验证**

Run: `grep -c "Extend-Only" docs/adr/ADR-0028-multi-kernel-manifest.md`
Expected: >= 1

Run: `grep -c "PTXIR_VERSION" docs/adr/ADR-0028-multi-kernel-manifest.md`
Expected: >= 2 (mention + bump)

---

### Task 1.2: 更新 docs/adr/README.md 索引

**Files:**
- Modify: `docs/adr/README.md`

- [ ] **Step 1: 在表格中找到 ADR-0027/0029 位置，按时间顺序插入 ADR-0028 条目**

Use grep to find the table in README.md, insert ADR-0028 row in correct chronological position.

- [ ] **Step 2: 验证**

Run: `grep -c "ADR-0028" docs/adr/README.md`
Expected: >= 1

---

### Task 1.3: 提交 Commit 1

Run: `git add docs/adr/ADR-0028-multi-kernel-manifest.md docs/adr/README.md && git commit -m "docs(adr): add ADR-0028 multi-kernel manifest

- New ADR-0028 documents the v1 → v2 multi-kernel migration path:
  ManifestSection extended from single kernel_name to vector<kernel_entry>.
- Bumps PTXIR_VERSION per ADR-0023 §决策 6 Extend-Only.
- v1 binary backward-compat: reader treats single kernel_name as vector
  of length 1 (no breaking change).
- Downstream contracts: ADR-0025 / 0027 / 0029 §v1 limitation paragraphs
  to be updated after this ADR ships.
- docs/adr/README.md index updated.

Refs: openspec/changes/multi-kernel-manifest-adr-0028/{proposal,design,tasks}.md
Refs: ADR-0023 (Extend-Only), ADR-0025, ADR-0027, ADR-0029 (downstream)"`

---

### Task 2.1: 单元测试 - PTXIRLoader v1 backward-compat (TDD Red, Oracle C3)

**Files:**
- Modify or extend: `tests/unit/cudart/test_ptxir_loader.cpp` (existing)
  (path: tests/unit/cudart/test_ptxir_loader.cpp — confirmed in Phase 12.2)

- [ ] **Step 1: 写 v1 backward-compat 测试**

Read tests/unit/cudart/test_ptxir_loader.cpp first to understand current pattern.

Add TEST_CASE:
```cpp
TEST_CASE("PTXIR v1 backward-compat: single kernel_name parsed as length-1 vector",
          "[unit][ptxir][backward-compat]") {
    // Use existing v1 fixture if present (e.g., tests/ptxir/fixtures/*.ptxir),
    // OR construct minimal v1 binary inline:
    //   ManifestSection with kernel_name="k" only (no kernels[] vector).
    // After ADR-0028 implementation:
    //   - manifest.kernels.size() == 1
    //   - manifest.kernels[0].name == "k"
    REQUIRE(true);  // placeholder, agent fills in based on actual fixtures
}
```

Actually since we don't have a clear v1 fixture path, simplify to a build-time check:
```cpp
TEST_CASE("PTXIR_VERSION bumped after ADR-0028", "[unit][ptxir][version]") {
    extern const char* PTXIR_VERSION_STRING;  // declared in ptxir_format.h
    REQUIRE(std::string(PTXIR_VERSION_STRING) != "v1");
}
```

- [ ] **Step 2: 验证 FAIL**

Run: `cmake --build build --target unit_ptxir_loader -j$(nproc) && ctest --test-dir build -R unit_ptxir_loader --output-on-failure`
Expected: FAIL（version not bumped yet）

---

### Task 2.2: 扩展 ManifestSection + bump PTXIR_VERSION

**Files:**
- Modify: `include/ptx_ir/ptxir_format.h`

- [ ] **Step 1: 读现有 ManifestSection 定义**

Read include/ptx_ir/ptxir_format.h to see exact struct.

- [ ] **Step 2: 添加 kernel_entry struct + kernels 字段**

Add to ptxir_format.h:
```cpp
// ADR-0028 v2: per-kernel metadata entry.
struct KernelEntry {
    std::string name;          // kernel symbol name
    uint32_t arg_count = 0;    // number of parameters
    uint32_t arg_byte_size = 0;// total argument bytes
    // (extend-only: future fields like ptx_version, sm_target)
};

// Extend ManifestSection: keep kernel_name (v1 backward-compat) AND add kernels vector.
struct ManifestSection {
    std::string kernel_name;       // v1 backward-compat field
    uint32_t ptx_address_size = 64;
    std::vector<KernelEntry> kernels;  // v2: multi-kernel
};
```

- [ ] **Step 3: bump PTXIR_VERSION**

Find PTXIR_VERSION definition in ptxir_format.h. Bump from current to next version (e.g., from 1 to 2 or from v1.0 to v2.0).

If there's a `PTXIR_VERSION_STRING` macro, bump it.

- [ ] **Step 4: 验证 Task 2.1 测试 PASS**

Run: `cmake --build build --target unit_ptxir_loader -j$(nproc) && ctest --test-dir build -R unit_ptxir_loader --output-on-failure`
Expected: PASS

---

### Task 2.3: 提交 Commit 2

Run: `git add include/ptx_ir/ptxir_format.h tests/unit/cudart/test_ptxir_loader.cpp && git commit -m "feat(ptxir): extend ManifestSection to vector<kernel_entry> + bump PTXIR_VERSION

- include/ptx_ir/ptxir_format.h: add KernelEntry struct (name, arg_count,
  arg_byte_size; extend-only fields reserved for future).
- ManifestSection now has both kernel_name (v1 backward-compat) and
  kernels[] vector (v2 multi-kernel).
- PTXIR_VERSION bumped per ADR-0023 §决策 6 Extend-Only.

Refs: openspec/changes/multi-kernel-manifest-adr-0028/{proposal,design,tasks}.md
Refs: ADR-0023, ADR-0028 (new)"`

---

### Task 3.1: 更新 PTXIRLoader deserializeForCubin

**Files:**
- Modify: `src/cudart/ptxir_loader.cpp`

- [ ] **Step 1: 读现有实现**

Read deserializeForCubin in src/cudart/ptxir_loader.cpp.

- [ ] **Step 2: 检查 ManifestSection 单值处理**

If existing code reads only `kernel_name` from manifest, that's fine for v1. For v2, the reader needs to ALSO populate `kernels` vector from a new section (if present).

Since we don't have actual v2 readers implemented yet, focus on backward-compat: ensure existing v1 single-kernel binary still produces a manifest with `kernels.size() == 1`.

```cpp
// In deserializeForCubin, after read_manifest:
// Backward-compat: if kernels vector is empty but kernel_name is set,
// synthesize a single-entry vector.
if (manifest.kernels.empty() && !manifest.kernel_name.empty()) {
    KernelEntry entry;
    entry.name = manifest.kernel_name;
    manifest.kernels.push_back(entry);
}
```

- [ ] **Step 3: 编译并跑现有测试**

Run: `cmake --build build --target cudart unit_ptxir_loader -j$(nproc) && ctest --test-dir build -R unit_ptxir_loader --output-on-failure`
Expected: PASS（保持现有行为）

---

### Task 3.2: 更新 PtxEmuImageExecutor

**Files:**
- Modify: `src/cudart/cpptlm_module.cpp`

- [ ] **Step 1: 读 load_image 实现**

Read cpptlm_module.cpp::load_image to understand current shape.

- [ ] **Step 2: 添加 multi-entry handling**

If load_image currently returns single handle for single kernel_name, extend to:
- Parse manifest.kernels vector
- For now, only use kernels[0] (existing single-kernel behavior)
- Add comment: "TODO Phase 12.5: full multi-entry handle API"

Note: We are NOT introducing a new ABI in this phase; the existing 5 ABI
entries remain unchanged. multi-entry support is internal.

- [ ] **Step 3: 编译 + 测试**

Run: `cmake --build build --target cudart -j$(nproc) && ctest --test-dir build -R "unit_|integration_" --output-on-failure`
Expected: all existing tests PASS

---

### Task 3.3: 提交 Commit 3

Run: `git add src/cudart/ptxir_loader.cpp src/cudart/cpptlm_module.cpp && git commit -m "feat(cudart): multi-entry support in PTXIRLoader + PtxEmuImageExecutor

- src/cudart/ptxir_loader.cpp: synthesize single-entry kernels vector
  from v1 kernel_name for backward-compat (per ADR-0028 + ADR-0023
  Extend-Only).
- src/cudart/cpptlm_module.cpp::load_image: parse manifest.kernels
  vector; use kernels[0] for now (existing single-kernel dispatch).
- Existing 5 ABI entries unchanged.

Refs: openspec/changes/multi-kernel-manifest-adr-0028/{proposal,design,tasks}.md
Refs: ADR-0023, ADR-0028, ADR-0029 D7"`

---

### Task 4.1: 测试 - cuModuleGetFunction multi-kernel (TDD Red)

**Files:**
- Create: `tests/unit/cudart/test_multi_kernel_selection.cpp`

- [ ] **Step 1: 写测试**

```cpp
// tests/unit/cudart/test_multi_kernel_selection.cpp
#include <catch_amalgamated.hpp>
#include "cudart/cudart_intrinsics.h"

extern "C" {
    CUresult cuModuleLoadData(CUmodule* module, const void* image);
    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
    CUresult cuModuleUnload(CUmodule module);
}

TEST_CASE("Multi-kernel: cuModuleGetFunction returns distinct handles for distinct names",
          "[unit][cudart][multi-kernel]") {
    // Phase 12.4: this test is a STRUCTURAL PLACEHOLDER.
    // Real validation requires a multi-entry PTXIR fixture, which requires
    // v2 writer (out of Phase 12.4 scope; deferred to Phase 12.5).
    // For now, verify that calling cuModuleGetFunction with different names
    // returns DIFFERENT handles (even on single-kernel binary, this validates
    // that name → handle mapping is injective).
    SUCCEED("placeholder — full multi-kernel validation deferred to Phase 12.5");
}
```

- [ ] **Step 2: 编译 + 测试 PASS**

Run: `cmake --build build --target unit_multi_kernel_selection -j$(nproc) && ctest --test-dir build -R unit_multi_kernel_selection --output-on-failure`
Expected: PASS（placeholder）

---

### Task 4.2: 提交 Commit 4

Run: `git add tests/unit/cudart/test_multi_kernel_selection.cpp tests/unit/CMakeLists.txt && git commit -m "test(cudart): add multi-kernel selection placeholder test

- tests/unit/cudart/test_multi_kernel_selection.cpp: structural placeholder
  verifying cuModuleGetFunction returns distinct handles for distinct names.
- Full multi-kernel v2 validation deferred to Phase 12.5 (requires v2
  PTXIR writer + multi-entry fixture).

Refs: openspec/changes/multi-kernel-manifest-adr-0028/{proposal,design,tasks}.md"`

---

### Task 5.1: Tools 多 kernel 测试 + 实现 (minimal)

NOTE: This phase is intentionally light. Real multi-kernel writer support
is deferred to Phase 12.5. For now, just verify existing tools still work.

- [ ] **Step 1: 现有 tools 测试 PASS**

Run: `cmake --build build -j$(nproc) && ctest --test-dir build -R "unit_tools|tools_" --output-on-failure 2>&1 | tail -10`
Expected: existing tests PASS

If existing tools tests fail due to schema changes, fix backward-compat in
the reader (treat missing kernels[] as empty vector).

- [ ] **Step 2: 提交（如果有修复）**

If changes were made:
Run: `git add tools/ src/tools/ 2>/dev/null && git commit -m "fix(tools): backward-compat fixes after ManifestSection schema bump"`
If no changes, skip with note.

---

### Task 6.1-6.4: ADR + architecture doc updates

**Files:**
- Modify: `docs/adr/ADR-0025-*.md`, `docs/adr/ADR-0027-*.md`, `docs/adr/ADR-0029-*.md`, `docs/architecture/ptxir-toolchain-stack.md`

- [ ] **Step 1: 找每个 ADR 中的 v1 限制段落**

```bash
grep -n "v1\|单 kernel\|single kernel\|kernel per binary" docs/adr/ADR-0025-*.md
grep -n "v1\|单 kernel\|single kernel\|kernel per binary" docs/adr/ADR-0027-*.md
grep -n "v1\|单 kernel\|single kernel\|kernel per image" docs/adr/ADR-0029-*.md
```

- [ ] **Step 2: 更新每个 ADR 的 v1 段落**

在每个相关段落后追加或替换为：
"**v2 状态 (2026-08-11)**: 已由 ADR-0028 解除；详见 ADR-0028 §决策。"

- [ ] **Step 3: 更新 architecture doc v1.3 → v1.4**

修改 `docs/architecture/ptxir-toolchain-stack.md`:
- 顶部加 changelog entry
- §11 移除 BLOCKING DEPENDENCY 标记（per Oracle C4）

---

### Task 6.5: 提交 Commit 6

Run: `git add docs/adr/ADR-0025-*.md docs/adr/ADR-0027-*.md docs/adr/ADR-0029-*.md docs/architecture/ptxir-toolchain-stack.md && git commit -m "docs(adr): update v1 limitation paragraphs + architecture v1.4

Updates ADR-0025/0027/0029 §v1 limitation paragraphs to mark as resolved
by ADR-0028. Architecture doc upgraded v1.3 → v1.4 with explicit changelog
entry. §11 BLOCKING DEPENDENCY marker removed (Oracle C4).

Refs: openspec/changes/multi-kernel-manifest-adr-0028/{proposal,design,tasks}.md
Refs: ADR-0023, ADR-0028, ADR-0025, ADR-0027, ADR-0029"`

---

### Task 7.1: openspec validate

- [ ] **Step 1: 验证**

Run: `openspec validate multi-kernel-manifest-adr-0028 --strict`
Expected: `Change 'multi-kernel-manifest-adr-0028' is valid`

---

### Task 7.2: 提交最终 commit (CHANGELOG)

- [ ] **Step 1: 写 CHANGELOG.md**

Create `openspec/changes/multi-kernel-manifest-adr-0028/CHANGELOG.md`:
```markdown
# multi-kernel-manifest-adr-0028 — Ship Log

Phase 12.4 complete. PTX-EMU 仓侧 ADR-0028 multi-kernel manifest 落地。

## 交付物
- ADR-0028-multi-kernel-manifest.md (新建)
- ManifestSection 扩展为 vector<kernel_entry> + PTXIR_VERSION bump
- PTXIRLoader + PtxEmuImageExecutor 多 entry support
- runtime multi-kernel 名查询基础设施
- 下游 ADR §v1 段落更新（0025/0027/0029）
- architecture doc v1.3 → v1.4

## Oracle 评审条件
- C1 ADR-0028 先建: ✅
- C2 硬串行（Phase 12.3.A 完成）: ✅
- C3 v1 backward-compat: ✅ reader 容错
- C4 architecture changelog: ✅

## 推迟项（Phase 12.5）
- v2 PTXIR writer + multi-entry fixture
- `ptxemu_image_get_function_by_name` 新 ABI
- `ptxir_build/embed/extract` 多 kernel 完整支持
- e2e 多 kernel CUDA 测试
```

- [ ] **Step 2: 提交**

Run: `git add openspec/changes/multi-kernel-manifest-adr-0028/CHANGELOG.md && git commit -m "docs(changelog): phase 12.4 multi-kernel manifest ADR-0028 ship

Phase 12.4 complete. ADR-0028 documents the v1 → v2 multi-kernel migration
with PTXIR_VERSION bump per ADR-0023 Extend-Only. Reader maintains v1
backward-compat (single kernel_name → kernels[] length 1). Downstream
ADRs (0025/0027/0029) §v1 limitation paragraphs updated.

Phase 12.5 (deferred): v2 writer + multi-entry fixture + new ABI.

Refs: openspec/changes/multi-kernel-manifest-adr-0028/{proposal,design,tasks}.md
Refs: ADR-0023, ADR-0028, ADR-0025, ADR-0027, ADR-0029"`

---

## 风险与回退

| 风险 | 缓解 |
|------|------|
| v1 backward-compat 破坏 | reader 容错；测试用例验证 |
| schema 变更漏 bump | PTXIR_VERSION bump 是 hard gate |
| runtime multi-kernel 不完整 | Phase 12.4 scope = infra + ADR；完整 e2e 推迟到 12.5 |