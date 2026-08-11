# ADR-0028: Multi-Kernel Manifest

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-08-11 |
| **关联任务** | Phase 12.4 (multi-kernel-manifest-adr-0028) |
| **关联 OpenSpec change** | [openspec/changes/multi-kernel-manifest-adr-0028/](../openspec/changes/multi-kernel-manifest-adr-0028/) |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle (architecture review) |

---

## 上下文

### 问题背景

`include/ptx_ir/ptxir_format.h:36-41` 的 `ManifestSection` 只有单 `kernel_name` 字段，导致 3 个已 Accepted ADR 同时受 v1 单 kernel 限制拖累：

- **ADR-0025** (`ptxir_build` CLI) — wrapper 拒绝 multi-entry PTX
- **ADR-0027** (`ptx-nvcc` wrapper) — 同样限制
- **ADR-0029** (image executor) D4 — `libptxemu_device.so` 的 `ptxemu_image_kernel_name` 只返回首个

架构 §11 明示 **ADR-0028 是 BLOCKING DEPENDENCY**，状态从"预留占位"于 2026-08-09 升级。

### 触发事件

1. **2026-08-09** — ADR-0029 F2 跨仓评审修订，ADR-0028 从"预留占位"升级为 BLOCKING DEPENDENCY
2. **2026-08-09** — `docs/architecture/ptxir-toolchain-stack.md` v1.3 §11 记录 BLOCKING DEPENDENCY 约束
3. **2026-08-11** — Phase 12.4 实施启动

### 技术约束

- **ADR-0023 Extend-Only 版本管理**：`PTXIR_VERSION` bump 是 hard gate，version 不 bump 不允许改 schema
- **backward-compat**：旧 v1 单 kernel binary 在 ADR-0028 后运行时仍可加载
- **不修改 ANTLR 解析路径**：复用 `PTXIRLoader::deserializeForCubin()` 扩展
- **不修改 cpptlm_bridge.h ABI**：与 Phase 12.3.A 共享约束

---

## 决策内容

### Decision 1: 扩展 ManifestSection 为 vector<kernel_entry>

**选择**: `ManifestSection` 新增 `std::vector<KernelEntry> kernels` 字段，保留 `kernel_name`（v1 backward-compat）。

```cpp
// ADR-0028 v2: per-kernel metadata entry.
struct KernelEntry {
    std::string name;          // kernel symbol name
    uint32_t arg_count = 0;    // number of parameters
    uint32_t arg_byte_size = 0; // total argument bytes
    // (extend-only: future fields like ptx_version, sm_target)
};

// Extend ManifestSection: keep kernel_name (v1 backward-compat) AND add kernels vector.
struct ManifestSection {
    std::vector<uint8_t> cubin_hash;   // SHA-256 (32 bytes)
    std::string kernel_name;           // v1 backward-compat field
    uint8_t ptx_address_size = 64;     // 32 or 64
    std::vector<ManifestParam> params;
    std::vector<KernelEntry> kernels;  // v2: multi-kernel
};
```

**理由**:
- `kernel_name` 保留使 v1 binary 不需要转换，reader 可以继续使用
- `kernels` vector 让 v2 writer 可以序列化多个 entry
- extend-only 字段（`arg_count`、`arg_byte_size`）为 future extension 预留

### Decision 2: PTXIR_VERSION bump

**选择**: `PTXIR_VERSION` 从 3 bump 到 4。

**理由**: per ADR-0023 §决策 6 Extend-Only，version bump 是 schema 变更的 hard gate。

### Decision 3: backward-compat 策略

**选择**: reader 端若 `kernels` vector 为空但 `kernel_name` 非空，synthesize 单-entry vector。

```cpp
// Backward-compat: if kernels vector is empty but kernel_name is set,
// synthesize a single-entry vector.
if (manifest.kernels.empty() && !manifest.kernel_name.empty()) {
    KernelEntry entry;
    entry.name = manifest.kernel_name;
    manifest.kernels.push_back(entry);
}
```

**理由**: v1 binary 没有 `kernels` vector，reader 必须容错。writer 端 v2 binary 同时写入 `kernel_name` 和 `kernels`，保证两者一致。

---

## 下游契约

实施 ADR-0028 后，以下 ADR §v1 限制段落须更新：

| ADR | 段落 | 更新内容 |
|-----|------|---------|
| ADR-0025 | §技术约束 / §v1 单 kernel 限制 | 替换为："**v2 状态 (2026-08-11)**: 已由 ADR-0028 解除；详见 ADR-0028 §Decision 1。" |
| ADR-0027 | §技术约束 / §v1 单 kernel 限制 | 同上 |
| ADR-0029 | D4 v1 单 kernel per image | 同上 |

同时，`docs/architecture/ptxir-toolchain-stack.md` v1.3 → v1.4 changelog entry + §11 BLOCKING DEPENDENCY 标记移除。

---

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptx_ir/ptxir_format.h` | 修改 | `KernelEntry` struct + `ManifestSection.kernels` + `PTXIR_VERSION` bump |
| `src/cudart/ptxir_loader.cpp` | 修改 | backward-compat synthesis |
| `src/cudart/cpptlm_module.cpp` | 修改 | `load_image` / `get_kernel_name` / `execute` 多 entry handle |
| `docs/adr/ADR-0025/0027/0029` | 修改 | §v1 限制段落更新 |
| `docs/architecture/ptxir-toolchain-stack.md` | 修改 | v1.4 changelog + §11 BLOCKING 移除 |

**不变组件**:
- `cpptlm_bridge.h` ABI — 与 Phase 12.3.A 共享约束
- `libptxemu_device.so` 新 ABI — 仅扩展，不破坏现有 5 函数

---

## 合规检查

- [ ] `PTXIR_VERSION` bump 是 hard gate（ADR-0023 §决策 6）
- [ ] backward-compat: v1 single-kernel binary 在新 runtime 下加载正常
- [ ] downstream ADR §v1 段落已更新
- [ ] `ptxir-toolchain-stack.md` §11 BLOCKING DEPENDENCY 标记已移除
- [ ] architecture doc changelog v1.4 entry 已添加

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-11 | 初始版本（Phase 12.4 multi-kernel-manifest-adr-0028 实施） | PTX-EMU Architecture Team |

---

## 参考

- [ADR-0023](./ADR-0023-ptxir-binary-format.md) — PTXIR 二进制序列化格式（Extend-Only 版本管理原则）
- [ADR-0024](./ADR-0024-ptxir-cubin-embed-extension.md) — PTXIR-Embedded CUBIN 格式
- [ADR-0025](./ADR-0025-ptxir-build-cli.md) — `ptxir_build` CLI（§v1 待更新）
- [ADR-0027](./ADR-0027-ptx-nvcc-wrapper.md) — `ptx-nvcc` wrapper（§v1 待更新）
- [ADR-0029](./ADR-0029-ptxemu-image-executor.md) — PTX-EMU Image Executor（§D4 待更新）
- [docs/architecture/ptxir-toolchain-stack.md](../architecture/ptxir-toolchain-stack.md) — 工具链栈架构（§11 BLOCKING DEPENDENCY）
