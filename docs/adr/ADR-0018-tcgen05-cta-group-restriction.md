# ADR-0018: tcgen05 cta_group::2 throws UnsupportedInstructionException (cluster abstraction deferred)

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-07-12 |
| **关联任务** | `openspec/changes/fix-tcgen05-commit-wait-group/` |
| **关联 PR** | TBD |
| **作者** | project architect (Metis pre-impl review 2026-07-12) |
| **审核人** | TBD |
| **Supersedes** | 无 |
| **Related** | [ADR-0016](./ADR-0016-blackwell-only-tcgen05.md) (Blackwell-only scope + cluster prerequisites) |

## 上下文

PTX ISA §9.7.16 `tcgen05.*` 指令族支持 `cta_group::N` 限定符：

- `cta_group::1` — 单 CTA 内部同步（默认，当前完整支持）
- `cta_group::2` — 跨 2 个 CTA 的 distributed shared memory 同步（依赖 cluster mode）

PTX-EMU 在 `ADR-0016`（Blackwell-only tcgen05 范围）中明确：
> **Hopper (sm_90+) cluster 抽象未实现**

这意味着 `cta_group::2` 需要的 cluster abstraction（distributed shared memory、
cross-CTA arrive/wait、TMA multicast descriptors）在 PTX-EMU 中尚未实现。

如果 `cta_group::2` 静默退化为 `cta_group::1` 行为（当前默认 fallback），
会产生 **silent correctness bug**：用户写 `cta_group::2` 期望跨 CTA 同步，
实际只跑了单 CTA 同步，结果不可预测且难以调试。

## 决策驱动因素

1. **Fail-fast 优于 silent fallback**：违反 `replace-silent-stub-failures` 合约（archived 2026-07-04）
2. **Exception message 必须包含 ADR 引用**：让用户能跳转文档理解为何不支持
3. **保持 `cta_group::1` 完整工作路径**：不因 `cta_group::2` throw 影响 `cta_group::1` 用户
4. **Scope discipline**：`cta_group::2` 实现属于更大范围的 cluster abstraction，
   不在 `fix-tcgen05-commit-wait-group` 或 `implement-tcgen05-handlers-extended` scope

## 考虑的替代方案

### 方案 A: Silent fallback to cta_group::1 (❌ 拒绝)

**描述**: `cta_group::2` 直接走 `cta_group::1` handler 路径，不报错

**优点**:
- 用户代码"看似"工作

**缺点**:
- 违反 `replace-silent-stub-failures` 合约
- 静默 correctness bug（多 CTA 同步语义丢失）
- 调试噩梦（用户期望分布式行为，实际串行）

### 方案 B: Throw UnsupportedInstructionException with ADR reference (✅ 选中)

**描述**: `tcgen05.*.cta_group::2.*` 抛 `UnsupportedInstructionException`,
消息包含 `cluster abstraction not yet implemented (ADR-0018)`

**优点**:
- Fail-fast，立即告知用户不支持
- 消息中 ADR 引用让用户能查到原因
- 与现有 `wmma.mma.sync.*`（pre-Blackwell）抛异常模式一致

**缺点**:
- 用户必须修改 PTX 代码（去掉 `cta_group::2` 或改写为 `cta_group::1`）

**选择理由**: 与项目已建立的"未实现功能必须显式失败"合约保持一致。

### 方案 C: 实现 cta_group::2 partial emulation (❌ 拒绝)

**描述**: 模拟两个串行 CTA 的 commit/wait（通过 TBD CTA 抽象层）

**优点**:
- 看起来支持 `cta_group::2`

**缺点**:
- 需要先实现 cluster abstraction（~2000+ LoC 新基础设施）
- scope 远大于本 change / `implement-tcgen05-handlers-extended`
- 应作为独立 change 单独 propose（如 `implement-cluster-abstraction`）

## 决策内容

### 设计原则

1. **Throw at handler dispatch, not at parse time**: parsing `cta_group::2` 是允许的
   （`Tcgen05Instr::cta_group` 字段已填充，值为 `2`）；throw 发生在 handler dispatch 阶段
2. **统一所有 11 handler**: `cta_group::2` 必须 throw 自每个 handler（mma/ld/st/commit/wait/
   alloc/dealloc/relinquish_alloc_permit/cp/mma_ws/fence）
3. **Exception message 包含 ADR 引用**: 让用户能跳转 ADR 文档理解为何不支持
4. **测试覆盖**: 现有 `tests/integration/tcgen05/test_tcgen05_extended_parse.cpp`
   已验证 throw 模式（per 2026-07-08 commit `718095a`）

### 实现要点

```cpp
// 11 个 handler 统一的 cta_group::2 检查模式（伪代码）
if (instr.cta_group == 2) {
    throw UnsupportedInstructionException(
        "tcgen05.<subop>.cta_group::2",
        "cluster abstraction not yet implemented (ADR-0018); "
        "use cta_group::1 or remove .cta_group qualifier for single-CTA semantics");
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| 11 `processTcgen05Xxx` handler | 修改 | 每个 handler 添加 `cta_group==2` throw |
| `tests/integration/tcgen05/test_tcgen05_extended_parse.cpp` | 已存在 | 验证 throw 模式（不需要新增） |
| 4 个 OpenSpec change artifacts | 更新 | 引用 ADR-0018（之前引用悬空） |

## 后果

### 正面影响

- 用户写 `cta_group::2` 立即得到清晰错误
- 不破坏现有 `cta_group::1` 测试（默认 1，行为不变）
- 与 `wmma.mma.sync.*` 抛异常模式一致

### 负面影响

- 用户必须修改 PTX 代码（去掉 `cta_group::2`）— 但这是正确做法
- 部分 FlashAttention kernel（FA3 等）依赖 `cta_group::2` → 不可在 PTX-EMU 跑（已知限制）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 现有测试因 `cta_group::2` 抛出失败 | 低 | 测试已有 throw 路径验证 | per `test_tcgen05_extended_parse.cpp` 已覆盖 |
| 用户希望"至少部分模拟" | 中 | 用户体验下降 | ADR-0018 文档明确说明 cluster abstraction deferred |
| Cluster abstraction 长期未实现 | 中 | 限制持续存在 | `docs/dev-process/post-tcgen05-roadmap.md` 跟踪 |

## 合规检查

后续相关开发应检查：

- [ ] 新增 tcgen05 handler 必须包含 `cta_group::2` throw 分支
- [ ] 修改 tcgen05 handler 不能删除 `cta_group::2` throw 分支
- [ ] 所有 `tcgen05.*` 测试要么用 `cta_group::1`（或默认），要么 catch `UnsupportedInstructionException`

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-07-12 | 初始版本（formalize the cta_group::2 throw semantics that was implicitly scattered across 11 handlers） | Metis pre-impl review of `fix-tcgen05-commit-wait-group` |

## 参考

- [ADR-0016: Blackwell-only tcgen05](./ADR-0016-blackwell-only-tcgen05.md) — 主范围 ADR（cluster prerequisites 标记为未实现）
- [docs/dev-process/post-tcgen05-roadmap.md](../dev-process/post-tcgen05-roadmap.md) — 跟踪 cluster abstraction 长期 roadmap
- PTX ISA §9.7.16 — `tcgen05.*` 指令族 cta_group 语义
- `openspec/changes/fix-tcgen05-commit-wait-group/` — 本 ADR 第一个引用方