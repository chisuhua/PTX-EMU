# PTX-EMU rdd-workflow 合规差距分析

> **日期**: 2026-07-23
> **状态**: Closed (Archived 2026-07-28)
> **关闭原因**: 所有 🔴 阻塞差距（G1-G3）已在 2026-07-28 前全部解决；roadmap.md、.arch-handoff.json、.rddf/ 均已就绪。该分析已过时，归档至 docs/archive/。
> **触发**: guide-arch Phase 3 (architecture gap analysis)
> **范围**: rdd-workflow 三阶段流程合规性、目录结构、关键交付物

---

## 1. 目标

对比 PTX-EMU 当前状态与 rdd-workflow v2.0 规范（arch → plan → ship 三阶段架构，ADR-0003）的差距，识别缺失组件并排定优先级。

---

## 2. 合规状态矩阵

### 2.1 目录结构

| 路径 | 要求 | 现状 | 差距 |
|------|------|------|:--:|
| `docs/adr/ADR-*.md` | ADR-NNNN-slug.md 格式，≥1 个 | ✅ 21 个，已重命名为 ADR-NNNN 格式 | — |
| `docs/adr/template.md` | ADR 模板 | ✅ | — |
| `docs/adr/README.md` | ADR 索引 | ✅ 已同步 | — |
| `docs/architecture/` | 架构文档 + 差距分析 | ✅ 4 个架构文档；❌ 0 个差距分析 | 🔴 |
| `roadmap.md` (root) | rdd-workflow 格式路线图 | ❌ 不存在（有 docs/roadmap/ 但非标准格式） | 🔴 |
| `docs/roadmap/roadmap-meta.yaml` | roadmap 元数据 | ❌ | 🟡 |
| `.rddf/state/` | 状态持久化目录 | ❌ | 🔴 |
| `.rddf/state/.arch-handoff.json` | arch→plan 交接 | ❌ | 🔴 |
| `.rddf/state/.plan-handoff.json` | plan→ship 交接 | ❌ | 🟡 |
| `.rddf/state/sessions.json` | rddf-session 记录 | ❌ | 🟡 |
| `.rddf/state/.roadmap-state.json` | roadmap 状态 | ❌ | 🟡 |
| `openspec/changes/` | 活跃变更 | ❌ 空（仅 archive） | 🟡 |

### 2.2 流程阶段

| 阶段 | 要求 | 现状 | 差距 |
|------|------|------|:--:|
| **arch** | ADR ≥ 1 + roadmap.md + arch-handoff | ADR: ✅ 21；roadmap: ❌；handoff: ❌ | 🔴 |
| **plan** | scan → propose → deps → plan-done | ❌ 未启动 | 🟡 |
| **ship** | plan → execute → archive → cleanup | ❌ 未启动 | 🟡 |

### 2.3 ADR 命名

| 要求 | 现状 | 差距 |
|------|------|:--:|
| `ADR-NNNN-slug.md` 格式 | ✅ 已在 2026-07-23 会话中全部重命名 | — |
| 内部交叉引用一致性 | ✅ 已同步更新（docs/adr/ + .opencode/ + docs/ + openspec/） | — |

---

## 3. 差距分级

### 🔴 阻塞（arch-done 门控）

| # | 差距 | 影响 | 工作量 |
|---|------|------|:--:|
| G1 | **root 级 roadmap.md 缺失** | arch-done 双重门控之一不满足 | 中 |
| G2 | **.rddf/state/.arch-handoff.json 缺失** | arch→plan 无法交接 | 小 |
| G3 | **.rddf/ 目录不存在** | 工作流状态无处持久化 | 小 |

### 🟡 重要（plan 阶段需要）

| # | 差距 | 影响 | 工作量 |
|---|------|------|:--:|
| G4 | **架构差距分析缺失** | 本文件即为此创建 — 满足要求 | — |
| G5 | **openspec/changes/ 无活跃变更** | plan 阶段开始后自然产生 | — |
| G6 | **roadmap-meta.yaml 缺失** | roadmap 技能需要 | 小 |

### 🟢 远期

| # | 差距 | 影响 | 工作量 |
|---|------|------|:--:|
| G7 | **.plan-handoff.json 缺失** | ship 阶段未启动时不需要 | 小 |
| G8 | **sessions.json 缺失** | 首次 rddf-session 时自动创建 | 自动 |

---

## 4. 修复顺序

```
Phase 1 (本次): G3 → G1 → G2 → arch-done
  .rddf/ dir → roadmap.md → arch-handoff → ✅ arch-done

Phase 2 (后续): G6 → G4 → plan 阶段
  roadmap-meta.yaml → gap analysis → guide-plan

Phase 3 (远期): G5 → plan → ship
  openspec changes → execute → archive
```

---

## 5. 已有优势

PTX-EMU 在以下方面已超出 rdd-workflow 基础要求：

| 方面 | 现状 |
|------|------|
| ADR 覆盖 | 21 个（远超 ≥1 的要求），涵盖异常体系、PC 管理、SIMT、Barrier、Tensor Core、CppTLM 集成 |
| openspec specs | 37 个已归档 spec |
| 架构文档 | 4 个（SIMT v2 43KB + GPGPU-SIM 分析 + sm_90/100 + README） |
| 项目文档导航 | docs/README.md 16 个子目录索引 |
| OpenSpec CLI | 1.4.1（≥1.3.1 要求） |

**核心差距集中在流程治理层**（roadmap + .rddf/ state），而非技术架构层。修复成本低。

---

## 6. 建议

1. **立即**: 创建 root 级 `roadmap.md`（基于 docs/roadmap/ 现有内容 + rdd-workflow 模板）
2. **立即**: 初始化 `.rddf/` 目录 + 写 `.arch-handoff.json`
3. **后续**: 从 arch 阶段进入 plan 阶段（`guide-plan`），开始活跃 change 管理

---

**维护**: PTX-EMU Architecture Team
**日期**: 2026-07-23
