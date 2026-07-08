# tcgen05 Documentation Sync + Archive (Final Documentation Update)

> **架构依据**: [ADR-0016](../../../docs/adr/0016-blackwell-only-tcgen05.md) Accepted
> **前置 changes**(全部必须 archive 后才能执行本 change):
>   - `archive/2026-07-06-implement-tcgen05-syntax-ir` (Change-1, ✅ archived)
>   - `archive/2026-07-06-extend-blackwell-tcgen05-infra` (Change-2, ✅ archived)
>   - `archive/2026-07-07-fix-tcgen05-grammar-mr3` (Change-3a, ✅ archived 2026-07-07)
>   - `archive/2026-07-07-implement-tcgen05-handlers-core` (Change-3b, ✅ archived 2026-07-07 @ `df6dde7`)
>   - `implement-tcgen05-handlers-extended` (Change-3d, pending,可选 — 若未实施,本 change 仍可执行)
> **设计时教训**: `ptx-lessons-learned` §3(分 Phase commit)+ §6(artifacts-first)+ Checklist G(OpenSpec lifecycle)+ Checklist I(重大功能交付清单)

## Why

Change-1(grammar + IR) + Change-2(infra 审计) + Change-3a(grammar fix) + Change-3b(core handlers) + Change-3d(extended handlers,可选) 全部 archive 后,**用户和未来 maintainer 看到的状态与实际代码不一致**:

1. **根 `AGENTS.md` 已知限制表**仍标注 "pre-Blackwell tcgen05 永久抛异常" → 不准确,handler 已实施
2. **`src/grammar/AGENTS.md`** 仍描述旧 wmma 路径 → 需标注 tcgen05 替代
3. **`src/ptxsim/instructions/AGENTS.md`** 仍主要描述 wmma.cpp → 需标注 tcgen05.cpp
4. **`docs/adr/0016-blackwell-only-tcgen05.md`** 更新记录缺失 Phase 1-3 archive 引用
5. **`docs/dev-process/lessons-learned.md`** 缺 §24 新案例(per `ptx-lessons-learned` §24 重大功能交付清单)
6. **OpenSpec `openspec/specs/`** 缺最终 spec 状态(各 change 的 spec delta 未 publish)

本 change 是 4-change 路线图的**最终 documentation sync**,无代码改动,只同步文档 + archive 整理。

## What Changes

### 修改

| 文件 | 范围 |
|------|------|
| `AGENTS.md`(根) | 更新已知限制表:pre-Blackwell → 标注"handler 已实现,详见 ADR-0016" |
| `src/grammar/AGENTS.md` | 更新 lexer/parser 规则说明,标注 tcgen05 替代 wmma |
| `src/ptxsim/instructions/AGENTS.md` | 更新目录说明,标注 `tcgen05.cpp`(5 core handler + 可选 extended) |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 在 "更新记录" 追加 5-6 个 archive commit 引用 |
| `docs/dev-process/lessons-learned.md` | 追加 §24:重大功能交付清单(per Checklist I) |
| `docs/audits/` | 引用 Change-2 审计报告路径(若已 archive) |
| `openspec/specs/` | 验证 Change-1 spec delta 已 publish(tcgen05-{grammar,ir-types,parse-tests}) |

### 不修改(范围外)

- ❌ 不修改任何源码(handler 实现已完成)
- ❌ 不修改任何测试(测试已就位)
- ❌ 不修改 OpenSpec proposal/design/tasks(已 archive)
- ❌ 不删除 `wmma.cpp`(Change-4 scope)
- ❌ 不实现新 handler(已通过 Change-3b/d 实施)

## Non-Goals

### 显式拒绝

- ❌ 不修改源码、测试、proposal(纯文档 sync)
- ❌ 不删除 wmma.cpp(Change-4 scope)
- ❌ 不实现新功能(per Checklist I "重大功能交付" = 文档同步,不包含新代码)
- ❌ 不修改 `docs/audits/` 内容(Change-2 产出,本 change 仅引用)
- ❌ 不添加新 ADR(per `ptx-lessons-learned` §22 ADR 追加应与代码 commit 同步,handler 实施时已应追加)

### 范围限制

- 仅文档同步 + 最终验证
- 仅 1 个 commit(纯文档,无需分 Phase)

## Goals

### Phase 1: 文档同步(1 commit)

1. **根 `AGENTS.md`**:更新已知限制表
   - "pre-Blackwell tcgen05 永久抛异常" → "pre-Blackwell WMMA 永久抛异常(per ADR-0016);Blackwell tcgen05 handler 已实现(per Change-1/3a/3b)"
2. **`src/grammar/AGENTS.md`**:更新 lexer/parser 规则说明
   - 标注 `tcgen05Inst` 替代 `wmmaInst`
3. **`src/ptxsim/instructions/AGENTS.md`**:更新目录说明
   - 标注 `tcgen05.cpp` 存在
   - 标注 `wmma.cpp` 保留 pre-Blackwell 路径
4. **`docs/adr/0016-blackwell-only-tcgen05.md`**:追加 "更新记录" 段落
   - 2026-07-XX:tokens: tcgen05 added
   - 2026-07-XX:tcgen05 core handlers implemented
   - 2026-07-XX:tcgen05 extended handlers (optional)
5. **`docs/dev-process/lessons-learned.md`**:追加 §24
   - 标题: "重大功能交付清单" (per Checklist I)
   - 内容: 4-change 路线图 + Metis 审查关键教训 + 4-change 拆分的价值
6. **`openspec/specs/`**:验证 3 个 spec delta 已 publish(无新 spec,只是验证)

### Phase 2: 验证(本 change 范围内)

1. 跑 `cmake --build build` 验证编译通过
2. 跑 `cd build && ctest --output-on-failure` 全量验证
3. 跑 `./tests/ptx/test_all_ptx.sh` 验证
4. 跑 `cd build && ctest -L "e2e" -V` 验证 E2E
5. 跑 `openspec list` 验证所有 active change 已 archive

### Phase 3: Archive(per Checklist G)

1. 跑 `openspec archive tcgen05-docs-and-archive --yes`
2. 跑 `cd build && ctest --output-on-failure` 最终验证
3. commit archive 目录

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `tcgen05-grammar`:spec 修订(标记为 "implemented" 状态)
- `tcgen05-ir-types`:spec 修订(标记为 "implemented" 状态)
- `tcgen05-parse-tests`:spec 修订(标记为 "implemented" 状态)

## Impact

### 影响的文件(预计,纯文档)

| 文件 | 变更类型 | LoC 估计 |
|---|---|---|
| `AGENTS.md` | 修改(已知限制表) | +20 |
| `src/grammar/AGENTS.md` | 修改(规则说明) | +20 |
| `src/ptxsim/instructions/AGENTS.md` | 修改(目录说明) | +20 |
| `docs/adr/0016-blackwell-only-tcgen05.md` | 修改(更新记录) | +30 |
| `docs/dev-process/lessons-learned.md` | 追加 §24 | +100 |
| `openspec/changes/tcgen05-docs-and-archive/` | 新增(proposal + design + tasks) | +500 |
| **总计** | | **+690** |

### 影响的依赖

- 无(纯文档,无新代码依赖)

### 不影响的依赖

- 任何源码文件
- 任何测试文件
- 任何 OpenSpec proposal(已 archive)

### 影响的文档

- 根 `AGENTS.md`
- `src/grammar/AGENTS.md`
- `src/ptxsim/instructions/AGENTS.md`
- `docs/adr/0016-blackwell-only-tcgen05.md`
- `docs/dev-process/lessons-learned.md`
- `openspec/specs/`(验证,不修改)

## Design-Time Checklist (Lessons-Learned)

### Checklist I: 重大功能交付清单(必填)

- [x] 代码 + 单元测试 + e2e + 根 README 同步(per Checklist I 第 1 项)
- [x] 实施阶段:根 AGENTS.md "状态" 章节随代码同步更新(不延后到 archive)— 本 change 修复
- [x] Archive commit 前 grep 验证:`grep -n "stub\|TODO\|FIXME"` 应为空(本 change 不需验证,handler 已实施)
- [x] 任何 feat-*/implement-* change archive 前必跑本 checklist
- [x] 新 sync-* / fix-* change 处理已归档案例:通过 Ref 链接 + 不 amend(本 change 不需)

### Checklist G: OpenSpec lifecycle(必填)

- [x] 验证 4-change 路线图全部 archive(per `openspec list`)
- [x] 已归档的 change 不 amend(本 change 不修改已 archive artifacts)
- [x] 本 change archive 后,`openspec/specs/` 是最终 spec 状态

### 多 Phase 推进

- [x] 仅 1 个 commit(纯文档,无需分 Phase)
- [x] 基线 worktree 计划:`.worktrees/baseline-tcgen05-docs`(per `ptx-lessons-learned` §4)
- [x] 失败处理策略:文档 regression → 立即 revert + 修订

### 文档同步(per Checklist I)

- [x] 根 AGENTS.md 同步项已列出
- [x] ADR 追加段落已规划
- [x] lessons-learned §24 预留

### 实施前必跑(per `ptx-lessons-learned` §7)

- [ ] **Metis pre-implementation review**:验证文档 sync 清单完整性
- [ ] 跑 `openspec list` 确认所有前置 change 已 archive
- [ ] 跑 `cd build && ctest --output-on-failure` 确认 baseline 全绿
- [ ] 跑 `./tests/ptx/test_all_ptx.sh` 确认 13 fixtures 全 PASS

## 跨 Change 依赖

| 上游 | 本 change | 下游 |
|------|----------|------|
| Change-1, 2, 3a, 3b(全部必须 archive) | **tcgen05-docs-and-archive** | (无 — 这是 4-change 路线图终点) |
| Change-3d(extended handlers,可选) | | |

- **5 个前置 change 必须全部 archive**(`openspec list` 验证)
- **本 change 完成后,4-change 路线图结束**
- **下游**:无新 change,仅 Change-4 (cleanup wmma) 可作为后续 5th change(独立)
- **不依赖** Change-3d(若 extended handler 未实施,本 change 仍可执行)

## 本 change 特有设计决策(per Metis F.2)

**决策 D1:文档 sync 优先级**
- 优先:`AGENTS.md` + `docs/adr/0016-*.md` + `docs/dev-process/lessons-learned.md`(per Checklist I)
- 次要:各 `src/**/AGENTS.md`(per-change 局部)
- 拒绝:为每个 Phase 都新建 lessons-learned 章节(过度文档化,应每 change 1 节)

**决策 D2:lessons-learned §24 内容**
- 必含:4-change 路线图回顾 + Metis 审查关键教训 + Change 拆分价值
- 拒绝:逐 commit 复述(冗余,git log 已记录)
- 风格:per `ptx-lessons-learned` 模板("现象/教训/检查工具/真实案例")

**决策 D3:archive 策略**
- 1 个 commit(per `ptx-lessons-learned` Checklist G)
- 包含:本 change 的 proposal/design/tasks 全部 + `openspec archive` 生成的 archive 目录
- 拒绝:为 docs + archive 拆 2 commit(过度拆分,docs 与 archive 强耦合)

**决策 D4:是否本 change 完成时启动 Change-4**
- 拒绝:本 change 不提议 Change-4(已在 `cleanup-wmma-namespace/proposal.md` 单独 propose)
- 理由:Change-4 是独立 scope(删除 wmma 路径),不应混入 docs sync
