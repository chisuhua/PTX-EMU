## Context

### 当前状态（基于 git verify 实证）

**tcgen05 完整实施链**（git log 验证，已于 `66e3e2e` 之前完成）：

| Commit | Phase | 影响 |
|--------|-------|------|
| `ac1a8d4` | Phase 0 merge | Blackwell tcgen05 基础设施（Fix #1-#12） |
| `35808d6` | Fix #12 | tcgen05.ld/st with TMA + TMEM integration |
| `0213ff1` | Fix #13 | tcgen05.commit/wait async flow |
| `4151268` | Fix #14 | e2e GEMM test + AGENTS sync |
| `79fc236` | - | archive tcgen05 Phase 1-3 |

**已实施组件清单**（按需求追溯）：

- ✅ `tcgen05.mma`（fragment arithmetic）— `src/ptxsim/instructions/wmma.cpp` lines 100+
- ✅ `tcgen05.ld`（TMEM → registers）— `tests/unit/ptx/test_tcgen05_ld_st.cpp`
- ✅ `tcgen05.st`（registers → TMEM）— 同上
- ✅ `tcgen05.commit`（async flow）— `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp`
- ✅ `tcgen05.wait`（async wait）— 同上
- ✅ TMA descriptor parser — `ad527f5` (Fix #5)
- ✅ Per-CTA TMEM — `758edb0` (Fix #6)
- ✅ Cluster arrive/wait — `e513235` (Fix #7)
- ✅ TcQueue (commit-group + wait-aware scheduling) — `c0fa43f` (Fix #8)
- ✅ e2e GEMM kernel — `4151268` (Fix #14) → `tests/e2e/kernel/test_blackwell_gemm.cu`

### README.md 现状（stale 描述）

```markdown
## 已知限制

- **PTX 指令覆盖**：核心 ISA ~67%（详见审计 §3）
- **WMMA / Tensor Core**：是 stub    ← ❌ STALE
- **ANTLR 版本**：4.11.1 完全 vendored
- **CUDA Toolkit**：11.4.4 测试通过   ← ❌ 硬编码
```

**根因分析**：2026-07-04 之前，WMMA 确实是 stub；之后 tcgen05 完整实施，但 README.md 未同步。`fix-cvt-strategy-actual-split` commit `43edf55` 同步了 `docs/audits/debt-audit-2026-07-02.md` 但未触及根 README.md。

### 影响范围

- **README.md line 49-53**（"已知限制" 章节，4 条全部/部分 stale）
- **README.md line 3**（"状态" 描述）
- **README.md line 16**（"CUDA Toolkit" 描述）

### 期望状态

README.md 应反映：
1. **状态**: SIMT v2.0 完成；Blackwell tcgen05 完整实施；H5 规划中
2. **PTX 指令覆盖**: 链接到自动统计（避免硬编码）
3. **WMMA/Tensor Core**: 已完整实现，含 .mma/.ld/.st/.commit/.wait（commit `4151268` Fix #14）
4. **CUDA Toolkit**: 环境自适应（env.sh 自动检测）
5. **文档导航**: 新增 `docs/adr/0016-blackwell-only-tcgen05.md` 链接

## Decisions

### Decision 1: 用"已实现功能"章节代替"已知限制"中的 WMMA 条目

**Options**:
- (A) 直接删除"已知限制"中 WMMA 行（保留 4 条）
- (B) 添加"已实现功能"章节（并列"已知限制"）

**选择**: (B) — 新章节可承载未来扩展（其他已实现功能如 TMA/TMEM/cluster 共 5 项），避免"已知限制"清单"反向膨胀"

### Decision 2: PTX 指令覆盖采用"指向自动统计"模式（不硬编码）

**Options**:
- (A) 删除数字，添加链接到 `docs/audits/`
- (B) 更新为最新数字（手动追踪）
- (C) 引用 `scripts/check-docs-index.sh` 自动生成

**选择**: (A)+(C) 组合 — README.md line 22 已说明"统计自动生成"，但 README 本身仍在硬编码数字 "67%"。本次同步删除硬编码 + 引用 docs/audits/debt-audit-2026-07-02.md

### Decision 3: CUDA Toolkit 描述改为环境自适应

**Options**:
- (A) 更新到实际测试版本（手动追踪）
- (B) 描述环境自适应机制

**选择**: (B) — `env.sh` 用 `NVCC_PATH=$(which nvcc)` 自动检测，硬件允许任意版本。README 应说"支持任意 CUDA 版本（环境自适应）"而非硬编码版本号

### Decision 4: 添加 tcgen05 文档导航（3 个引用）

**引用列表**:
1. `docs/adr/0016-blackwell-only-tcgen05.md` — 架构决策（Blackwell-only 限制 + pre-Blackwell 抛 `UnsupportedInstructionException`）
2. `docs/dev-process/post-tcgen05-roadmap.md` — H5 后续规划
3. `docs/dev-process/lessons-learned.md` — §19 跨模块状态翻译成功案例

**理由**: 这 3 个文档是 tcgen05 实施的"知识三角"，README 链接让新开发者快速了解全貌

### Decision 5: 不重写 README 整体结构

**原则**: 本 change 是**增量同步**，不是重写。`commit 04a62c4` 已完成 SIMT v2.0 重写，本次仅更新 4 个具体章节。

**反例**: 不要因为 README "陈旧" 就重写整个文件。重写是 review 噩梦，难以 review diff。本次仅修改 line 3 + line 16 + line 49-53，diff < 20 行。

### Decision 6: Phase 强制 0-FIRST（避免 lessons-learned §6 反模式）

**Lessons-learned §6 Checklist E 要求**: artifacts FIRST，代码 SECOND

**实施顺序**:
1. Phase 0: 创建 `openspec/changes/sync-readme-after-tcgen05/` 全部 5 个 artifact + commit
2. Phase 1: 修改 `README.md` 4 章节
3. Phase 2: 验证（grep 对比、行数检查、链接验证）
4. Phase 3: 归档（Checklist G）

**Commit 数量预算**: 4 commits（artifacts + 4 章节各一个 + 归档 commit）

## Alternatives Considered

### Alt 1: 不创建 OpenSpec change，直接 commit "docs(readme): sync after tcgen05"

**拒绝理由**: 违反 lessons-learned §6。重大文档同步（如根 README）应通过 OpenSpec change 跟踪，便于审计与回溯。

### Alt 2: 合并到 `implement-wmma-tensor-core-tcgen05` archive 修改

**拒绝理由**: 违反 Checklist G（生命周期约束）。已归档 change 不可 amend；任何后续修补必须新建 change + `Ref:` 链接。

### Alt 3: 不动 README，等下一次 docs-readme-rebuild

**拒绝理由**: 已知限制章节误导新开发者越久，风险越大。docs-readme-rebuild 未来何时触发不确定，立即同步是正确选择。

## Risk Assessment

- **代码风险**: 🟢 零（无代码改动）
- **测试风险**: 🟢 零（无测试改动）
- **构建风险**: 🟢 零（无 CMake/build 改动）
- **文档风险**: 🟡 低（README 修改可能引入链接错误）— 由 Phase 2 验证缓解
- **回退风险**: 🟢 极低（`git revert HEAD` 即可）

**估算总工时**: < 1 小时（含 review 与归档）
