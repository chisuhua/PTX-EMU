# SIMT v2.0 架构升级 - 批准请求

**日期**: 2026-04-09  
**请求**: 批准开始 Phase 1 实施  
**预计时间**: 9 个工作日  
**风险等级**: 中

---

## 📋 执行摘要

PTX-EMU 项目计划升级其 SIMT（Single Instruction Multiple Thread）架构至 v2.0 版本，以精确模拟 NVIDIA Hopper/Blackwell 架构的控制流管理行为。

本次升级是基于：
- ✅ 对 SIMT 学术文献的深入研究（2024 arXiv 论文，GPGPU-Sim 实现）
- ✅ 对 NVIDIA PTX ISA 9.1 规范的详细分析
- ✅ 当前架构在 divergent barrier 场景下的局限性分析

**关键改进**：
1. **Post-Dominator-based CFG 分析**: 精确计算 branch reconvergence 点
2. **Per-Thread PC**: 支持 Hopper 的 Independent Thread Scheduling
3. **SIMT Stack**: 显式跟踪分支和收敛状态
4. **Barrier 验证**: Debug 模式下验证 memory fence 语义

---

## 📚 已交付文档

### 1. 架构设计文档

**文件**: [`docs/architecture-upgrade/SIMT-ARCHITECTURE-V2.md`](docs/architecture-upgrade/SIMT-ARCHITECTURE-V2.md)

**内容**:
- 理论基础（PTX ISA 规范，Post-Dominator 理论，Barrier 语义）
- 架构设计（SIMT Stack, Per-Thread PC, CFG 分析模块，Barrier 机制）
- 执行流程（Branch 处理，Reconvergence 检查，Barrier 执行）
- 与 GPGPU-Sim 的对比
- 设计决策与权衡（3 个关键决策均有详细理由）
- 实施路线图

**大小**: 约 900 行，包含详细的代码示例和流程图

---

### 2. 实施计划文档

**文件**: [`docs/architecture-upgrade/SIMT-IMPLEMENTATION-PLAN.md`](docs/architecture-upgrade/SIMT-IMPLEMENTATION-PLAN.md)

**内容**:
- Phase 1: CFG 分析基础设施（2 天）
  - 基本块识别，控制流图构建，Post-Dominator 计算
- Phase 2: SIMT Stack 实现（2 天）
  - Stack entry 数据结构，push/pop 操作，reconvergence 检查
- Phase 3: Per-Thread PC 集成（2 天）
  - ThreadState 重构，WarpContext 更新，调度器适配
- Phase 4: Barrier 增强（1 天）
  - Wbar 与 SIMT stack 集成，Memory fence 验证
- Phase 5: 测试与验证（2 天）
  - 单元测试，集成测试，性能基准

**每个 Phase 包含**:
- 详细任务分解（文件级，小时级估算）
- 关键代码示例
- 验收标准（具体测试命令）
- 回滚策略（git 命令）

**大小**: 约 700 行

---

### 3. 进度追踪文档

**文件**: [`docs/architecture-upgrade/PROGRESS.md`](docs/architecture-upgrade/PROGRESS.md)

**内容**:
- 总体进度追踪
- 已完成工作清单
- 待批准事项
- 待办任务（Phase 1-5 详细清单）
- 当前状态和需要决策的问题
- 时间线
- 历史进度参考（v1.0 Stage 1-3）

---

## 🔑 关键设计决策（已确定）

### 决策 1: CFG 分析策略

**选择**: ✅ 完整 Post-Dominator 计算

**选项对比**:
- A: 完整 CFG 分析 (准确度高，复杂度中) ← **选择**
- B: 简化 Heuristic (准确度中，复杂度低)

**理由**:
- 简化 heuristic 可能在边缘场景失败
- 完整 CFG 是一次性开销（kernel 加载时），无运行时开销
- 符合 "匹配真实硬件" 的目标

---

### 决策 2: Barrier 验证策略

**选择**: ✅ DEBUG Only

**选项对比**:
- A: Always On (完整保护，性能开销 5-10%)
- B: Debug Only (开发友好，Production 无开销) ← **选择**
- C: 完全禁用 (最快，无保护)

**理由**:
- Barrier 语义错误是开发期 bug，不应出现在 production
- PTX_DEBUG 模式已足够覆盖开发和测试场景
- 性能敏感场景不应承担额外开销

---

### 决策 3: Per-Thread PC 实现

**选择**: ✅ 显式 ThreadState 数组

**选项对比**:
- A: 显式 ThreadState 数组 (内存占用大，访问快) ← **选择**
- B: 隐式通过 SIMT stack (内存占用小，访问稍慢)

**理由**:
- Hopper/Blackwell 支持 Independent Thread Scheduling
- 显式数组更容易调试和验证
- 内存开销可接受 (32 threads × sizeof(ThreadState) ≈ 2KB per warp)

---

## 📊 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| CFG 构建错误 | 中 | 高 | 单元测试覆盖所有分支模式 |
| 性能回归 > 5% | 低 | 中 | 每 Phase 后 benchmark 对比 |
| 测试失败增加 | 中 | 高 | 详细日志，快速定位 |
| 需要大规模重构 | 低 | 高 | 分 Phase 开发，每阶段可回滚 |

---

## 🎯 验收标准

### 功能验收
- [ ] `test_warp_divergence` 3/3 PASS
- [ ] `test_syncthreads` 3/3 PASS
- [ ] `test_simt_reconvergence` (new) 5/5 PASS
- [ ] `test_barrier_semantics` (new) 3/3 PASS

### 性能验收
- [ ] 无 regression (> 5% 视为 regression)
- [ ] CFG 分析开销 < 1ms per kernel
- [ ] SIMT stack 操作开销 < 1 cycle per instruction

### 代码质量
- [ ] LSP diagnostics clean
- [ ] 所有新增代码通过 clang-tidy
- [ ] 单元测试覆盖率 > 80%

---

## 📅 时间线

```
2026-04-09: 文档评审 (当前)
2026-04-10: Phase 1 (CFG 分析)
2026-04-12: Phase 2 (SIMT Stack)
2026-04-15: Phase 3 (Per-Thread PC)
2026-04-16: Phase 4 (Barrier)
2026-04-17: Phase 5 (Testing)
2026-04-18: 架构评审与合并
```

**总计**: 9 个工作日

---

## 🔧 回滚策略

每个 Phase 在独立分支开发，可随时回滚：

```bash
# 如果 Phase N 失败
git checkout main
git branch -D feature/simt-v2-phaseN

# 或使用 worktree 隔离
./zcf_git worktree add ../simt-v2-dev -b feature/simt-v2
cd ../simt-v2-dev
# ... development ...

# 如果需要回滚
cd /workspace/project/PTX-EMU
rm -rf ../simt-v2-dev  # 删除 worktree
git branch -D feature/simt-v2
```

---

## ✅ 批准检查清单

在开始编码前，请确认：

- [x] **架构文档审查**: SIMT-ARCHITECTURE-V2.md 已被审阅和批准
- [x] **实施计划审查**: SIMT-IMPLEMENTATION-PLAN.md 已被审阅和批准
- [x] **备份分支**: 已创建 `backup/pre-simt-v2` 分支
- [ ] **测试基线**: 当前所有测试已通过（待确认）
- [ ] **性能基线**: Benchmark 已运行并记录（待确认）
- [ ] **时间承诺**: 确认有 9 个工作日的开发时间
- [ ] **回滚计划**: 理解并准备好回滚策略

---

## 📝 批准签名

**架构评审委员会**:

- [ ] _________________ (主席)
- [ ] _________________ (技术负责人)
- [ ] _________________ (项目经理)

**批准日期**: _________________

**预计开始日期**: _________________

**预计完成日期**: _________________

---

## 📋 下一步行动

### 如果批准：

1. 创建 `feature/simt-v2` 主分支
2. 为每个 Phase 创建子分支
3. 开始 Phase 1 实施
4. 每日更新进度到 PROGRESS.md

### 如果不批准：

1. 记录不批准原因
2. 修改设计或实施计划
3. 重新提交评审

---

## 🔗 相关文档

- [架构设计文档](docs/architecture-upgrade/SIMT-ARCHITECTURE-V2.md)
- [实施计划](docs/architecture-upgrade/SIMT-IMPLEMENTATION-PLAN.md)
- [进度追踪](docs/architecture-upgrade/PROGRESS.md)
- [备份分支](https://github.com/your-org/PTX-EMU/tree/backup/pre-simt-v2)

---

**文档结束** - 等待架构评审委员会批准
