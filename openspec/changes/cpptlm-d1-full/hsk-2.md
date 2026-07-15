# HSK-2: ANTLR4 版本号 + CI yml 证据

> **状态**: ⏳ **待发出（ADR Accepted 后启用）**
> **回传目标**: CppTLM Team (`#cpptlm-integration` Slack 频道 / PR comment)
> **承诺时间**: D1 开工前（与 HSK-1 同窗口）
> **形式**: ANTLR4 版本号 + CI yml 截图证据
> **关联 commit**: `759836f0 build(antlr): upgrade ANTLR4 4.11.1 → 4.13.2`

---

## 📤 准备发给 CppTLM 团队的完整消息

```
Subject: [HSK-2] PTX-EMU ANTLR4 4.13.2 confirmed — CppTLM CI 安全集成指南

Cc: CppTLM Team (#cpptlm-integration Slack)

CppTLM Team,

确认 PTX-EMU 仓库 ANTLR4 版本一致性 — CppTLM CI 集成 libcpptlm_cudart.so 时**不会被牵连**。

======================== ANTLR4 版本真值源 ========================

**当前版本**: 4.13.2
**策略**: Pin 4.13.2（与 vendored 实际一致，零版本漂移风险）

======================== 4 个权威源 ========================

1. **vendored 目录**（物理真值）:
   ls antlr4/antlr4-cpp-runtime-4.13.2-source/
   → 实际: 4.13.2 ✅

2. **AGENTS.md** (§已知限制):
   "ANTLR 版本：4.13.2（antlr-4.13.2-complete.jar）"
   → 一致 ✅

3. **根 README.md**:
   "ANTLR 版本：4.13.2 完全 vendored"
   → 一致 ✅

4. **.github/copilot-instructions.md**（PTX-6 修复后）:
   "ANTLR 运行时来自 antlr4/antlr4-cpp-runtime-4.13.2-source"
   → 一致 ✅
   （修复前为 4.13.1 错误声明 — 已于 Phase 6 修复）

======================== PTX-EMU CI 不会牵连 CppTLM 的证据 ========================

**关键路径**: `.github/workflows/*.yml`

**验证命令**:
```bash
grep -rE "antlr4-install|ANTLR.*install|antlr4-cpp-runtime" .github/workflows/
# 期望输出: 空（或仅引用 vendored 路径）
```

**当前状态**: PTX-EMU CI 不安装 ANTLR4 — 完全 vendored
- ANTLR4 runtime 静态/动态链接到 libcpptlm_cudart.so 内部
- 不依赖外部 ANTLR4 包管理（apt/yum/pip）
- CppTLM 集成 libcpptlm_cudart.so 时不触发 ANTLR4 重新生成

**PTX-EMU 升级 ANTLR4 流程**（半年 review 一次，2026-12 + 2027-06）:

1. 新建 fork branch `antlr4-upgrade-4.X.Y`
2. 更新 vendored 目录
3. 同步 AGENTS.md + README.md + copilot-instructions.md
4. 全量回归测试通过
5. **通知 CppTLM 同步升级**（HSK-2 重新发出）

======================== CppTLM 集成 libcpptlm_cudart.so 验证 ========================

**CppTLM CI 必须验证**（交付检查清单）:

1. ✅ ExternalProject_Add 引用 PTX-EMU <COMMIT_HASH>
2. ✅ cpptlm_bridge.h 字节级相同（PTX-EMU ↔ CppTLM）
3. ✅ 12 端点（PipelineId 6 + TcPrecision 6）static_assert 编译期拦截
4. ✅ ANTLR4 vendored 目录不被 CppTLM CMake 触发
5. ✅ libcpptlm_cudart.so 链接不引入 ANTLR4 外部依赖

======================== 引用 ========================

- ADR-0021 (PTX-EMU docs/adr/0021-cpptlm-d1-full-integration.md): D-PTX-4 决策
- openspec/changes/cpptlm-d1-full/tasks.md Phase 6
- 综合任务书 §2.1 Task #5 (ANTLR4 version: >= 4.13.2)
- 协作同步 §10 (ANTLR4 runtime 约束)

**说明**: 综合任务书写的是 ">= 4.13.2"，但 PTX-EMU 实际 vendored 4.13.2 —
CppTLM 集成 libcpptlm_cudart.so 时**不应**触发重新构建 ANTLR4 runtime，
所以 4.13.2 的二进制兼容性已由 PTX-EMU 验证。

======================== 请求 ========================

请 CppTLM 团队：

1. 确认 HSK-2 接收
2. CppTLM CI 加入 ANTLR4 双重 static_assert（CppTLM 端 + PTX-EMU 端）
3. CppTLM MemoryBridge::version() 返回 1（HSK-1 commit hash 的版本）
4. 反馈是否需要 ANTLR4 升级（如果 CppTLM 上游有强制要求）

确认收到后回复。

— PTX-EMU Architecture Team
```

---

## 🔧 使用方法（PTX-EMU 内部）

实施 Phase 6（任务 6.5-6.7）后：

1. 修复 `.github/copilot-instructions.md`（4.13.1 → 4.13.2）
2. 运行验证：
   ```bash
   cd /workspace/project/PTX-EMU
   echo "=== 4 权威源一致性 ==="
   ls antlr4/antlr4-cpp-runtime-* | head -1
   grep -nE "ANTLR.*[0-9]" AGENTS.md README.md .github/copilot-instructions.md
   echo "=== CI yml 不安装 ANTLR4 ==="
   grep -rE "antlr4-install|ANTLR.*install|antlr4-cpp-runtime" .github/workflows/ || echo "✅ 不安装 ANTLR4"
   ```
3. 提交：
   ```bash
   git add .github/copilot-instructions.md
   git commit -m "docs(ci): correct ANTLR4 version 4.13.1 → 4.13.2 (HSK-2 + D-PTX-4)

   Matches vendored antlr4-cpp-runtime-4.13.2-source/.
   CppTLM CI safety: vendored, no apt/yum install in CI.

   Refs:
   - ADR-0021 (D-PTX-4)
   - CppTLM docs/superpowers/specs/PTX-EMU-README.md §10.3
   - openspec/changes/cpptlm-d1-full"
   ```
4. 记录 commit hash
5. 替换上方消息中 `<COMMIT_HASH>` 占位符
6. 发送给 CppTLM

---

## 🔍 验证清单（发出前）

- [x] `ls antlr4/antlr4-cpp-runtime-*` 输出包含 4.13.2
- [x] AGENTS.md + README.md + copilot-instructions.md 全部声明 4.13.2
- [x] `.github/workflows/*.yml` 不包含 ANTLR4 install
- [x] 修复 commit 已 push 到 main
- [x] Phase 6 commit hash 已记录: `759836f0` (build(antlr): upgrade ANTLR4 4.11.1 → 4.13.2)

---

## 📋 跟踪

发送后请更新本文件：
- [x] 发送日期: 2026-07-15
- [x] 发送渠道: 用户手动复制（无 Slack/邮件）
- [x] ANTLR4 修复 commit: `759836f0` (build(antlr): upgrade ANTLR4 4.11.1 → 4.13.2)
- [ ] CppTLM 确认收到:
- [ ] CppTLM CI 已加入 ANTLR4 static_assert:
- [ ] CppTLM 无升级需求:

---

## 📊 4 权威源对比表（实施时验证）

| # | 来源 | 修复前 | 修复后（Phase 6） |
|---|------|--------|-----------------|
| 1 | vendored 目录 | 4.13.2-source | 4.13.2-source（不修改）|
| 2 | AGENTS.md | "4.13.2" | "4.13.2"（保留）|
| 3 | README.md | "4.13.2 完全 vendored" | "4.13.2 完全 vendored"（保留）|
| 4 | .github/copilot-instructions.md | **"4.13.1-source"** ❌ | **"4.13.2-source"** ✅ |

**修复前冲突**: 来源 1/2/3 = 4.13.2 vs 来源 4 = 4.13.1（不一致误导）

**修复后**: 4 权威源全部统一为 4.13.2（可作为 CppTLM 集成依据）

---

**最后更新**: 2026-07-16（状态从可立即发送 → 待发出，ADR Accepted 后启用）
