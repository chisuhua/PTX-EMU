# HSK-2 草稿：ANTLR4 4.13.2 验证 + CI yml 不安装 ANTLR

> **生成时间**: 2026-07-16  
> **来源**: PTX-EMU `cpptlm-d1-full` change（OpenSpec / ADR-0021 §D-PTX-4）  
> **目标**: CppTLM 团队 (`chisuhua/CppTLM`)  
> **上下文**: F12b-LD MemoryBridge 集成第二阶段交付

---

## 📧 Send-to (待用户填写)

- **C.C.**: CppTLM CI 团队
- **Channel**: GitHub issue `@chisuhua/CppTLM` + Slack `#cpptlm-ptxemu-bridge`
- **Subject**: `[CppTLM D1-Full] HSK-2: ANTLR4 4.13.2 版本契约 + CppTLM CI 信任 PTX-EMU vendored ANTLR4`

---

## 📋 Message Body

### 1. 状态锁定

PTX-EMU 端 ANTLR4 4.13.2 升级已完成，4 权威源验证一致；CppTLM CI 不应再安装/重装 ANTLR4。

### 2. ANTLR4 版本契约：4.13.2

CppTLM 端在 MemoryBridge 项目中使用 ANTLR4 时，**必须**锁到 `4.13.2`，**不可**跟随 PTX-EMU 或独立升级。理由：

1. **PTX-EMU vendored 全集**: `antlr4/antlr4-cpp-runtime-4.13.2-source/`
2. **Cpptlm CI 双重静态断言** (`ADB-0021:189`): PipelineId 6 + TcPrecision 6 双方一致
3. **不传播 ANTLR install**: 若 CppTLM CI 安装独立 ANTLR4，可能引入 graph-level 双重符号冲突

### 3. 4 权威源验证矩阵（PTX-EMU 端）

```
┌─────────────────────────────────┬───────────────────┬──────────┐
│ 权威源                            │ 报告版本           │ 验证方式  │
├─────────────────────────────────┼───────────────────┼──────────┤
│ 1. AGENTS.md §已知限制             │ "ANTLR 4.13.2"    │ grep     │
│ 2. README.md                     │ "ANTLR 4.13.2 fully vendored" │ grep │
│ 3. .github/copilot-instructions.md│ "antlr4-cpp-runtime-4.13.2-source" │ grep │
│ 4. 实际 vendored 目录             │ 4.13.2-source     │ ls       │
└─────────────────────────────────┴───────────────────┴──────────┘
```

**所有 4 源一致为 4.13.2**（commits `9c992e26`（hsk-2 修复）+ `741da807`（stale references 修复）落地）。

### 4. PTX-EMU 端的修改与新增

| 文件 | 修改 | commit |
|------|------|--------|
| `AGENTS.md` §已知限制 | 已升级 | pre-existing |
| 根 `README.md` | "4.13.2 fully vendored" 已声明 | pre-existing |
| `.github/copilot-instructions.md` | "4.13.1" 修正为 "4.13.2" | `741da807` |
| `antlr4/antlr4-cpp-runtime-4.13.2-source/` | 已升级至 4.13.2 | pre-existing |

### 5. CppTLM CI 必须满足的 5 条

```bash
# 1. find_package(ANTLR) 不应该触发下载
grep -E "find_package.*ANTLR|antlr4_runtime" CMakeLists.txt
# 期望：无 ANTLR4 install or find_package 调用（PTX-EMU vendored 是 single source）

# 2. libcpptlm_cudart.so 不应该链接 ANTLR4
ldd build/lib/libcpptlm_cudart.so 2>&1 | grep -E "antlr" || echo "✅ 不依赖 ANTLR4"
# 期望：✅ 不依赖（避免 graph-level 双重符号）

# 3. CppTLM CI 不应该从 apt/pip/npm 安装 ANTLR4
grep -rE "antlr4-install|ANTLR.*install|antlr4-cpp-runtime" .github/workflows/
# 期望：空输出（PTX-EMU vendored 已隔离 ANTLR4）

# 4. PipelineId 双向 static_assert
static_assert(static_cast<uint32_t>(CppTLM::PipelineId::P0_INT_FP32) == 0, "PTX-EMU drift");
static_assert(static_cast<uint32_t>(CppTLM::PipelineId::P4_TC) == 5, "PTX-EMU drift");

# 5. TcPrecision 双向 static_assert
static_assert(static_cast<uint32_t>(CppTLM::TcPrecision::FP4) == 0, "PTX-EMU drift");
static_assert(static_cast<uint32_t>(CppTLM::TcPrecision::TF32) == 5, "PTX-EMU drift");
```

### 6. 升级路径（如未来确实需要升级）

```
Pipeline:
  Step 1: PTX-EMU 端开 fork branch `antlr4-upgrade-4.X.Y`
  Step 2: 更新 vendored 目录
  Step 3: 同步 4 权威源（AGENTS.md + README.md + copilot-instructions.md + 目录名）
  Step 4: 全量回归测试 PASS
  Step 5: 通知 CppTLM 同步升级（HSK-2 重新发出新 commit hash）
  Step 6: CppTLM 端 rebase + static_assert 重新对账
```

**半年 review**: 2026-12 + 2027-06 各一次；触发条件：上游 ANTLR4 安全修复 或 PTX 语法关键 bug 修复。

---

## 📎 交叉引用

- PTX-EMU 端 ADR-0021 §D-PTX-4: https://github.com/chisuhua/PTX-EMU/blob/380a8b6a/docs/adr/ADR-0021-cpptlm-d1-full-integration.md#决策-d-ptx-4-antlr4-版本策略
- PTX-EMU 端 commit `9c992e26`（hsk-2 stale references）: https://github.com/chisuhua/PTX-EMU/commit/9c992e26
- 协作同步: https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-14-ptxemu-comprehensive-modification-plan.md §2.1 Task #5
- 姊妹 change 设计: https://github.com/chisuhua/PTX-EMU/blob/main/openspec/changes/cpptlm-phase8b-injection-points/design.md §2

---

## ⏱️ 等待 CppTLM 端的反馈

- **期望反馈类型**: PR → `chisuhua/CppTLM` + 静态断言测试 PASS 报告
- **本 PR 应包含**:
  - `libcpptlm_cudart.so` 不依赖 ANTLR4 链接（`ldd` 验证）
  - 12 端点双向 static_assert 落地
  - `cmake --build --target cpptlm_bridge` 通过 PTX-EMU vendored ANTLR4
- **不在本 PR 范围**:
  - APT/PIP 安装 ANTLR4 — 禁止
  - 强行升级 ANTLR4 4.X.Y — 必须先 PTX-EMU 端发起

---

**发送方**: PTX-EMU Architecture Team  
**ADR-0021 状态**: Active (2026-07-16)  
**本 HSK 版本**: HSK-2 v1  
**签发**: ⏳ 待 PTX-EMU Architecture Team 发出
