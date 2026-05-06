# PTX-EMU 技能索引

> 本目录技能由 opencode 自动发现并可在调试/开发时加载。

## 调试类

| 技能 | 触发场景 |
|------|---------|
| `ptx-debug` | PTX-EMU 通用调试 — 配置选择、场景化调试方法 |
| `regression-bisect` | 重构后测试回归 — git bisect + 语义对比 + 最小修复 |
| `state-modification-audit` | 状态值异常 — 全项目读写交叉引用审计 |
| `oracle-prompting` | 咨询 Oracle 时 — 防幻觉提示词模板 |

## PTX 仿真类

| 技能 | 触发场景 |
|------|---------|
| `ptx-instruction-pipeline` | 指令执行流水线 — ExecPipe / Handler / PC 管理 / 危险区 |
| `ptx-barrier-mechanism` | 屏障机制 — S_BAR vs S_BAR_WARP_SYNC / Wbar / PC 覆写链 |

## 架构与合规类

| 技能 | 触发场景 |
|------|---------|
| `adr-compliance-check` | ADR 合规检查 — 开发完成后对照 ADR 检查清单验证实现 |

## 语法与解析类

| 技能 | 触发场景 |
|------|---------|
| `ptx-grammar-modification` | ANTLR 解析错误 / 修改 .g4 文件 — 强制 TDD 流程 |
| `ptxir-serialization` | PTX 加载慢 — 二进制序列化与反序列化 |

## 测试类

| 技能 | 触发场景 |
|------|---------|
| `three-mode-testing` | 生成 PTX 测试用例 — 从 CUDA 程序自动生成 |

---

## 技能调用关系

```
ptx-debug (入口)
  ├─ regression-bisect (测试回归 → 找 root cause)
  │   ├─ state-modification-audit (值被覆盖 → 交叉引用)
  │   └─ oracle-prompting (咨询 Oracle → 防幻觉)
  ├─ ptx-instruction-pipeline (PC/ExecPipe 问题)
  │   └─ ptx-barrier-mechanism (屏障问题)
  ├─ ptx-grammar-modification (ANTLR 解析错误)
  └─ cpp-debug (C++ 崩溃/内存)

adr-compliance-check (独立使用)
  └─ 开发完成后 / 代码审查前检查 ADR 合规性
```

**最后更新**: 2026-05-06
