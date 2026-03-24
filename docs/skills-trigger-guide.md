# PTX-EMU 项目技能触发指南

> **重要**: 项目级技能（位于 `docs/skills/`）需要正确的触发机制才能被 Agent 加载使用。

---

## 📦 项目技能列表

### 1. ptx-grammar-modification（PTX 语法修复）

**触发方式**: **AGENTS.md 自动触发**

**触发条件**（满足任一即🛑停止并加载技能）:

| 触发场景 | 关键词/错误 | 自动行动 |
|---------|------------|---------|
| 用户请求修复解析错误 | "PTX 解析错误", "语法错误", "ANTLR 错误" | 🛑 → 加载技能 → 运行测试 |
| ANTLR 解析错误 | `no viable alternative at input` | 🛑 → 加载技能 → 运行测试 |
| 意外 Token | `mismatched input 'X' expecting Y` | 🛑 → 加载技能 → 运行测试 |
| 修改语法文件 | 改动 `src/grammar/*.g4` | 🛑 → 加载技能 → 运行测试 |
| 解析阶段崩溃 | `Segmentation fault` 在 parser 阶段 | 🛑 → 加载技能 → 运行测试 |

**触发机制**:
```
AGENTS.md 中定义了强制流程
  ↓
用户提到触发关键词
  ↓
Agent 识别为 PTX 语法问题
  ↓
🛑 停止当前操作
  ↓
加载技能：docs/skills/ptx-grammar-modification.md
  ↓
遵循技能流程执行
```

---

### 2. ptx-debug（运行时调试）

**触发方式**: **关键词触发 + 手动加载**

#### 方式 1: 自动触发（推荐）

当用户提到以下关键词时，**自动加载 ptx-debug 技能**：

**问题类型触发**:
- "测试失败", "ctest 不通过", "单元测试失败"
- "程序崩溃", "segfault", "SIGSEGV", "core dumped"
- "内存错误", "非法访问", "越界", "内存泄漏"
- "指令错误", "执行结果不对", "结果不对"
- "性能慢", "性能优化", "benchmark"
- "调试这个", "分析一下问题", "查看日志"

**场景触发**:
- 运行测试后失败
- 查看日志文件
- 使用调试配置
- 分析内存访问
- 跟踪指令执行

**触发示例**:
```
用户："test_memory 测试失败了"
→ 自动加载 ptx-debug 技能
→ 选择 debug_config.ini
→ 运行测试收集日志

用户："程序崩溃了，帮我分析"
→ 自动加载 ptx-debug 技能
→ 选择 verbose_trace.ini
→ 收集崩溃前日志

用户："这个内存访问有问题"
→ 自动加载 ptx-debug 技能
→ 选择 memory_debug.ini
→ 跟踪内存操作
```

#### 方式 2: 手动加载

**明确要求加载技能**:
```
"请加载 ptx-debug 技能分析这个问题"
"使用 ptx-debug 技能调试这个测试"
"帮我调试一下，用 ptx-debug 技能"
```

**使用 skill 工具**:
```bash
# 在对话中要求
skill name="ptx-debug"
```

---

## 🎯 技能选择决策

### 自动识别逻辑

```
用户报告问题
  ↓
分析问题类型
  ↓
┌─────────────────────────────────────┐
│ IF 问题是 ANTLR 语法错误：          │
│   - no viable alternative           │
│   - mismatched input                │
│   - 语法文件修改需求                │
│   ↓                                 │
│   加载 ptx-grammar-modification     │
│   → 遵循 TDD 流程                   │
│   → 运行 ./tests/ptx/test_all_ptx.sh│
│                                     │
│ ELSE 问题是运行时错误：             │
│   - 测试失败                        │
│   - 程序崩溃                        │
│   - 内存问题                        │
│   - 指令错误                        │
│   - 性能问题                        │
│   ↓                                 │
│   加载 ptx-debug                    │
│   → 自动选择配置                    │
│   → 收集日志分析                    │
│                                     │
│ ELSE 问题是 CUDA 编程：             │
│   ↓                                 │
│   加载 cuda-ptx（全局技能）         │
│                                     │
│ ELSE 问题是 C++ 调试：              │
│   ↓                                 │
│   加载 cpp-debug（全局技能）        │
└─────────────────────────────────────┘
```

---

## 📋 使用示例

### 示例 1: 语法错误（自动触发）

**用户**: "ANTLR 报错 no viable alternative at input '.param'"

**自动流程**:
```
1. Agent 识别错误类型
2. 🛑 停止当前操作
3. 加载技能：ptx-grammar-modification
4. 阅读 docs/ptx/ 相关章节
5. 运行 ./tests/ptx/test_all_ptx.sh
6. 修改 .g4 文件
7. 重新生成解析器
8. 验证测试通过
```

---

### 示例 2: 测试失败（关键词触发）

**用户**: "test_memory 测试失败"

**自动流程**:
```
1. Agent 识别为"测试失败"
2. 自动加载 ptx-debug 技能
3. 选择 debug_config.ini
4. 运行测试收集日志
5. 分析日志定位问题
6. 生成修复方案
7. 验证修复
```

---

### 示例 3: 手动加载技能

**用户**: "请加载 ptx-debug 技能分析这个内存问题"

**手动流程**:
```
1. 用户明确要求加载技能
2. Agent 加载 ptx-debug
3. 选择 memory_debug.ini
4. 运行程序跟踪内存
5. 分析访问模式
6. 定位问题
```

---

### 示例 4: 技能协作

**用户**: "PTX 解析后指令执行结果不对"

**协作流程**:
```
1. 先加载 ptx-grammar-modification
   → 确认语法正确
   → 运行 ./tests/ptx/test_all_ptx.sh

2. 再加载 ptx-debug
   → 分析执行日志
   → 选择 instruction_debug.ini

3. 必要时加载 cuda-ptx
   → 查询 PTX ISA 规范
   → 对比硬件行为
```

---

## ⚠️ 注意事项

### 1. 项目级技能 vs 全局技能

| 类型 | 位置 | 触发方式 | 示例 |
|------|------|---------|------|
| **项目级** | `docs/skills/` | AGENTS.md 自动触发或手动加载 | ptx-grammar-modification, ptx-debug |
| **全局级** | `~/.config/opencode/skills/` | 自动触发（基于通用关键词） | cuda-ptx, cpp-debug |

### 2. 技能冲突处理

**潜在冲突**: `ptx-grammar-modification` 和 `ptx-debug` 都可能被"PTX 解析错误"触发

**解决方案**:
```
ptx-debug 技能包含自动识别逻辑：
  - 如果是 ANTLR 错误 → 建议使用 ptx-grammar-modification
  - 如果是运行时错误 → 继续调试流程
```

### 3. 确保技能被加载

**检查方法**:
1. 查看 Agent 是否提到了技能名称
2. 查看是否遵循了技能流程
3. 查看是否使用了技能推荐的配置

**如果技能未被加载**:
```
明确要求："请加载 ptx-debug 技能"
或使用 skill 工具：skill name="ptx-debug"
```

---

## 🔧 配置说明

### AGENTS.md 配置

项目级技能的触发规则在 `AGENTS.md` 中定义：

```markdown
## 🎯 项目技能自动触发

### 项目技能位置

**项目技能目录**: `docs/skills/`

| 技能 | 文件 | 触发方式 |
|------|------|---------|
| ptx-grammar-modification | docs/skills/ptx-grammar-modification.md | AGENTS.md 自动触发 |
| ptx-debug | docs/skills/ptx-debug/SKILL.md | 关键词触发或手动加载 |

### ptx-debug 技能触发关键词

当用户提到以下关键词时，自动加载 ptx-debug 技能：
- "测试失败"
- "程序崩溃"
- "内存错误"
- ...
```

### 更新触发规则

要添加新的触发关键词，编辑 `AGENTS.md`：

```bash
# 编辑 AGENTS.md
vim AGENTS.md

# 在"ptx-debug 技能触发关键词"部分添加新关键词
```

---

## 📚 相关文档

- [项目技能总览](docs/skills/README.md)
- [技能系统说明](docs/skills-overview.md)
- [AGENTS.md](AGENTS.md)
- [PTX 语法修复](docs/skills/ptx-grammar-modification.md)
- [PTX 调试技能](docs/skills/ptx-debug/SKILL.md)

---

## ✅ 快速参考

### 触发方式总结

| 技能 | 自动触发 | 关键词触发 | 手动加载 |
|------|---------|-----------|---------|
| ptx-grammar-modification | ✅ (AGENTS.md) | ❌ | ✅ |
| ptx-debug | ❌ | ✅ | ✅ |
| cuda-ptx | ✅ (全局) | ✅ | ✅ |
| cpp-debug | ✅ (全局) | ✅ | ✅ |

### 常用触发词

**ptx-grammar-modification**:
- "PTX 解析错误"
- "ANTLR 错误"
- "语法错误"
- "no viable alternative"
- "mismatched input"

**ptx-debug**:
- "测试失败"
- "程序崩溃"
- "内存错误"
- "指令错误"
- "性能优化"
- "调试这个"

---

## 🚀 最佳实践

1. **使用关键词触发** - 让技能自动加载
2. **明确说明问题** - 帮助 Agent 正确识别
3. **必要时手动加载** - 确保技能被使用
4. **遵循技能流程** - 不要跳过步骤
5. **技能协作** - 复杂问题使用多个技能

---

**通过正确的触发机制，确保项目技能在需要时自动加载！**
