# PTX-EMU 技能系统说明

## 技能列表

PTX-EMU 项目使用以下技能系统：

### 1. `ptx-grammar-modification` - PTX 语法修复专家

**位置**: `docs/skills/ptx-grammar-modification.md`

**职责**: **专门处理 ANTLR 语法文件修改和解析错误**

**触发条件**:
- ANTLR 解析错误（`no viable alternative at input`）
- Token 不匹配（`mismatched input 'X' expecting Y`）
- 需要修改 `.g4` 语法文件
- 解析阶段崩溃

**核心流程**:
```
🛑 停止 → 阅读 docs/ptx/ 文档 → 运行 ./tests/ptx/test_all_ptx.sh 
→ 修改 .g4 文件 → 重新生成解析器 → 验证测试通过
```

**典型场景**:
- "PTX 解析失败，报告语法错误"
- "ANTLR 报错 no viable alternative"
- "需要添加新的 PTX 指令语法"

**强制要求**:
- ✅ 必须先阅读 `docs/ptx/` 对应章节
- ✅ 必须先运行 `./tests/ptx/test_all_ptx.sh`
- ✅ 必须使用 TDD 流程（RED→GREEN→REFACTOR）
- ✅ 必须验证所有测试通过

---

### 2. `ptx-debug` - 通用调试助手

**位置**: `~/.config/opencode/skills/ptx-debug/SKILL.md`

**职责**: **运行时调试、日志分析、问题定位**

**触发条件**:
- 测试失败（ctest 不通过）
- 程序崩溃（segfault、core dump）
- 内存问题（非法访问、越界）
- 指令错误（执行结果不对）
- 性能问题（运行太慢）
- 日常调试需求

**核心流程**:
```
识别问题类型 → 自动选择调试配置 → 运行程序收集日志 
→ 分析日志定位问题 → 生成修复方案 → 验证修复
```

**典型场景**:
- "test_memory 测试失败"
- "程序访问了非法内存"
- "指令执行结果不对"
- "RAY benchmark 太慢"

**自动配置选择**:
| 问题类型 | 自动配置 |
|---------|---------|
| 测试失败 | debug_config.ini |
| 内存问题 | memory_debug_config.ini |
| 指令错误 | instruction_debug_config.ini |
| 性能问题 | perf_config.ini |
| 程序崩溃 | verbose_trace_config.ini |

---

### 3. `cuda-ptx` - CUDA/PTX 专家

**位置**: `~/.config/opencode/skills/cuda-ptx/SKILL.md`

**职责**: **CUDA 编程、PTX ISA 参考、性能优化**

**核心能力**:
- CUDA kernel 开发和优化
- PTX ISA 指令参考（本地文档 2.3MB）
- 性能分析（nsys、ncu）
- GPU 调试（compute-sanitizer、cuda-gdb）

**典型场景**:
- "优化这个 CUDA kernel"
- "查询 PTX 指令说明"
- "分析 GPU 性能瓶颈"

---

### 4. `cpp-debug` - C++ 调试专家

**位置**: `~/.config/opencode/skills/cpp-debug/SKILL.md`

**职责**: **C++ 运行时故障排查**

**核心能力**:
- 段错误（SIGSEGV）诊断
- 死锁检测
- 内存泄漏检查（valgrind、ASan）

---

### 5. `systematic-debugging` - 系统化调试方法

**位置**: `~/.config/opencode/superpowers/skills/systematic-debugging/SKILL.md`

**职责**: **提供系统化调试流程**

**核心原则**:
- NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
- 四阶段流程：Root Cause → Pattern → Hypothesis → Implementation

---

## 技能分工矩阵

### 问题类型 vs 推荐技能

| 问题类型 | 首选技能 | 辅助技能 | 说明 |
|---------|---------|---------|------|
| **ANTLR 解析错误** | `ptx-grammar-modification` | `ptx-debug` | 语法修复流程 |
| **语法文件修改** | `ptx-grammar-modification` | - | 必须遵循 TDD |
| **测试失败** | `ptx-debug` | `systematic-debugging` | 运行时调试 |
| **程序崩溃** | `ptx-debug` | `cpp-debug` | 堆栈分析 |
| **内存问题** | `ptx-debug` | `cpp-debug` | 内存跟踪 |
| **指令错误** | `ptx-debug` | `cuda-ptx` | ISA 参考 |
| **性能优化** | `ptx-debug` | `cuda-ptx` | 性能分析 |
| **CUDA kernel** | `cuda-ptx` | - | GPU 编程 |

---

## 技能冲突处理

### ⚠️ `ptx-grammar-modification` vs `ptx-debug`

**潜在冲突**: 两者都可能被"PTX 解析错误"触发

**解决方案**:

```
用户报告 "PTX 解析错误"
  ↓
ptx-debug 技能触发
  ↓
分析错误类型
  ↓
┌─────────────────────────────────────┐
│ IF 错误是 ANTLR 解析错误：          │
│   - no viable alternative           │
│   - mismatched input                │
│   - 语法文件修改需求                │
│   ↓                                 │
│   🛑 停止调试流程                   │
│   → 建议使用 ptx-grammar-modification│
│   → 提示运行 ./tests/ptx/test_all_ptx.sh │
│                                     │
│ ELSE 错误是运行时错误：             │
│   ↓                                 │
│   继续调试流程                      │
│   → 选择 verbose_trace 配置         │
│   → 收集日志分析                    │
└─────────────────────────────────────┘
```

### 实际示例

**示例 1: 语法错误**
```
用户："ANTLR 报错 no viable alternative at input '.param.u64.ptr'"

正确流程:
1. ptx-debug 识别为语法问题
2. 建议："这是 ANTLR 语法错误，应该使用 ptx-grammar-modification 技能"
3. 提示："请先运行 ./tests/ptx/test_all_ptx.sh，然后阅读 docs/ptx/ 相关章节"
```

**示例 2: 运行时错误**
```
用户："test_memory 测试失败，内存访问错误"

正确流程:
1. ptx-debug 识别为运行时问题
2. 自动选择 debug_config.ini
3. 运行测试收集日志
4. 分析内存访问模式
5. 定位问题根源
```

---

## 技能加载

### 自动加载

技能会根据问题类型自动加载：

```yaml
"PTX 解析错误" → ptx-grammar-modification (强制)
"测试失败" → ptx-debug
"内存错误" → ptx-debug + cpp-debug
"性能优化" → ptx-debug + cuda-ptx
"CUDA kernel" → cuda-ptx
```

### 手动加载

明确要求使用特定技能：

```
"请加载 ptx-grammar-modification 技能修复语法错误"
"使用 ptx-debug 技能分析这个测试失败"
"用 cuda-ptx 技能优化这个 kernel"
```

---

## 最佳实践

### 1. 正确识别问题类型

**关键**: 区分**解析时错误**和**运行时错误**

```
解析时错误 → ptx-grammar-modification
  - ANTLR 报错
  - 语法不匹配
  - .g4 文件修改

运行时错误 → ptx-debug
  - 测试失败
  - 程序崩溃
  - 内存问题
  - 指令错误
```

### 2. 遵循技能流程

**`ptx-grammar-modification`**:
- ✅ 必须阅读文档
- ✅ 必须运行测试
- ✅ 必须使用 TDD 流程
- ✅ 必须验证通过

**`ptx-debug`**:
- ✅ 自动选择配置
- ✅ 收集证据
- ✅ 分析定位
- ✅ 验证修复

### 3. 技能协作

复杂问题可能需要多个技能协作：

```
问题："PTX 解析后指令执行结果不对"

流程:
1. ptx-grammar-modification: 确认语法正确
2. ptx-debug: 分析执行日志
3. cuda-ptx: 查询 PTX ISA 规范
4. systematic-debugging: 遵循系统流程
```

---

## 快速参考

### 技能选择决策树

```
遇到问题
  ↓
是 ANTLR 语法错误吗？
  ├─ 是 → 🛑 → ptx-grammar-modification
  │         → 阅读 docs/ptx/
  │         → 运行 ./tests/ptx/test_all_ptx.sh
  │
  └─ 否 → 是运行时错误吗？
           ├─ 是 → ptx-debug
           │        → 自动选择配置
           │        → 收集日志分析
           │
           └─ 否 → 是 CUDA 编程问题吗？
                    ├─ 是 → cuda-ptx
                    │        → 性能分析
                    │        → PTX ISA 查询
                    │
                    └─ 否 → cpp-debug 或其他技能
```

### 常用命令

```bash
# 语法修复流程
./tests/ptx/test_all_ptx.sh
cmake --build build --target GenerateParser

# 调试配置
./scripts/debug-run.sh debug ./build/bin/程序
./scripts/debug-run.sh trace ./build/bin/程序
./scripts/debug-run.sh memory ./build/bin/程序

# 日志分析
grep "ERROR" ptx_emu_*.log
grep "\[mem\]" ptx_emu_*.log
grep "\[instr\]" ptx_emu_*.log
```

---

## 相关文档

- [调试配置指南](debug-config-guide.md)
- [调试速查表](DEBUG_QUICK_REFERENCE.md)
- [PTX 语法修复](skills/ptx-grammar-modification.md)
- [系统调试方法](~/config/opencode/superpowers/skills/systematic-debugging/SKILL.md)
