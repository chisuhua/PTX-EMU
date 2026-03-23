# PTX-EMU 项目技能

> **位置**: `docs/skills/`  
> **适用范围**: 仅适用于 PTX-EMU 项目  
> **加载方式**: 通过 AGENTS.md 自动触发或手动加载

---

## 📦 项目技能列表

### 1. `ptx-grammar-modification` - PTX 语法修复

**文件**: `ptx-grammar-modification.md`

**职责**: 专门处理 ANTLR4 语法文件修改和解析错误

**触发条件**:
- ANTLR 解析错误（`no viable alternative at input`）
- Token 不匹配（`mismatched input 'X' expecting Y`）
- 需要修改 `src/grammar/*.g4` 文件
- 解析阶段崩溃

**核心流程**:
```
🛑 停止 → 阅读 docs/ptx/ → 运行 ./tests/ptx/test_all_ptx.sh
→ 修改 .g4 → 重新生成解析器 → 验证测试通过
```

**强制要求**:
- ✅ 必须遵循 TDD 流程（RED→GREEN→REFACTOR）
- ✅ 必须先阅读 `docs/ptx/` 对应章节
- ✅ 必须先运行 `./tests/ptx/test_all_ptx.sh`
- ✅ 必须验证所有测试通过

**加载方式**:
```bash
# 通过 AGENTS.md 自动触发
# 或明确要求加载技能
```

---

### 2. `ptx-debug` - PTX-EMU 调试助手

**文件**: `ptx-debug/SKILL.md`

**职责**: 运行时调试、日志分析、问题定位

**触发条件**:
- 测试失败（ctest 不通过）
- 程序崩溃（segfault、core dump）
- 内存问题（非法访问、越界）
- 指令错误（执行结果不对）
- 性能问题（运行太慢）
- 日常调试需求

**核心能力**:
- 自动选择调试配置（6 种场景）
- 场景化调试方法
- 日志分析技术
- 快捷脚本支持

**自动配置选择**:
| 问题类型 | 自动配置 | 日志文件 |
|---------|---------|---------|
| PTX 解析错误 | verbose_trace.ini | ptx_emu_trace.log |
| 测试失败 | debug_config.ini | ptx_emu_debug.log |
| 程序崩溃 | verbose_trace.ini | ptx_emu_trace.log |
| 内存问题 | memory_debug.ini | ptx_emu_memory_debug.log |
| 指令错误 | instruction_debug.ini | ptx_emu_instr_debug.log |
| 性能问题 | perf_config.ini | 控制台 |
| 日常调试 | debug_config.ini | ptx_emu_debug.log |

**⚠️ 与 ptx-grammar-modification 的分工**:

| 问题类型 | 使用技能 | 说明 |
|---------|---------|------|
| **ANTLR 解析错误** | `ptx-grammar-modification` | 语法文件修改 |
| **语法文件修改** | `ptx-grammar-modification` | 修改 `.g4` 文件 |
| **`no viable alternative`** | `ptx-grammar-modification` | 语法规则修复 |
| **`mismatched input`** | `ptx-grammar-modification` | Token 定义修复 |
| **测试失败** | `ptx-debug` | 运行时错误 |
| **内存问题** | `ptx-debug` | 内存访问调试 |
| **指令错误** | `ptx-debug` | 指令执行分析 |
| **程序崩溃** | `ptx-debug` | 堆栈分析 |
| **性能问题** | `ptx-debug` | 性能测试 |

**自动识别逻辑**:
```
当用户提到 "PTX 解析错误" 时:
  1. ptx-debug 技能触发
  2. 检查错误类型
  3. IF 错误是 ANTLR 解析错误:
       → 🛑 停止调试流程
       → 建议使用 ptx-grammar-modification
       → 提示运行 ./tests/ptx/test_all_ptx.sh
  4. ELSE 错误是运行时错误:
       → 继续调试流程
       → 选择 verbose_trace 配置
       → 收集日志分析
```

---

## 🎯 技能选择决策树

```
遇到问题
  ↓
是 ANTLR 语法错误吗？
  ├─ 是（no viable alternative, mismatched input）
  │     ↓
  │   🛑 STOP → 使用 ptx-grammar-modification
  │             → 阅读 docs/ptx/
  │             → 运行 ./tests/ptx/test_all_ptx.sh
  │
  └─ 否
      ↓
    是运行时错误吗？
      ├─ 是（测试失败、崩溃、内存、指令、性能）
      │     ↓
      │   使用 ptx-debug
      │     → 自动选择配置
      │     → 收集日志分析
      │
      └─ 否
          ↓
        其他问题 → cpp-debug / cuda-ptx / 其他通用技能
```

---

## 📋 使用示例

### 示例 1: 语法错误

**用户**: "ANTLR 报错 no viable alternative at input '.param'"

**正确流程**:
1. 识别为**语法错误**
2. 🛑 停止调试流程
3. 使用 `ptx-grammar-modification`
4. 运行 `./tests/ptx/test_all_ptx.sh`
5. 阅读 `docs/ptx/` 相关章节
6. 修改 `.g4` 文件
7. 重新生成解析器
8. 验证测试通过

---

### 示例 2: 测试失败

**用户**: "test_memory 测试失败"

**正确流程**:
1. 识别为**运行时错误**
2. 使用 `ptx-debug`
3. 自动选择 `debug_config.ini`
4. 运行测试收集日志
5. 分析日志定位问题
6. 生成修复方案
7. 验证修复

---

### 示例 3: 内存问题

**用户**: "程序访问了非法内存地址"

**正确流程**:
1. 识别为**内存问题**
2. 使用 `ptx-debug`
3. 自动选择 `memory_debug_config.ini`
4. 运行程序跟踪内存操作
5. 分析访问模式
6. 定位非法访问

---

### 示例 4: 性能优化

**用户**: "RAY benchmark 太慢，需要优化"

**正确流程**:
1. 识别为**性能问题**
2. 使用 `ptx-debug`
3. 自动选择 `perf_config.ini`
4. 运行性能测试
5. 分析性能瓶颈
6. 生成优化建议

---

## 🛠️ 快捷使用

### 调试快捷脚本

```bash
# 查看所有配置
./scripts/debug-run.sh help

# 使用 debug 配置
./scripts/debug-run.sh debug ./build/bin/dummy-args

# 使用 trace 配置
./scripts/debug-run.sh trace ./build/bin/RAY 512 512

# 使用 memory 配置
./scripts/debug-run.sh memory ./build/bin/dummy-args
```

### 手动配置

```bash
# 复制配置文件
cp configs/debug_config.ini ./ptx_debug.conf

# 运行程序
./build/bin/dummy-args

# 查看日志
tail -f ptx_emu_debug.log
```

---

## 📚 相关文档

- [技能系统总览](../skills-overview.md)
- [调试配置指南](../debug-config-guide.md)
- [调试速查表](../DEBUG_QUICK_REFERENCE.md)
- [PTX 语法修复](ptx-grammar-modification.md)
- [AGENTS.md](../../AGENTS.md)

---

## ✅ 最佳实践

### 1. 正确识别问题类型

**关键**: 区分**解析时错误**和**运行时错误**

```
解析时错误（ANTLR 报错） → ptx-grammar-modification
运行时错误（测试/崩溃） → ptx-debug
```

### 2. 遵循技能流程

**每个技能都有自己的流程，必须严格遵守**:

- `ptx-grammar-modification`: TDD 流程（RED→GREEN→REFACTOR）
- `ptx-debug`: 自动配置 + 日志分析

### 3. 使用快捷脚本

快捷脚本自动处理配置选择：

```bash
./scripts/debug-run.sh debug ./build/bin/程序
```

### 4. 技能协作

复杂问题可能需要多个技能：

```
"PTX 解析后指令执行不对"
  ↓
1. ptx-grammar-modification: 确认语法正确
2. ptx-debug: 分析执行日志
3. cuda-ptx: 查询 PTX ISA 规范
```

---

## 🚀 更新记录

- **2026-03-23**: 初始版本（ptx-grammar-modification）
- **2026-03-23**: 添加 ptx-debug 技能
- **2026-03-23**: 明确两个技能的职责分工
- **2026-03-23**: 移到项目级别管理
