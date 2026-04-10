# PTX-EMU 调试技能使用说明

## 概述

`ptx-debug` 是一个专门为 PTX-EMU 项目创建的调试技能，能够根据问题类型自动选择最合适的调试配置和方法。

## 技能位置

```
~/.config/opencode/skills/ptx-debug/SKILL.md
```

## 自动触发

### 触发关键词

技能会在以下情况下自动触发：

**问题类型触发**:
- "PTX 解析错误", "语法错误", "ANTLR 错误"
- "测试失败", "单元测试不通过", "ctest"
- "程序崩溃", "segfault", "core dumped"
- "内存错误", "非法访问", "越界"
- "指令执行错误", "结果不对"
- "性能慢", "需要优化", "benchmark"

**关键词触发**:
- "ptx", "cubin", "GPU", "kernel"
- "test_", "单元测试"
- "debug", "调试", "analyze", "分析"
- "为什么失败", "怎么回事", "哪里错了"

## 自动配置选择

技能会根据问题类型自动选择调试配置：

| 问题类型 | 自动选择配置 | 说明 |
|---------|-------------|------|
| PTX 解析错误 | `configs/verbose_trace_config.ini` | 详细跟踪解析过程 |
| 测试失败 | `configs/dev_debug_config.ini` | 平衡的日志级别 |
| 程序崩溃 | `configs/verbose_trace_config.ini` | 详细跟踪定位崩溃点 |
| 内存问题 | `configs/memory_debug_config.ini` | 专注内存操作跟踪 |
| 指令错误 | `configs/instruction_debug_config.ini` | 专注指令执行跟踪 |
| 性能问题 | `configs/perf_config.ini` | 最小日志开销 |
| 日常调试 | `configs/dev_debug_config.ini` | 默认调试配置 |

## 使用示例

### 示例 1: PTX 解析错误

**用户**: "PTX 解析失败，报告语法错误"

**技能自动行动**:
1. ✅ 识别问题类型：PTX 解析错误
2. ✅ 选择配置：`verbose_trace_config.ini`
3. ✅ 执行调试：
    ```bash
    ./scripts/debug-run.sh verbose ./build/bin/dummy-args
    grep "parser\|lexer" ptx_emu_trace.log
    ```
4. ✅ 分析错误位置
5. ✅ 生成修复方案

### 示例 2: 测试失败

**用户**: "test_memory_manager 测试失败"

**技能自动行动**:
1. ✅ 识别问题类型：测试失败
2. ✅ 选择配置：`debug_config.ini`
3. ✅ 执行调试：
    ```bash
    cd build && ctest -R test_memory_manager -V
    tail -100 ptx_emu_debug.log
    ```
4. ✅ 分析失败原因
5. ✅ 生成修复

### 示例 3: 内存问题

**用户**: "程序访问了非法内存地址"

**技能自动行动**:
1. ✅ 识别问题类型：内存问题
2. ✅ 选择配置：`memory_debug_config.ini`
3. ✅ 执行调试：
    ```bash
    ./scripts/debug-run.sh memory ./build/bin/dummy-args
    grep "\[mem\]" ptx_emu_memory_debug.log
    ```
4. ✅ 分析内存访问模式
5. ✅ 定位非法访问

### 示例 4: 性能优化

**用户**: "RAY benchmark 运行太慢，需要优化"

**技能自动行动**:
1. ✅ 识别问题类型：性能问题
2. ✅ 选择配置：`perf_config.ini`
3. ✅ 执行调试：
    ```bash
    ./scripts/debug-run.sh perf ./build/bin/RAY 512 512
    ```
4. ✅ 分析性能瓶颈
5. ✅ 提出优化建议

## 手动触发

如果技能没有自动触发，可以手动要求：

**方式 1: 直接提到技能名**
```
请使用 ptx-debug 技能分析这个问题
```

**方式 2: 描述调试需求**
```
帮我调试一下这个 PTX 程序，看看为什么失败
```

**方式 3: 指定调试场景**
```
用内存调试配置分析一下这个问题
```

## 技能功能

### 1. 自动化调试配置选择

根据问题关键词自动匹配最合适的调试配置，无需手动查找和复制配置文件。

### 2. 场景化调试方法

针对 6 种常见调试场景提供专门的调试流程：
- PTX 解析错误调试
- 测试失败调试
- 程序崩溃调试
- 内存问题调试
- 指令执行错误调试
- 性能问题调试

### 3. 日志分析自动化

提供常用的日志分析命令和技术：
- 自动提取关键日志
- grep 命令模板
- 日志分析模式

### 4. 调试流程标准化

遵循系统化的调试流程：
1. 问题识别
2. 配置选择
3. 证据收集
4. 分析定位
5. 修复方案
6. 验证修复

## 与其他技能协作

`ptx-debug` 技能会与其他技能协作：

- **cpp-debug**: 使用通用 C++ 调试方法
- **systematic-debugging**: 遵循系统化调试流程
- **cuda-ptx**: 利用 CUDA/PTX 专业知识

## 调试输出位置

技能会自动将日志输出到以下文件：

| 配置文件 | 日志文件 | 用途 |
|---------|---------|------|
| `configs/dev_debug_config.ini` | `ptx_emu_debug.log` | 日常调试 |
| `configs/verbose_trace_config.ini` | `ptx_emu_trace.log` | 详细跟踪 |
| `configs/memory_debug_config.ini` | `ptx_emu_memory_debug.log` | 内存调试 |
| `configs/instruction_debug_config.ini` | `ptx_emu_instr_debug.log` | 指令调试 |
| `configs/perf_config.ini` | - | 控制台输出 |

## 常用命令

### 查看日志

```bash
# 实时查看
tail -f ptx_emu_*.log

# 查看最新 N 行
tail -100 ptx_emu_debug.log

# 搜索关键词
grep "ERROR" ptx_emu_debug.log

# 分页查看
less -R ptx_emu_trace.log
```

### 使用快捷脚本

```bash
# 手动选择配置运行
./scripts/debug-run.sh debug ./build/bin/dummy-args
./scripts/debug-run.sh trace ./build/bin/RAY 512 512
./scripts/debug-run.sh memory ./build/bin/dummy-args
./scripts/debug-run.sh perf ./build/bin/RAY 1024 1024
```

## 最佳实践

1. **及时清理日志**: 详细日志可能很大，定期清理
2. **选择合适的配置**: 不要总是用 trace，日常调试用 debug 即可
3. **保留日志文件**: 分析问题时保留相关日志
4. **恢复配置**: 调试完成后恢复到 release 配置

## 故障排除

### 技能未触发

**问题**: 技能没有自动触发

**解决**:
1. 检查问题描述是否包含触发关键词
2. 手动提到"ptx-debug"技能名
3. 使用"调试"、"分析"等明确词汇

### 配置未生效

**问题**: 调试配置没有生效

**解决**:
1. 确认使用 `./scripts/debug-run.sh` 或 `configs/` 中的配置文件
2. 确认程序会读取正确的配置路径
3. 重启程序使配置生效

### 日志文件未生成

**问题**: 没有生成日志文件

**解决**:
1. 检查配置的 `target` 设置（console/file/both）
2. 检查日志文件路径权限
3. 查看控制台输出

## 更新日志

- **2026-03-23**: 初始版本发布
  - 支持 6 种调试场景
  - 自动配置选择
  - 场景化调试方法
  - 日志分析技术

## 相关资源

- [调试配置完整指南](docs/debug-config-guide.md)
- [调试快捷脚本](scripts/debug-run.sh)
- [系统调试方法](~/config/opencode/superpowers/skills/systematic-debugging/SKILL.md)
