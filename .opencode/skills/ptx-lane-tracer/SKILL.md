# PTX Lane Tracer Skill

## 用途
从 PTX 文件中提取指定 kernel 的每个 lane (tid.x 0~31) 的指令执行序列，结合 LLM 分析 register 赋值链，生成 Markdown 格式的路径审查报告。

## 核心思想

PTX 中的分支条件依赖于 register 的值（如 `%r1 = %tid.x`）。Python 只能看到原始值，LLM 能理解语义关系。

**混合架构**：
1. **LLM 分析** → 理解 register 赋值链、predicate 逻辑、分支条件语义
2. **Python 执行** → 基于 LLM 分析结果，为每个 lane 模拟执行路径

## 工作流程

### 步骤 1: 生成 LLM 分析模板

```bash
python3 ptx_lane_tracer.py test.ptx _Z16kernelIiEvPT 25 --generate-analysis > analysis_template.json
```

### 步骤 2: 让 LLM 分析 PTX

把 PTX 文件和模板发给 LLM，让它填充分析结果。

**关键分析点**：

1. **Register 追踪**：
   - `mov.u32 %r1, %tid.x` → %r1 存储 thread ID
   - `and.b32 %r2, %r1, 31` → %r2 存储 lane_id (0-31)

2. **Predicate 条件推导**：
   - `setp.lt.u32 %p1, %r2, 16` → %p1 = (lane_id < 16)
   - `setp.ne.s32 %p3, %r1, 0` → %p3 = (tid.x != 0)

3. **Loop 迭代次数**：
   - `loop_iterations = max(0, lane_id - 15)`

### 步骤 3: 用 LLM 分析结果运行 Python

```bash
python3 ptx_lane_tracer.py test.ptx _Z16kernelIiEvPT 25 -a analysis.json -o report.md
```

## 命令行选项

| 选项 | 说明 |
|------|------|
| `ptx_file` | PTX 文件路径 |
| `kernel_name` | Kernel 函数名 |
| `start_line` | 起始行号 (默认 25) |
| `-o, --output FILE` | 输出报告文件 |
| `-a, --analysis FILE` | LLM 分析 JSON 文件 |
| `--generate-analysis` | 输出分析模板 |

## 输出报告结构

```markdown
# PTX Lane Execution Path Review Report

## Metadata
| Item | Value |
|------|-------|
| File | `test.ptx` |
| Kernel | `_Z16kernel_i32_iPv` |
| Start Line | 25 |
| Analyzed Lanes | 32 (tid.x 0-31) |
| Unique Paths | 2 |

## Register & Predicate Analysis (LLM)
- **%p1**: lane_id < 16 (from setp.lt.u32 %r2, 16)
- **%p3**: tid.x != 0 (from setp.ne.s32 %r1, 0)

## Path Summary
| Path | tid.x Range | Lanes | Total PC |
|------|-------------|-------|----------|
| Path 1 | 0-15 | 16 | 85 |
| Path 2 | 16-31 | 16 | 35 |

## Execution Matrix
...

## Path 1 Detail
...

## Divergence Analysis
...
```

## 示例

### 生成模板
```bash
python3 ptx_lane_tracer.py test_divergence_sync_standalone.1.sm_100.ptx \
  _Z27test_divergence_sync_kernelIiEvPT_ 25 --generate-analysis > analysis.json
```

### LLM 分析后填充的 JSON
```json
{
  "predicates": {
    "%p1": "lane_id < 16 (from setp.lt.u32 %p1, %r2, 16 at PC=29)",
    "%p2": "loop_iteration < loop_iterations (from setp.ne.s32 %p2, %r88, -15)",
    "%p3": "tid.x != 0 (from setp.ne.s32 %p3, %r1, 0 at PC=47)"
  },
  "branches": [
    {"pattern": "@%p1 bra", "condition": "lane_id < 16", "target_bb": 7},
    {"pattern": "@%p2 bra", "condition": "loop_iteration < loop_iterations", "target_bb": 2},
    {"pattern": "@%p3 bra", "condition": "tid.x != 0", "target_bb": 5}
  ],
  "loop": {
    "has_loop": true,
    "iterations_func": "max(0, lane_id - 15)"
  }
}
```

### 生成报告
```bash
python3 ptx_lane_tracer.py test_divergence_sync_standalone.1.sm_100.ptx \
  _Z27test_divergence_sync_kernelIiEvPT_ 25 -a analysis.json -o REPORT.md
```

## 注意事项

1. **分工明确**：LLM 负责语义理解（register 赋值链、predicate 条件），Python 负责执行模拟
2. **精度保证**：Python 使用 LLM 分析的精确条件来模拟每个 lane 的执行
3. **可验证性**：报告包含完整的 register 追踪，方便验证分析结果的正确性