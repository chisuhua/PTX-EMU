# ADR-0025: `ptxir_build` CLI（PTX → PTXIR 序列化命令行）

| 属性 | 值 |
|------|-----|
| **状态** | Proposed |
| **日期** | 2026-08-08 |
| **关联任务** | T13.1（`feat-ptxir-nvcc-toolchain` Phase 2） |
| **关联 PR** | TBD |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle（实现 review）、Metis（决策完备性） |

---

## 上下文

### 问题背景

`feat-implement-ptxir-cubin-embed-extension`（commit `40fa1423`，ADR-0024）归档后，端到端工具链在 `ptx-serializer build` 这一步断裂：

- **已 ship**：`tools/ptxir_embed`、`tools/ptxir_extract`、运行时 `libcudart.so`（含 PTXIR dispatch）
- **未 ship**：将 `.ptx`（nvcc/cuobjdump 输出）转为 `.ptxir` 二进制的 CLI
- `tools/README.md` §场景 1 文档中明确写 `ptx-serializer build --in kernel.ptx --out kernel.ptxir`，但实际工具未构建——这是上一轮 change 的 **遗留债**

库函数 `src/ptxir/ptxir_serialization.cpp::generate_ptxir(ptx_path, ptxir_path, kernel_name)` 已存在并被 `unit_ptxir_serialization` 测试覆盖（7 个 TEST_CASE），但无 CLI wrapper，用户（包括新工具链 `ptx-nvcc` ADR-0027）无法从 shell 调用。

### 触发事件

1. **2026-08-07**：ADR-0024 §影响范围列出 `ptx-serializer build` CLI，ship 时遗漏
2. **2026-08-08**：`ptx-nvcc` wrapper 工具链 ADR-0027 提议，依赖本 CLI 补齐 PTX→PTXIR 这一步
3. 用户明确要求提供 NVIDIA SDK 兼容工具链（"原生 cuda sdk 一样的运行方式"）

### 技术约束

- **不重复造轮子**：直接调用已存在的 `ptxir::generate_ptxir()` 库函数
- **保持 PTXIR 格式兼容**：不修改 ADR-0023 §PTXIRHeader / §Section TOC
- **单 kernel v1**：当前 `ManifestSection` 仅 `kernel_name` 单值（ADR-0024 §决策内容 §决策 §1）；`generate_ptxir` 按单 kernel 处理。**v2 状态 (2026-08-11)**: 已由 ADR-0028 解除；详见 ADR-0028 §Decision 1。backward-compat 策略：旧 v1 单 kernel binary 仍可被新 loader 读取。
- **POSIX CLI 习惯**：GNU-style `--option value`，使用明确且稳定的退出码契约：0 成功，1 用法或参数错误，2 PTX/kernel 数据错误，3 I/O、内部或工具失败。

---

## 决策驱动因素

1. **factor 1 — 端到端工具链完整**：用户从 `.cu` 到 `./myapp` 不应有 manual gap
2. **factor 2 — 与 `ptx-nvcc` 解耦**：CLI 必须独立可调用，不强制依赖 wrapper
3. **factor 3 — 库函数复用**：避免在 CLI 中重写 ANTLR parse + manifest 序列化逻辑
4. **factor 4 — 单 SRP**：CLI 只做 PTX→PTXIR 转换，不做 embed/extract（后者已独立）
5. **可演进**：v1 保持单 kernel。多 kernel 语义见 ADR-0028 §Decision 1（已 ship）。

---

## 考虑的替代方案

### 方案 A: 新 CLI `ptxir_build`（✅ 选中）

**描述**：在 `tools/` 新增独立 CLI，调用 `ptxir::generate_ptxir()`

**优点**：
- 与现有 `ptxir_embed` / `ptxir_extract` 命名一致
- 独立可调用，被 `ptx-nvcc` wrapper 直接 subprocess
- 库函数已实现 + 测试覆盖，CLI 只是 thin wrapper
- SRP 清晰

**缺点**：
- 增加一个工具（但这是补齐缺口，不是新增）
- v1 单 kernel 限制，后续多 kernel 设计延期。ADR-0028 文件目前不存在。

**选择理由**：唯一同时满足端到端完整性、SRP、可演进的方案。

### 方案 B: 合并到 `ptxir_embed --in-ptx`（❌ 未采用）

**描述**：`ptxir_embed` 增加 `--in-ptx <ptx>` 参数，内部先调 `generate_ptxir` 再 embed

**优点**：
- 工具数 -1

**缺点**：
- **违反 SRP**：embed 应只做 IO 拼接（bin + ptxir → bin），不应嵌入 PTX 解析/序列化逻辑
- 复用性差：想单独生成 PTXIR（不 embed）的人被迫走 embed
- 测试复杂度：embed 单测要 mock PTX parse

**未采用理由**：工具数 -1 不值得用架构清晰度换。

### 方案 C: 用现有 `unit_ptxir_serialization` binary（❌ 未采用）

**描述**：把测试 binary 当 tool 用

**优点**：
- 0 新增代码

**缺点**：
- 测试 binary 不可作 production tool（链接 catch2 等）
- CLI 接口不可控（测试 runner 输出格式固定）

**未采用理由**：测试 ≠ tool。

---

## 决策内容

### 设计原则

1. **薄 wrapper**：CLI 仅做参数解析 + 错误信息友好化 + 库函数调用
2. **POSIX 兼容**：GNU-style long options，退出码固定为 0 成功，1 用法或参数错误，2 PTX/kernel 数据错误，3 I/O、内部或工具失败。
3. **错误定位**：解析失败时打印 PTX file path + 行号（如可定位）
4. **v1 单 kernel**：显式接受 `--kernel-name <K>`。多 kernel 选项见 ADR-0028 §Decision 1（已 ship）。

### 实现要点

#### CLI 接口

```
ptxir_build --in <ptx-file> --kernel-name <name> --out <ptxir-file>
```

- `--in <path>`：输入 PTX 文件（必填）
- `--kernel-name <name>`：目标 kernel 名（必填，匹配 `.entry <name>`）
- `--out <path>`：输出 PTXIR 文件（必填）
- `--help`：打印 usage
- exit code：0 成功；1 用法或参数错误；2 PTX/kernel 数据错误；3 I/O、内部或工具失败

`ptx-nvcc` wrapper 先从 `cuobjdump` 导出的 PTX 中自动检测一个 kernel 名，再将该名称作为显式 `--kernel-name <K>` 参数传给 `ptxir_build`。CLI 不负责自动选择 kernel，v1 保持单 kernel 输入契约。

#### 实现骨架

```cpp
// tools/ptxir_build.cpp (~80 行)
#include "ptxir/ptxir_serialization.h"
#include <CLI11/CLI11.hpp>  // 或手工解析；当前 tools/ 不引 CLI11

int main(int argc, char** argv) {
    std::string in_path, kernel_name, out_path;
    // 解析 --in / --kernel-name / --out
    if (解析失败) print_usage_and_exit(1);

    try {
        if (!ptxir::generate_ptxir(in_path, out_path, kernel_name)) {
            std::cerr << "ptxir_build: failed to generate PTXIR (kernel '"
                      << kernel_name << "' not found in " << in_path << ")\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "ptxir_build: " << e.what() << "\n";
        return 3;
    }
    return 0;
}
```

#### 与现有工具对齐

| 工具 | 职责 | 调用链 |
|---|---|---|
| `ptxir_build` | .ptx → .ptxir | `generate_ptxir()` |
| `ptxir_embed` | binary + .ptxir → 嵌入 binary | (无库依赖) |
| `ptxir_extract` | 嵌入 binary → 纯 cubin / 纯 PTXIR | (无库依赖) |

三工具完全独立，可单独使用；`ptx-nvcc` wrapper 按序调用。

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `tools/CMakeLists.txt` | 修改 | 新增 `add_executable(ptxir_build ptxir_build.cpp)` + `target_link_libraries(ptxir_build PRIVATE ptxir ptxir_writer ptxir_reader)` |
| `tools/ptxir_build.cpp` | 新增 | ~80 行 |
| `tools/README.md` | 修改 | §场景 1 标注 CLI 已 ship；新增 `ptxir_build` 段 |
| `tests/unit/tools/test_ptxir_build.cpp` | 新增 | roundtrip test（用 nvcc 生成的 dummy.ptx）+ kernel-name 不存在 → exit 2 + 文件缺失 → exit 1 |
| ADR-0024 §影响范围 | 修改 | 追加 `tools/ptxir_build.cpp` |
| `docs/adr/README.md` | 修改 | ADR-0025 加入 Proposed 索引 |

---

## 后果

### 正面影响

1. **端到端工具链完整**：`.cu` → `.ptx` → `.ptxir` → binary（ADR-0027 wrapper 依赖）
2. **债务补齐**：消除 ADR-0024 §影响范围 文档与实现的 gap
3. **库函数 public 化**：`generate_ptxir` 从内部测试可见升级为 production CLI
4. **可独立调试**：用户可手动 `.ptx → .ptxir` 验证 serializer 行为

### 负面影响

1. **工具数 +1**：但这是补缺，不是新增 surface
2. **v1 单 kernel**：`generate_ptxir` 一次处理一个 kernel。多 kernel 支持见 ADR-0028 §Decision 1（已 ship，backward-compat 保留）。

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| nvcc 生成的 PTX 与 `generate_ptxir` 预期格式不匹配 | 中 | 高 | `tests/unit/tools/test_ptxir_build.cpp` 用真实 nvcc 生成的 PTX；e2e `e2e_ptxir_nvcc_wrapper` 覆盖 |
| `--kernel-name` 不匹配 PTX 内 `.entry` | 中 | 中 | 友好错误："kernel '<X>' not found; available: A, B" |
| argv 解析与现有 `ptxir_embed` / `ptxir_extract` 不一致 | 低 | 低 | 三个工具统一 `--in` / `--out` / `--kernel-name` 命名 |
| CLI 链接依赖膨胀（ptxir + ptxir_writer + ptxir_reader） | 低 | 低 | ptxir 已是 static lib，size 可忽略 |

---

## 合规检查

后续相关开发应检查：

- [ ] `ptxir_build` 必须保留 PTXIRHeader magic + version 兼容性（不破坏 ADR-0023）
- [ ] v1 单 kernel 限制在 CLI usage 文本中明示
- [ ] `tools/ptxir_embed` 不内嵌 PTX→PTXIR 逻辑（SRP）
- `ptx-nvcc` wrapper 先从 `cuobjdump` 导出的 PTX 中自动检测 kernel 名，再将检测结果作为显式 `--kernel-name <K>` 参数传给 `ptxir_build`；v1 仍只处理一个 kernel。
- [ ] magic literal `PTXIR_EMBED_MAGIC` 变更（ADR-0024 §合规 #6）需同步更新本 CLI 不受影响

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-08 | 初始版本（端到端工具链债务补齐 + SRP） | PTX-EMU Architecture Team |
| 2026-08-09 | **跨仓评审修订**: §技术约束 v1 单 kernel 限制明示 ADR-0028 BLOCKING DEPENDENCY 标注 + backward-compat 策略 | PTX-EMU Architecture Team |

---

## 参考

- [ADR-0023 PTXIR 二进制序列化格式与 7 项架构决策](./ADR-0023-ptxir-binary-format.md) — sibling 决策，PTXIR 格式
- [ADR-0024 PTXIR-Embedded CUBIN](./ADR-0024-ptxir-cubin-embed-extension.md) — 本 CLI 补齐其 §影响范围 PTX 生成步骤
- [ADR-0027 ptx-nvcc wrapper toolchain](./ADR-0027-ptx-nvcc-wrapper.md) — 下游依赖本 CLI
- [docs/architecture/ptxir-toolchain-stack.md](../architecture/ptxir-toolchain-stack.md) — 工具链栈架构总览
- [src/ptxir/ptxir_serialization.cpp](../../src/ptxir/ptxir_serialization.cpp) — `generate_ptxir()` 库函数