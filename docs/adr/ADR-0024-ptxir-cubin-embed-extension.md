# ADR-0024: Cubin 嵌入 PTXIR 的混合二进制格式（PTXIR-Embedded CUBIN）

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-08-06 |
| **关联任务** | TBD（由 `guide-design` 阶段创建 proposal-suggestions 条目后分配） |
| **关联 OpenSpec change** | TBD（待 design 阶段创建） |
| **关联差距分析** | [docs/architecture/ptxir-serialization-gaps-gap-analysis.md](../architecture/ptxir-serialization-gaps-gap-analysis.md)（沿用） |
| **关联技能** | [.opencode/skills/ptxir-serialization/SKILL.md](../../.opencode/skills/ptxir-serialization/SKILL.md) |
| **关联 ADR** | [ADR-0010](./ADR-0010-fake-cuda-runtime.md)（Fake CUDA Runtime）、[ADR-0011](./ADR-0011-pipeline-architecture.md)（PTX→PTXIR Pipeline）、[ADR-0023](./ADR-0023-ptxir-binary-format.md)（PTXIR 二进制格式 — sibling 决策） |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle (architecture review), Metis (decision completeness) |

---

## 上下文

### 问题背景

PTX-EMU 当前 fake libcudart.so 提供两条独立的执行路径：

1. **ANTLR 路径（生产路径）**：`.cu` → nvcc → `.ptx` → ANTLR 解析 → `StatementContext[]` → 执行
2. **PTXIR 路径（测试加速路径）**：`.ptx` → ANTLR（一次）→ `.ptxir` 二进制 → 反序列化 → 执行

但**实际部署场景**中，Cubin 才是 NVIDIA driver/cuModuleLoad 链路的最终形态。Cubin 不携带 PTXIR 信息，因此 PTX-EMU 现有架构无法直接加载标准 cubin。

需求定义场景：
- **场景 A（生成）**：将 PTX 编译（或对现有 cubin 重新生成）时，把 PTXIR 嵌入到 cubin 末尾的 `.ptxir.section`，形成"PTXIR-embedded CUBIN"
- **场景 B（执行）**：fake libcudart.so 加载该混合格式后，**可选**走 PTX-EMU PTXIR 执行路径，或**回退**到 `cuModuleLoadData` 标准路径
- **场景 C（提取）**：从混合格式剥离 PTXIR section，恢复纯 cubin，**走标准 CUDA 二进制程序执行类似的流程**

### 触发事件

1. **2026-08-01** — `ptxir-format-compliance` 提案在 `proposal-suggestions.md` 被拒绝（与 G1-G9 + D1-D5 7 项决策不完全一致），但讨论中识别 cubin + PTXIR 兼容路径缺失
2. **2026-08-04** — `god-class-refactor-sm-context-phase3` 归档，SM/CTA 解耦后 fake libcudart 可独立承担新加载逻辑
3. **2026-08-06** — 用户提出本 ADR 的核心需求：用 PTXIR 编译 CUDA 代码生成含 PTXIR 的 cubin，可提取纯 cubin 走标准 CUDA 流程

### 技术约束

- **不破坏现有 cubin 路径**：标准 cubin 必须仍能被 nvcc + driver 加载执行
- **复用 ADR-0023 的 Section TOC**：PTXIR 嵌入段须符合 `.ptxir.section` 命名约定
- **fake libcudart ABI 不变**：仅添加新加载分支，不修改现有 `__cudaRegisterFatBinary` 签名
- **NVIDIA cubin 格式约束**：嵌入段须位于 cubin 文件尾（NVIDIA `cuModuleLoadData` 容忍尾部 unknown data）

### 目标架构

```
                ┌─────────────────────────────────┐
                │      PTXIR-Embedded CUBIN       │
                ├─────────────────────────────────┤
                │  ┌───────────────────────────┐  │
                │  │   NVIDIA CUBIN (prefix)   │  │  ← cuModuleLoadData 可识别
                │  │   - Header (.nv.module)  │  │
                │  │   - Code sections        │  │
                │  │   - Symbol table         │  │
                │  └───────────────────────────┘  │
                │  ┌───────────────────────────┐  │
                │  │  .ptxir.section (suffix)  │  │  ← 仅 PTX-EMU 读取
                │  │   - PTXIRHeader (24B)     │  │
                │  │   - Section TOC          │  │
                │  │   - KERNEL/REGDECL/STRTAB│  │
                │  │   - .ptxir.magic (footer)│  │
                │  └───────────────────────────┘  │
                └─────────────────────────────────┘

加载决策树:
  loader.detect(input):
    if input 末尾匹配 .ptxir.magic:
      if PTXIR_MODE env / config enabled:
        return ptxir_loader.load_embedded_cubin(input)
      else:
        log "PTXIR section ignored (PTXIR_MODE off)"
        return standard_cubin_path(input)  // 仅前半段
    else:
      return standard_cubin_path(input)

提取路径:
  ptxir_extract <embedded.cubin> [--out-dir <dir>]
    └─> 写出 pure.cubin (前半段) + pure.ptxir (后半段)
```

---

## 决策驱动因素

1. **factor 1 — 复用现有 PTXIR 基建**：ADR-0023 已定义 `.ptxir` 格式（扁平二进制 + Section TOC），本 ADR 直接复用 KERNEL/REGDECL/STRTAB 三个 section，避免重新发明
2. **factor 2 — 不与 NVIDIA cubin 路径对抗**：嵌入段必须位于文件尾，cuModuleLoadData 解析时遇到 unknown section 应跳过（NVIDIA driver 实际行为）
3. **factor 3 — 决策可逆性**：必须能提取出纯 cubin，确保与标准 CUDA 工具链（`nvcc --cubin`、`cuobjdump`、`ncu` profiler）兼容
4. **factor 4 — 与现有 fake libcudart 协同**：ADR-0010 定义了 `__cudaRegisterFatBinary` 入口，本 ADR 在该入口下增加 cubin 决策分支
5. **factor 5 — 独立可演进**：PTXIR section 与 cubin 解耦，未来 cubin 格式演进（如 Hopper 升级）不影响 PTXIR section

---

## 考虑的替代方案

### 方案 A: 纯 metadata 标识（cubin 不动）（❌ 未采用）

**描述**: 仅在 cubin header 的 metadata 段加 magic，不嵌入完整 PTXIR

**优点**:
- cubin 不被破坏，NVIDIA 工具链完全兼容
- 实现最简单（仅加 8-byte magic）

**缺点**:
- 携带的 PTXIR 数据有限（最多只放 registration metadata）
- 实际 PTX-EMU 执行仍需重新解析 PTX 文本，无法独立执行

**未采用理由**: 无法满足"独立 PTXIR 执行路径"的核心需求。

### 方案 B: 只嵌入不提取（❌ 未采用）

**描述**: 仅支持 PTXIR 嵌入 cubin，不提供 ptxir-extract 工具

**优点**:
- 实现简单（仅一侧）
- 边界明确

**缺点**:
- 一旦嵌入，无法回退到标准 cubin 路径
- 与 NVIDIA 工具链兼容性窗口关闭（不能 extract → cuobjdump debug）

**未采用理由**: 违反 factor 3（决策可逆性）。

### 方案 C: 独立定义新格式 `.pcubin`（❌ 未采用）

**描述**: 完全定义新格式 `.pcubin`，不与 NVIDIA cubin 共享

**优点**:
- 完全自主设计，无需考虑 NVIDIA 兼容性
- 可包含丰富元数据

**缺点**:
- nvcc 不识别 `.pcubin`，需要专门工具生成
- 与现有 CUDA 工具链完全割裂
- 实现成本最高（全新格式 spec + 工具链）

**未采用理由**: 违反 factor 2（不与 NVIDIA cubin 路径对抗），实现成本与收益不匹配。

### 方案 D: Cubin 嵌入 PTXIR Section + 标准 magic 后缀（✅ 选中）

**描述**: cubin 末尾追加 `.ptxir.section` + `.ptxir.magic` 后缀，loader 决策，extract 工具复原

**优点**:
- 复用 ADR-0023 的 Section TOC 格式（无重复设计）
- 与 NVIDIA cubin 共存（cuModuleLoadData 容忍尾部）
- 提供双向：嵌入 + 提取

**缺点**:
- cubin 体积增加（典型的 5-20% 取决于 kernel 复杂度）
- 需要确保 `.ptxir.magic` magic number 与 NVIDIA 已有 magic 不冲突

**选择理由**: 满足全部 5 个 factor（factor 1-5）。

---

## 决策内容

### 设计原则

1. **最小破坏性**: 标准 cubin 前缀保持原样，`cuModuleLoadData` 不感知嵌入段
2. **决策可逆**: 必须提供 ptxir-extract 工具
3. **复用 ADR-0023**: 嵌入的 PTXIR section 复用 7 项决策（扁平二进制 + Section TOC + 字符串表末尾 + Extend-Only 等）
4. **loader 决策透明**: `__cudaRegisterFatBinary` 入口增加 dispatch 但 ABI 不变
5. **配置驱动**: PTXIR mode 通过环境变量或 config 启用，默认 OFF（与现有行为完全一致）

### 实现要点

#### 1. PTXIR Section 嵌入格式

```cpp
// PTXIR-Embedded CUBIN 文件结构
struct EmbeddedCubinLayout {
    uint8_t  cubin_prefix[N];        // NVIDIA cubin (cuModuleLoadData 可见)
    uint32_t ptxir_section_size;     // sizeof(.ptxir.section)
    uint8_t  ptxir_section[M];       // .ptxir.section = PTXIRHeader + TOC + sections
    uint8_t  ptxir_magic[8];         // = "PTXIR\x00\x01\x00" (8 bytes, 唯一标识)
};

// 嵌入段末尾 magic 后缀的目的:
// 1. loader 可以 O(1) 检测末尾，避免扫描整个 cubin
// 2. 校验完整，避免 cubin 末尾恰好匹配部分前缀的伪阳性
// 3. 8 字节 magic 与 NVIDIA 已用 magic 不冲突
constexpr uint8_t PTXIR_EMBED_MAGIC[8] = {'P','T','X','I','R','\x00','\x01','\x00'};
```

#### 2. Loader 决策逻辑

```cpp
// src/cudart/ptxir_loader.{h,cpp}
// 函数职责: 检测/提取/反序列化 PTXIR section + 选择执行路径

class PTXIRLoader {
public:
    // 检测嵌入段
    static bool hasEmbeddedPTXIR(const uint8_t* data, size_t size);

    // 提取 PTXIR section (返回 allocated buffer + size)
    static std::unique_ptr<uint8_t[]> extractPTXIR(
        const uint8_t* data, size_t size, size_t* out_size);

    // 提取纯 cubin (去除嵌入段, 写出或返回)
    static std::vector<uint8_t> extractPureCubin(
        const uint8_t* data, size_t size);

    // 从 PTXIR section 反序列化为 StatementContext[]
    static std::vector<StatementContext> deserializeForCubin(
        const uint8_t* ptxir_data, size_t ptxir_size);
};

// src/cudart/cudart_sim.cpp 修改点:
// 在 __cudaRegisterFatBinary 入口增加 dispatch
extern "C" void __cudaRegisterFatBinary(void* fatCubin) {
    auto data = reinterpret_cast<uint8_t*>(fatCubin);

    if (PTXIRLoader::hasEmbeddedPTXIR(data, size) &&
        config::isPTXIRModeEnabled()) {  // env PTXIR_MODE / config 字段
        // 提取 PTXIR → 反序列化 → 注册到 GPU registry
        // 注意: 仍走现有 gpu.registerFatBinary() 主路径
    } else {
        // 现有路径不变
        gpu.registerFatBinary(fatCubin);
    }
}
```

#### 3. PTXIR-Extract 工具

```cpp
// tools/ptxir_extract.cpp
// CLI: ptxir_extract <input.cubin|input.ptxir-embedded> [--out-cubin <X>] [--out-ptxir <Y>]
// 默认: input.cubin → input.pure.cubin + input.extracted.ptxir

int main(int argc, char** argv) {
    auto input = read_file(argv[1]);

    if (!PTXIRLoader::hasEmbeddedPTXIR(input)) {
        // 普通 cubin, 直接 copy
        write_file("input.pure.cubin", input);
        return 0;
    }

    // 是嵌入式, 同时输出纯 cubin + 纯 PTXIR
    auto pure_cubin = PTXIRLoader::extractPureCubin(input);
    write_file(out_cubin_path, pure_cubin);

    auto pure_ptxir = PTXIRLoader::extractPTXIR(input);
    write_file(out_ptxir_path, pure_ptxir);

    return 0;
}
```

#### 4. 编译嵌入工具

```cpp
// tools/ptxir_embed.cpp (与 extract 对偶)
// CLI: ptxir_embed <input.cubin> <input.ptxir> [--out <X>]
// 读取已有 cubin 和 .ptxir, 拼成 embedding

int main(int argc, char** argv) {
    auto cubin = read_file(cubin_path);
    auto ptxir = read_file(ptxir_path);

    EmbeddedCubinLayout layout;
    layout.cubin_prefix = cubin;
    layout.ptxir_section_size = ptxir.size();
    layout.ptxir_section = ptxir;
    layout.ptxir_magic = PTXIR_EMBED_MAGIC;

    write_file(out_path, layout);
}
```

#### 5. 三类使用场景示例

```
场景 1: nvcc 编译并嵌入 PTXIR
  nvcc -ptx kernel.cu -o kernel.ptx
  ptx-serializer build --in kernel.ptx --out kernel.ptxir
  ptxir_embed --in-cubin kernel.cubin --in-ptxir kernel.ptxir --out kernel.embedded.cubin

场景 2: PTX-EMU 加载嵌入式 cubin (PTXIR_MODE=auto)
  LD_PRELOAD=./libcudart.so ./myapp
    → __cudaRegisterFatBinary 检测嵌入段 → 反序列化 PTXIR → 执行

场景 3: 提取纯 cubin 用于 NVIDIA driver
  ptxir_extract --in kernel.embedded.cubin --out-cubin kernel.pure.cubin
  cuobjdump --dump-sass kernel.pure.cubin  # 标准 NVIDIA 工具
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/cudart/cudart_sim.cpp` | **修改** | `__cudaRegisterFatBinary` 增加 PTXIR 检测分支（30 行） |
| `src/cudart/cudart_sim.cpp` | **修改** | 新增 `config::isPTXIRModeEnabled()` 函数（10 行） |
| `src/cudart/CMakeLists.txt` | **修改** | 添加 `ptxir_loader.cpp` 子目标 |
| `src/cudart/ptxir_loader.{h,cpp}` | **新增** | PTXIRLoader 类（检测/提取/反序列化，约 250 行） |
| `tools/ptxir_extract.cpp` | **新增** | CLI 提取工具（约 80 行） |
| `tools/ptxir_embed.cpp` | **新增** | CLI 嵌入工具（约 60 行） |
| `tools/CMakeLists.txt` | **修改** | 注册两个工具目标 |
| `include/cudart/ptxir_loader.h` | **新增** | PTXIRLoader 公开 API |
| ADR-0023 (PTXIR 格式) | **依赖** | 复用 Section TOC + PTXIRHeader 格式 |
| ADR-0010 (Fake CUDA Runtime) | **依赖** | 修改 `__cudaRegisterFatBinary` 入口 |
| ADR-0011 (Pipeline 架构) | **依赖** | 复用 PTXIR 反序列化路径 |

### 前置依赖

- **ADR-0023** 必须为 Accepted（✅ 已于 2026-07-30 Accepted）
- **ADR-0010** 必须为 Active（✅ 当前 Active）
- **`ptxir_serialization.cpp` 已实现**（✅ `src/ptxir/ptxir_serialization.cpp` 存在）
- **`config::isPTXIRModeEnabled()` 全局配置函数**：依赖 configs/ 的全局 config 机制（✅ 已存在）

### 后续依赖

- **`openspec/changes/<future>/ptxir-cubin-embed/`**: 由 guide-design 阶段创建 proposal 后由 guide-plan 实施
- **测试 3 层覆盖**: unit (PTXIRLoader 类)、integration (loader dispatch 流程)、e2e (nvcc + 嵌入 + PTX-EMU 执行)

---

## 后果

### 正面影响

1. **复用 ADR-0023 基建**：不重复发明二进制格式，节省 1 周实现成本
2. **与 NVIDIA 工具链共存**：extract 路径保证 `cuobjdump` / `ncu` 等工具仍能用
3. **决策可逆**：嵌入后可提取，破坏性窗口<24h（任何嵌入即时可 revert）
4. **性能优势保留**：PTX-EMU 执行嵌入 cubin 仍走 ~5ms 反序列化，不重 ANTLR
5. **选项灵活**：PTXIR_MODE=off 时完全等价于现有行为

### 负面影响

1. **cubin 体积增加**：典型 kernel 的 .ptxir section 占 5-20%（实测待 e2e 验证）
2. **loader ABI 决策增加复杂度**：`__cudaRegisterFatBinary` 检测分支需测试覆盖
3. **magic number 维护**：PTXIR_EMBED_MAGIC 必须与 NVIDIA 已用 magic 不冲突（需要 case-by-case 验证）
4. **ABI v1 冻结**：嵌入段格式被外部 `.ptxir-embedded.cubin` 文件依赖，format 改动需兼容旧版本

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| NVIDIA driver 不识别 magic 后缀并报错 | 低 | 高 | 用真实 cubin + 标准 driver 跑 e2e 测试；magic 选择 `.ptxir\x00\x01\x00` (尾部非 NVIDIA 已知 magic) |
| 嵌入段位置错误导致 cuModuleLoadData 失败 | 中 | 高 | 通过 `ptxir_extract → cuobjdump` 双向测试 |
| PTXIR 反序列化与嵌入 cubin 携带的 cubin 不一致 | 中 | 中 | 在 Section TOC 中显式嵌入 `cubin_hash`，loader 校验 |
| PTXIR section v1 → v2 兼容性破坏 | 低 | 中 | Section TOC header 已含 `version` 字段 (ADR-0023 Decision 6) |
| loader 检测伪阳性（cubin 末尾恰好含 magic） | 极低 | 高 | 校验 PTXIR_EMBED_MAGIC 前 4 字节 + 后 4 字节双重检查 |

### 合规检查

后续相关开发应检查：

- [ ] 任何修改 `__cudaRegisterFatBinary` 必须确保 PTXIR 检测分支可被 env var PTXIR_MODE=off 完全绕过
- [ ] 任何 ptxir_extract 应保留原 cubin 字节内容（avoid re-serialization corruption）
- [ ] 嵌入段 .ptxir.section 必须使用 ADR-0023 定义的 Section TOC 格式
- [ ] PTXIRLoader 类的所有函数必须有 unit 测试（包括 magic 边界、size=0、伪 cubin 输入）
- [ ] e2e 测试必须用真实 nvcc + cuobjdump 验证嵌入 cubin 的 NVIDIA 兼容性
- [ ] magic number `PTXIR_EMBED_MAGIC` 变更需要 ADR 重新审视

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-06 | 初始版本（架构决策与 5 项 factor + 4 方案对比） | PTX-EMU Architecture Team |

---

## 参考

- [ADR-0023 PTXIR 二进制序列化格式与 7 项架构决策](./ADR-0023-ptxir-binary-format.md) — sibling 决策，Section TOC 格式
- [ADR-0010 Fake CUDA Runtime 拦截机制](./ADR-0010-fake-cuda-runtime.md) — `__cudaRegisterFatBinary` 入口
- [ADR-0011 PTX→PTXIR 多阶段 Pipeline 架构](./ADR-0011-pipeline-architecture.md) — 反序列化路径
- [NVIDIA CUDA cuModuleLoadData 文档](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUmodule.html)
- [NVIDIA cubin 格式参考 (cubin.h)](https://github.com/NVIDIA/cuda-parallel-compute-sdk)
- [docs/architecture/ptxir-serialization-gaps-gap-analysis.md](../architecture/ptxir-serialization-gaps-gap-analysis.md) — 9 项差距 + 5 项格式偏差
- [.opencode/skills/ptxir-serialization/SKILL.md](../../.opencode/skills/ptxir-serialization/SKILL.md) — PTXIR 序列化技能
