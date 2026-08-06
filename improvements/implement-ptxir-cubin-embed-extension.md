# implement-ptxir-cubin-embed-extension

**优先级**: P1 | **来源**: ADR-0024
**阶段**: Phase 12.2 (PTXIR Cubin 集成) | **分类**: core-impl
**类型**: functional

## 架构依据

- **ADR-0024**: PTXIR-Embedded CUBIN 格式（已 Accepted 2026-08-06, commit `18ad58cb`）
- **ADR-0023**: PTXIR 二进制序列化格式与 7 项架构决策（sibling, Accepted 2026-07-30）
- **ADR-0010**: Fake CUDA Runtime 拦截机制（`__cudaRegisterFatBinary` 入口）
- **ADR-0011**: PTX→PTXIR 多阶段 Pipeline 架构（反序列化路径复用）
- **触发事件**：`ptxir-format-compliance` 提案 2026-08-01 被拒绝（与 G1-G9/D1-D5 7 决策不完全一致），但 cubin + PTXIR 兼容路径缺失需填补
- **架构依据**：标准 cubin 是 NVIDIA driver/cuModuleLoadData 链路的最终形态，PTX-EMU 现有架构无法直接加载。本 proposal 将 PTXIR 嵌入 cubin 末尾，使 PTX-EMU 能执行标准 cubin 同时保留 PTXIR 快速加载优势

## 范围

**In Scope**:

- `src/cudart/ptxir_loader.{h,cpp}` — 新增 PTXIRLoader 类（约 250 行）
- `src/cudart/cudart_sim.cpp` — 修改 `__cudaRegisterFatBinary` 增加 PTXIR 检测分支（约 +30-40 行，包含 PTXIRLoader dispatch + `config::isPTXIRModeEnabled()` 调用）
- `include/cudart/ptxir_loader.h` — 新增 PTXIRLoader 公开 API
- `tools/ptxir_extract.cpp` — 新增 CLI 提取工具（约 80 行）
- `tools/ptxir_embed.cpp` — 新增 CLI 嵌入工具（约 60 行）
- `src/cudart/CMakeLists.txt` — 注册 `ptxir_loader.cpp` 子目标
- `tools/CMakeLists.txt` — 注册两个工具目标
- **3 层测试覆盖**：
  - unit (PTXIRLoader 类的所有 static 方法，含 magic 边界、size=0、伪 cubin 输入)
  - integration (loader dispatch 流程，含 PTXIR_MODE on/off 分支)
  - e2e (nvcc + ptxir_embed + PTX-EMU 加载 + ptxir_extract → cuobjdump 双向验证)

**Out Scope**:

- NVIDIA cubin 格式本身的修改（仅追加，cuModuleLoadData 不感知）
- ANTLR 解析路径的修改（保持独立）
- CppTLM bridge 集成（独立 ADR-0021 范围）
- PTXIR Section TOC 格式的扩展（v1 → v2 升级是另一个 change）
- `config::isPTXIRModeEnabled()` 函数的**新设计**（复用 `configs/` 现有全局 config 机制；函数本身的实现 In Scope，MUST 级）

## 关键场景

### 场景 1: 生成嵌入式 cubin

- **GIVEN** nvcc 编译 `kernel.cu` 输出 `kernel.cubin`，且 `kernel.ptx` 通过 ptx-serializer 生成 `kernel.ptxir`
- **WHEN** 执行 `ptxir_embed --in-cubin kernel.cubin --in-ptxir kernel.ptxir --out kernel.embedded.cubin`
- **THEN** 生成 `kernel.embedded.cubin`，含 cubin 前缀 + `.ptxir.section` + `.ptxir.magic`，标准 NVIDIA 工具（`cuobjdump`）仍能解析提取后的纯 cubin

### 场景 2: PTX-EMU 加载嵌入式 cubin

- **GIVEN** PTX-EMU 加载 `kernel.embedded.cubin`（`PTXIR_MODE=auto` 环境变量已设置）
- **WHEN** `__cudaRegisterFatBinary` 被调用
- **THEN** PTXIRLoader 检测嵌入段 → 提取 PTXIR → 反序列化为 `StatementContext[]` → 走现有 `gpu.registerFatBinary()` 主路径（不修改后续执行）

### 场景 3: PTXIR_MODE 关闭时回退到标准路径

- **GIVEN** 用户设置 `PTXIR_MODE=off` 环境变量
- **WHEN** 加载任意 cubin（含/不含嵌入段）
- **THEN** 行为与现状完全一致 — PTXIRLoader 检测到 `PTXIR_MODE=off` → 跳过嵌入段处理 → 走标准 cubin 路径（dispatch 等价于无嵌入段情况）

### 场景 4: 提取纯 cubin 用于 NVIDIA 标准工具链

- **GIVEN** `kernel.embedded.cubin` 需要用 NVIDIA 标准工具链 debug
- **WHEN** 执行 `ptxir_extract --in kernel.embedded.cubin --out-cubin kernel.pure.cubin --out-ptxir kernel.pure.ptxir`
- **THEN** 输出 `kernel.pure.cubin` 与 `kernel.pure.ptxir`；`pure.cubin` 与 NVIDIA driver/cuobjdump 完全兼容，且字节级等于原始 `kernel.cubin`

## 技术约束

### MUST

- **MUST** 复用 ADR-0023 的 Section TOC + PTXIRHeader 格式（不重新发明二进制格式）
- **MUST** 提供 `PTXIR_EMBED_MAGIC` 8 字节 magic 后缀（loader O(1) 检测末尾），且 magic 与 NVIDIA 已有 magic 不冲突
- **MUST** `__cudaRegisterFatBinary` ABI 不变，仅添加 dispatch 分支（不修改现有签名）
- **MUST** `PTXIR_MODE` 环境变量可完全 bypass 检测分支（默认 OFF，行为等价于当前）
- **MUST** `Section TOC` 中显式嵌入 `cubin_hash` 字段，loader 校验一致性
- **MUST** 任何修改 `__cudaRegisterFatBinary` 必须确保 PTXIR 检测分支可被 env var `PTXIR_MODE=off` 完全绕过（CI 守门）
- **MUST** 实现 `config::isPTXIRModeEnabled()` 读取 `PTXIR_MODE` env var + `configs/*.ini` 配置项（loader dispatch 行为依赖此函数，不可降级为 SHOULD）

### MUST NOT

- **MUST NOT** 修改 ANTLR 解析路径（保持独立）
- **MUST NOT** 修改 NVIDIA cubin 格式前缀（仅追加嵌入段）
- **MUST NOT** 在 GPU registry / WarpContext / ThreadContext 等核心执行路径添加新依赖（仅 cudart 入口层增加 dispatch）
- **MUST NOT** 在没有 PTXIRLoader 测试覆盖的情况下合入主分支
- **MUST NOT** 假设 cubin 末尾永远有充足空间（magic 检测必须在 cubin size 上限内）
- **MUST NOT** 单方面修改 `PTXIR_EMBED_MAGIC` 的字面值 — magic 任何变更必须触发 ADR-0024 重新审视（governance check，不可绕过 proposal 层面单方面修改）

### SHOULD

- **SHOULD** 提取后纯 cubin 与原始 cubin 字节内容完全一致（avoid re-serialization corruption，hash 相等）
- **SHOULD** PTXIRLoader 类的所有 public static 方法都有 unit 测试覆盖（magic 边界、size=0、伪 cubin 输入、hash mismatch 场景）
- **SHOULD** e2e 测试用真实 nvcc + cuobjdump 验证嵌入 cubin 的 NVIDIA 兼容性
- **SHOULD** 嵌入/提取工具提供 `--help` 与 version 输出
- **SHOULD** tools/ 添加 README.md 说明用法与限制

## 验收标准

### 单元测试（unit）

- [ ] `tests/unit/test_ptxir_loader.cpp` 编译通过，覆盖所有 4 个 public static 方法
- [ ] `hasEmbeddedPTXIR()` 测试：合法嵌入 cubin / 普通 cubin / 空输入 / 截断输入（无 magic）/ 假 magic（首 4 字节匹配但后 4 字节不匹配）
- [ ] `extractPTXIR()` 测试：合法嵌入 / size=0 / 非嵌入 cubin（返回 null）
- [ ] `extractPureCubin()` 测试：合法嵌入 / 普通 cubin（透传）/ hash 不匹配
- [ ] `deserializeForCubin()` 测试：合法 PTXIR section / 损坏 header / cubin_hash 校验失败场景
- [ ] 测试覆盖率 ≥ 90%

### 集成测试（integration）

- [ ] `tests/integration/test_ptxir_cubin_loader.cpp` 编译通过，覆盖 `__cudaRegisterFatBinary` dispatch
- [ ] 场景：嵌入 cubin + PTXIR_MODE=auto → PTX-EMU 执行路径
- [ ] 场景：嵌入 cubin + PTXIR_MODE=off → 标准 cubin 路径
- [ ] 场景：普通 cubin + PTXIR_MODE=auto → 标准 cubin 路径（不报错）
- [ ] 场景：PTXIR 损坏 + PTXIR_MODE=auto → 优雅降级到标准 cubin 路径
- [ ] 场景：mock cubin_size 超过 magic 检测窗口 → 错误日志 + 标准路径
- [ ] 集成测试 ≥ 5 个 loader dispatch 场景

### 端到端测试（e2e）

- [ ] `tests/e2e/test_ptxir_cubin_embed.cu` 编译通过
- [ ] 场景：nvcc 编译真实 CUDA kernel（≥3 个不同复杂度）→ embed → PTX-EMU 执行 → 结果与原始 kernel 一致
- [ ] 场景：嵌入 cubin → extract → cuobjdump --dump-sass 输出与原始 cubin 字节级一致（hash 相等）
- [ ] 场景：嵌入 cubin → extract → cuobjdump --dump-ptx 正常输出
- [ ] **场景（关键，Oracle review blocking fix）**：`cuobjdump --dump-sass kernel.embedded.cubin` 直接对嵌入 cubin 解析成功，输出 SASS 与原始 cubin 一致 — 验证 cuModuleLoadData 容忍尾部 magic 的假设（ADR-0024 §风险 risk 1 / factor 2）
- [ ] **场景（关键，Oracle review blocking fix）**：在真实 NVIDIA driver 可用环境下，`cuModuleLoadData(kernel.embedded.cubin)` 不报错 — 验证嵌入 cubin 的 NVIDIA 工具链兼容性；若环境无真实 driver，测试必须输出显式 SKIP 日志（`[SKIP] cuModuleLoadData test — no driver`）而非静默通过
- [ ] e2e 测试 ≥ 5 个真实 CUDA kernel 嵌入场景（含上述 2 个新增）

### 构建与 CI

- [ ] `cmake --build build && ctest --output-on-failure` 全部通过
- [ ] `./scripts/sanity.sh` 全部通过
- [ ] `ptxir_extract --help` 与 `ptxir_embed --help` 输出正常
- [ ] 静态分析工具（clang-tidy / cppcheck）无新增 warning

### ADR 合规

- [ ] ADR-0024 §合规检查 6 项全部通过：
  1. PTXIR_MODE=off 完全 bypass 检测分支
  2. ptxir_extract 保留原 cubin 字节内容
  3. 嵌入段 .ptxir.section 使用 ADR-0023 Section TOC
  4. PTXIRLoader 所有函数有 unit 测试
  5. e2e 测试用 nvcc + cuobjdump 验证 NVIDIA 兼容性（含 Oracle review 新增的 2 个直接对 embedded cubin 解析场景）
  6. **magic number `PTXIR_EMBED_MAGIC` 变更必须触发 ADR-0024 重新审视（governance check，不可绕过 proposal 层面单方面修改）**

## 后续依赖

- 本 proposal 经 guide-arch Phase 5.5 审批通过后，由 guide-plan 生成实施计划，最终由 guide-ship 执行
- 实施时建议拆分为 3 个独立 commits（参考 `worktree-archive-workflow` v2.0.5+ 提案）：
  - Commit 1: PTXIRLoader 类 + unit 测试（独立可合并）
  - Commit 2: cudart_sim.cpp dispatch 集成 + integration 测试（依赖 Commit 1）
  - Commit 3: tools/ptxir_extract.cpp + tools/ptxir_embed.cpp + e2e 测试（依赖 Commit 1）
