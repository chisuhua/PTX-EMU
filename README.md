# PTX-EMU

> **状态**：SIMT v2.0 完成；Blackwell tcgen05 完整实施；H5 规划中
> **核心特性**：C++20/CUDA PTX 模拟器，ANTLR4 解析 PTX，fake libcudart.so 拦截 CUDA runtime
> **文档入口**：[docs/README.md](./docs/README.md)

PTX-EMU 是一个 PTX（Parallel Thread Execution）指令级模拟器，用于在无 NVIDIA GPU 环境下仿真执行 CUDA 程序。

## 快速开始

```bash
# 1. 设置环境（必须！）
. env.sh

# 2. 配置 + 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. 跑测试
cd build && ctest --output-on-failure
```

## 架构概览

- **执行层次**：GPUContext → SMContext → CTAContext → WarpContext → ThreadContext
- **PTX 解析**：ANTLR4 → IR (StatementContext) → 解释执行
- **测试三类物理隔离**：
  - `tests/unit/` — 直接单元测试（数据结构/算法）
  - `tests/integration/` — 指令序列集成测试（通过 `execute_warp_instruction`）
  - `tests/e2e/` — CUDA Kernel 端到端测试（nvcc 编译 + 拦截）

## 文档导航

| 类别 | 路径 |
|---|---|
| 项目总入口 | [AGENTS.md](./AGENTS.md) |
| 文档索引 | [docs/README.md](./docs/README.md) |
| SIMT 架构 | [docs/architecture/SIMT-ARCHITECTURE-V2.md](./docs/architecture/SIMT-ARCHITECTURE-V2.md) |
| 开发指南 | [docs/developer-guide/](./docs/developer-guide/) |
| ADR 索引 | [docs/adr/README.md](./docs/adr/README.md) |
| 健康审计 | [docs/audits/HEALTH-AUDIT-2026-06-21.md](./docs/audits/HEALTH-AUDIT-2026-06-21.md) |
| 审计勘误 | [docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md](./docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md) |
| Roadmap | [openspec/changes/](./openspec/changes/)（活跃 changes 即未来 roadmap 项）|

## 已知限制

- **PTX 指令覆盖**：核心 ISA ~67%（详见审计 §3）
- **WMMA / Tensor Core**：是 stub
- **ANTLR 版本**：4.11.1 完全 vendored
- **CUDA Toolkit**：11.4.4 测试通过

## 贡献指南

新增 PTX 指令时，遵循三步流程：

1. 在 `include/ptx_ir/ptx_op.def` 添加 X-Macro 条目
2. 在 `src/ptxsim/instructions/<category>.cpp` 实现 handler
3. 添加测试（unit + integration + e2e 三层）

详见 [docs/developer-guide/](./docs/developer-guide/)。

## 相关参考

- [PTX ISA 规范](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
- [GPGPU-Sim](https://github.com/accel-sim/gpgpu-sim_distribution)

## 许可证

[按项目实际情况填写]
