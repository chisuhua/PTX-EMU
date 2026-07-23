# ADR-0022: CppTLM + PTX-EMU 统一构建链路（--whole-archive 替代独立 .so + dlopen）

| 属性 | 值 |
|------|-----|
| **状态** | Accepted |
| **日期** | 2026-07-23 |
| **关联任务** | CppTLM D1-Full MemoryBridge 构建集成 |
| **关联 OpenSpec change** | [openspec/changes/cpptlm-d1-full/](../../openspec/changes/cpptlm-d1-full/) |
| **关联审查** | [CppTLM docs/superpowers/specs/2026-07-23-ptxemu-build-review-feedback.md](../../../CppTLM/docs/superpowers/specs/2026-07-23-ptxemu-build-review-feedback.md) |
| **姊妹 ADR** | [ADR-0020](./ADR-0020-cpptlm-injection-points.md)（注入点）+ [ADR-0021](./ADR-0021-cpptlm-d1-full-integration.md)（MemoryBridge 集成）|
| **作者** | PTX-EMU Architecture Team |
| **审核人** | CppTLM Team (Oracle review) |

---

## 上下文

### 问题背景

CppTLM D1-Full MemoryBridge 需要 PTX-EMU 的 `libcudart.so` 在运行时与 CppTLM 进行双向 ABI 通信。原方案设计为独立编译 `libcpptlm_cudart.so`，通过 `dlopen` / `LD_PRELOAD` 在运行时加载并完成强/弱符号覆盖。

### 触发事件

1. **Commit `84212a9d`** — PTX-EMU 侧将 CppTLM 集成从"可选选项 `BUILD_LIB_CPPTLM_CUDART`"改为"常驻链接"：
   - `PtxEmuDriverShim` 源文件始终编译（不再受 `BUILD_LIB_CPPTLM_CUDART` 保护）
   - CppTLM bridge 符号始终存在于 `libcudart.so`
   - `EMU_COSIM=1` 环境变量作为运行时门控，替代编译期选项
2. **2026-07-23** — CppTLM 团队对 HEAD (`adb12d77`) 进行 Oracle 审查，确认构建方案正确

### 技术约束

- `libcudart.so` 必须能被标准 CUDA 程序通过 `LD_LIBRARY_PATH` 直接加载，无额外 `.so` 依赖
- 双向 ABI：`cpptlm_set_driver`（PTX-EMU → CppTLM）+ `cpptlm_attach_bridge`（CppTLM → PTX-EMU）
- 无 CppTLM 时，bridge 符号必须安全降级为 no-op（零影响原始 PTX-EMU 行为）
- `include/cudart/cpptlm_bridge.h` 零 CppTLM 依赖（ABI 真值源原则）

---

## 决策驱动因素

1. **部署复杂度**：需要 `LD_PRELOAD` + `libcpptlm_cudart.so` 路径管理 → 对用户不友好
2. **符号可见性**：`cudart.so` 默认无 `-fvisibility=hidden`，`PTXEMU_BRIDGE_API` 宏保障未来兼容
3. **编译期 vs 运行时门控**：`EMU_COSIM` 环境变量比 CMake 选项更灵活（无需重编译）
4. **ABI 稳定性**：`PtxEmuDriverApi` vtable 12 端点 + 4 签名 `static_assert` 已在两仓库同步

---

## 考虑的替代方案

### 方案 A: 独立 `libcpptlm_cudart.so` + dlopen（❌ 未采用）

**描述**: 编译独立 `.so`，运行时通过 `dlopen` 加载，强符号覆盖 `libcudart.so` 中的弱符号

**优点**:
- 构建完全解耦，CppTLM 和 PTX-EMU 可独立编译
- `BUILD_LIB_CPPTLM_CUDART=OFF` 时完全无 CppTLM 代码

**缺点**:
- 需要管理 `.so` 路径和加载时机
- 用户必须 `LD_PRELOAD=libcpptlm_cudart.so` 或调用 `dlopen`
- 两阶段加载（先 `libcudart.so`，后 `libcpptlm_cudart.so`）时序敏感
- 运行时配置错误静默回退到 no-op（难以排查）

### 方案 B: `cpptlm_core.a` + `--whole-archive` 链接（✅ 选中）

**描述**: CppTLM 编译为静态库 `cpptlm_core.a`，PTX-EMU 通过 `--whole-archive` 将其完整链接进 `libcudart.so`，`cpptlm_set_driver` 在 `.so` 中为强符号

**优点**:
- 单 `.so` 部署，零运行时配置
- `nm -D libcudart.so | grep cpptlm_set_driver` → `T`（strong text），可直接验证
- `EMU_COSIM=1` 环境变量作为运行时门控（无需重编译切换）
- 无 dlopen 时序问题

**缺点**:
- 将全部 CppTLM 模块（RouterTLM 805 行、ModuleFactory 604 行等）链接进 `cudart.so`，增大 `.so` 体积
- 构建耦合：CppTLM 代码变更需要 PTX-EMU 重编
- 未来需要拆分 `cpptlm_core_minimal`（仅 bridge + IPtxEmuDriver + MemoryBridge）以减小体积

**选择理由**: 部署简单性 > `.so` 体积。40MB 的 `libcudart.so` 对开发/测试场景可接受，远期可通过 `cpptlm_core_minimal` 优化。

---

## 决策内容

### 设计原则

1. **强/弱符号覆盖链路**:
   - PTX-EMU `libcudart.so` 提供 `__attribute__((weak))` 的 `cpptlm_set_driver` 空实现
   - CppTLM `cpptlm_core.a`（通过 `--whole-archive` 链接）提供同名强定义
   - 链接器选择强定义 → `cpptlm_set_driver` 在最终 `.so` 中导出为 `T`（strong text）
2. **运行时门控替代编译期门控**:
   - `EMU_COSIM=1` → StubBridge auto-attach + auto-advance
   - 默认（未设置）→ 字节级兼容原有同步路径
3. **ABI 双向验证**:
   - 12 端点 + 4 签名 `static_assert` 在 PTX-EMU 和 CppTLM 两侧同步
   - `sizeof(cudaStream_t) <= sizeof(uint64_t)` 编译期断言

### 实现要点

- `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` 在 `add_subdirectory()` 之前设置
- `PtxEmuDriverShim` 始终编译（不再受 `BUILD_LIB_CPPTLM_CUDART` 保护）
- `PTXEMU_BRIDGE_API` 标注所有 ABI 边界符号（`cpptlm_set_driver`、`cpptlm_attach_bridge`、`cpptlm_detach_bridge`）
- `PTX_EMU_MAX_ADVANCE_CYCLES` 默认 10M，ceiling 耗尽时返回 `cudaError_t(999)` + 错误日志

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/cudart/cpptlm_bridge.h` | 修改 | 加 `PTXEMU_BRIDGE_API` 到 `cpptlm_set_driver`；更新注释 |
| `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` | 修改 | 取消 `BUILD_LIB_CPPTLM_CUDART` 条件编译 |
| `src/cudart/cpptlm_bridge/cpptlm_cudart_lib.cpp` | 删除 | 旧方案 `libcpptlm_cudart.so` 入口（commit `84212a9d`） |
| `src/cudart/cudart_sim.cpp` | 修改 | `cpptlm_set_driver()` 调用始终执行；advance ceiling 日志增强 |
| `CMakeLists.txt` | 修改 | `--whole-archive` 链接 `cpptlm_core.a` |
| CppTLM `CMakeLists.txt` | 修改 | 编译 `cpptlm_core.a` 静态库 |

---

## 后果

### 正面影响

- 部署从"管理 2 个 .so"简化为"单 .so"——用户只需 `LD_LIBRARY_PATH=build/lib ./app`
- 运行时门控（`EMU_COSIM`）比编译期门控（CMake 选项）更灵活——同一 build 可切换模式
- 符号正确性可通过 `nm -D` 脚本化验证（适合 CI 集成）

### 负面影响

- `libcudart.so` 体积增大（~41MB，含 CppTLM 全部模块）
- CppTLM 代码变更需要 PTX-EMU 重新 CMake 配置 + 重编
- `cpptlm_set_driver` 缺少 `PTXEMU_BRIDGE_API` 时，若未来加 `-fvisibility=hidden` 会静默退化

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| `--whole-archive` 误移除 | 低 | 强符号退化→CppTLM 静默失效 | CI `nm -D` 断言脚本 |
| `cpptlm_core.a` 体积膨胀 | 中 | `.so` 过大 | 远期拆分 `cpptlm_core_minimal` |
| 可见性宏遗漏 | 低 | `-fvisibility=hidden` 时符号隐藏 | `PTXEMU_BRIDGE_API` 统一标注所有 ABI 符号 |

---

## 合规检查

后续相关开发应检查：

- [x] `PTXEMU_BRIDGE_API` 标注 `cpptlm_set_driver`、`cpptlm_attach_bridge`、`cpptlm_detach_bridge`
- [x] 注释更新：`libcpptlm_cudart.so` → `cpptlm_core（--whole-archive）`
- [x] 构建残留 `build/lib/libcpptlm_cudart.so` 已清理
- [x] advance ceiling 日志提示 `PTX_EMU_MAX_ADVANCE_CYCLES`
- [ ] CI 增加 `nm -D build/lib/libcudart.so | grep 'T cpptlm_set_driver'` 符号覆盖测试
- [ ] 未来评估 `cpptlm_core_minimal` 拆分时机

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-07-23 | 初始版本 — 基于 CppTLM Oracle 审查 + PTX-EMU commit `84212a9d` | PTX-EMU Architecture Team |

---

## 参考

- [CppTLM 构建审查反馈](https://github.com/chisuhua/CppTLM/blob/main/docs/superpowers/specs/2026-07-23-ptxemu-build-review-feedback.md)
- [ADR-0020: CppTLM 注入点](./ADR-0020-cpptlm-injection-points.md)
- [ADR-0021: CppTLM D1-Full MemoryBridge 集成](./ADR-0021-cpptlm-d1-full-integration.md)
- [ADR-0010: Fake CUDA Runtime 拦截机制](./ADR-0010-fake-cuda-runtime.md)
- [Commit `84212a9d`: 统一 CppTLM 构建](https://github.com/chisuhua/PTX-EMU/commit/84212a9d)
- [include/cudart/cpptlm_bridge.h](../../include/cudart/cpptlm_bridge.h) — ABI 真值源
