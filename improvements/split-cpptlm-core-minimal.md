# split-cpptlm-core-minimal

**优先级**: P2 | **来源**: ADR-0022 §未来 — cpptlm_core_minimal 拆分评估
**阶段**: Phase-11 | **分类**: infra-setup
**类型**: refactor

## 架构依据

- ADR-0022-cpptlm-unified-build.md:79 — `--whole-archive cpptlm_core` 导致 libcudart.so ~40MB，决策时接受此代价，标注"远期可通过 cpptlm_core_minimal 优化"
- ADR-0022-cpptlm-unified-build.md:137 — 风险表：体积膨胀 | 概率: 中 | 缓解: 远期拆分
- ADR-0022-cpptlm-unified-build.md:151 — "未来评估 cpptlm_core_minimal 拆分时机"（未勾项）
- Oracle 评估完成：当前 40MB 对开发/测试场景可接受

## 范围

- **In Scope**:
  - 从 cpptlm_core 拆分出 minimal 子集（bridge + IPtxEmuDriver + MemoryBridge）以减小 libcudart.so 体积
  - CppTLM 侧新建 CMake target（cpptlm_core_minimal 静态库）
  - PTX-EMU 侧 CMakeLists.txt:144-145 换 target 名链接

- **Out Scope**:
  - 不更改 ABI（cpptlm_bridge.h 5 虚方法 + 8 函数指针 vtable）
  - 不重排 CppTLM 项目目录结构（仅加 CMake target）
  - 不改现有链接行为（`--whole-archive` 保留）

- **当前结论**: 推迟 — 维持 `--whole-archive cpptlm_core` 现状

## 关键场景

- GIVEN 构建 PTX-EMU, WHEN `nm -D libcudart.so`, THEN `cpptlm_set_driver` 为 `T` (strong)
- GIVEN `EMU_COSIM=1`, WHEN 启动, THEN CppTLM 协同仿真路径走强定义（非 PTX-EMU 的 weak no-op）
- GIVEN `g_cpptlm_bridge == nullptr`（无 CppTLM）, WHEN 运行, THEN 同步路径与现状字节级一致

## 技术约束

- MUST 保持 `__attribute__((weak))` cpptlm_set_driver 模式（无 CppTLM 时安全降级 no-op）
- MUST 保留 `--whole-archive` 链接方式（已验证 `-Wl,-u,cpptlm_set_driver` 不可行 — weak symbol 在 PTX-EMU 对象文件已解析，链接器不搜 archive）
- SHOULD 满足触发条件时再执行

## 验收标准

已触发条件（满足任意一条时重新评估此提案）：

- [ ] `libcudart.so` > 80MB
- [ ] CppTLM 向 `cpptlm_core.a` 新增 ≥2 模块
- [ ] 需要外部分发 `libcudart.so`
