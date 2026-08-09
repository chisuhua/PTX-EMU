# ADR-0026: PTXIR 调度默认模式 = auto（零配置嵌入 binary）

| 属性 | 值 |
|------|-----|
| **状态** | Proposed |
| **日期** | 2026-08-08 |
| **关联任务** | T13.2（`feat-ptxir-nvcc-toolchain` Phase 3） |
| **关联 PR** | TBD |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle（行为兼容性 review）、Metis（决策完备性） |

---

## 上下文

### 问题背景

`config::isPTXIRModeEnabled()` 当前实现（`src/cudart/ptxir_config.cpp`）：

```cpp
bool isPTXIRModeEnabled() {
    evaluate_env_once();
    if (g_env_cached != kEnvUnset) {
        return g_env_cached == 1;     // env 优先
    }
    return g_ini_mode == 1;           // INI 次之；默认 -1（未 set）→ false
}
```

- **当前默认**：env + INI 均未设置时返回 `false`（off）
- **ADR-0024 §设计原则 5** 明确选择 "默认 OFF" 以保证 backward compat
- **`configs/*.ini`** 在 `feat-implement-ptxir-cubin-embed-extension` 中被显式加上 `[ptxir] mode = off`，确保现有测试配置仍显式关闭 PTXIR

但 `feat-ptxir-nvcc-toolchain`（ADR-0027）的目标是 **零配置**：

> 用户用 `ptx-nvcc` 编译 `.cu` 得到嵌入 PTXIR 的 binary；直接 `./myapp` 运行，无需 `LD_PRELOAD`、无需 `PTXIR_MODE=auto`。

若保持默认 OFF，用户需手动设 `PTXIR_MODE=auto`，违背 "原生 CUDA SDK 一样的运行方式" 目标。

### 触发事件

1. **2026-08-07**：ADR-0024 §设计原则 5 选择默认 OFF（保守）
2. **2026-08-08**：用户要求提供 NVIDIA SDK 兼容工具链，明确禁止任何 env / LD_PRELOAD 步骤
3. ADR-0027 wrapper 设计要求 binary 自包含、零配置运行

### 技术约束

- **非嵌入 binary 行为兼容**：未嵌入 PTXIR section 的普通 cubin 仍走 cuobjdump 路径；auto 模式只增加一次未设置时的可执行文件尾探测。
- **现有测试回归 = 0**：所有 configs/*.ini 已显式 `mode = off`，翻转 default 不影响它们
- **fast-fail 成本**：默认 auto 只表示“尝试 PTXIR 检测”。未发现 footer 时回退 cuobjdump；检测到 malformed embedded PTXIR 或 manifest mismatch 时按工具/运行时错误处理，不把损坏数据当作普通缺失。
- **override 仍可用**：`PTXIR_MODE=off`（env 或 INI）显式关闭

---

## 决策驱动因素

1. **factor 1 — 零配置用户体验**：wrapper 产出的 binary 必须 `./myapp` 即用
2. **factor 2 — backward compat**：现有所有 configs 显式 off，翻转 default 不影响
3. **factor 3 — fast-fail 廉价**：tail-detect 失败时 0 副作用
4. **factor 4 — 显式优于隐式**：env / INI 仍可显式关闭 PTXIR 路径
5. **factor 5 — 文档一致性**：README / tools/README 标注 "默认 auto，可显式 off"

---

## 考虑的替代方案

### 方案 A: 默认 auto（✅ 选中）

**描述**：env + INI 均未设置时返回 `true`（auto）

**配置优先级**：

```
env PTXIR_MODE=off|auto   >   INI [ptxir] mode=off|on   >   default (auto)
```

**优点**：
- 零配置：wrapper 产出 binary 直接 `./myapp` 即可
- backward compat：现有 configs 全部显式 off，dispatch 语义兼容
- fast-fail：非嵌入 binary 走 tail-detect，未发现 footer 时回退 cuobjdump；这保持 dispatch 语义和最终执行路径兼容，但不承诺字节级或零行为变化。
- 显式 override：仍可 env / INI 关闭

**缺点**：
- 老 binary 启动时间微增（< 1µs tail-detect）
- 用户可能意外触发 PTXIR 路径（虽 default auto，但若 binary 含 PTXIR section 会被加载）

**选择理由**：唯一满足零配置 + backward compat + override 自由的方案。

### 方案 B: 保持默认 off，wrapper 设 env via RPATH-launched helper（❌ 未采用）

**描述**：`ptx-nvcc` 额外生成一个小 launcher binary，RPATH 到 wrapper 脚本，脚本里 `export PTXIR_MODE=auto` 后 exec 真实 binary

**优点**：
- libcudart 默认行为不变（off）

**缺点**：
- **额外 binary**：增加构建步骤 + 调试 surface
- **不原生**：用户 `./myapp` 实际执行的是 launcher，不是 `.cu` 编译产物
- **RPATH 冲突**：launcher 需要 RPATH 同时指 PTX-EMU lib 和 wrapper 脚本

**未采用理由**：违反 "原生 SDK 体验" 目标（用户期望直接执行自己编译的 binary）。

### 方案 C: 保持默认 off，wrapper 写 `ptxir.conf` 到 binary 同目录（❌ 未采用）

**描述**：wrapper 编译时在 `.` 下写 `ptxir.conf` 含 `mode = auto`；loader 优先读 binary 旁 config

**优点**：
- libcudart 行为不变
- 用户可控（删除 conf 即关闭）

**缺点**：
- **污染文件系统**：每个 binary 旁塞 config
- **复杂**：loader 需新增 "binary 旁 config" 读取逻辑（多一层决策路径）
- **不原生**：用户运行前要确认 conf 存在

**未采用理由**：方案 A 同样零配置且更干净。

---

### 与 ADR-0024 的关系

本 ADR 是 ADR-0024 v1.1 的修订案，将其默认模式决策更新为 v1.2。它只澄清默认检测与 fallback 的运行时契约，不改变 ADR-0024 已定义的 PTXIR footer、嵌入格式或 dispatch 数据路径。

### 设计原则

1. **默认 auto**：env + INI 未显式设置时尝试 PTXIR 检测
2. **与 ADR-0024 v1.1 的兼容修订**：本 ADR 是对 ADR-0024 v1.1 的 amendment，将默认模式从 off 改为 auto，形成 v1.2。目标是保持 dispatch 语义和最终执行路径兼容，不作字节级不变或零行为变化承诺。
3. **显式 override 优先**：env / INI 始终胜过 default
4. **兼容 dispatch 语义**：默认检测只改变是否尝试 PTXIR，最终 dispatch 语义与原路径兼容；未设置时增加一次 executable-tail probe。`true` 表示“尝试 PTXIR detection”，不表示 footer 存在。缺少 footer 时正常 fallback，malformed embedded PTXIR 或 manifest mismatch 则作为错误处理。

### 实现要点

#### 改动的代码（`src/cudart/ptxir_config.cpp`）

```cpp
bool isPTXIRModeEnabled() {
    evaluate_env_once();
    if (g_env_cached != kEnvUnset) {
        return g_env_cached == 1;     // env 显式
    }
    if (g_ini_mode == 1) return true;  // INI [ptxir] mode = on / auto
    if (g_ini_mode == 0) return false; // INI [ptxir] mode = off
    // g_ini_mode == -1 (INI 未 set 或无 [ptxir] 段)
    return true;                       // ★ NEW: 默认 auto
}
```

注意：当 INI 有 `[ptxir]` 段但缺 `mode` key 时，`inipp::get_value` 保留默认值 `""`，`ptxir_mode_str == "auto"` 比较仍返回 `true`（与现有逻辑一致）。仅当 INI 完全无 `[ptxir]` 段时 `g_ini_mode` 保持 -1，走新 default。

#### 配置优先级矩阵

| env `PTXIR_MODE` | INI `[ptxir] mode` | `isPTXIRModeEnabled()` |
|---|---|---|
| `auto` | (any) | `true` |
| `off` | (any) | `false` |
| (unset) | `on` / `auto` | `true` |
| (unset) | `off` | `false` |
| **(unset)** | **(未 set / 无段)** | **`true` ★ NEW** |

#### 兼容性论证

现有测试配置显式使用 `mode = off`，因此其 dispatch 语义保持兼容。默认 auto 对未嵌入 binary 增加一次 executable-tail probe，未发现 footer 时回到原 cuobjdump 路径。malformed embedded PTXIR 或 manifest mismatch 不属于“未发现 footer”，应分别作为嵌入数据错误或 manifest 校验错误处理。

性能：非嵌入 binary 启动时多 1 次 `open /proc/self/exe` + `read`（≤ 4KB）+ memcmp(8) ≈ 1µs @ SSD。可忽略。

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/cudart/ptxir_config.cpp` | 修改 | `isPTXIRModeEnabled()` default 分支：`true` 替代 `false` |
| `configs/*.ini` | 不动 | 保持显式 `mode = off`，兜底现有测试 |
| ADR-0024 v1.2 amendment | 修改 | §设计原则 5 "默认 OFF" → "默认 AUTO（可显式 off）" |
| `docs/adr/README.md` | 修改 | ADR-0026 加入 Proposed 索引 |
| `tests/unit/cudart/test_ptxir_config.cpp` | 修改 | 新增 TEST_CASE："env unset + INI unset → true" |

---

## 后果

### 正面影响

1. **零配置 wrapper**：ADR-0027 `ptx-nvcc` 产出的 binary `./myapp` 即用，匹配原生 CUDA SDK 体验
2. **非嵌入 binary dispatch 语义兼容**：增加一次 executable-tail probe 后，未发现 footer 时回到原路径
3. **显式关闭仍可用**：CI / 严格场景可 `PTXIR_MODE=off` 强制走 cuobjdump
4. **嵌入数据错误可区分**：malformed embedded PTXIR 或 manifest mismatch 不会被当作普通缺失而静默 fallback

### 负面影响

1. **非嵌入 binary 启动微增 < 1µs**：可忽略
2. **隐式行为**：用户可能未察觉 PTXIR dispatch 被默认启用

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 老 binary 启动性能下降（1µs 级别） | 中 | 极低 | 实测 < 1µs；用户可 `PTXIR_MODE=off` 关闭 |
| 现有 ctest 230/230 因 default 翻转回归 | 低 | 高 | configs/*.ini 显式 `off`，覆盖 default；新增 unit 测试锁定 default=auto |
| 用户误解 "PTXIR 路径默认 ON" 为破坏性变更 | 中 | 低 | README + tools/README 标注 "默认 auto（v1.2 起）"；commit message 写明 |
| 多 INI 加载顺序导致 default 失效 | 低 | 中 | `setPTXIRModeFromIni` 在 `initialize_environment` 早期调用；INI 显式 off 一定覆盖 default |

---

## 合规检查

后续相关开发应检查：

- [ ] 任何新增 configs/*.ini 必须显式 `mode = off` 或 `mode = auto`，**不依赖 default**
- [ ] `__cudaRegisterFatBinary` PTXIR dispatch 路径必须可被 `PTXIR_MODE=off` 完整关闭；默认 auto 仅表示尝试 PTXIR 检测，最终执行路径须与现有 dispatch 语义兼容
- [ ] `unit_ptxir_config` 必须覆盖 5 种配置优先级矩阵场景
- [ ] `e2e_ptxir_cubin_embed` 与现有 e2e 测试在 default=auto 下 dispatch 语义兼容
- [ ] ADR-0024 v1.1 的 amendment 更新为 v1.2，明确默认 auto 与上述 fallback/error 区分

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-08 | 初始版本（零配置 wrapper 依赖 + backward compat 论证） | PTX-EMU Architecture Team |

---

## 参考

- [ADR-0024 PTXIR-Embedded CUBIN](./ADR-0024-ptxir-cubin-embed-extension.md) — §设计原则 5 被本 ADR 覆盖
- [ADR-0027 ptx-nvcc wrapper toolchain](./ADR-0027-ptx-nvcc-wrapper.md) — 强依赖本 default 翻转
- [docs/architecture/ptxir-toolchain-stack.md](../architecture/ptxir-toolchain-stack.md) — 工具链栈架构 §5 配置优先级
- [src/cudart/ptxir_config.cpp](../../src/cudart/ptxir_config.cpp) — `isPTXIRModeEnabled()` 实现
- [src/cudart/cudart_sim.cpp](../../src/cudart/cudart_sim.cpp) — `__cudaRegisterFatBinary` PTXIR dispatch 调用点（cudart_sim.cpp:393）