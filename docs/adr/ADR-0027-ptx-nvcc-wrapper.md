# ADR-0027: `ptx-nvcc` nvcc 兼容 wrapper 工具链

| 属性 | 值 |
|------|-----|
| **状态** | Proposed |
| **日期** | 2026-08-08 |
| **关联任务** | T13.3（`feat-ptxir-nvcc-toolchain` Phase 4） |
| **关联 PR** | TBD |
| **作者** | PTX-EMU Architecture Team |
| **审核人** | Oracle（wrapper 行为 review）、Metis（决策完备性） |

---

## 上下文

### 问题背景

`feat-implement-ptxir-cubin-embed-extension` 归档后，工具链 5 步断成 2 段：

```
# 用户手动 5 步
1. nvcc -arch=sm_100 myapp.cu -o myapp             # 编译
2. cuobjdump --ptx myapp > /tmp/myapp.ptx           # 提 PTX
3. ptxir_build --in /tmp/myapp.ptx \                # PTX→PTXIR
                --kernel-name <K> --out /tmp/myapp.ptxir
4. ptxir_embed --in-exe myapp \                    # 嵌入
                --in-ptxir /tmp/myapp.ptxir \
                --kernel-name <K> --out myapp
5. LD_PRELOAD=./build/lib/libcudart.so.12 \
   PTXIR_MODE=auto ./myapp                         # 运行
```

5 步中有 4 步手动，与 "原生 CUDA SDK 一样的运行方式" 目标冲突。用户期望：

```bash
# 原生 CUDA SDK 体验
ptx-nvcc -arch=sm_100 myapp.cu -o myapp
./myapp   # 无 LD_PRELOAD / 无 env
```

### 触发事件

1. 用户明确要求 "NVIDIA sdk 工具兼容的工具链" + "无感的用原来 cuda sdk 一样的运行方式"
2. ADR-0025（`ptxir_build`）+ ADR-0026（默认 auto）补齐了第 3 步与第 5 步的零配置基础
3. `tools/README.md` §场景 1-3 已写出 end-to-end 流程，但缺自动化

### 技术约束

- **不修改 nvcc**：避免与 CUDA toolkit 共存冲突；wrapper 是 standalone 工具
- **不修改 libcudart ABI**：DT_RUNPATH 解决新构建 binary 的 runtime 加载，不引入新 API
- **PATH 可执行**：wrapper 放在 `build/bin/` 或 `tools/`，用户 `export PATH=...`
- **POSIX shell 兼容**：Python 3 实现，无 bash 专有语法
- **单 kernel v1**：wrapper 只接受恰好一个 `.entry`；显式 `--kernel-name` 仅选择该唯一 entry，不绕过多 kernel 校验。多 kernel 行为延后，ADR-0028 文件目前不存在，待 manifest 与 runtime selection 语义明确后再提出

---

## 决策驱动因素

1. **factor 1 — 用户体验等价原生 CUDA SDK**：`./myapp` 即用，无额外步骤
2. **factor 2 — 不污染 nvcc**：wrapper 独立，不修改 CUDA toolkit
3. **factor 3 — 与现有工具解耦**：subprocess 调用 `ptxir_build` / `ptxir_embed`，不内嵌逻辑
4. **factor 4 — 透明 passthrough**：未知 nvcc 参数透传给真 nvcc
5. **factor 5 — 可移植**：`DT_RUNPATH` 默认使用绝对路径，支持 `--ptxemu-root` 覆盖；该选项只影响新构建 binary。

---

## 考虑的替代方案

### 方案 A: Python wrapper script `tools/ptx-nvcc`（✅ 选中）

**描述**：~120 行 Python，argparse 解析 + subprocess 编排

**优点**：
- 与 nvcc 共存好（独立 PATH entry）
- subprocess 调用现成工具（ADR-0025 `ptxir_build` + `ptxir_embed`），零重复逻辑
- argparse 易测试、易扩展
- Python 3 在 PTX-EMU dev 环境已是依赖（`env.sh` 检测）

**缺点**：
- Python 解释器依赖（系统默认有）
- 启动延迟 ~30ms（argparse + subprocess fork）— 对编译流水线可忽略

**选择理由**：最低风险实现，与 ADR-0025/0026 解耦。

### 方案 B: 自定义 Clang/LLVM 编译器驱动（❌ 未采用）

**描述**：基于 LLVM 写 `-cc1` 风格的 PTX-EMU 编译器驱动，拦截 nvcc 调用

**优点**：
- 与 nvcc 真正的"编译驱动"等价

**缺点**：
- **工作量极大**：需 fork 或包装 clang driver（数月）
- **PTX-EMU 目标是模拟**，不替代 nvcc 的 PTX 生成
- **维护负担**：CUDA toolkit 升级时同步

**未采用理由**：远超本 change scope。

### 方案 C: CMake `ptxemu_add_executable()`（✅ v2 候选，v1 不采用）

**描述**：CMake 函数封装 wrapper 调用，CMakeLists 一行替换 `add_executable`

**优点**：
- CMake-native，用户友好
- 与 build system 集成

**缺点**：
- v1 用户可能不用 CMake（直接 Makefile / shell）
- 范围 > wrapper

**未采用理由**：v2 增强，v1 优先 wrapper 的最大兼容性。CMake 函数可包装 wrapper。

### 方案 D: LD_PRELOAD wrapper runner `ptxrun`（❌ 未采用）

**描述**：`ptxrun ./myapp` → 设 env → exec myapp

**优点**：
- libcudart 无改动

**缺点**：
- **不原生**：用户必须用 `ptxrun` 而非 `./myapp`
- 违背 "原生 SDK 体验" 目标

**未采用理由**：明确违反用户核心需求。

---

## 决策内容

### 设计原则

1. **Wrapper 形式**：Python 脚本（`tools/ptx-nvcc`），PATH 可执行
2. **薄编排**：仅 argparse + subprocess；不内嵌 PTX/PTXIR/ELF 逻辑
3. **透传 nvcc args**：未识别的参数原样传给真 `nvcc`
4. **DT_RUNPATH 注入**：链接时 `-Wl,-rpath,<ptxemu_root>/build/lib` + `-L <ptxemu_root>/build/lib -lcudart`，由链接器生成 `DT_RUNPATH`。
5. **可覆盖**：用户 `--ptxemu-root <path>` 只影响本次新链接 binary 的 runtime search path 和工具位置，不修改已有 binary。

### 实现要点

#### CLI 接口

```
ptx-nvcc [nvcc args...] -o <exe> [--ptxemu-root <path>] [--kernel-name <K>] [--no-embed]
```

- **透传**：所有 nvcc 参数原样转发（`-arch`, `-O3`, `-std=c++17`, `-I`, `-L`, `-l`, 等）
- **`-o <exe>`**：最终可执行文件路径（必填，nvcc 要求）
- **`--ptxemu-root <path>`**：PTX-EMU 项目根（默认 = `os.path.dirname(__file__)/..`，即 wrapper 脚本相对路径）
- **`--kernel-name <K>`**：目标 kernel 名（默认 = 从 PTX 文本 grep `\.entry\s+(\w+)` 首个匹配）
- **`--no-embed`**：跳过 PTXIR embed（仅 link 并注入 DT_RUNPATH libcudart；用于调试）
- **`--help`**：打印 usage + 示例

#### 编排流程（伪代码）

```python
def main(argv):
    nvcc_args, custom = split_custom_args(argv)
    # custom contains --ptxemu-root, --kernel-name, --no-embed.
    # Every other argument remains in nvcc_args, in its original order.
    exe = require_output_executable(nvcc_args)  # missing -o is a usage error
    root = custom.ptxemu_root or default_ptxemu_root()
    libcudart_dir = path.join(root, "build", "lib")
    tools_dir = path.join(root, "build", "bin")
    temp_dir = tempfile.mkdtemp(prefix="ptx-nvcc-")
    obj_path = path.join(temp_dir, "kernel.o")
    ptx_path = path.join(temp_dir, "kernel.ptx")
    ptxir_path = path.join(temp_dir, "kernel.ptxir")

    try:
        # Compile only. Do not ask nvcc to link before the PTX is extracted.
        run(["nvcc", *nvcc_args, "-c", "-o", obj_path])
        link_args = [
            *nvcc_args_without_compile_only_or_output,
            obj_path,
            "-L", libcudart_dir,
            "-lcudart",
            f"-Wl,-rpath,{libcudart_dir}",
            "-o", exe,
        ]
        run(["nvcc", *link_args])

        if not custom.no_embed:
            run(["cuobjdump", "--ptx", exe], stdout_path=ptx_path)
            kernel_name = custom.kernel_name or first_entry_from_ptx(ptx_path)
            if kernel_name is None:
                fail_data(2, "no .entry found; specify --kernel-name")
            if more_than_one_entry(ptx_path):
                fail_data(2, "v1 supports one kernel per binary")
            run([path.join(tools_dir, "ptxir_build"), "--in", ptx_path,
                 "--kernel-name", kernel_name, "--out", ptxir_path])
            run([path.join(tools_dir, "ptxir_embed"), "--in-exe", exe,
                 "--in-ptxir", ptxir_path, "--kernel-name", kernel_name,
                 "--out", exe])
    except ToolFailure as error:
        return propagate_tool_exit(error)
    finally:
        # Remove the temporary directory and every known temporary file.
        # No shell wildcard is used.
        shutil.rmtree(temp_dir, ignore_errors=True)
```

The wrapper must create the temporary directory and the three named files explicitly. It must never pass a shell wildcard such as `*.o` to `nvcc`, and it must remove the object as well as the PTX and PTXIR files on success or failure. A subprocess failure is propagated as a tool failure, preserving the child exit status when it is available and otherwise returning exit code 3.

`--no-embed` still performs compile-only object creation and the final link, then stops before `cuobjdump`, `ptxir_build`, and `ptxir_embed`. The wrapper preserves the user's passthrough arguments and does not rewrite unrelated nvcc options. `--ptxemu-root` selects the PTX-EMU library and tools for newly built binaries only. It does not alter already built binaries or runtime selection for binaries produced elsewhere.

#### Runtime 行为

- 链接后，binary 的 `.dynamic` section 含 `NEEDED libcudart.so.12` + `DT_RUNPATH <ptxemu_root>/build/lib`
- 运行 `./myapp` 时 dynamic loader 解析 `DT_RUNPATH`，加载 PTX-EMU libcudart
- 默认 `PTXIR_MODE=auto`（ADR-0026）尝试 executable-tail detection，成功后走 PTXIR dispatch
- 用户体验：`./myapp` 与 `nvcc` + 系统 libcudart 的调用方式一致，运行时实现仍是 PTX-EMU 模拟器


### v1 限制（明示在 `--help`）

- **单 kernel per binary**：自动检测到的 `.entry` 必须只有一个；显式 `--kernel-name` 也不能绕过 v1 的单 kernel 限制。**[ADR-0028]**（**[BLOCKING DEPENDENCY]**）待 ship 后解除；解除时需 bump `PTXIR_VERSION` per ADR-0023 Extend-Only 原则。
- **DT_RUNPATH 绝对路径**：默认 `<wrapper_dir>/../build/lib`；可用 `--ptxemu-root` 为新构建 binary 覆盖
- **nvcc 透传**：所有非自定义参数透传；wrapper 不解析 nvcc 输出
- **Linux only**：POSIX ELF 与 DT_RUNPATH；macOS / Windows 需后续适配

### 与 ADR-0029 in-memory 路径的边界（**互斥关系**）

> **2026-08-09 修订**：本 wrapper 路径（DT_RUNPATH + libcudart.so）与 ADR-0029 in-memory 路径（`cuModuleLoadData` + `libptxemu_device.so`）是 **两条互斥的运行时集成路径**，分别面向不同使用场景。

| 场景 | 推荐路径 | 原因 |
|------|---------|------|
| 用户直接编译 `.cu` → 运行 `./myapp`（无 KMD/CP 介入） | **本 wrapper**（ADR-0027） | DT_RUNPATH 注入 libcudart.so 即可；最简用户体验 |
| UsrLinuxEmu/TaskRunner CP 端集成（KMD/CP 调度） | **HAL 扩展方案**（ADR-0029 D8） | 单一 GPU 状态来源；TaskRunner 不直接 link PTX-EMU；符合 UsrLinuxEmu 三区分架构 |
| Standalone PTX-EMU 用户直接调 `cuModuleLoadData` | **D8-Alt 直链**（ADR-0029 记录备查） | 仅在无 UsrLinuxEmu 介入的极简 PTX 仿真场景使用 |

**互斥约束**：
- 同一 binary **不能同时**（a）DT_RUNPATH 依赖 PTX-EMU `libcudart.so.12` **且**（b）通过 `libptxemu_device.so` link 加载 in-memory module——会出现两个 GPU 仿真器实例（PTX-EMU `GPUContext` ×2）并互相覆盖全局状态
- 如需切换路径，必须重新编译 binary 并移除对侧的依赖
- 检测方法：`ldd <binary>` 若同时输出 `libcudart.so.12 → <ptxemu>` 与 `libptxemu_device.so → <ptxemu>` 则配置错误

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `tools/ptx-nvcc` | 新增 | ~120 行 Python 脚本 + shebang `#!/usr/bin/env python3` |
| `tools/CMakeLists.txt` | 修改 | `install(PROGRAMS ptx-nvcc ...)` 或 symlink 到 `build/bin/ptx-nvcc` |
| `tools/README.md` | 修改 | 新增 §ptx-nvcc 段（end-to-end 示例 + v1 限制） |
| `README.md` | 修改 | "快速开始" 段加 ptx-nvcc 用法 |
| `tests/e2e/kernel/test_ptxir_nvcc_wrapper.cpp` | 新增 | e2e：用 wrapper 编译 `bench/dummy/dummy.cu`，运行 `./dummy`，断言 PASS |
| `docs/architecture/ptxir-toolchain-stack.md` | 引用 | §3 build-time data flow §4 runtime data flow |

---

## 后果

### 正面影响

1. **零配置原生体验**：用户 `ptx-nvcc` + `./myapp` 即用，匹配原生 CUDA SDK
2. **端到端工具链完整**：5 步变 1 步
3. **DT_RUNPATH 自动注入**：runtime libcudart 加载无需 LD_PRELOAD
4. **wrapper 复用现成工具**：subprocess 调用 `ptxir_build` / `ptxir_embed`，无逻辑重复
5. **可调试性**：用户可手动跑 wrapper 各 step 验证

### 负面影响

1. **Python 依赖**：wrapper 需 Python 3（系统默认有）
2. **v1 单 kernel**：多 kernel binary 失败
3. **DT_RUNPATH 绝对路径**：可移植性受限（用户 `--ptxemu-root` 缓解）
4. **wrapper 启动延迟 ~30ms**：可忽略

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| nvcc 参数透传不全（如 `-Xptxas`、`--use_fast_math`、`-gencode`） | 中 | 中 | 透传所有非 `--ptxemu-root/--kernel-name/--no-embed/--help` 参数；e2e 测试覆盖常见 flag |
| 自动检测 kernel 名失败（PTX 内无 `.entry`） | 低 | 高 | 报错 "no .entry found in <ptx>; specify --kernel-name" |
| DT_RUNPATH 硬编码绝对路径限制 install 到 `/opt/ptxemu` 等 | 中 | 中 | `--ptxemu-root` 仅影响新构建 binary；v2 再评估 `$ORIGIN` |
| 用户系统 nvcc 与 PTX-EMU build 不同步（如 arch 9.0 vs 10.0） | 中 | 低 | DT_RUNPATH 指向的库不依赖 nvcc 版本；embed 后 PTX 走 PTX-EMU 解析，与 nvcc 编译参数无关 |
| 单 kernel 限制让多 kernel app 失败 | 低 | 中 | v1 报错明示；v2 多 kernel 设计延期，ADR-0028 文件目前不存在 |
| wrapper Python 与系统 Python 不兼容 | 极低 | 中 | shebang `#!/usr/bin/env python3`；无 Python 3 专属语法 |

---

## 合规检查

后续相关开发应检查：

- [ ] `bench/dummy/dummy.cu` 通过 `ptx-nvcc` 编译 + `./dummy` 运行 PASS（e2e）
- [ ] `bench/test_syncthreads/test_syncthreads.cu` 通过 wrapper 编译 + 运行 PASS
- [ ] `tools/README.md` 给出 end-to-end 命令示例 + v1 限制
- [ ] README.md "快速开始" 加 wrapper 用法
- [ ] 不影响现有 `bench/` 直接 nvcc 编译路径（保留两条路径）
- [ ] `--ptxemu-root` 默认值在 wrapper 路径异常时（pip install / 移动位置）报错并 exit 1
- [ ] v2: 多 kernel manifest，ADR-0028 文件目前不存在+ `$ORIGIN` DT_RUNPATH + CMake `ptxemu_add_executable()`

---

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-08-08 | 初始版本（Python wrapper + subprocess 编排 + DT_RUNPATH 注入） | PTX-EMU Architecture Team |
| 2026-08-09 | **跨仓评审修订**: §v1 限制明示 ADR-0028 BLOCKING DEPENDENCY 标注；新增 §互斥关系段，clarify 本 wrapper 路径与 ADR-0029 in-memory 路径是两条互斥的运行时集成路径（同一 binary 不能同时 DT_RUNPATH libcudart.so + link libptxemu_device.so，会导致双 GPUContext 冲突） | PTX-EMU Architecture Team |

---

## 参考

- [ADR-0024 PTXIR-Embedded CUBIN](./ADR-0024-ptxir-cubin-embed-extension.md) — `libcudart.so` PTXIR dispatch 实现
- [ADR-0025 ptxir_build CLI](./ADR-0025-ptxir-build-cli.md) — wrapper Step 3 依赖
- [ADR-0026 PTXIR dispatch default auto](./ADR-0026-ptxir-default-mode-auto.md) — wrapper runtime 零配置基础
- [docs/architecture/ptxir-toolchain-stack.md](../architecture/ptxir-toolchain-stack.md) — 工具链栈架构总览
- NVIDIA CUDA Compiler Driver NVCC Documentation (https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)