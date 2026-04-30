# 三模式测试框架指南

**版本**: v1.0
**最后更新**: 2026-04-30
**适合人群**: 测试工程师，开发者

---

## 概述

三模式测试框架提供三种 PTX 加载模式，用于测试 PTX-EMU 模拟器，支持从端到端到精确单元测试的调试。

| 模式 | 描述 | 使用场景 |
|------|------|---------|
| **Mode 1** | cuobjdump 动态提取 | 端到端集成测试，CI/CD |
| **Mode 2** | 预提取的 PTX 文件 | 稳定复现，版本控制 |
| **Mode 3** | 直接构造 StatementContext | 单元测试，精确定位 |

---

## 快速开始

### 构建和运行

```bash
# 构建所有模式
cmake --build build --target test_mode1 test_mode2 test_mode3

# 运行所有三模式测试
ctest -R three_mode -V

# 运行特定模式
./build/bin/tests/test_mode3
```

### 自动生成测试

```bash
# 生成四模式测试
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark your_kernel

# 指定模式
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark your_kernel --mode mode3b

# 查看生成计划（不实际生成）
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark your_kernel --dry-run
```

---

## 模式详解

### Mode 1：cuobjdump 动态提取

完整端到端流程：

```
CUDA Binary → cuobjdump 提取 PTX → 解析 → CFG 构建 → 执行
```

```bash
# 提取 PTX
cuobjdump -xptx build/bin/YOUR_BINARY -arch=sm_100 > tests/three_mode_testing/ptx/your_test.ptx
```

**优点**：测试真实编译流程，发现编译器和链接器引入的问题。

---

### Mode 2：预提取 PTX 文件

跳过 cuobjdump 步骤，直接加载已保存的 PTX 文件：

```cpp
std::vector<StatementContext> stmts = load_ptx_file("tests/three_mode_testing/ptx/your_test.ptx");
```

**优点**：
- 稳定可复现
- 适合版本控制
- 不依赖 CUDA 工具链

---

### Mode 3：直接构造 StatementContext

最细粒度测试，直接用 helper 函数构造指令序列：

```cpp
std::vector<StatementContext> stmts = {
    make_mov("%r_lane", "%tid.x"),
    make_setp_lt("%p1", "%r_lane", "16"),
    make_bra_pred("L_path_b", "%p1", true),  // 条件分支
    make_bar_sync(0),                          // 屏障同步
    // ...
};
```

**Mode 3a vs Mode 3b**：

| 阶段 | `reconvergence_pc` | 说明 |
|------|-------------------|------|
| **Mode 3a** (CFG 构建前) | `-1` | 原始解析状态 |
| **Mode 3b** (CFG 构建后) | 已填充 | 最终执行版本 |

```
Mode 3a (BEFORE CFG):
  BranchInstr { reconvergence_pc = -1 }
  BarWarpSyncInstr { operands[1] = ? }  // 原始值

Mode 3b (AFTER CFG):
  BranchInstr { reconvergence_pc = 15 }  // CFG builder 填充
  BarWarpSyncInstr { operands[1] = 16 }  // 更新为 i+1
```

---

## 添加新测试

### 步骤 1：提取 PTX

```bash
cuobjdump -xptx build/bin/YOUR_KERNEL -arch=sm_100 > tests/three_mode_testing/ptx/your_kernel.ptx
```

### 步骤 2：分析 PTX 结构

```bash
# 检查关键指令
grep -E "bar\.|ld\.shared|st\.shared|bra" tests/three_mode_testing/ptx/your_kernel.ptx
```

### 步骤 3：生成测试文件

```bash
python3 docs/skills/three-mode-testing/generate_tests.py --benchmark your_kernel --force
```

### 步骤 4：构建

```bash
cmake --build build --target test_your_kernel_mode1 test_your_kernel_mode2 test_your_kernel_mode3a test_your_kernel_mode3b
```

### 步骤 5：运行

```bash
ctest -R "your_kernel" -V
```

---

## 常用 Helper 函数

详见 [`test_helpers.hpp`](https://github.com/chisuhua/PTX-EMU/blob/main/tests/three_mode_testing/test_helpers.hpp)。

### 指令构造

```cpp
make_mov(dest, src)           // 寄存器移动
make_add(dest, src1, src2)    // 加法
make_setp_lt(pred, src1, src2) // 条件设置
make_bra_pred(label, pred, is_uniform) // 条件分支
make_bar_sync(bar_id)         // 屏障同步
```

### 屏障测试

```cpp
Wbar& wbar = warp.get_warp_state().wbars[0];
wbar.init(0xFFFFFFFF, reconvergence_pc);
for (int i = 0; i < 32; i++) wbar.arrive(i);
warp.set_active_mask(wbar.arrived_mask);
```

### 共享内存测试

```cpp
void* shmem = allocate_shared(32);
write_shared(shmem, lane, value);
uint32_t val = read_shared(shmem, lane);
```

### 模式 3 核心函数

```cpp
// 从 StatementContext 向量创建 KernelLaunchRequest (Mode 3)
inline KernelLaunchRequest make_kernel_request(
    std::vector<StatementContext>& statements,
    std::map<std::string, Symtable*>& name2Sym,
    std::map<std::string, int>& label2pc,
    void** args = nullptr,
    Dim3 gridDim = {1, 1, 1},
    Dim3 blockDim = {32, 1, 1},
    size_t sharedMem = 0);
```

---

## 调试工作流

```
Mode 1 (发现问题)
    ↓ 提取 PTX
Mode 2 (稳定复现)
    ↓ 分析结构
Mode 3a (原始解析状态)
    ↓ 理解解析
Mode 3b (CFG 处理后)
    ↓ 对比 3a 观察 CFG 效果
定位根因
    ↓
修复源码
    ↓ 验证
Mode 2 (回归测试)
    ↓
Mode 1 (端到端)
```

---

## 目录结构

```
tests/three_mode_testing/
├── CMakeLists.txt           # 自动检测所有 *_mode*.cpp
├── README.md                # 本文档索引
├── SKILL.md                 # 技能文档
├── test_helpers.hpp         # 公共辅助函数
├── test_mode1.cpp           # Mode 1 模板
├── test_mode2.cpp           # Mode 2 模板
├── test_mode3.cpp           # Mode 3 模板
├── ptx/                     # 预提取 PTX 文件
│   └── *.ptx
└── golden/                  # 期望输出
    └── *.expected
```

---

## CMake 自动检测

CMakeLists.txt 自动检测所有 `*_mode*.cpp` 文件：

```cmake
file(GLOB THREE_MODE_SOURCES CONFIGURE_DEPENDS "*.cpp")
foreach(source IN LISTS THREE_MODE_SOURCES)
    if(basename MATCHES "_mode[0-9]+a?\\.cpp$")
        add_executable(${test_name} ${source} ${THREE_MODE_BASE})
        # ...
    endif()
endforeach()
```

**无需手动注册** —— 只需添加 `test_foo_modeN.cpp`，重新配置即可。

---

## 示例：分析分支重汇合

```cpp
// Mode 3a: CFG 构建前
std::vector<StatementContext> stmts = parse_ptx(ptx);
auto& bra = std::get<BranchInstr>(stmts[5].data);
INFO("Mode3a reconvergence_pc = " << bra.reconvergence_pc);  // -1

// Mode 3b: CFG 构建后
apply_cfg_builder(stmts, label2pc);
auto& bra2 = std::get<BranchInstr>(stmts[5].data);
INFO("Mode3b reconvergence_pc = " << bra2.reconvergence_pc);  // 已填充
```

---

## 技能脚本使用

生成器脚本：`docs/skills/three-mode-testing/generate_tests.py`

```bash
# 所有模式（默认）
python3 generate_tests.py --benchmark dummy

# 仅特定模式
python3 generate_tests.py --benchmark dummy --mode mode3b

# 从 CUDA 源码直接生成
python3 generate_tests.py --cuda-source path/to/kernel.cu

# 从二进制文件生成
python3 generate_tests.py --binary build/bin/kernel

# 从现有 PTX 生成
python3 generate_tests.py --ptx path/to/kernel.ptx

# 预览模式（不实际生成）
python3 generate_tests.py --benchmark dummy --dry-run

# 强制覆盖已有文件
python3 generate_tests.py --benchmark dummy --force
```

---

## 相关文档

| 文档 | 路径 |
|------|------|
| 测试指南 | [`TESTING-GUIDE.md`](./TESTING-GUIDE.md) |
| 技能文档 | [`../skills/three-mode-testing/SKILL.md`](../../skills/three-mode-testing/SKILL.md) |
| 测试生成器 | [`../skills/three-mode-testing/generate_tests.py`](../../skills/three-mode-testing/generate_tests.py) |

---

**最后更新**: 2026-04-30