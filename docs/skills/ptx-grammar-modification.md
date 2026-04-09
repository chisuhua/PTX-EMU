# PTX 语法修改技能指南

> **技能名称**: `ptx-grammar-modification`  
> **适用范围**: PTX-EMU 项目的 ANTLR4 语法文件修改  
> **最后更新**: 2026 年 3 月

---

## 🚨 触发条件识别（READ FIRST）

**在开始任何修复工作前，先检查是否有以下错误模式。如有任一匹配，🛑 立即停止并遵循本流程**：

| 错误模式 | 典型错误信息 | 触发流程？ |
|---------|-------------|-----------|
| **ANTLR 解析错误** | `no viable alternative at input '...'` | ✅ 是 |
| **意外 Token** | `mismatched input 'X' expecting Y` | ✅ 是 |
| **缺少 Token** | `extraneous input 'X' expecting EOF` | ✅ 是 |
| **PTX 文件解析失败** | `Failed to parse PTX file: ...` | ✅ 是 |
| **Segmentation Fault** | 在 parser 阶段崩溃 | ✅ 是 |
| **修改了 .g4 文件** | 任何 `src/grammar/*.g4` 的改动 | ✅ 是 |
| **普通 C++ 编译错误** | 与 `.cpp`/`.h` 相关的错误 | ❌ 否 |
| **运行时逻辑错误** | 指令执行结果不对 | ❌ 否 |

**快速决策树**：

```
遇到错误 → 是解析阶段？ → 是 → 🛑 STOP → 阅读本文档 → 运行 test_all_ptx.sh
                          → 否 → 继续普通调试流程
```

---

## 何时使用此技能

### ✅ 适用场景
- 更新 `src/grammar/ptxLexer.g4` 或 `src/grammar/ptxParser.g4` 中的语法规则
- 添加新的 PTX 指令语法以支持额外操作
- 修复解析错误（如 "no viable alternative"、意外 token 等）
- 语法修改后重新生成解析器
- 理解 PTX-EMU 项目中的 ANTLR4 集成

### ❌ 不适用场景
- 修改 PTX 指令实现（`src/ptxsim/instructions/` 中的执行逻辑）
- 更改 CUDA 运行时 API 行为（`src/cudart/` 中的文件）
- 与语法文件无关的通用 C++ 开发
- 添加新的 GPU 架构配置（`configs/` 中的 JSON 文件）

---

## 快速参考

| 任务 | 位置/说明 |
|------|----------|
| **语法文件** | `src/grammar/ptxLexer.g4`、`src/grammar/ptxParser.g4` |
| **重新生成解析器** | `cmake --build build --target GenerateParser` |
| **生成的解析器输出** | `build/antlr4_generated_src/` |
| **ANTLR 版本** | 4.13.1（见 `CMakeLists.txt`） |
| **测试解析** | `./build/bin/test-ptx <file.ptx>` |
| **批量 PTX 测试** | `./tests/ptx/test_all_ptx.sh` |
| **语法检查** | 使用 `antlr4` 工具（需要 Java） |
| **环境配置** | 构建前运行 `. env.sh`（设置 CUDA_PATH、LD_LIBRARY_PATH） |
| **PTX 文档参考** | `docs/ptx/README.md` - 按指令分类的章节 |

---

## 核心概念

- **ANTLR4 语法结构**: 分为词法分析器（`ptxLexer.g4`）和语法分析器（`src/grammar/ptxParser.g4`）
- **词法规则**: 匹配 token（大写字母名称，如 `PARAM: '.param' ;`）
- **语法规则**: 定义语法结构（小写字母名称，如 `paramDecl : PARAM type ... ;`）
- **项目集成**: 生成的解析器编译到 `cudart` 目标中
- **重新生成流程**: CMake 调用 ANTLR4 Java 工具重新生成 C++ 解析器代码
- **测试策略**: PTX 测试套件在语法更改后验证解析正确性
- **PtxListener/PtxVisitor 模式**: 生成的解析器使用访问者模式；语义分析在 `ptx_visitor.cpp` 中
- **X-Macro 与语法关系**: 语法 token 对应 `include/ptx_ir/ptx_op.def` 中的 X-Macro 条目

---

## 开发工作流程（MANDATORY）

> ⚠️ **重要**: 修改 PTX 语法时，**必须**遵循以下流程。这是测试驱动开发（TDD）原则在项目中的具体应用。

### 📋 开始前的自我检查清单

**在写任何代码之前，先确认以下几点**：

- [ ] **我确认这是语法解析问题**（见"触发条件识别"表格）
- [ ] **我已经阅读了 `docs/ptx/` 对应章节**（了解 PTX 语法规范）
- [ ] **我已经运行了 `./tests/ptx/test_all_ptx.sh`**（验证当前状态）
- [ ] **我准备好了测试用例**（如果有真实 binary，已用 `cuobjdump` 提取）
- [ ] **我理解了 TDD 流程**（RED → GREEN → REFACTOR）

**如果以上任一为 ❌，请🛑 停止并先完成该项**。

---

### 流程概览

```
┌─────────────────────────────────────────────────────────┐
│  PTX 语法解析开发工作流程                                │
├─────────────────────────────────────────────────────────┤
│  1️⃣ 代码改动前                                          │
│     → 阅读 docs/ptx/ 目录下对应章节                      │
│     → 理解 PTX 语法规范                                  │
│                                                         │
│  2️⃣ 验证方式                                            │
│     → 执行 ./tests/ptx/test_all_ptx.sh                  │
│     → 快速验证解析是否通过                              │
│                                                         │
│  3️⃣ 调试流程（测试失败时）                              │
│     → cuobjdump -xptx <binary>  dump PTX               │
│     → cp /tmp/dumped.ptx tests/ptx/<test_name>.ptx     │
│     → 修改 test_all_ptx.sh 添加新用例                   │
│     → 运行脚本修正语法解析问题                          │
└─────────────────────────────────────────────────────────┘
```

### 详细步骤

#### 步骤 1: 阅读 PTX 文档（代码改动前）

在修改任何语法文件**之前**，必须先理解 PTX 语法规范：

```bash
# 1. 查看 docs/ptx/ 目录结构
ls docs/ptx/

# 2. 阅读对应章节
# 例如：修改整数指令语法 → 阅读 docs/ptx/9.7.1_integer_arith.md
#      修改控制流语法 → 阅读 docs/ptx/9.7.12_control_flow.md

# 3. 参考章节索引
cat docs/ptx/README.md
```

**文档章节对应关系**：

| 语法类别 | 文档章节 |
|---------|---------|
| 整数算术 | `docs/ptx/9.7.1_integer_arith.md` |
| 扩展精度整数 | `docs/ptx/9.7.2_integer_extended.md` |
| 浮点与混合精度 | `docs/ptx/9.7.3-5_float.md` |
| 比较与选择 | `docs/ptx/9.7.6-7_cmp_sel.md` |
| 位操作 | `docs/ptx/9.7.8_bitwise.md` |
| 数据移动与转换 | `docs/ptx/9.7.9_datatransfer_cvtx.md` |
| 纹理 | `docs/ptx/9.7.10-11_texture.md` |
| 控制流 | `docs/ptx/9.7.12_control_flow.md` |
| 同步与通信 | `docs/ptx/9.7.13_sync_comm.md` |
| 栈/视频/杂项 | `docs/ptx/9.7.17-19_dbg_misc.md` |
| Tensor Core | `docs/ptx/9.7.16_tcgen05.md` |

#### 步骤 2: 运行测试验证

```bash
# 运行批量 PTX 语法测试
./tests/ptx/test_all_ptx.sh

# 或运行特定测试
cd build && ctest -L ptx -V
```

#### 步骤 3: 调试流程（测试失败时）

当测试失败时，使用以下 TDD 流程：

##### RED - 识别失败

```bash
# 1. 运行测试查看失败
./tests/ptx/test_all_ptx.sh

# 2. 记录解析错误信息
# 例如："no viable alternative at input '.param.u64.ptr'"
```

##### GREEN - 修复语法

```bash
# 1. 从二进制文件提取真实 PTX 代码（如果有真实用例）
cuobjdump -xptx <binary_file> > /tmp/dumped.ptx

# 2. 复制到测试目录
cp /tmp/dumped.ptx tests/ptx/<test_name>.ptx

# 3. 修改词法分析器（src/grammar/ptxLexer.g4）
# 添加缺失的 token，例如：
#   PTR   : '.ptr' ;
#   ALIGN : '.align' ;

# 4. 修改语法分析器（src/grammar/ptxParser.g4）
# 更新相关规则以支持新语法

# 5. 重新生成解析器
cmake --build build --target GenerateParser

# 6. 重新编译受影响的目标
cmake --build build --target cudart

# 7. 运行测试验证修复
./tests/ptx/test_all_ptx.sh
```

##### REFACTOR - 验证并清理

```bash
# 1. 运行完整测试套件
cd build && ctest

# 2. 运行 PTX 相关测试
ctest -L ptx

# 3. 确认相关测试通过
ctest -R test_memory_manager -V

# 4. 一起提交语法文件 + 生成的文件
git add src/grammar/*.g4 build/antlr4_generated_src/
```

---

## 实战案例：修复 `.param .u64 .ptr .align` 声明

**场景**: PTX 文件包含 `.param .u64 .ptr .align 1`，但解析器报错 "no viable alternative"。

### RED - 识别失败

```bash
# 1. 运行测试查看解析失败
cd build && ctest -R test_memory_manager -V
# 查找输出中的解析错误

# 2. 验证语法规则缺失 token
# 检查 src/grammar/ptxDeclarations.g4 中的 paramDecl 规则
```

### GREEN - 修复语法

```bash
# 1. 在词法分析器中添加缺失的 token（ptxLexer.g4）
PTR   : '.ptr' ;
ALIGN : '.align' ;

# 2. 更新语法规则（ptxDeclarations.g4）
paramDecl
    : PARAM paramTokens
    | typeSpecifier? vectorSpec? ID
    ;

paramTokens
    : (typeSpecifier | PTR | ALIGN | IMMEDIATE | ID)+
    ;

# 3. 重新生成解析器
cmake --build build --target GenerateParser

# 4. 重新编译受影响的目标
cmake --build build --target cudart

# 5. 运行测试验证修复
cd build && ctest -R test_memory_manager -V
```

### REFACTOR - 验证并清理

```bash
# 运行完整测试套件
cd build && ctest

# 运行 PTX 相关测试
ctest -L ptx

# 一起提交语法文件和生成的文件
```

---

## 失败反思与常见错误模式

### 🤔 为什么 AI 会跳过流程直接修改？

根据历史记录，AI 代理犯错的典型原因：

| 错误原因 | 表现 | 如何避免 |
|---------|------|---------|
| **误判任务性质** | 看到"修复"就当作普通 bug | 先检查错误模式表格，确认是否为解析错误 |
| **没有读取 AGENTS.md** | 忽略项目顶层流程要求 | 开始工作前必读 AGENTS.md 前 50 行 |
| **急躁冒进** | 直接修改 .g4 文件 | 🛑 STOP 原则：看到错误先停止，再思考 |
| **缺少流程意识** | 不知道有 TDD 流程 | 本文档开头的"触发条件识别"就是为此设计 |

### 🛡️ 防御性编程：AI 自我检查提示词

**当你准备修复 PTX 相关问题时，先问自己**：

```
1. 这是什么阶段的错误？
   - 编译时？运行时？解析时？
   - 解析时 → 🛑 触发 PTX 语法修改流程

2. 错误信息包含什么关键词？
   - "no viable alternative" → 🛑 语法流程
   - "mismatched input" → 🛑 语法流程
   - "Segmentation fault" + parser → 🛑 语法流程

3. 我要修改什么文件？
   - src/grammar/*.g4 → 🛑 语法流程
   - src/ptxsim/*.cpp → 普通 C++ 调试

4. 我是否已经阅读了 docs/ptx/ 对应章节？
   - 否 → 🛑 先阅读文档
   - 是 → 继续

5. 我是否已经运行了 test_all_ptx.sh？
   - 否 → 🛑 先运行测试
   - 是 → 继续
```

**如果以上任一答案为🛑，立即停止并遵循流程**！

---

## 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| **忘记重新生成解析器** | 语法更改后未运行 `GenerateParser` 目标 | 语法修改后**必须**运行 `cmake --build build --target GenerateParser` |
| **破坏现有语法** | 更改的规则被其他地方使用 | 始终运行完整 `ctest -L ptx` 检查回归 |
| **Token 定义错误** | 词法 token 与 PTX 语法不匹配 | Token 必须精确匹配 PTX 语法（包括前导点，如 `.param`） |
| **手动编辑生成的文件** | 直接修改 `build/antlr4_generated_src/` 中的文件 | **永远不要**手动编辑生成的文件 |
| **未运行完整 PTX 测试** | 某些边界情况只在特定 PTX 文件中出现 | 始终运行 `ctest -L ptx` 或 `./tests/ptx/test_all_ptx.sh` |
| **忽略 AGENTS.md 中的反模式** | 已知限制（WMMA stub、原子操作、Hopper 不支持） | 参见 AGENTS.md 中的 ANTI-PATTERNS 章节 |
| **未阅读 PTX 文档** | 修改前不理解 PTX 语法规范 | **必须**先阅读 `docs/ptx/` 对应章节 |

---

## 真实影响

**最近修复**（commit b157c55）：语法更改从 `paramList` 规则中移除 COMMA，并添加 `PTR` token 以支持 `.param .u64 .ptr .align 1` 声明。

- **问题**: 解析器遇到 `.ptr .align` 序列时报错 "no viable alternative"
- **修复**: 在词法分析器中添加 `PTR` token，修改 `paramDecl` 规则接受 `paramTokens`，重新生成解析器
- **测试**: 始终使用 `./build/bin/test-ptx` 验证，并在提交前运行 `ctest -L ptx`

---

## 相关文档

- [AGENTS.md](../../AGENTS.md) - 构建命令和测试工作流
- [docs/ptx/README.md](../ptx/README.md) - PTX ISA 章节索引
- [docs/debugging_guide.md](../debugging_guide.md) - 日志和调试配置
- [docs/arch.md](../arch.md) - 系统架构

---

## 外部资源

- [ANTLR4 入门](https://github.com/antlr/antlr4/blob/master/doc/getting-started.md) - 官方入门指南
- [ANTLR4 语法结构](https://github.com/antlr/antlr4/blob/master/doc/grammars.md) - 语法文件结构和语法
- [ANTLR4 词法规则](https://github.com/antlr/antlr4/blob/master/doc/lexer-rules.md) - 词法规则语法和模式
- [ANTLR4 C++ 示例](https://github.com/teverett/antlr4-cpp-example) - 使用 ANTLR4 和 C++ 的示例项目

---

## Agent 使用说明

### 如何加载此技能

当您需要修改 PTX 语法时，Agent 会自动加载此技能。您也可以明确要求：

```
请加载 ptx-grammar-modification 技能帮我修复语法解析错误
```

### Agent 工作流程

1. Agent 读取此技能文档
2. 遵循"开发工作流程"中的步骤
3. 修改前阅读 `docs/ptx/` 对应章节
4. 运行 `test_all_ptx.sh` 验证
5. 如有失败，使用 `cuobjdump` 提取真实 PTX 并添加测试用例
