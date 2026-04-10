commit 7ccd4b21cf853596e56c98cd1fa0170f9246840f
Author: PTX-EMU Developer <dev@ptx-emu.local>
Date:   Fri Apr 10 20:09:23 2026 +0800

    refactor(config): 统一配置文件到 configs/ 目录
    
    - 添加 configs/dev_debug_config.ini 作为默认开发调试配置
    - 删除根目录 ptx_debug.conf 和 ptx_verbose.conf
    - 更新 debug-run.sh: debug → dev_debug_config.ini, 新增 dev 别名
    - 更新 debug_example.cpp: 默认配置路径改为 configs/dev_debug_config.ini
    - 更新 ptx-debug SKILL.md: 统一使用 configs/ 配置路径和 debug-run.sh
    - 更新所有归档文档中的配置引用
    
    统一配置管理，避免工作目录中散落配置文件。

diff --git a/docs/archive/misc/ptx-debug-skill-usage.md b/docs/archive/misc/ptx-debug-skill-usage.md
index 054e2b5..b4b5e95 100644
--- a/docs/archive/misc/ptx-debug-skill-usage.md
+++ b/docs/archive/misc/ptx-debug-skill-usage.md
@@ -36,13 +36,13 @@
 
 | 问题类型 | 自动选择配置 | 说明 |
 |---------|-------------|------|
-| PTX 解析错误 | `verbose_trace_config.ini` | 详细跟踪解析过程 |
-| 测试失败 | `debug_config.ini` | 平衡的日志级别 |
-| 程序崩溃 | `verbose_trace_config.ini` | 详细跟踪定位崩溃点 |
-| 内存问题 | `memory_debug_config.ini` | 专注内存操作跟踪 |
-| 指令错误 | `instruction_debug_config.ini` | 专注指令执行跟踪 |
-| 性能问题 | `perf_config.ini` | 最小日志开销 |
-| 日常调试 | `debug_config.ini` | 默认调试配置 |
+| PTX 解析错误 | `configs/verbose_trace_config.ini` | 详细跟踪解析过程 |
+| 测试失败 | `configs/dev_debug_config.ini` | 平衡的日志级别 |
+| 程序崩溃 | `configs/verbose_trace_config.ini` | 详细跟踪定位崩溃点 |
+| 内存问题 | `configs/memory_debug_config.ini` | 专注内存操作跟踪 |
+| 指令错误 | `configs/instruction_debug_config.ini` | 专注指令执行跟踪 |
+| 性能问题 | `configs/perf_config.ini` | 最小日志开销 |
+| 日常调试 | `configs/dev_debug_config.ini` | 默认调试配置 |
 
 ## 使用示例
 
@@ -54,11 +54,10 @@
 1. ✅ 识别问题类型：PTX 解析错误
 2. ✅ 选择配置：`verbose_trace_config.ini`
 3. ✅ 执行调试：
-   ```bash
-   cp configs/verbose_trace_config.ini ./ptx_debug.conf
-   ./build/bin/dummy-args
-   grep "parser\|lexer" ptx_emu_trace.log
-   ```
+    ```bash
+    ./scripts/debug-run.sh verbose ./build/bin/dummy-args
+    grep "parser\|lexer" ptx_emu_trace.log
+    ```
 4. ✅ 分析错误位置
 5. ✅ 生成修复方案
 
@@ -70,11 +69,10 @@
 1. ✅ 识别问题类型：测试失败
 2. ✅ 选择配置：`debug_config.ini`
 3. ✅ 执行调试：
-   ```bash
-   cp configs/debug_config.ini ./ptx_debug.conf
-   cd build && ctest -R test_memory_manager -V
-   tail -100 ptx_emu_debug.log
-   ```
+    ```bash
+    cd build && ctest -R test_memory_manager -V
+    tail -100 ptx_emu_debug.log
+    ```
 4. ✅ 分析失败原因
 5. ✅ 生成修复
 
@@ -86,11 +84,10 @@
 1. ✅ 识别问题类型：内存问题
 2. ✅ 选择配置：`memory_debug_config.ini`
 3. ✅ 执行调试：
-   ```bash
-   cp configs/memory_debug_config.ini ./ptx_debug.conf
-   ./build/bin/dummy-args
-   grep "\[mem\]" ptx_emu_memory_debug.log
-   ```
+    ```bash
+    ./scripts/debug-run.sh memory ./build/bin/dummy-args
+    grep "\[mem\]" ptx_emu_memory_debug.log
+    ```
 4. ✅ 分析内存访问模式
 5. ✅ 定位非法访问
 
@@ -102,10 +99,9 @@
 1. ✅ 识别问题类型：性能问题
 2. ✅ 选择配置：`perf_config.ini`
 3. ✅ 执行调试：
-   ```bash
-   cp configs/perf_config.ini ./ptx_debug.conf
-   time ./build/bin/RAY 512 512
-   ```
+    ```bash
+    ./scripts/debug-run.sh perf ./build/bin/RAY 512 512
+    ```
 4. ✅ 分析性能瓶颈
 5. ✅ 提出优化建议
 
@@ -175,11 +171,11 @@
 
 | 配置文件 | 日志文件 | 用途 |
 |---------|---------|------|
-| `debug_config.ini` | `ptx_emu_debug.log` | 日常调试 |
-| `verbose_trace_config.ini` | `ptx_emu_trace.log` | 详细跟踪 |
-| `memory_debug_config.ini` | `ptx_emu_memory_debug.log` | 内存调试 |
-| `instruction_debug_config.ini` | `ptx_emu_instr_debug.log` | 指令调试 |
-| `perf_config.ini` | - | 控制台输出 |
+| `configs/dev_debug_config.ini` | `ptx_emu_debug.log` | 日常调试 |
+| `configs/verbose_trace_config.ini` | `ptx_emu_trace.log` | 详细跟踪 |
+| `configs/memory_debug_config.ini` | `ptx_emu_memory_debug.log` | 内存调试 |
+| `configs/instruction_debug_config.ini` | `ptx_emu_instr_debug.log` | 指令调试 |
+| `configs/perf_config.ini` | - | 控制台输出 |
 
 ## 常用命令
 
@@ -232,8 +228,8 @@ less -R ptx_emu_trace.log
 **问题**: 调试配置没有生效
 
 **解决**:
-1. 确认已复制配置文件到 `./ptx_debug.conf`
-2. 确认程序会读取该配置文件
+1. 确认使用 `./scripts/debug-run.sh` 或 `configs/` 中的配置文件
+2. 确认程序会读取正确的配置路径
 3. 重启程序使配置生效
 
 ### 日志文件未生成
