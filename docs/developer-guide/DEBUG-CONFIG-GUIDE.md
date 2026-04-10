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

diff --git a/docs/archive/misc/debug-config-guide.md b/docs/archive/misc/debug-config-guide.md
index dd31a4a..c48c443 100644
--- a/docs/archive/misc/debug-config-guide.md
+++ b/docs/archive/misc/debug-config-guide.md
@@ -6,59 +6,55 @@
 
 | 配置文件 | 用途 | 日志级别 | 输出目标 | 适用场景 |
 |---------|------|---------|---------|---------|
-| `release_config.ini` | 生产运行 | info | console | 正常运行程序，最佳性能 |
-| `debug_config.ini` | 日常开发 | debug | both | 日常调试，平衡性能和信息 |
-| `verbose_trace_config.ini` | 详细跟踪 | trace | file | 深入分析问题，理解执行流程 |
-| `memory_debug_config.ini` | 内存调试 | debug/trace | both | 调试内存相关问题 |
-| `instruction_debug_config.ini` | 指令调试 | info/trace | both | 调试特定指令问题 |
-| `perf_config.ini` | 性能分析 | warning | console | 性能测试，最小开销 |
+| `configs/release_config.ini` | 生产运行 | info | console | 正常运行程序，最佳性能 |
+| `configs/dev_debug_config.ini` | 日常开发 | debug | both | 日常调试，平衡性能和信息 |
+| `configs/verbose_trace_config.ini` | 详细跟踪 | trace | file | 深入分析问题，理解执行流程 |
+| `configs/memory_debug_config.ini` | 内存调试 | debug/trace | both | 调试内存相关问题 |
+| `configs/instruction_debug_config.ini` | 指令调试 | info/trace | both | 调试特定指令问题 |
+| `configs/perf_config.ini` | 性能分析 | warning | console | 性能测试，最小开销 |
 
 ## 🚀 快速使用方法
 
-### 方法 1：复制到工作目录（推荐）
+### 方法 1：使用快捷脚本（推荐）
 
 ```bash
-# 选择需要的配置文件
-cp configs/debug_config.ini ./ptx_debug.conf
+# 使用开发调试配置
+./scripts/debug-run.sh debug ./build/bin/dummy-args
 
-# 运行程序（程序会自动查找 ptx_debug.conf）
-./build/bin/dummy-args
+# 使用详细跟踪配置
+./scripts/debug-run.sh trace ./build/bin/dummy-args
+
+# 使用内存调试配置
+./scripts/debug-run.sh memory ./build/bin/dummy-args
 ```
 
-### 方法 2：直接指定配置文件
+### 方法 2：直接读取配置文件
 
 ```bash
-# 使用详细跟踪配置
-cp configs/verbose_trace_config.ini ./ptx_debug.conf
-./build/bin/dummy-args
+# 程序直接使用 configs/ 中的配置
+# 示例：在代码中指定配置路径
+load_config("configs/dev_debug_config.ini");
 
-# 或使用内存调试配置
-cp configs/memory_debug_config.ini ./ptx_debug.conf
-./build/bin/dummy-args
+# 或运行程序并指定配置
+./build/bin/dummy-args --config=configs/dev_debug_config.ini
 ```
 
-### 方法 3：使用快捷脚本
+### 方法 3：手动复制（不推荐，保留向后兼容）
 
 ```bash
-# 使用调试配置
-./debug-run.sh debug ./build/bin/dummy-args
-
-# 使用详细跟踪配置
-./debug-run.sh trace ./build/bin/dummy-args
-
-# 使用内存调试配置
-./debug-run.sh memory ./build/bin/dummy-args
+# 如需复制（例如程序固定读取特定路径）
+cp configs/dev_debug_config.ini ./ptx_debug.conf
+./build/bin/dummy-args
 ```
 
 ## 🔍 调试场景和配置选择
 
 ### 场景 1：程序崩溃或行为异常
 
-**推荐配置**: `verbose_trace_config.ini`
+**推荐配置**: `configs/verbose_trace_config.ini`
 
 ```bash
-cp configs/verbose_trace_config.ini ./ptx_debug.conf
-./build/bin/your_program
+./scripts/debug-run.sh verbose ./build/bin/your_program
 
 # 查看日志文件
 tail -f ptx_emu_trace.log
@@ -71,16 +67,45 @@ tail -f ptx_emu_trace.log
 
 ### 场景 2：内存访问错误
 
-**推荐配置**: `memory_debug_config.ini`
+**推荐配置**: `configs/memory_debug_config.ini`
 
 ```bash
-cp configs/memory_debug_config.ini ./ptx_debug.conf
-./build/bin/your_program
+./scripts/debug-run.sh memory ./build/bin/your_program
 
 # 查看内存相关日志
 grep "mem" ptx_emu_memory_debug.log | tail -50
 ```
 
+### 场景 3：指令执行错误
+
+**推荐配置**: `configs/instruction_debug_config.ini`
+
+```bash
+./scripts/debug-run.sh instruction ./build/bin/your_program
+
+# 查看指令执行日志
+grep "instr" ptx_emu_instr_debug.log | tail -100
+```
+
+### 场景 4：性能问题
+
+**推荐配置**: `configs/perf_config.ini`
+
+```bash
+./scripts/debug-run.sh perf ./build/bin/your_program
+```
+
+### 场景 5：日常开发调试
+
+**推荐配置**: `configs/dev_debug_config.ini`
+
+```bash
+./scripts/debug-run.sh debug ./build/bin/your_program
+
+# 实时查看日志
+tail -f ptx_emu_debug.log
+```
+
 **分析步骤**:
 1. 启用内存组件的 trace 级别日志
 2. 跟踪所有内存读写操作
