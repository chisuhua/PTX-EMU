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

diff --git a/docs/archive/misc/DEBUG_QUICK_REFERENCE.md b/docs/archive/misc/DEBUG_QUICK_REFERENCE.md
index 5cb1ea2..e6f77c6 100644
--- a/docs/archive/misc/DEBUG_QUICK_REFERENCE.md
+++ b/docs/archive/misc/DEBUG_QUICK_REFERENCE.md
@@ -27,11 +27,11 @@
 ### 手动配置
 
 ```bash
-# 复制配置文件
-cp configs/debug_config.ini ./ptx_debug.conf
+# 直接使用 configs/ 中的配置
+./scripts/debug-run.sh debug ./build/bin/dummy-args
 
-# 运行程序
-./build/bin/dummy-args
+# 或程序直接指定配置路径
+# ./build/bin/dummy-args --config=configs/dev_debug_config.ini
 
 # 查看日志
 tail -f ptx_emu_debug.log
@@ -43,12 +43,12 @@ tail -f ptx_emu_debug.log
 
 | 场景 | 命令 | 配置文件 | 日志文件 |
 |------|------|---------|---------|
-| **日常调试** | `./scripts/debug-run.sh debug ...` | `debug_config.ini` | `ptx_emu_debug.log` |
-| **详细跟踪** | `./scripts/debug-run.sh trace ...` | `verbose_trace.ini` | `ptx_emu_trace.log` |
-| **内存问题** | `./scripts/debug-run.sh memory ...` | `memory_debug.ini` | `ptx_emu_memory_debug.log` |
-| **指令错误** | `./scripts/debug-run.sh instruction ...` | `instruction_debug.ini` | `ptx_emu_instr_debug.log` |
+| **日常调试** | `./scripts/debug-run.sh debug ...` | `dev_debug_config.ini` | `ptx_emu_debug.log` |
+| **详细跟踪** | `./scripts/debug-run.sh trace ...` | `verbose_trace_config.ini` | `ptx_emu_trace.log` |
+| **内存问题** | `./scripts/debug-run.sh memory ...` | `memory_debug_config.ini` | `ptx_emu_memory_debug.log` |
+| **指令错误** | `./scripts/debug-run.sh instruction ...` | `instruction_debug_config.ini` | `ptx_emu_instr_debug.log` |
 | **性能测试** | `./scripts/debug-run.sh perf ...` | `perf_config.ini` | 控制台 |
-| **生产运行** | `./scripts/debug-run.sh release ...` | `release_config.ini` | 控制台 |
+| **生产运行** | `./scripts/debug-run.sh release ...` | `config.ini` | 控制台 |
 
 ---
 
