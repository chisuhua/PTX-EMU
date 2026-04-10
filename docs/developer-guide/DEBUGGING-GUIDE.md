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

diff --git a/docs/archive/misc/debugging_guide.md b/docs/archive/misc/debugging_guide.md
index 4ace806..9cf89fd 100644
--- a/docs/archive/misc/debugging_guide.md
+++ b/docs/archive/misc/debugging_guide.md
@@ -110,8 +110,8 @@ PTX-EMU 提供性能分析功能，可以帮助用户了解程序的执行性能
 或者，如果程序支持默认配置文件名：
 
 ```bash
-./your_program
-# 程序将在当前目录查找 ptx_debug.conf 文件
+./scripts/debug-run.sh debug ./your_program
+# 脚本会自动选择 configs/dev_debug_config.ini 配置文件
 ```
 
 ### 4.2 分析日志输出
