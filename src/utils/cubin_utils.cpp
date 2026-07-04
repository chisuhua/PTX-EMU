#include "utils/cubin_utils.h"
#include "ptx_ir/ptx_types.h"
#include "utils/logger.h"
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

#define PTX_ERROR(fmt, ...) PTX_ERROR_EMU(fmt, ##__VA_ARGS__)
#define PTX_DEBUG(fmt, ...) PTX_DEBUG_EMU(fmt, ##__VA_ARGS__)

// 使用编译时定义的 cuobjdump 路径，如果未定义则使用默认命令
#ifndef CUOBJDUMP_PATH
#define CUOBJDUMP_PATH "cuobjdump"
#endif

// 移除PTX中的内联汇编块
// NVVM生成PTX时会在内联汇编周围插入 "// begin inline asm" 和 "// end inline asm" 注释
// 或者直接输出 {} 包裹的内联汇编块，解析器无法处理这些语法
//
// 内联汇编块的特征：
// 1. 以 // begin inline asm 注释开始（如果有）
// 2. 或者 { 出现在独立行，前面是 ; (指令结束)
// 3. 块内包含PTX指令如 mov.s32, add.s32 等
static std::string strip_inline_asm(const std::string& ptx_code) {
    std::string result;
    std::istringstream stream(ptx_code);
    std::string line;
    bool in_inline_asm = false;
    bool in_brace_block = false;
    // 跟踪上一个非空行是否以 ; 结尾
    bool last_nonempty_ends_with_semi = false;

    while (std::getline(stream, line)) {
        // 移除前后空白进行检测
        std::string trimmed = line;
        // 去除前导空白
        size_t start = 0;
        while (start < trimmed.size() && (trimmed[start] == ' ' || trimmed[start] == '\t')) {
            start++;
        }
        if (start > 0) {
            trimmed = trimmed.substr(start);
        }
        // 去除尾部空白
        while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t')) {
            trimmed.pop_back();
        }

        // 跟踪非空行是否以 ; 结尾
        if (!trimmed.empty()) {
            last_nonempty_ends_with_semi = (trimmed.back() == ';');
        }

        // 检查内联汇编开始标记 (带注释的格式)
        if (line.find("// begin inline asm") != std::string::npos) {
            in_inline_asm = true;
            continue;
        }
        // 检查内联汇编结束标记 (带注释的格式)
        if (line.find("// end inline asm") != std::string::npos) {
            in_inline_asm = false;
            continue;
        }

        // 检查裸的大括号块 (无注释的格式)
        // 只有当行只包含 { 且上一个非空行以 ; 结束时，才认为是内联汇编块
        // 函数体的 { 前面是 ) 而不是 ;
        if (trimmed == "{") {
            if (last_nonempty_ends_with_semi) {
                in_brace_block = true;
                continue;  // 跳过内联汇编的 {
            }
            // 否则这是函数体的 {，继续正常处理（下面会添加到结果）
        }
        if (trimmed == "}" && in_brace_block) {
            // 只有在 brace_block 内才跳过 }
            in_brace_block = false;
            continue;
        }

        // 如果不在内联汇编块中，保留该行
        if (!in_inline_asm && !in_brace_block) {
            result += line;
            result += "\n";
        }
    }
    return result;
}

std::string extract_ptx_with_cuobjdump(const std::string &executable_path) {
    // Get absolute path to working directory
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        PTX_ERROR("Failed to get current working directory");
        return "";
    }
    
    char ptx_list_cmd[1024];
    snprintf(ptx_list_cmd, 1024,
             CUOBJDUMP_PATH " -lptx %s | cut -d : -f 2 | awk '{$1=$1}1' > "
             "%s/__ptx_list_temp__",
             executable_path.c_str(), cwd);

    if (system(ptx_list_cmd) != 0) {
        PTX_ERROR("Failed to execute: %s", ptx_list_cmd);
        return "";
    }

    std::string ptx_list_path = std::string(cwd) + "/__ptx_list_temp__";
    std::ifstream ptx_list_file(ptx_list_path);
    if (!ptx_list_file.is_open()) {
        PTX_ERROR("Failed to open PTX list file");
        return "";
    }

    std::string ptx_codes;
    std::string ptx_file;
    // Count how many .ptx sections cuobjdump found in the cubin. The
    // extracted text is appended for all sections (legacy behavior, do
    // not change to "first only" without breaking multi-cubin tests);
    // the warning below is purely diagnostic. See c5 Fix #3.
    int ptx_section_count = 0;
    while (std::getline(ptx_list_file, ptx_file)) {
        char extract_cmd[1024];
        snprintf(extract_cmd, 1024, CUOBJDUMP_PATH " -xptx %s %s",
                 ptx_file.c_str(), executable_path.c_str());

        if (system(extract_cmd) != 0) {
            PTX_ERROR("Failed to extract PTX: %s", extract_cmd);
            continue;
        }

        // PTX file is extracted to current directory
        std::string ptx_file_path = std::string(cwd) + "/" + ptx_file;
        std::ifstream extracted_ptx_file(ptx_file_path);
        if (!extracted_ptx_file.is_open()) {
            PTX_ERROR("Failed to open extracted PTX file: %s", ptx_file_path.c_str());
            continue;
        }

        std::string line;
        while (std::getline(extracted_ptx_file, line)) {
            ptx_codes += line;
            ptx_codes += "\n";
        }
        extracted_ptx_file.close();

        ++ptx_section_count;

        char cleanup_cmd[1024];
        snprintf(cleanup_cmd, 1024, "rm %s", ptx_file_path.c_str());
        system(cleanup_cmd);
    }
    ptx_list_file.close();

    if (ptx_section_count > 1) {
        PTX_WARN_EMU("Multiple PTX sections found in cubin (count=%d) - "
                     "all sections extracted (c5 Fix #3)",
                     ptx_section_count);
    }

    char cleanup_cmd[1024];
    snprintf(cleanup_cmd, 1024, "rm %s", ptx_list_path.c_str());
    system(cleanup_cmd);

    // 移除内联汇编块，避免解析错误
    return strip_inline_asm(ptx_codes);
}

std::vector<uint8_t> parse_cubin(const std::string &cubin_path) {
    std::ifstream file(cubin_path, std::ios::binary | std::ios::ate);
    if (!file) {
        PTX_ERROR("Failed to open file: %s", cubin_path.c_str());
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
        PTX_ERROR("Failed to read file: %s", cubin_path.c_str());
        return {};
    }

    return buffer;
}

std::string cubin_to_ptx(const std::string &cubin_path) {
    std::string extract_cmd = CUOBJDUMP_PATH " -ptx " + cubin_path + " > __cubin_temp__";
    if (system(extract_cmd.c_str()) != 0) {
        PTX_ERROR("Failed to extract PTX from cubin");
        return "";
    }

    std::ifstream ptx_file("__cubin_temp__");
    if (!ptx_file.is_open()) {
        PTX_ERROR("Failed to open extracted PTX");
        return "";
    }

    std::string ptx_code;
    std::string line;
    bool in_ptx_section = false;
    while (std::getline(ptx_file, line)) {
        if (line.find("PTX") != std::string::npos) {
            in_ptx_section = true;
            continue;
        }
        if (in_ptx_section) {
            if (line.empty() || line[0] == '.') {
                if (line == "")
                    break;
            }
            ptx_code += line;
            ptx_code += "\n";
        }
    }
    ptx_file.close();
    system("rm __cubin_temp__");
    return ptx_code;
}
