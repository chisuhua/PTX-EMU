#include "utils/cubin_utils.h"
#include "ptx_ir/ptx_types.h"
#include "utils/logger.h"
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define PTX_ERROR(fmt, ...) PTX_ERROR_EMU(fmt, ##__VA_ARGS__)
#define PTX_DEBUG(fmt, ...) PTX_DEBUG_EMU(fmt, ##__VA_ARGS__)

std::string extract_ptx_with_cuobjdump(const std::string &executable_path) {
    char ptx_list_cmd[1024];
    snprintf(ptx_list_cmd, 1024,
             "cuobjdump -lptx %s | cut -d : -f 2 | awk '{$1=$1}1' > "
             "__ptx_list_temp__",
             executable_path.c_str());

    if (system(ptx_list_cmd) != 0) {
        PTX_ERROR("Failed to execute: %s", ptx_list_cmd);
        return "";
    }

    std::ifstream ptx_list_file("__ptx_list_temp__");
    if (!ptx_list_file.is_open()) {
        PTX_ERROR("Failed to open PTX list file");
        return "";
    }

    std::string ptx_codes;
    std::string ptx_file;
    while (std::getline(ptx_list_file, ptx_file)) {
        char extract_cmd[1024];
        snprintf(extract_cmd, 1024, "cuobjdump -xptx %s %s >/dev/null",
                 ptx_file.c_str(), executable_path.c_str());

        if (system(extract_cmd) != 0) {
            PTX_ERROR("Failed to extract PTX: %s", extract_cmd);
            continue;
        }

        std::ifstream extracted_ptx_file(ptx_file);
        if (!extracted_ptx_file.is_open()) {
            PTX_ERROR("Failed to open extracted PTX file: %s", ptx_file.c_str());
            continue;
        }

        std::string line;
        while (std::getline(extracted_ptx_file, line)) {
            ptx_codes += line;
            ptx_codes += "\n";
        }
        extracted_ptx_file.close();

        char cleanup_cmd[1024];
        snprintf(cleanup_cmd, 1024, "rm %s", ptx_file.c_str());
        system(cleanup_cmd);
    }
    ptx_list_file.close();

    system("rm __ptx_list_temp__");
    return ptx_codes;
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
    std::string extract_cmd = "cuobjdump -ptx " + cubin_path + " > __cubin_temp__";
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
