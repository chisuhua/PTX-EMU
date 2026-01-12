#ifndef CUBIN_UTILS_H
#define CUBIN_UTILS_H

#include <string>

/**
 * @brief 从可执行文件中使用cuobjdump提取PTX代码
 * 
 * @param executable_path 可执行文件路径
 * @return std::string 提取的PTX代码字符串，如果失败则返回空字符串
 */
std::string extract_ptx_with_cuobjdump(const std::string &executable_path);

#endif // CUBIN_UTILS_H