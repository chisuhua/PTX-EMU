#ifndef PTX_SYNTAX_UTILS_H
#define PTX_SYNTAX_UTILS_H

#include "operand_context.h"
#include <string>

/**
 * @file ptx_syntax_utils.h
 * @brief PTX 语法解析工具函数
 * 
 * 提供 PTX 文本格式相关的解析工具，包括：
 * - 寄存器名称解析（%r1, %rd4, %f5 等）
 * - 地址表达式解析（[global::buf + %rd4]）
 * - 特殊寄存器识别（%tid.x, %ctaid.y 等）
 * 
 * @note 这些函数属于语义解析层，由解析器和执行器共享使用
 * @note 长期目标：所有语法解析应在解析阶段完成，执行器只使用预解析结果
 */

namespace ptx {
namespace syntax {

/**
 * @brief 从文本解析 PTX 寄存器名称
 * 
 * 支持 PTX 寄存器家族：
 * - 整数寄存器：r, rd, rs (如 %r1, %rd4)
 * - 浮点寄存器：f, fd (如 %f5, %fd3)
 * - 谓词寄存器：p (如 %p0, %p1)
 * - 位寄存器：b (如 %b1, %b2)
 * - 半精度寄存器：h (如 %h1, %h2)
 * 
 * 不支持：
 * - 特殊寄存器（%tid.x, %ctaid.y 等）- 使用 isSpecialRegister()
 * - 符号变量（buf1, param_0 等）
 * 
 * @param raw 原始文本（可带 % 或 $ 前缀）
 * @param regOut 输出寄存器操作数
 * @return true 解析成功，false 不是有效的 PTX 寄存器
 * 
 * @example
 * ```cpp
 * RegOperand reg;
 * if (ptx::syntax::parseRegisterFromText("%rd4", reg)) {
 *     // reg.name == "rd", reg.index == 4
 * }
 * ```
 */
bool parseRegisterFromText(const std::string &raw, RegOperand &regOut);

/**
 * @brief 检查文本是否为 PTX 特殊寄存器
 * 
 * 特殊寄存器包括：
 * - 线程索引：%tid.x, %tid.y, %tid.z
 * - CTA 索引：%ctaid.x, %ctaid.y, %ctaid.z
 * - 网格维度：%gridDim.x, %gridDim.y, %gridDim.z
 * - 块维度：%blockDim.x, %blockDim.y, %blockDim.z
 * - 程序计数器：%pc
 * - 寄存器数量：%nregs
 * - 性能计数器：%pm0, %pm1, 等
 * 
 * @param text 待检查文本（可带 % 前缀）
 * @return true 是特殊寄存器
 */
bool isSpecialRegister(const std::string &text);

/**
 * @brief 解析特殊寄存器名称
 * 
 * @param text 特殊寄存器文本（如 "%tid.x"）
 * @param nameOut 输出寄存器名称（如 "tid"）
 * @param componentOut 输出组件（如 "x"），如果没有组件则为空
 * @return true 解析成功
 */
bool parseSpecialRegister(const std::string &text, std::string &nameOut, 
                          std::string &componentOut);

} // namespace syntax
} // namespace ptx

#endif // PTX_SYNTAX_UTILS_H
