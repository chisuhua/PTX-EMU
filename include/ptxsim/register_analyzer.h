#ifndef PTXSIM_REGISTER_ANALYZER_H
#define PTXSIM_REGISTER_ANALYZER_H

#include "ptx_ir/operand_context.h"
#include "ptx_ir/ptx_types.h"
#include "ptx_ir/statement_context.h"
#include <string>
#include <unordered_set>
#include <vector>

struct RegisterInfo {
    std::string name;
    int index;
    size_t size;

    RegisterInfo(const std::string &n, int i, size_t s)
        : name(n), index(i), size(s) {}

    bool operator==(const RegisterInfo &other) const {
        return name == other.name && index == other.index;
    }
};

// 自定义哈希函数，只考虑寄存器名和索引
struct RegisterInfoHash {
    std::size_t operator()(const RegisterInfo &reg) const {
        std::size_t h1 = std::hash<std::string>{}(reg.name);
        std::size_t h2 = std::hash<int>{}(reg.index);
        return h1 ^ (h2 << 1);
    }
};

class RegisterAnalyzer {
public:
    // 分析语句向量，收集所有需要的寄存器信息
    static std::vector<RegisterInfo>
    analyze_registers(const std::vector<StatementContext> &statements);

private:
    // 从单个语句中提取寄存器信息
    static void extract_registers_from_statement(
        const StatementContext &stmt,
        std::unordered_set<RegisterInfo, RegisterInfoHash> &registers);

    // 从语句的所有操作数中提取寄存器信息
    static void extract_registers_from_all_operands(
        const StatementContext &stmt,
        std::unordered_set<RegisterInfo, RegisterInfoHash> &registers);

    // 从操作数中提取寄存器信息
    static void extract_registers_from_operand(
        const OperandContext &op, const std::vector<Qualifier> &qualifiers,
        std::unordered_set<RegisterInfo, RegisterInfoHash> &registers);

    // 从语句结构体中提取寄存器信息
    template <typename StmtType>
    static void extract_registers_from_struct(
        const StmtType &stmt,
        std::unordered_set<RegisterInfo, RegisterInfoHash> &registers);
};

#endif // PTXSIM_REGISTER_ANALYZER_H