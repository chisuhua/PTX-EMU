#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include "ptx_ir/ptx_types.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

// 符号表项定义，用于存储常量、参数、局部变量等信息
class Symtable {
public:
    Qualifier symType; // 符号类型（如CONST、PARAM、LOCAL等）
    int byteNum;       // 每个元素的字节数
    int elementNum;    // 元素数量
    std::string name;  // 符号名称
    uint64_t val;      // 符号的值或地址
};

// 寄存器定义，用于存储寄存器信息
class Reg {
public:
    Qualifier regType; // 寄存器类型
    int byteNum;       // 每个元素的字节数
    int elementNum;    // 元素数量
    std::string name;  // 寄存器名称
    void *addr;        // 寄存器地址
};

// 立即数定义，用于存储立即数操作数
class IMM {
public:
    Qualifier type; // 立即数类型
    union Data {
        uint8_t u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        float f32;
        double f64;

        // 为union提供默认构造函数
        Data() : u64(0) {}
    };
    Data data; // 立即数数据
};

// 向量定义，用于存储向量操作数
class VEC {
public:
    std::vector<void *> vec; // 向量元素列表
};

#endif // COMMON_TYPES_H