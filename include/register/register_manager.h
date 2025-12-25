#ifndef PTXSIM_REGISTER_MANAGER_H
#define PTXSIM_REGISTER_MANAGER_H

#include "ptx_ir/ptx_types.h"
#include "ptxsim/execution_types.h"
#include "ptxsim/ptx_debug.h"
#include "utils/logger.h"
#include <any>
#include <cstdio>  // 添加snprintf所需头文件
#include <cstring> // 添加memset所需头文件
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// 寄存器接口，定义寄存器的基本操作
class RegisterInterface {
public:
    virtual ~RegisterInterface() = default;

    // 获取寄存器的物理地址（用于兼容现有代码）
    virtual void *get_physical_address() = 0;

    // 获取寄存器大小（字节）
    virtual size_t get_size() const = 0;

    // 检查操作是否完成（用于模拟多周期操作）
    virtual bool is_read_complete() const = 0;
    virtual bool is_write_complete() const = 0;

    // 获取操作剩余周期数
    virtual int get_remaining_read_cycles() const = 0;
    virtual int get_remaining_write_cycles() const = 0;

    // 开始读取操作（非阻塞）
    virtual void start_read() = 0;

    // 开始写入操作（非阻塞）
    virtual void start_write(const void *buffer, size_t size) = 0;

    // 前进一个周期（用于模拟多周期操作）
    virtual void tick() = 0;
};

// 简单寄存器实现，操作立即完成
class SimpleRegister : public RegisterInterface {
public:
    SimpleRegister(size_t size) : data_(new char[size]), size_(size) {
        memset(data_.get(), 0, size_);
    }

    void *get_physical_address() override { return data_.get(); }

    size_t get_size() const override { return size_; }

    bool is_read_complete() const override {
        return true; // 简单寄存器操作立即完成
    }

    bool is_write_complete() const override {
        return true; // 简单寄存器操作立即完成
    }

    int get_remaining_read_cycles() const override { return 0; }

    int get_remaining_write_cycles() const override { return 0; }

    void start_read() override {
        // 简单寄存器不需要特殊处理
    }

    void start_write(const void *buffer, size_t size) override {
        if (size > size_) {
            PTX_DEBUG_EMU("Register write size %zu exceeds register size %zu",
                          size, size_);
            size = size_;
        }
        memcpy(data_.get(), buffer, size);
    }

    void tick() override {
        // 简单寄存器不需要tick
    }

private:
    std::unique_ptr<char[]> data_;
    size_t size_;
};

// 寄存器管理器，负责创建、管理和访问寄存器
class RegisterManager {
public:
    // 创建寄存器
    bool create_register(const std::string &name, size_t size) {
        if (registers_.find(name) != registers_.end()) {
            PTX_DEBUG_EMU("Register %s already exists", name.c_str());
            return false; // 寄存器已存在
        }
        registers_[name] = std::make_unique<SimpleRegister>(size);
        PTX_DEBUG_EMU("Created register %s with size %zu", name.c_str(), size);
        return true;
    }

    // 获取寄存器
    RegisterInterface *get_register(const std::string &name) {
        auto it = registers_.find(name);
        if (it == registers_.end()) {
            PTX_DEBUG_EMU("Register %s not found", name.c_str());
            return nullptr;
        }
        return it->second.get();
    }

    // 读取寄存器值（立即返回，但数据可能尚未就绪）
    bool read_register(const std::string &name, void *buffer, size_t size,
                       const Dim3 &block_idx = Dim3(0, 0, 0),
                       const Dim3 &thread_idx = Dim3(0, 0, 0),
                       bool enable_trace = true) {
        auto reg = get_register(name);
        if (!reg)
            return false;

        reg->start_read();
        // 注意：这里返回的是可能未完成的读取
        // 调用者需要检查is_read_complete()
        memcpy(buffer, reg->get_physical_address(),
               std::min(size, reg->get_size()));

        // 如果启用trace，记录寄存器读取
        if (enable_trace &&
            should_trace_register_access(block_idx, thread_idx)) {
            std::any reg_value = convert_to_any(buffer, size);
            ptxsim::PTXDebugger::get().trace_register_access(
                name, reg_value, false, block_idx, thread_idx);
        }

        return true;
    }

    // 写入寄存器值（启动写入过程）
    bool write_register(const std::string &name, const void *buffer,
                        size_t size, const Dim3 &block_idx = Dim3(0, 0, 0),
                        const Dim3 &thread_idx = Dim3(0, 0, 0),
                        bool enable_trace = true) {
        auto reg = get_register(name);
        if (!reg)
            return false;

        reg->start_write(buffer, size);

        // 如果启用trace，记录寄存器写入
        if (enable_trace &&
            should_trace_register_access(block_idx, thread_idx)) {
            std::any reg_value = convert_to_any(buffer, size);
            ptxsim::PTXDebugger::get().trace_register_access(
                name, reg_value, true, block_idx, thread_idx);
        }

        return true;
    }

    // 前进模拟周期
    void tick() {
        for (auto &pair : registers_) {
            pair.second->tick();
        }
    }

    // 检查所有寄存器操作是否完成
    bool is_ready() const {
        for (const auto &pair : registers_) {
            if (!pair.second->is_read_complete() ||
                !pair.second->is_write_complete()) {
                return false;
            }
        }
        return true;
    }

    // 获取所有寄存器的名称和接口对
    std::vector<std::pair<std::string, RegisterInterface *>>
    get_all_registers() const {
        std::vector<std::pair<std::string, RegisterInterface *>> result;
        for (const auto &pair : registers_) {
            result.push_back({pair.first, pair.second.get()});
        }
        return result;
    }

private:
    std::unordered_map<std::string, std::unique_ptr<RegisterInterface>>
        registers_;

    // 检查是否应该对指定的block_idx和thread_idx进行寄存器访问trace
    bool should_trace_register_access(const Dim3 &block_idx,
                                      const Dim3 &thread_idx) const {
        // 检查是否启用了寄存器trace
        if (!ptxsim::DebugConfig::get().is_register_traced()) {
            return false;
        }

        // 检查是否有特定的block/thread过滤条件
        // 这里可以扩展以从配置中读取特定的block_idx和thread_idx设置
        // 如果没有配置特定的过滤条件，则默认对所有线程进行trace
        return true;
    }

    // 将内存数据转换为std::any类型，以便trace使用
    std::any convert_to_any(const void *data, size_t size) const {
        std::any reg_value;

        switch (size) {
        case 1:
            reg_value = *(uint8_t *)data;
            break;
        case 2:
            reg_value = *(uint16_t *)data;
            break;
        case 4:
            reg_value = *(uint32_t *)data;
            break;
        case 8:
            reg_value = *(uint64_t *)data;
            break;
        default:
            // 对于非标准大小，我们记录为十六进制字符串
            std::string hex_str = "0x";
            const unsigned char *bytes =
                static_cast<const unsigned char *>(data);
            for (size_t i = 0; i < size; ++i) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", bytes[i]);
                hex_str += buf;
            }
            reg_value = hex_str;
            break;
        }

        return reg_value;
    }
};

#endif // PTXSIM_REGISTER_MANAGER_H