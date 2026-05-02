/**
 * @file ptx_exceptions.h
 * @brief PTX 模拟器异常类层次结构
 *
 * 提供运行时错误报告机制，替代 assert(false) 和 TODO 注释。
 * 使模拟器能够在遇到错误时正确报告并处理，而不是直接崩溃。
 *
 * @author PTX-EMU Team
 * @date 2026-05-03
 */

#ifndef PTX_SIM_EXCEPTIONS_H
#define PTX_SIM_EXCEPTIONS_H

#include <stdexcept>
#include <string>
#include <cstdint>

/**
 * @brief PTX 模拟器错误码枚举
 */
enum class PtxEmuErrorCode {
    UNSUPPORTED_INSTRUCTION = 1,  //!< 不支持的 PTX 指令
    INVALID_MEMORY_ACCESS = 2,     //!< 非法内存访问（越界、未对齐等）
    PTX_PARSE_ERROR = 3,           //!< PTX 解析错误
    EXECUTION_STATE_ERROR = 4,    //!< 执行状态错误（PC 异常、分支错误等）
    INTERNAL_ERROR = 5            //!< 内部错误（未预期的代码路径）
};

/**
 * @brief PTX 模拟器异常基类
 *
 * 所有模拟器特定异常的基类，继承自 std::runtime_error。
 * 提供统一的错误报告接口。
 */
class PtxEmuException : public std::runtime_error {
public:
    /**
     * @brief 构造函数
     * @param message 错误描述信息
     * @param error_code 错误码，默认为 INTERNAL_ERROR
     */
    explicit PtxEmuException(
        const std::string& message,
        PtxEmuErrorCode error_code = PtxEmuErrorCode::INTERNAL_ERROR) noexcept
        : std::runtime_error(message),
          error_code_(error_code) {}

    /**
     * @brief 虚析构函数
     */
    virtual ~PtxEmuException() override = default;

    /**
     * @brief 获取错误码
     * @return PtxEmuErrorCode 错误码枚举值
     */
    PtxEmuErrorCode get_error_code() const noexcept { return error_code_; }

    /**
     * @brief 获取错误码对应的整数值
     * @return int 错误码的整数值
     */
    int get_error_code_value() const noexcept {
        return static_cast<int>(error_code_);
    }

    /**
     * @brief 获取错误码名称
     * @return std::string 错误码的字符串表示
     */
    std::string get_error_code_name() const noexcept {
        switch (error_code_) {
            case PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION:
                return "UNSUPPORTED_INSTRUCTION";
            case PtxEmuErrorCode::INVALID_MEMORY_ACCESS:
                return "INVALID_MEMORY_ACCESS";
            case PtxEmuErrorCode::PTX_PARSE_ERROR:
                return "PTX_PARSE_ERROR";
            case PtxEmuErrorCode::EXECUTION_STATE_ERROR:
                return "EXECUTION_STATE_ERROR";
            case PtxEmuErrorCode::INTERNAL_ERROR:
            default:
                return "INTERNAL_ERROR";
        }
    }

protected:
    PtxEmuErrorCode error_code_;  //!< 错误码
};

/**
 * @brief 不支持的 PTX 指令异常
 *
 * 当模拟器遇到尚未实现的 PTX 指令时抛出。
 */
class UnsupportedInstructionException : public PtxEmuException {
public:
    /**
     * @brief 构造函数
     * @param instruction_name 指令名称（如 "wmma", "mma"）
     * @param details 详细错误信息
     */
    explicit UnsupportedInstructionException(
        const std::string& instruction_name,
        const std::string& details = "") noexcept
        : PtxEmuException(
              build_message(instruction_name, details),
              PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION),
          instruction_name_(instruction_name),
          details_(details) {}

    /**
     * @brief 获取指令名称
     */
    const std::string& get_instruction_name() const noexcept {
        return instruction_name_;
    }

    /**
     * @brief 获取详细信息
     */
    const std::string& get_details() const noexcept { return details_; }

private:
    static std::string build_message(
        const std::string& instruction_name,
        const std::string& details) {
        std::string msg = "Unsupported PTX instruction: " + instruction_name;
        if (!details.empty()) {
            msg += " (" + details + ")";
        }
        return msg;
    }

    std::string instruction_name_;  //!< 指令名称
    std::string details_;            //!< 详细错误信息
};

/**
 * @brief 非法内存访问异常
 *
 * 当模拟器检测到内存访问违规时抛出，包括越界、未对齐等。
 */
class InvalidMemoryAccessException : public PtxEmuException {
public:
    /**
     * @brief 构造函数
     * @param address 访问的内存地址
     * @param access_size 访问大小（字节）
     * @param bounds 信息，如 "out of bounds", "misaligned"
     * @param details 详细错误信息
     */
    explicit InvalidMemoryAccessException(
        uint64_t address,
        size_t access_size,
        const std::string& bounds = "",
        const std::string& details = "") noexcept
        : PtxEmuException(
              build_message(address, access_size, bounds, details),
              PtxEmuErrorCode::INVALID_MEMORY_ACCESS),
          address_(address),
          access_size_(access_size),
          bounds_(bounds) {}

    /**
     * @brief 获取访问的内存地址
     */
    uint64_t get_address() const noexcept { return address_; }

    /**
     * @brief 获取访问大小（字节）
     */
    size_t get_access_size() const noexcept { return access_size_; }

    /**
     * @brief 获取边界信息
     */
    const std::string& get_bounds() const noexcept { return bounds_; }

private:
    static std::string build_message(
        uint64_t address,
        size_t access_size,
        const std::string& bounds,
        const std::string& details) {
        std::string msg = "Invalid memory access at address 0x" +
                          std::to_string(address) +
                          ", size=" + std::to_string(access_size);
        if (!bounds.empty()) {
            msg += " (" + bounds + ")";
        }
        if (!details.empty()) {
            msg += ": " + details;
        }
        return msg;
    }

    uint64_t address_;      //!< 访问的内存地址
    size_t access_size_;    //!< 访问大小（字节）
    std::string bounds_;    //!< 边界信息
};

/**
 * @brief PTX 解析错误异常
 *
 * 当 ANTLR 解析器遇到语法错误时抛出。
 */
class PTXParseException : public PtxEmuException {
public:
    /**
     * @brief 构造函数
     * @param message 错误描述信息
     * @param line_number 行号（如有）
     * @param column 列号（如有）
     */
    explicit PTXParseException(
        const std::string& message,
        int line_number = -1,
        int column = -1) noexcept
        : PtxEmuException(
              build_message(message, line_number, column),
              PtxEmuErrorCode::PTX_PARSE_ERROR),
          line_number_(line_number),
          column_(column) {}

    /**
     * @brief 获取行号
     */
    int get_line_number() const noexcept { return line_number_; }

    /**
     * @brief 获取列号
     */
    int get_column() const noexcept { return column_; }

    /**
     * @brief 是否有位置信息
     */
    bool has_location() const noexcept {
        return line_number_ > 0;
    }

private:
    static std::string build_message(
        const std::string& message,
        int line_number,
        int column) {
        std::string msg = "PTX parse error";
        if (line_number > 0) {
            msg += " at line " + std::to_string(line_number);
            if (column > 0) {
                msg += ":" + std::to_string(column);
            }
        }
        msg += ": " + message;
        return msg;
    }

    int line_number_;  //!< 行号
    int column_;       //!< 列号
};

/**
 * @brief 执行状态错误异常
 *
 * 当线程/SM 执行状态异常时抛出，如 PC 异常、分支错误等。
 */
class ExecutionStateException : public PtxEmuException {
public:
    /**
     * @brief 构造函数
     * @param thread_id 线程 ID（如有）
     * @param state 信息，如 "invalid PC", "branch prediction error"
     * @param details 详细错误信息
     */
    explicit ExecutionStateException(
        uint32_t thread_id = 0,
        const std::string& state = "",
        const std::string& details = "") noexcept
        : PtxEmuException(
              build_message(thread_id, state, details),
              PtxEmuErrorCode::EXECUTION_STATE_ERROR),
          thread_id_(thread_id),
          state_(state) {}

    /**
     * @brief 获取线程 ID
     */
    uint32_t get_thread_id() const noexcept { return thread_id_; }

    /**
     * @brief 获取状态信息
     */
    const std::string& get_state() const noexcept { return state_; }

private:
    static std::string build_message(
        uint32_t thread_id,
        const std::string& state,
        const std::string& details) {
        std::string msg = "Execution state error";
        if (thread_id > 0) {
            msg += " in thread " + std::to_string(thread_id);
        }
        if (!state.empty()) {
            msg += " (" + state + ")";
            if (!details.empty()) {
                msg += ": " + details;
            }
        } else if (!details.empty()) {
            msg += ": " + details;
        }
        return msg;
    }

    uint32_t thread_id_;   //!< 线程 ID
    std::string state_;    //!< 状态信息
};

#endif  // PTX_SIM_EXCEPTIONS_H
