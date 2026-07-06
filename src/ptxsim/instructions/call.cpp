#include "memory/hardware_memory_manager.h"
#include "ptxsim/instruction_handlers.h"
#include "ptxsim/thread_context.h"
#include "ptxsim/thread_state.h"
#include "ptxsim/warp_context.h"
#include "ptxsim/utils/qualifier_utils.h"
#include <iostream>
#include <mutex>

static std::mutex g_printf_mutex;

// RET is a VOID_INSTR handler
void RetHandler::processOperation(ThreadContext *context, StatementContext &stmt) {
    if (context->call_stack.empty()) {
        context->set_state(EXIT);  // MR-1: bare-field access → method call
        WarpContext *wc = context->get_warp_context();
        if (wc != nullptr) {
            ptxsim::WarpState &ws = wc->get_warp_state();
            for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
                ws.threads[i].status = ptxsim::ThreadStatus::Exited;
                ws.threads[i].is_exited = true;
                ws.threads[i].is_active = false;
                ws.threads[i].is_blocked = false;
                ThreadContext *t = wc->get_thread(i);
                if (t != nullptr) {
                    t->set_state(EXIT);
                }
            }
            wc->set_active_mask(0u);
        }
    } else {
        int return_pc = context->call_stack.top();
        context->call_stack.pop();
        context->set_next_pc(return_pc);
    }
    (void)stmt;
}

// CALL is a CALL_INSTR handler
void CallHandler::executeCall(ThreadContext *context, const CallInstr &instr) {
    if (instr.funcName == "vprintf" || instr.funcName == "printf" || instr.funcName == "_printf") {
        handlePrintf(context, instr);
        return;
    }
}

static constexpr size_t MAX_FORMAT_LEN = 512;
static constexpr size_t MAX_ARGS_SIZE = 256;

void CallHandler::handlePrintf(ThreadContext *context, const CallInstr &instr) {
    std::string formatStr;
    std::vector<void *> args;

    if (!instr.operands.empty()) {
        void *formatPtrAddr = context->acquire_operand(instr.operands[0], instr.qualifiers);
        if (formatPtrAddr) {
            uint64_t formatPtr = *static_cast<uint64_t *>(formatPtrAddr);

            char formatBuf[MAX_FORMAT_LEN];
            memset(formatBuf, 0, sizeof(formatBuf));
            try {
                HardwareMemoryManager::instance().access(
                    reinterpret_cast<void *>(formatPtr),
                    formatBuf, MAX_FORMAT_LEN - 1, false, MemorySpace::GLOBAL);
                formatBuf[MAX_FORMAT_LEN - 1] = '\0';
                formatStr = formatBuf;
            } catch (const std::exception &e) {
                PTX_DEBUG_EMU("handlePrintf: failed to read format string at %p: %s",
                              reinterpret_cast<void *>(formatPtr), e.what());
            }
        }

        if (instr.operands.size() > 1) {
            void *argsPtrAddr = context->acquire_operand(instr.operands[1], instr.qualifiers);
            if (argsPtrAddr) {
                uint64_t gpuAddr = *static_cast<uint64_t *>(argsPtrAddr);
                if (gpuAddr != 0) {
                    char addrBuf[8];
                    try {
                        HardwareMemoryManager::instance().access(
                            reinterpret_cast<void *>(gpuAddr),
                            addrBuf, sizeof(addrBuf), false, MemorySpace::GLOBAL);
                        uint64_t argsPtr = *reinterpret_cast<uint64_t *>(addrBuf);
                        if (argsPtr != 0) {
                            char actualArgsBuf[MAX_ARGS_SIZE];
                            HardwareMemoryManager::instance().access(
                                reinterpret_cast<void *>(argsPtr),
                                actualArgsBuf, sizeof(actualArgsBuf), false, MemorySpace::GLOBAL);
                            uint64_t *argsArray = reinterpret_cast<uint64_t *>(actualArgsBuf);
                            size_t maxArgs = sizeof(actualArgsBuf) / sizeof(uint64_t);
                            for (size_t i = 0; i < maxArgs; i++) {
                                args.push_back(&argsArray[i]);
                            }
                        }
                    } catch (const std::exception &e) {
                        PTX_DEBUG_EMU("handlePrintf: failed to read args at %p: %s",
                                      reinterpret_cast<void *>(gpuAddr), e.what());
                    }
                }
            }
        }
    }

    if (formatStr.empty()) {
        std::lock_guard<std::mutex> lock(g_printf_mutex);
        printf("[kernel printf]");
        fflush(stdout);
        return;
    }

    if (args.empty()) {
        std::lock_guard<std::mutex> lock(g_printf_mutex);
        printf("%s", formatStr.c_str());
        fflush(stdout);
        return;
    }

    parseAndPrintFormat(context, formatStr, args);
}

void CallHandler::parseAndPrintFormat(ThreadContext *context,
                                       const std::string &format,
                                       const std::vector<void *> &args) {
    std::string result;
    size_t argIndex = 0;

    for (size_t i = 0; i < format.length(); i++) {
        if (format[i] == '%' && i + 1 < format.length()) {
            i++;
            while (format[i] &&
                   (format[i] == '-' || format[i] == '+' || format[i] == '#' ||
                    format[i] == ' ' || format[i] == '*' ||
                    format[i] == '.' ||
                    (format[i] >= '0' && format[i] <= '9'))) {
                i++;
            }

            if (format[i] == '%') {
                result += '%';
                continue;
            }

            if (argIndex >= args.size()) {
                break;
            }

            switch (format[i]) {
            case 'd':
            case 'i': {
                int val = *static_cast<int *>(args[argIndex]);
                result += std::to_string(val);
            } break;
            case 'u': {
                unsigned int val = *static_cast<unsigned int *>(args[argIndex]);
                result += std::to_string(val);
            } break;
            case 'x':
            case 'X': {
                unsigned int val = *static_cast<unsigned int *>(args[argIndex]);
                char buf[32];
                snprintf(buf, sizeof(buf), (format[i] == 'x') ? "%x" : "%X", val);
                result += buf;
            } break;
            case 'o': {
                unsigned int val = *static_cast<unsigned int *>(args[argIndex]);
                char buf[32];
                snprintf(buf, sizeof(buf), "%o", val);
                result += buf;
            } break;
            case 'f':
            case 'F':
            case 'e':
            case 'E':
            case 'g':
            case 'G': {
                double val = *static_cast<double *>(args[argIndex]);
                char buf[128];
                snprintf(buf, sizeof(buf), "%g", val);
                result += buf;
            } break;
            case 'c': {
                int val = *static_cast<int *>(args[argIndex]);
                result += static_cast<char>(val);
            } break;
            case 's': {
                uint64_t str_ptr = *static_cast<uint64_t *>(args[argIndex]);
                if (str_ptr != 0) {
                    char str_buf[1024];
                    size_t max_len = sizeof(str_buf) - 1;
                    HardwareMemoryManager::instance().access(
                        reinterpret_cast<void *>(str_ptr), str_buf, max_len, false, MemorySpace::GLOBAL);
                    str_buf[max_len] = '\0';
                    result += str_buf;
                }
            } break;
            case 'p': {
                uint64_t ptr = *static_cast<uint64_t *>(args[argIndex]);
                char buf[32];
                snprintf(buf, sizeof(buf), "0x%lx", ptr);
                result += buf;
            } break;
            default:
                result += '%';
                result += format[i];
                continue;
            }
            argIndex++;
        } else {
            result += format[i];
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_printf_mutex);
        printf("%s", result.c_str());
        fflush(stdout);
    }
}