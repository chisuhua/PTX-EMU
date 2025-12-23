#ifndef MEMORY_INTERFACE_H
#define MEMORY_INTERFACE_H

#include <cstddef>
#include <cstdint>

enum class MemorySpace { SHARED, GLOBAL, LOCAL, CONST, PARAM };

struct MemoryAccess {
    MemorySpace space;
    uint64_t address;
    size_t size;
    bool is_write;
    void *data; // 读：目标缓冲区；写：源数据
};

class MemoryInterface {
public:
    virtual ~MemoryInterface() = default;

    // 当前使用：同步访问（立即完成）
    virtual void access(const MemoryAccess &req) = 0;

    // 未来扩展：异步访问（cycle-accurate）
    virtual uint64_t issue_access(const MemoryAccess &req) {
        access(req);
        return 0;
    }
    virtual bool is_complete(uint64_t token) { return true; }
};
#endif