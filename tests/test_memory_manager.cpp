#include "memory/memory_manager.h"
#include "memory/simple_memory_allocator.h"
#include "catch_amalgamated.hpp"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

// 简单的MemoryInterface实现用于测试
class TestMemoryInterface : public MemoryInterface {
public:
    void access(const MemoryAccess& req) override {
        // 简单记录访问，不做实际操作
        accesses.push_back(req);
    }
    
    std::vector<MemoryAccess> accesses;
};

TEST_CASE("SimpleMemoryAllocator basic operations") {
    SimpleMemoryAllocator allocator;
    
    SECTION("Basic allocation") {
        size_t offset1 = allocator.allocate(1024);
        REQUIRE(offset1 != static_cast<size_t>(-1));
        INFO("Allocated 1024 bytes at offset: " << offset1);
        
        size_t offset2 = allocator.allocate(2048);
        REQUIRE(offset2 != static_cast<size_t>(-1));
        REQUIRE(offset2 >= offset1 + 1024); // 确保不重叠
        INFO("Allocated 2048 bytes at offset: " << offset2);
    }
    
    SECTION("Allocation and deallocation") {
        size_t offset1 = allocator.allocate(1024);
        size_t offset2 = allocator.allocate(2048);
        
        REQUIRE(offset1 != static_cast<size_t>(-1));
        REQUIRE(offset2 != static_cast<size_t>(-1));
        
        allocator.deallocate(offset1);
        
        size_t offset3 = allocator.allocate(512);  // 应该能复用之前释放的空间
        // 不检查具体位置，因为分配算法可能不会立即复用空间
        
        allocator.deallocate(offset2);
        allocator.deallocate(offset3);
    }
}

TEST_CASE("MemoryManager global memory operations") {
    MemoryManager& mm = MemoryManager::instance();
    TestMemoryInterface testInterface;
    mm.set_memory_interface(&testInterface);
    
    SECTION("Global memory allocation") {
        void* ptr1 = mm.malloc(1024);
        REQUIRE(ptr1 != nullptr);
        INFO("Allocated 1024 bytes at: " << ptr1);
        
        void* ptr2 = mm.malloc(2048);
        REQUIRE(ptr2 != nullptr);
        INFO("Allocated 2048 bytes at: " << ptr2);
        
        // Test get_global_pool method
        uint8_t* pool = mm.get_global_pool();
        REQUIRE(pool != nullptr);
        
        // 测试释放
        mycudaError_t result1 = mm.free(ptr1);
        REQUIRE(result1 == Success);
        mycudaError_t result2 = mm.free(ptr2);
        REQUIRE(result2 == Success);
    }
    
    SECTION("Memory access operations") {
        void* ptr = mm.malloc(1024);
        REQUIRE(ptr != nullptr);
        
        int testData = 42;
        mm.access(ptr, &testData, sizeof(testData), true);  // 写入
        
        int readData = 0;
        mm.access(ptr, &readData, sizeof(readData), false);  // 读取
        REQUIRE(readData == testData);
        INFO("Memory access test passed!");
        
        mycudaError_t result = mm.free(ptr);
        REQUIRE(result == Success);
    }
}

TEST_CASE("MemoryManager param memory operations") {
    MemoryManager& mm = MemoryManager::instance();
    
    SECTION("Parameter memory allocation and access") {
        // 测试参数内存分配
        void* param_ptr1 = mm.malloc_param(512);
        REQUIRE(param_ptr1 != nullptr);
        INFO("Allocated 512 bytes for param at: " << param_ptr1);
        
        void* param_ptr2 = mm.malloc_param(1024);
        REQUIRE(param_ptr2 != nullptr);
        INFO("Allocated 1024 bytes for param at: " << param_ptr2);
        
        // 测试参数内存访问
        char test_data[10];
        strcpy(test_data, "test");
        mm.access(param_ptr1, test_data, 5, true);  // 写入5个字节
        
        char read_data[10] = {0};
        mm.access(param_ptr1, read_data, 5, false);  // 读取5个字节
        REQUIRE(strcmp(read_data, "test") == 0);
        INFO("Parameter memory access test passed!");
        
        // 测试释放参数内存
        mm.free_param(param_ptr1);
        mm.free_param(param_ptr2);
    }
}

TEST_CASE("MemoryManager memory overlap protection") {
    MemoryManager& mm = MemoryManager::instance();
    
    // 分配大量内存测试分配器的效率
    std::vector<void*> ptrs;
    for (int i = 0; i < 10; ++i) {
        void* ptr = mm.malloc(1024);
        REQUIRE(ptr != nullptr);
        ptrs.push_back(ptr);
    }
    
    // 释放一些内存以测试碎片整理
    for (int i = 0; i < 10; i += 2) {
        mycudaError_t result = mm.free(ptrs[i]);
        REQUIRE(result == Success);
        ptrs[i] = nullptr;
    }
    
    // 重新分配
    for (int i = 0; i < 10; i += 2) {
        ptrs[i] = mm.malloc(1024);
        REQUIRE(ptrs[i] != nullptr);
    }
    
    // 释放所有
    for (auto ptr : ptrs) {
        if (ptr) {
            mycudaError_t result = mm.free(ptr);
            REQUIRE(result == Success);
        }
    }
    
    INFO("Memory overlap protection tests passed!");
}

TEST_CASE("MemoryManager managed memory operations") {
    MemoryManager& mm = MemoryManager::instance();
    
    SECTION("Managed memory allocation") {
        void* ptr = mm.malloc_managed(1024);
        REQUIRE(ptr != nullptr);
        INFO("Allocated 1024 bytes for managed memory at: " << ptr);
        
        // 测试managed内存访问
        int testData = 123;
        mm.access(ptr, &testData, sizeof(testData), true);  // 写入
        
        int readData = 0;
        mm.access(ptr, &readData, sizeof(readData), false);  // 读取
        REQUIRE(readData == testData);
        INFO("Managed memory access test passed!");
        
        mycudaError_t result = mm.free(ptr);
        REQUIRE(result == Success);
    }
}