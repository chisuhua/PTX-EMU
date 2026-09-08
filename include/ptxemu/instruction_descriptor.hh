// include/ptxemu/instruction_descriptor.hh
// InstrDescriptor POD: PTX-EMU 端 mirror of cpptlm::gpu::InstrDescriptor
// (per HSK-9 OpenSpec change hsk9-icompute-api-v1-consumer-pinning Phase 2 task 2.0)
//
// 跨仓契约 (per HSK-9 §3 + 2027-02-09-hsk-9-icompute-api-v1-sm-rewrite.md):
//   - PTX-EMU 端 PtxEmuDeviceImpl::set_instr_descriptor_buf(const InstrDescriptor*, uint32_t)
//     注入 PTX-EMU 已解码的指令描述符 (producer 侧, per §3 R9.2)
//   - CppTLM 端 IComputeDevice::set_instr_descriptor_buf (15-方法接口) 接收 buffer
//   - POD 字段与 cpptlm::gpu::InstrDescriptor 1:1 对应 (47 bytes 估计, ≤128 bytes 扩展)
//   - **Phase 2 PR scope**: PTX-EMU 端 mirror POD, 用于 producer 侧注入 (task 2.1)
//   - **HSK-9 协议**: PTX-EMU owner ack 14 天窗口内确认字段对齐 (PTXEMU_API_VERSION=1 冻结,
//     POD 字段变更必须签发 HSK-N bump VERSION)
//
// 命名约定: PTX-EMU 端使用 ptxemu::InstrDescriptor (与 device_api.h namespace 一致),
// 不是 cpptlm::gpu::InstrDescriptor — PTX-EMU 仓独立, 不 link CppTLM header.
//
// 作者 CppTLM/PTX-EMU Team / 日期 2027-02-09 (HSK-9 Phase 2)
#ifndef PTXEMU_INSTRUCTION_DESCRIPTOR_HH
#define PTXEMU_INSTRUCTION_DESCRIPTOR_HH

#include <cstdint>
#include <cstddef>

namespace ptxemu {

// === Pipe 类别 (mirror of cpptlm::gpu::PipeClass) ===
enum class PipeClass : uint8_t {
    kScalarALU = 0,
    kVectorALU = 1,
    kMatrixCore = 2,
    kSIMTLane = 3,
    kLsuGlobal = 4,
    kLsuLDS = 5,
    kBranch = 6,
};

// === Latency 类别 (mirror of cpptlm::gpu::LatencyClass) ===
enum class LatencyClass : uint8_t {
    kFixed1Cycle = 0,
    kFixed4Cycle = 1,
    kFixed8Cycle = 2,
    kFixed16Cycle = 3,
    kFixed32Cycle = 4,
    kMemory = 5,
};

// === ISA 类别 (mirror of cpptlm::gpu::IsaType) ===
enum class IsaType : uint8_t {
    kUnknown = 0,
    kCDNA64 = 1,
    kPTX70 = 2,
    kSASS = 3,
};

// === 控制位 (mirror of cpptlm::gpu::CtrlBits) ===
struct CtrlBits {
    uint8_t branch_type = 0;
    uint8_t is_accvgpr = 0;
    uint8_t reserved_ctrl0 = 0;
    uint8_t reserved_ctrl1 = 0;
};

// === 完整指令描述符 POD (mirror of cpptlm::gpu::InstrDescriptor) ===
// 字段对齐: 与 CppTLM 端 1:1 同构 (per HSK-9 §3 cross-repo POD 冻结契约)
struct InstrDescriptor {
    // === 8 字节 header ===
    IsaType isa_type = IsaType::kUnknown;
    uint8_t  result_num = 0;
    uint8_t  num_src = 0;
    uint8_t  num_dst = 0;
    uint32_t reserved_hdr = 0;

    // === 16 字节 instr_id + pc + sm_id ===
    uint64_t instr_id = 0;
    uint32_t pc = 0;
    uint32_t sm_id = 0;

    // === 16 字节 exec info ===
    uint16_t exec_cycles = 0;
    uint16_t cycles_remaining = 0;
    uint8_t  warpid = 0;
    uint8_t  reserved_exec = 0;
    uint64_t exec_mask = 0xFFFFFFFFFFFFFFFFull;

    // === 48 字节 result_value[4] + dst_regs[4] + src_regs[4] ===
    uint64_t result_value[4] = {0, 0, 0, 0};
    uint16_t dst_regs[4] = {0, 0, 0, 0};
    uint16_t src_regs[4] = {0, 0, 0, 0};

    // === 24 字节 memory_data ===
    uint64_t memory_data = 0;
    uint8_t  memory_data_valid = 0;
    bool     is_memory = false;
    uint8_t  mem_size = 0;
    uint32_t reserved_mem = 0;
    uint64_t target_vaddr = 0;

    // === 2 字节 pipe + latency ===
    PipeClass    pipe = PipeClass::kScalarALU;
    LatencyClass latency_class = LatencyClass::kFixed1Cycle;

    // === 4 字节 CtrlBits ===
    CtrlBits ctrl{};

    // === 8 字节 lane_mask ===
    uint64_t lane_mask = 0xFFFFFFFFFFFFFFFFull;
};

// 静态断言: PTX-EMU 端 POD 与 CppTLM 端对齐 (size 必须一致)
// PTX-EMU 仓独立编译时无法直接访问 CppTLM 端 POD; 此 static_assert 仅校验
// sizeof <= 128 字节 (per HSK-9 §3 R9.2 47 bytes 估计 + Oracle P1 扩展).
static_assert(sizeof(InstrDescriptor) <= 128, "ptxemu::InstrDescriptor too large (>128 bytes)");

}  // namespace ptxemu

#endif  // PTXEMU_INSTRUCTION_DESCRIPTOR_HH