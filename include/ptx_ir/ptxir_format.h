// ptxir_format.h
#ifndef PTXIR_FORMAT_H
#define PTXIR_FORMAT_H

#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// PTXIR Binary Format
// Version: 3
// Layout: Header + Section TOC + Section Data + String Table (at end)
// ============================================================================

static constexpr char PTXIR_MAGIC[4] = {'P', 'T', 'X', 'I'};
static constexpr uint16_t PTXIR_VERSION = 3;

// Section types
enum class PtxirSectionType : uint8_t {
    REGDECL = 1,
    TYPE = 2,
    KERNEL = 3,
    CONSTANT = 4,
    STRING_TABLE = 5,
    MANIFEST = 6   // NEW: PTXIR-Embedded CUBIN manifest (cubin_hash + kernel_name + params)
};

enum class ParamKind : uint8_t { U8 = 1, U16 = 2, U32 = 4, U64 = 8, F32 = 9, F64 = 10 };

struct ManifestParam {
    std::string name;
    uint16_t size;
    ParamKind kind;
};

struct ManifestSection {
    std::vector<uint8_t> cubin_hash;   // SHA-256 (32 bytes)
    std::string kernel_name;
    uint8_t ptx_address_size = 64;       // 32 or 64
    std::vector<ManifestParam> params;
};

// Header: 24 bytes (little-endian)
struct PtxirHeader {
    char magic[4];                    // 0-3: "PTXIR"
    uint16_t version;                 // 4-5: PTXIR_VERSION
    uint16_t flags;                   // 6-7: reserved, must be 0
    uint16_t section_count;           // 8-9: number of TOC entries
    uint16_t reserved;                // 10-11: reserved
    uint32_t string_table_offset;    // 12-15: absolute file offset
    uint32_t string_table_size;       // 16-19: bytes
    uint32_t header_size;            // 20-23: sizeof(PtxirHeader) = 24
};

// TOC entry: 6 bytes
struct PtxirSectionTOC {
    uint8_t type;     // PtxirSectionType
    uint8_t reserved;
    uint32_t offset;  // absolute file offset to section start
};

// Instruction encoding helpers
namespace ptxir_encoding {

// Opcode: u16 StatementType enum value
// Pred: i32 (-1 = no predicate)
// BranchInstr: opcode + pred + target_id + pred_negated + reconvergence_pc
constexpr size_t BRANCH_ENCODED_SIZE = sizeof(uint16_t) + sizeof(int32_t) +
                                       sizeof(uint32_t) + sizeof(uint8_t) + sizeof(int32_t);

// GenericInstr: opcode + qualifier_count + qualifiers[] + dst_id + src_count + src_ids[]
// Variable size, caller must compute

// LabelInstr: opcode + label_str_id
constexpr size_t LABEL_ENCODED_SIZE = sizeof(uint16_t) + sizeof(uint32_t);

// VoidInstr: opcode only
constexpr size_t VOID_ENCODED_SIZE = sizeof(uint16_t);

// BarrierInstr: opcode + bar_id + reconvergence_pc
constexpr size_t BARRIER_ENCODED_SIZE = sizeof(uint16_t) + sizeof(int32_t) +
                                        sizeof(int32_t);

// DeclarationInstr: opcode + kind + type + name_str_id + array_size
constexpr size_t DECLARATION_ENCODED_SIZE = sizeof(uint16_t) + sizeof(uint8_t) +
                                            sizeof(uint8_t) + sizeof(uint32_t) + sizeof(int32_t);

}  // namespace ptxir_encoding

#endif  // PTXIR_FORMAT_H
