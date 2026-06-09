## ADDED Requirements

### Requirement: PTXIR 头部结构
`.ptxir` 文件 SHALL 以 24 字节固定头部开始，包含：Magic（"PTXIR" 4字节）、Version（uint16_t）、Flags（uint16_t）、SectionCount（uint16_t）、StringTableOffset（uint32_t）、StringTableSize（uint32_t）。

#### Scenario: 文件打开时读取头部
- **WHEN** `PtxirReader` 打开一个 `.ptxir` 文件
- **THEN** 读取并验证 Magic 为 "PTXIR"，否则抛出 `std::runtime_error`

#### Scenario: 版本不匹配时拒绝加载
- **WHEN** `PtxirReader` 读取的 Version 大于支持版本
- **THEN** 抛出 `std::runtime_error("Unsupported ptxir version")`

### Requirement: Section TOC 索引
头部后 SHALL 跟 SectionCount 个 TOC 条目，每个条目 6 字节：Type（uint8_t）、Offset（uint32_t）、Reserved（uint8_t=0）。

#### Scenario: 通过 TOC 随机访问 Section
- **WHEN** `PtxirReader` 需要读取 STRING_TABLE Section
- **THEN** 遍历 TOC 找到 Type=STRING_TABLE 的条目，Seek 到 Offset 读取

### Requirement: Section 类型枚举
Section 类型 SHALL 枚举为：`REGDECL(1)`, `TYPE(2)`, `KERNEL(3)`, `CONSTANT(4)`, `STRING_TABLE(5)`。

#### Scenario: 未知 Section 类型被跳过
- **WHEN** `PtxirReader` 遍历 TOC 发现未知 Type
- **THEN** 跳过该 Section（Seek 到下一 Section 开头），继续处理后续 Section

### Requirement: 指令 Opcode 编码
每条指令 SHALL 以 `opcode:u16`（StatementType 枚举值）开始，后跟类型特定的紧凑字段。

#### Scenario: BranchInstr 序列化格式
- **WHEN** `PtxirWriter` 序列化一条 `S_BRA` 指令
- **THEN** 写入 `opcode=S_BRA(u16)`, `pred_id:u32(-1 表示无 pred)`, `reconvergence_pc:i32`

#### Scenario: GenericInstr 序列化格式
- **WHEN** `PtxirWriter` 序列化一条 `S_ADD`（或其他 GenericInstr）指令
- **THEN** 写入 `opcode=S_ADD(u16)`, `qualifier_count:u8`, `qualifiers[]`, `dst_reg_id:u32`, `src_count:u8`, `src_reg_ids[]`

#### Scenario: 未知 Opcode 被跳过（前向兼容）
- **WHEN** `PtxirReader` 遇到无法识别的 opcode 值
- **THEN** 根据前一条已知指令的 operand_count 计算跳过字节数，继续解析

### Requirement: 值枚举（寄存器 ID 紧凑化）
序列化器 SHALL 在写入 KERNEL Section 前先遍历所有语句，为每个唯一的 `RegOperand` 分配从 0 开始的紧凑 `u32` ID，构建 REGDECL 表。

#### Scenario: 同一寄存器多次引用使用同一 ID
- **WHEN** `add.rn %r1, %r2, %r3` 和 `mov %r4, %r2` 被序列化
- **THEN** `%r2` 在 REGDECL 表中只出现一次（ID=1），两次指令中都写入 `src_reg_id=1`

#### Scenario: 反序列化重建 OperandContext
- **WHEN** `PtxirReader` 反序列化时读取 `src_reg_id=2`
- **THEN** 重建 `OperandContext{RegOperand{reg2id[2], 2}}`

### Requirement: 字符串表去重存储
所有字符串（立即数文本、变量名、寄存器名前缀）SHALL 通过字符串表去重，每个字符串编码为 `[length:u16, bytes:u8[N]]`。

#### Scenario: 重复字符串只存储一次
- **WHEN** 两条 `mov %r1, 0` 和 `mov %r2, 0` 被序列化
- **THEN** 字符串 "0" 只在 STRING_TABLE 中出现一次，所有对 "0" 的引用通过 offset 共享

### Requirement: 版本 1 支持的指令类型
Version 1 SHALL 支持以下指令类型的序列化：S_MOV, S_ADD, S_SUB, S_MUL, S_LD, S_ST, S_BRA, S_SETP, S_BAR, S_BAR_WARP_SYNC, S_EXIT, S_RET, S_LABEL, S_PRAGMA, S_NOP, S_CVT, S_MOV_IMM, S_LD_SHARED, S_ST_SHARED。

#### Scenario: 不支持的指令类型抛出异常
- **WHEN** `PtxirWriter` 尝试序列化 `S_WMMA` 指令（当前不支持）
- **THEN** 抛出 `std::runtime_error("Unsupported opcode for v1")`
