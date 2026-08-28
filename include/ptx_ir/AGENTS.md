# include/ptx_ir/ AGENTS.md

## OVERVIEW

**Dual-state during Phase 1.5 migration (HSK-8 follow-up, in progress)**.

`include/ptx_ir/` 与 `include/ptxemu/ir/` 当前并存:
- `include/ptxemu/ir/` (Phase 1 scaffolding commit `564174f7`): ptxemu::ir 命名空间包裹的新公共布局。
- `include/ptx_ir/`: 全局命名空间的等价类型, 仍为 PTX-EMU 内部 caller 实际引用路径。

**Phase 1.5 进行中**: 内部 src/ caller (177 个文件) 迁移到 `ptxemu::ir::*` 限定名; 完成后旧 `include/ptx_ir/` 3 个公共类型 header (ptx_types.h / operand_context.h / statement_context.h) 改为 forwarding shim (`#include <ptxemu/ir/...>` + `using namespace ::ptxemu::ir;`) — 与 openspec/specs/statement-ir-public/spec.md:46-48 一致。

**Phase 1.5+1 release cycle**: 旧路径 forwarding shim 移除, 全仓仅 `include/ptxemu/ir/`。

## KEY FILES (Phase 1.5 中状态)

| 文件 | 当前 | 迁移后 (Phase 1.5f 完成) |
|------|------|--------------------------|
| `include/ptxemu/ir/statement.h` | Phase 1 scaffolding, ptxemu::ir 命名空间, 0 caller (死代码但作为合同基线) | canonical |
| `include/ptxemu/ir/operand_context.h` | Phase 1 scaffolding, ptxemu::ir 命名空间, 0 caller | canonical |
| `include/ptxemu/ir/ptx_types.h` | Phase 1 scaffolding, ptxemu::ir 命名空间, 0 caller | canonical |
| `include/ptxemu/ir/execution_types.h` | Phase 1 scaffolding, InstructionState enum, 0 caller | canonical |
| `include/ptxemu/ir/ptx_op.def` | byte-equivalent to ptx_ir/ptx_op.def | canonical (Phase 1.5b 单源化) |
| `include/ptxemu/ir/ptx_qualifier.def` | byte-equivalent to ptx_ir/ptx_qualifier.def | canonical (Phase 1.5b 单源化) |
| `include/ptx_ir/statement_context.h` | 全局命名空间, 仍在 caller 使用 | forwarding shim (`#include <ptxemu/ir/statement.h>` + `using namespace ::ptxemu::ir;`) |
| `include/ptx_ir/operand_context.h` | 全局命名空间, 仍在 caller 使用 | forwarding shim |
| `include/ptx_ir/ptx_types.h` | 全局命名空间, 仍在 caller 使用 | forwarding shim |
| `include/ptx_ir/ptx_op.def` | byte-equivalent | `#include <ptxemu/ir/ptx_op.def>` (1 行) |
| `include/ptx_ir/ptx_qualifier.def` | byte-equivalent | `#include <ptxemu/ir/ptx_qualifier.def>` (1 行) |

剩余 PTX-EMU internal files (kernel_context / ptx_context / param_context / ptx_syntax_utils / instruction_latency* / ptxir_* / ptxir_serialization) 永不移至 `ptxemu/ir/` (openspec/specs/statement-ir-public 不包含; per `ptxemu-core-library/spec.md:17-18` 它们仍 PRIVATE)。

## CONVENTIONS

- **Phase 1.5d 进行中**: 新代码 SHOULD include `<ptxemu/ir/...>` directly + 用 `ptxemu::ir::*` 限定名。
- **Phase 1.5d 进行中**: 现有 src/ caller 仍使用 `#include "ptx_ir/foo.h"` + 全局未限定类型, 这是过渡态。
- **Phase 1.5f 后**: 旧路径 forwarding shim 提供 `#include + using namespace` 双层兼容, 老 caller 无需立即改。
- **1.5k closure**: scanner strengthened (ANTLR exclusion + single-line namespace fix); Invariant 8 wired; shim retained for backward compatibility.

## ANTI-PATTERNS

- ❌ **不要新增到 `include/ptx_ir/` 全局类型** — 所有新 IR 类型必须放入 `include/ptxemu/ir/`
- ❌ **不要绕过 forwarding shim 引入新限定名之外的依赖** — 旧 path 内部永远走 `using namespace`, 不要用 `ptxemu::ir::` 直接在迁移完成前的 caller
- ❌ **不要修改 `include/ptxemu/ir/` canonical 内容以兼容 legacy caller** — canonical 是 contract, legacy 必须迁就 canonical