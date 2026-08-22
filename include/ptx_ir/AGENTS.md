# include/ptx_ir/ AGENTS.md

## OVERVIEW

**[DEPRECATED — forwarding-only after Phase 1, HSK-8 ack 738b412c]**

Internal IR types for PTX-EMU implementation. These headers are kept
as forwarding shims to the new public layout at `include/ptxemu/ir/`
(per Phase 1 Migration Plan + design.md Decision 1).

**Migration timeline**: Old path forwarding headers in this directory
will be REMOVED in the next release cycle. New code MUST include
`<ptxemu/ir/...>` directly. Old path provides backward compat for one
release cycle.

## KEY FILES (forwarding to ptxemu/ir/)

| File | Role |
|------|------|
| `ptx_types.h` | Forwarding → `<ptxemu/ir/ptx_types.h>` (Qualifier, StatementType, OperandKind enums) |
| `operand_context.h` | Forwarding → `<ptxemu/ir/operand_context.h>` (OperandContext + 6 operand types, post-Phase 0.3d clean) |
| `statement_context.h` | Forwarding → `<ptxemu/ir/statement.h>` (20 instruction types + StatementContext class + InstrVariant) |

Other files (`kernel_context.h` / `ptx_context.h` / `param_context.h` /
`ptx_syntax_utils.h` / `instruction_latency*.h` / `ptxir_*.h` /
`ptx_qualifier.def` / `ptx_op.def`) remain PTX-EMU internal
implementation details — NOT part of HSK-8 public API.

## CONVENTIONS

- **Old path only as shim**: `#include "ptx_ir/foo.h"` works via
  forwarding. `#include <ptxemu/ir/foo.h>` is preferred for new code.
- **Namespace compatibility**: forwarding headers do `using namespace ::ptxemu::ir;` so old code using unqualified types (e.g., `Qualifier`) still compiles.
- **Phase 1 scaffolding (commit 564174f7)** introduced
  `include/ptxemu/ir/` with full content duplicated. Old path
  forwards to new path.

## ANTI-PATTERNS

- ❌ **不要新增到 `include/ptx_ir/`** — 所有新 IR 类型必须放入 `include/ptxemu/ir/`
- ❌ **不要修改此目录的 forwarding 内容** — 编辑会破坏 backward compat; 等 release 周期结束后直接删除整个目录
- ❌ **不要从 `src/ptxsim/` / `src/ptx_parser/` 等内部实现模块引用此目录** — 内部模块已用 `ptxemu::ir::` namespace; 仅 legacy code 仍依赖旧路径