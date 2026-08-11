# Spec: cross-repo-rfc

## ADDED Requirements

### Requirement: 跨仓 RFC 文档记录 commit 顺序 + 外部 ADR 引用

The system SHALL create `openspec/changes/hal-extension-ptxemu-usrlinu-emu-taskrunner/rfc-hal-extension.md` that:
- Cites **ADR-0029 §D8** as the HAL extension context
- Cites **TaskRunner ADR-035 R5.1** as the external cross-repo commit order specification
- Cites **UsrLinuxEmu ADR-036** as the 3-region architecture constraint
- Records cross-repo commit order (per Oracle C2)
- Explicitly states "PTX-EMU 仓不拥有跨仓协调责任；commit 顺序是 integrator 责任" (per Oracle C3)

#### Scenario: RFC 引用 3 个外部 ADR

- **WHEN** RFC is reviewed
- **THEN** it contains verbatim citations of ADR-0029 §D8, TaskRunner ADR-035 R5.1, and UsrLinuxEmu ADR-036

#### Scenario: RFC 显式声明 PTX-EMU 不拥有跨仓协调

- **WHEN** RFC is reviewed
- **THEN** a line reads "PTX-EMU 仓不拥有跨仓协调责任；commit 顺序是 integrator 责任"

### Requirement: cross-repo 端到端 acceptance（跨仓，不在本仓实施）

The system SHALL document (not implement) the TaskRunner-side acceptance criterion: `cuModuleLoadData(image)` → `cuLaunchKernel` → `cuModuleUnload` end-to-end test must pass on the TaskRunner side. PTX-EMU 仓 only validates its own compatibility via tests in `libptxemu-abi-freeze` spec.

#### Scenario: RFC 包含跨仓 acceptance 段

- **WHEN** RFC is reviewed
- **THEN** it has a section documenting the cross-repo end-to-end test as TaskRunner's responsibility (NOT PTX-EMU's)