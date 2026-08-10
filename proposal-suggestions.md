# 提案池（待架构讨论）

> arch 阶段输入。guide-arch Phase 5.5 逐个审查，批准后添加到 `proposal-approved.md`。

| 提案 | 优先级 | 来源 | 添加时间 | 状态 |
|------|--------|------|----------|------|
| [ptxir-format-compliance](openspec/changes/ptxir-format-compliance/) | P1 | 差距分析 G1-G9/D1-D5 + ADR-0023 7 决策实施 | 2026-07-30 | ❌ 已拒绝 (2026-08-01) |
| [implement-ptxir-cubin-embed-extension](improvements/implement-ptxir-cubin-embed-extension.md) | P1 | ADR-0024 (Accepted 2026-08-06, commit `18ad58cb`) | 2026-08-06 | ✅ 已批准 (2026-08-06, Oracle 2nd-pass APPROVED) |
| [ptxir-driver-api-front-door](improvements/ptxir-driver-api-front-door.md) | P0 | [roadmap.md](roadmap.md) Phase 12.3.A + 架构 §2 §4.2 + 2026-08-10 Oracle review (RISK MEDIUM-HIGH, 5 conditions) | 2026-08-10 | ✅ 已批准 (2026-08-10, Oracle 1st-pass APPROVED-WITH-CONDITIONS) |
| [multi-kernel-manifest-adr-0028](improvements/multi-kernel-manifest-adr-0028.md) | P1 BLOCKING | 架构 §11 BLOCKING DEPENDENCY + 3 个已 ship ADR v1 单 kernel 限制拖累 + 2026-08-10 Oracle review (RISK MEDIUM, 4 conditions, 硬串行依赖 Phase 12.3.A) | 2026-08-10 | ✅ 已批准 (2026-08-10, Oracle 1st-pass APPROVED-WITH-CONDITIONS) |
| [hal-extension-ptxemu-usrlinu-emu-taskrunner](improvements/hal-extension-ptxemu-usrlinu-emu-taskrunner.md) | P1 | 架构 §2 CP 端跨仓集成节点 + ADR-0029 §D8 + 2026-08-10 Oracle review (RISK LOW, 3 conditions) | 2026-08-10 | ✅ 已批准 (2026-08-10, Oracle 1st-pass APPROVED-WITH-CONDITIONS) |
| [split-cpptlm-core-minimal](improvements/split-cpptlm-core-minimal.md) | P2 | ADR-0022 §未来 — Oracle 评估（2026-07-28）：已评估推迟；触发条件未满足 | 2026-08-10 | ⏸️ 延迟 (2026-08-10, Oracle DEFER；触发条件见 proposal) |
