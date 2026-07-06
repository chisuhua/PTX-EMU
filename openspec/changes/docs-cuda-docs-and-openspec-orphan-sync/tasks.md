## 1. Phase 1: 验证当前状态（~0.5h）

- [ ] 1.1 D-1: 运行 `ls -d docs/*/ | wc -l` 与 `docs/README.md` 标题声称的数字对比；若一致仅记录结论
- [ ] 1.2 D-1: 运行 `for d in docs/*/; do echo "$d -> $(find "$d" -type f | wc -l) files"; done` 对照索引表格验证无遗漏子目录
- [ ] 1.3 D-3: 运行 `find .opencode/skills/ -maxdepth 1 -type d | wc -l` 与 `docs/skills/README.md` §统计 对比，确认 18=18
- [ ] 1.4 D-4: 运行 `for d in openspec/changes/archive/2026-06-24-*/; do echo "$d: $(find "$d" -type f | wc -l) files"; done` 确认 5 个缺 design.md
- [ ] 1.5 D-5: 运行 `ls docs/skills/` 列出过期副本（ptx-debug/, ptxir-serialization/, ptx-grammar-modification.md, three-mode-testing/）
- [ ] 1.6 D-6: 运行 `grep -c "E[1-8]" docs/audits/HEALTH-AUDIT-2026-06-21.md` 确认主审计尚未合并 ERRATA（预期 0 matches 或仅 §0 提及）

## 2. Phase 2: Fix D-4 — Retroactive design.md（~0.5h）

- [ ] 2.1 为 `2026-06-24-phase3-cvt-precision-bugfix` 合成 `design.md`（`git log --all --oneline -- openspec/changes/archive/2026-06-24-phase3-cvt-precision-bugfix/`）
- [ ] 2.2 为 `2026-06-24-phase3-half-precision-bugfix` 合成 `design.md`
- [ ] 2.3 为 `2026-06-24-phase3-t2-1-active-mask-unify` 合成 `design.md`
- [ ] 2.4 为 `2026-06-24-phase3-t2-3-god-class-split` 合成 `design.md`
- [ ] 2.5 为 `2026-06-24-phase3-t2-6-cvt-strategy-pattern` 合成 `design.md`
- [ ] 2.6 验证：运行 `for d in openspec/changes/archive/2026-06-24-*/; do echo "$d: $(test -f $d/design.md && echo HAS design.md || echo MISSING)"; done`
- [ ] 2.7 更新 `docs/roadmap/post-phase3-debt-roadmap.md` §1.3 D-4 状态标记为 RESOLVED

## 3. Phase 3: Fix D-5 — 清理 docs/skills/ 过期副本（~1h）

- [ ] 3.1 `git rm -r docs/skills/ptx-debug/`（权威版本在 `.opencode/skills/ptx-debug/`）
- [ ] 3.2 `git rm -r docs/skills/ptxir-serialization/`（权威版本在 `.opencode/skills/ptxir-serialization/`）
- [ ] 3.3 `git rm docs/skills/ptx-grammar-modification.md`（权威版本在 `.opencode/skills/ptx-grammar-modification/`）
- [ ] 3.4 `git rm -r docs/skills/three-mode-testing/`（含 `__pycache__/`；已移至 `.opencode/skills.disable/three-mode-testing/`）
- [ ] 3.5 验证 `docs/skills/README.md` 表格中的技能路径仍可链接到正确的 `.opencode/skills/` 位置
- [ ] 3.6 运行 `ls docs/skills/` 确认仅剩 `README.md`、`post-dominator-algorithm.md`、`simt-reconvergence.md`
- [ ] 3.7 更新 `docs/roadmap/post-phase3-debt-roadmap.md` §1.3 D-5 状态标记为 RESOLVED

## 4. Phase 4: Fix D-6 — 合并 ERRATA 到主审计（~1h）

- [ ] 4.1 在 `HEALTH-AUDIT-2026-06-21.md` §0.2 第五要点添加 `**[勘误 E1: 108→81 public 字段]**` 标记
- [ ] 4.2 在 §0.2 第三要点添加 `**[勘误 E2: 5→7 处 Symtable 泄漏]**` 标记
- [ ] 4.3 在 §1.2 添加 E3（14→18 引用）+ E4（H2 🔴→🟡 M）标记
- [ ] 4.4 在 §0.4/§3.5 添加 E5（2d→2-3d）+ E8（PTX 8.7+ 占位：A+PTX_WARN）标记
- [ ] 4.5 在 §8 添加 E6（Phase 1 顺序调整：P0-4→P0-3→P0-2→P0-1）标记
- [ ] 4.6 在 §2.2.1 添加 E7（cudaStream_t: destroy exists but type-unsafe, fake sync is real issue）标记
- [ ] 4.7 在 `HEALTH-AUDIT-2026-06-21-ERRATA.md` 顶部添加合并状态标记：`**[2026-07-06 已合并到主审计 by change docs-cuda-docs-and-openspec-orphan-sync]**`
- [ ] 4.8 更新 `docs/roadmap/post-phase3-debt-roadmap.md` §1.3 D-6 状态标记为 RESOLVED

## 5. Phase 5: 最终验证与归档（~0.5h）

- [ ] 5.1 验证 `docs/README.md` 索引无遗漏：`for d in docs/*/; do grep -q "$(basename $d)" docs/README.md || echo "MISSING: $d"; done`
- [ ] 5.2 验证 `docs/skills/` 仅含 3 个文件：`ls docs/skills/`
- [ ] 5.3 验证 6 个归档 change 均含 design.md（`integrate-barrier-module-cta-warp` 原本就有；5 个 retroactive 合成）
- [ ] 5.4 验证 `HEALTH-AUDIT-2026-06-21.md` 含 8 个 `**[勘误:**` 标记
- [ ] 5.5 运行 `git status` 确认仅文档文件被修改（无 `.cpp`/`.h` 变更）
- [ ] 5.6 更新 `docs/roadmap/post-phase3-debt-roadmap.md` §1.3 D-1/D-3 状态（若 Phase 1 验证均通过则标记 RESOLVED）