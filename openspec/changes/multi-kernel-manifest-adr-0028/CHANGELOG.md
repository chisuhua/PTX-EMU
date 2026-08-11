# multi-kernel-manifest-adr-0028 — Ship Log

Phase 12.4 complete. PTX-EMU 仓侧 ADR-0028 multi-kernel manifest 落地。

## 交付物
- ADR-0028-multi-kernel-manifest.md (新建)
- ManifestSection 扩展为 vector<kernel_entry> + PTXIR_VERSION bump (3 → 4)
- PTXIRLoader + PtxEmuImageExecutor 多 entry support
- runtime multi-kernel 名查询基础设施
- 下游 ADR §v1 段落更新（0025/0027/0029）
- architecture doc v1.3 → v1.4

## Oracle 评审条件
- C1 ADR-0028 先建: ✅
- C2 硬串行（Phase 12.3.A 完成）: ✅
- C3 v1 backward-compat: ✅ reader 容错
- C4 architecture changelog: ✅

## 推迟项（Phase 12.5）
- v2 PTXIR writer + multi-entry fixture
- `ptxemu_image_get_function_by_name` 新 ABI
- `ptxir_build/embed/extract` 多 kernel 完整支持
- e2e 多 kernel CUDA 测试

## Commits
- `e5fe7f2a` docs(adr): add ADR-0028 multi-kernel manifest
- `05504d0c` feat(ptxir): extend ManifestSection to vector<kernel_entry> + bump PTXIR_VERSION
- `757a8064` feat(cudart): multi-entry support in PTXIRLoader + PtxEmuImageExecutor
- `c6ac1176` test(cudart): add multi-kernel selection placeholder test
- `3d288cc2` docs(adr): update v1 limitation paragraphs + architecture v1.4
