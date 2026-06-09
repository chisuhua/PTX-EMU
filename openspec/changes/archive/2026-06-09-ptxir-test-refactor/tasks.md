## 1. Update test_helpers.hpp (MUST precede test creation)

- [x] 1.1 Add `serialize_to_string(const StatementContext&)` helper returning `std::string`
- [x] 1.2 Add `deserialize_from_string(const std::string&)` helper returning `StatementContext`
- [x] 1.3 Document serialize→deserialize flow in comments (flow: PTX → StatementContext → .ptxir → StatementContext)

## 2. Refactor test_ptxir_serialization.cpp

- [x] 2.1 **Remove raw tests**: Remove all 10 `[raw]`-tagged TEST_CASEs across these 4 files. Coverage gap analysis:
  - `test_warp_divergence_mode3a.cpp`: 2 raw tests → `[shared]` and `[divergence]` coverage exists in non-raw Mode3a counterparts
  - `test_warp_divergence_mode3b.cpp`: 3 raw tests → `[barrier]` (Wbar.arrive/is_complete) NOT covered elsewhere — **add barrier validation to non-raw Mode3b tests before removing**
  - `test_divergence_sync_standalone_mode3a.cpp`: 2 raw tests → coverage exists in non-raw counterparts
  - `test_divergence_sync_standalone_mode3b.cpp`: 3 raw tests → same barrier gap as above — **add barrier validation to non-raw Mode3b tests before removing**
- [x] 2.2 Add TEST_CASE: Mode 4 serialize→deserialize preserves operand values (values remain identical after roundtrip)
- [x] 2.3 Add TEST_CASE: Mode 4 deserialization produces valid StatementContext without ANTLR dependency (loads .ptxir directly)
- [x] 2.4 Update TEST_CASE tags to use `[mode4][roundtrip]` consistently

## 3. Create test_four_mode_flow.cpp

- [x] 3.1 Create `tests/three_mode_testing/test_four_mode_flow.cpp` (Mode 1→2→3→4 end-to-end pipeline)
- [x] 3.2 Add TEST_CASE: Mode 1 PTX extraction matches cuobjdump output (baseline extraction correctness)
- [x] 3.3 Add TEST_CASE: Mode 2 parsed statements structurally equivalent to Mode 3 hand-written (IR equivalence)
- [x] 3.4 Add TEST_CASE: Mode 3 serialization produces valid .ptxir text (format correctness)
- [x] 3.5 Add TEST_CASE: Mode 4 deserialization produces StatementContexts identical to Mode 2 input
- [x] 3.6 Add TEST_CASE: Full pipeline Mode 1→2→3→4 preserves kernel semantics end-to-end

## 4. Update CMakeLists.txt

- [x] 4.1 Add `test_four_mode_flow.cpp` to auto-detection pattern in `tests/three_mode_testing/CMakeLists.txt`
  - **NOTE**: Current pattern `test_.*_mode3[ab]\.cpp$` does NOT match `test_four_mode_flow.cpp`. Add `_four_mode_flow\.cpp$` to the regex or add it explicitly.
- [x] 4.2 Link `ptxir_writer` and `ptxir_reader` targets to both `test_ptxir_serialization` and `test_four_mode_flow` targets

## 5. Deprecate Mode 3c

- [x] 5.1 Add deprecation notice in `docs/skills/three-mode-testing/SKILL.md`: Mode 3c is deprecated and superseded by Mode 4. Existing Mode 3c tests remain for backwards compatibility but new tests should use Mode 4.

## 6. Update Documentation

- [x] 6.2 Update `docs/skills/three-mode-testing/SKILL.md` with four-mode flow description (Mode 1: raw PTX extraction, Mode 2: parsed IR, Mode 3: IR serialization, Mode 4: IR deserialization/resurrection)

Note: 6.1 skipped — THREE-MODE-TESTING-GUIDE.md does not exist in the repository. The SKILL.md already serves this purpose and has been updated with Mode 4 information.

---

### Raw Test Removal Detail (for 2.1)

**10 `[raw]` tests to remove — 2 have coverage gaps:**

| File | Count | Lines | Tags | Removal risk | Gap action needed |
|------|-------|-------|------|--------------|-------------------|
| `test_warp_divergence_mode3a.cpp` | 2 | 57, 119 | `[shared][raw]`, `[divergence][raw]` | Low | None — non-raw counterparts cover same WarpContext paths |
| `test_divergence_sync_standalone_mode3a.cpp` | 2 | 57, 119 | `[shared][raw]`, `[divergence][raw]` | Low | None — non-raw counterparts cover same WarpContext paths |
| `test_warp_divergence_mode3b.cpp` | 3 | 42, 92, 139 | `[barrier][raw]`, `[shared][raw]`, `[divergence][raw]` | **HIGH** | `[barrier][raw]` tests `Wbar.arrive()` + `is_complete()` — these paths NOT exercised by non-raw counterparts. Must add explicit barrier validation to non-raw Mode3b tests BEFORE removing. |
| `test_divergence_sync_standalone_mode3b.cpp` | 3 | 42, 92, 139 | `[barrier][raw]`, `[shared][raw]`, `[divergence][raw]` | **HIGH** | Same barrier gap — `Wbar.arrive()` + `is_complete()` not covered by StatementContext path. Must add barrier validation to non-raw Mode3b tests BEFORE removing. |

**Action**: Before removing any `[raw]` test, verify the non-raw counterparts actually call `wbar.arrive()` and `wbar.is_complete()`. If not, add explicit barrier TEST_CASEs to the non-raw Mode3b tests first.