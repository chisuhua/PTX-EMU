# tests/unit/scripts/

Tests for the `scripts/check-docs-index.py` documentation validator.

## Why This Directory Exists

These tests protect the `scripts-check-docs-index.py` validator (4 main checks + tier additions) against regressions. The validator's regex patterns are fragile (e.g., `_` excluded from `[a-z0-9-]` character class crashed `technical_design` during C4 implementation), so changes to its logic MUST have automated test coverage.

## Test Style

**These tests are Python, NOT Catch2.** The validator under test is a Python script. Catch2 (used elsewhere in this repo) only supports C++ unit tests. The pattern adopted here:

- Per-check test file: `test_check_<N>_<name>.py`
- Each test creates a `tempfile.TemporaryDirectory()` fixture with synthetic `docs/`, `openspec/changes/archive/`, etc.
- Each test shells out to `python3 scripts/check-docs-index.py --mock-root=<tmpdir>` and asserts on exit code
- Failures are isolated: each test's fixture is destroyed automatically

## Running

```bash
# Run all tests (via wrapper script)
bash tests/unit/scripts/run_all_tests.sh

# Run individual tests
python3 tests/unit/scripts/test_check_1_subdirs.py
python3 tests/unit/scripts/test_check_6_skills.py

# Run via ctest (after CMake integration)
cd build && ctest -R "unit_doc_validator"
```

## Test Coverage Matrix

| Check | PASS test | FAIL test | File |
|-------|-----------|-----------|------|
| Check 1 | ✓ | ✓ | test_check_1_subdirs.py + test_check_1_mismatch.py |
| Check 2 | ✓ | ✓ | test_check_2_links_pass.py + test_check_2_links_fail.py |
| Check 3 | (FAIL only — by design) | ✓ | test_check_3_stats.py |
| Check 4 | ✓ | ✓ | test_check_4_orphan.py |
| Check 5 | ✓ | ✓ | test_check_5_banner.py |
| Check 6 | ✓ | ✓ | test_check_6_skills.py |

## How to Add a New Check Test

1. Identify which Check the new logic corresponds to
2. Create `tests/unit/scripts/test_check_<N>_<name>.py`:
   ```python
   #!/usr/bin/env python3
   """Test [description]."""
   import subprocess, sys, tempfile, os
   SCRIPT = "scripts/check-docs-index.py"
   def run_validator(root):
       result = subprocess.run(
           ["python3", SCRIPT, "--mock-root", root],
           capture_output=True, text=True,
       )
       return result.returncode, result.stdout, result.stderr

   if __name__ == "__main__":
       with tempfile.TemporaryDirectory() as tmp:
           # Set up fixture
           os.makedirs(f"{tmp}/docs/adr")
           with open(f"{tmp}/docs/README.md", "w") as f:
               f.write("| adr/ | ... |\n")
           # Run
           exit_code, _, _ = run_validator(tmp)
           assert exit_code == 0, f"Expected PASS, got exit {exit_code}"
       print("PASS")
   ```
3. Run `bash tests/unit/scripts/run_all_tests.sh`
4. Update this README's coverage matrix
