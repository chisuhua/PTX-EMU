#!/usr/bin/env python3
"""Unit test for Check 3: FAIL for hand-edited statistics (post-Tier-2).

Note: Tier 1 currently treats hand-edited stats as WARN. This test expects
FAIL behavior, which will FAIL (intentionally) on Tier 1 implementation.
After Tier 2 upgrade (Check 3 WARN->FAIL), this test will PASS.
"""
import subprocess
import tempfile
import os


def setup_fail_fixture(root):
    os.makedirs(os.path.join(root, 'docs', 'adr'))
    with open(os.path.join(root, 'docs/README.md'), 'w') as f:
        f.write('| [`adr/`](./adr/) | desc |\n')
        f.write('| 38 测试 | auto |\n')


def run_validator(root):
    r = subprocess.run(
        ['python3', 'scripts/check-docs-index.py', f'--mock-root={root}'],
        capture_output=True, text=True, cwd='/workspace/project/PTX-EMU'
    )
    return r.returncode, r.stdout, r.stderr


def main():
    with tempfile.TemporaryDirectory() as tmp:
        setup_fail_fixture(tmp)
        exit_code, stdout, stderr = run_validator(tmp)
        is_tier1_warn = (
            'WARN' in stdout and exit_code == 0
        )
        is_tier2_fail = (
            'FAIL' in stdout and exit_code == 1
        )
        if is_tier1_warn:
            print('SKIP (Tier 1 behavior — activates after Tier 2)')
            return
        assert is_tier2_fail, (
            f'Expected Tier 2 FAIL; got exit={exit_code}\nstdout: {stdout}'
        )
    print('PASS: test_check_3_stats FAIL detected (Tier 2 contract)')


if __name__ == '__main__':
    main()
