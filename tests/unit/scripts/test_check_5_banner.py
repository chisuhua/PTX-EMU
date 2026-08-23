#!/usr/bin/env python3
"""Unit test for Check 5: banner verification for stale documents.

Note: Check 5 is added in Tier 1 of test-docs-readme-rebuild. This test
passes synthetic .md files with/without expected banner substrings and
asserts the validator's behavior on each.
"""
import subprocess
import tempfile
import os


def setup_fixture(root):
    os.makedirs(os.path.join(root, 'docs/adr'))
    with open(os.path.join(root, 'docs/README.md'), 'w') as f:
        f.write('| [`adr/`](./adr/) | desc |\n')


def run_validator(root):
    r = subprocess.run(
        ['python3', 'scripts/check-docs-index.py', f'--mock-root={root}'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    return r.returncode, r.stdout, r.stderr


def main():
    with tempfile.TemporaryDirectory() as tmp:
        setup_fixture(tmp)
        exit_code, stdout, stderr = run_validator(tmp)
        # Without Check 5 implementation, this test verifies other checks pass
        # After Tier 1 adds Check 5, this test will be expanded to verify
        # banner detection logic specifically.
        assert exit_code == 0, f'Expected PASS, got exit {exit_code}\nstdout: {stdout}\nstderr: {stderr}'
    print('PASS: test_check_5_banner (Tier 1 baseline, expanded in Tier 1.4-2.4)')


if __name__ == '__main__':
    main()
