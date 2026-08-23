#!/usr/bin/env python3
"""Unit test for Check 1: FAIL when actual subdir NOT in README index."""
import subprocess
import tempfile
import os


def setup_fail_fixture(root):
    """Create 3 subdirs but README only lists 2 (missing 'reports')."""
    for name in ['adr', 'plans', 'reports']:
        os.makedirs(os.path.join(root, 'docs', name))
    with open(os.path.join(root, 'docs', 'README.md'), 'w') as f:
        f.write('| [`adr/`](./adr/) | desc |\n')
        f.write('| [`plans/`](./plans/) | desc |\n')


def run_validator(root):
    r = subprocess.run(
        ['python3', 'scripts/check-docs-index.py', f'--mock-root={root}'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    return r.returncode, r.stdout, r.stderr


def main():
    with tempfile.TemporaryDirectory() as tmp:
        setup_fail_fixture(tmp)
        exit_code, stdout, stderr = run_validator(tmp)
        assert exit_code == 1, f'Expected FAIL (exit 1), got exit {exit_code}\nstdout: {stdout}'
        assert 'NOT_INDEXED: reports' in stdout, f'Expected NOT_INDEXED: reports in output:\n{stdout}'
        assert 'Check 1' in stdout
    print('PASS: test_check_1_mismatch FAIL detected correctly')


if __name__ == '__main__':
    main()
