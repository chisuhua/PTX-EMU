#!/usr/bin/env python3
"""Unit test for Check 1: docs/ subdirs matches README index (PASS case)."""
import subprocess
import tempfile
import os
import sys


def setup_pass_fixture(root):
    """Create 3 docs/ subdirs all listed in README."""
    for name in ['adr', 'plans', 'reports']:
        os.makedirs(os.path.join(root, 'docs', name))
    with open(os.path.join(root, 'docs', 'README.md'), 'w') as f:
        f.write('# Index\n\n')
        f.write('| [`adr/`](./adr/) | desc |\n')
        f.write('| [`plans/`](./plans/) | desc |\n')
        f.write('| [`reports/`](./reports/) | desc |\n')


def run_validator(root):
    r = subprocess.run(
        ['python3', 'scripts/check-docs-index.py', f'--mock-root={root}'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    return r.returncode, r.stdout, r.stderr


def main():
    with tempfile.TemporaryDirectory() as tmp:
        setup_pass_fixture(tmp)
        exit_code, stdout, stderr = run_validator(tmp)
        assert exit_code == 0, f'Expected PASS (exit 0), got exit {exit_code}\nstdout: {stdout}\nstderr: {stderr}'
        assert 'Check 1' in stdout
        assert 'PASS' in stdout
        assert 'Check 2' not in stdout or 'FAIL' not in stdout.split('Check 1')[1]
    print('PASS: test_check_1_subdirs PASS')


if __name__ == '__main__':
    main()
