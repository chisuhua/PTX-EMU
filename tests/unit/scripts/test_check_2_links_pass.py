#!/usr/bin/env python3
"""Unit test for Check 2: PASS when all internal links resolve."""
import subprocess
import tempfile
import os


def setup_pass_fixture(root):
    os.makedirs(os.path.join(root, 'docs', 'adr'))
    os.makedirs(os.path.join(root, 'docs', 'plans'))
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
        setup_pass_fixture(tmp)
        exit_code, stdout, stderr = run_validator(tmp)
        assert exit_code == 0, f'Expected PASS, got exit {exit_code}\nstdout: {stdout}\nstderr: {stderr}'
    print('PASS: test_check_2_links_pass')


if __name__ == '__main__':
    main()
