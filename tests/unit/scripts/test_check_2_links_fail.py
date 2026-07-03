#!/usr/bin/env python3
"""Unit test for Check 2: FAIL when README has broken link."""
import subprocess
import tempfile
import os


def setup_fail_fixture(root):
    os.makedirs(os.path.join(root, 'docs/adr'))
    os.makedirs(os.path.join(root, 'docs/plans'))
    with open(os.path.join(root, 'docs/README.md'), 'w') as f:
        f.write('| [`adr/`](./adr/) | desc |\n')
        f.write('| [`plans/`](./plans/) | desc |\n')
        f.write('| broken | [missing](./nonexistent/) |\n')


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
        assert exit_code == 1, f'Expected FAIL (broken link), got exit {exit_code}\nstdout: {stdout}'
        assert 'FAIL' in stdout, f'Expected FAIL keyword in output:\n{stdout}'
        assert 'broken' in stdout, f'Expected broken marker in output:\n{stdout}'
    print('PASS: test_check_2_links_fail')


if __name__ == '__main__':
    main()
