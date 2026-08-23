#!/usr/bin/env python3
"""Unit test for Check 4: PASS for orphan with README, FAIL for orphan without."""
import subprocess
import tempfile
import os


def setup_mixed_fixture(root):
    # Create 1 well-documented orphan (has README)
    os.makedirs(os.path.join(root, 'openspec/changes/archive/2026-01-01-good-orphan'))
    with open(os.path.join(root, 'openspec/changes/archive/2026-01-01-good-orphan/proposal.md'), 'w') as f:
        f.write('# Good orphan\n')
    with open(os.path.join(root, 'openspec/changes/archive/2026-01-01-good-orphan/README.md'), 'w') as f:
        f.write('# Good orphan\n\nImplementation: xxxxxx\n')

    # Create 1 missing-README orphan (has proposal but no README)
    os.makedirs(os.path.join(root, 'openspec/changes/archive/2026-01-01-bad-orphan'))
    with open(os.path.join(root, 'openspec/changes/archive/2026-01-01-bad-orphan/proposal.md'), 'w') as f:
        f.write('# Bad orphan\n')


def setup_docs_min(root):
    os.makedirs(os.path.join(root, 'docs/adr'))
    with open(os.path.join(root, 'docs/README.md'), 'w') as f:
        f.write('| adr/ | desc |\n')


def run_validator(root):
    r = subprocess.run(
        ['python3', 'scripts/check-docs-index.py', f'--mock-root={root}'],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    return r.returncode, r.stdout, r.stderr


def setup_docs_complete(root):
    os.makedirs(os.path.join(root, 'docs/adr'))
    with open(os.path.join(root, 'docs/README.md'), 'w') as f:
        f.write('| [`adr/`](./adr/) | desc |\n')


def main():
    with tempfile.TemporaryDirectory() as tmp:
        setup_docs_complete(tmp)
        setup_mixed_fixture(tmp)
        exit_code, stdout, stderr = run_validator(tmp)
        assert exit_code == 1, f'Expected FAIL (1 missing README), got exit {exit_code}\nstdout: {stdout}'
        assert 'FAIL' in stdout, f'Expected FAIL keyword:\n{stdout}'
        assert 'orphans lack README' in stdout, f'Expected orphans message:\n{stdout}'
    print('PASS: test_check_4_orphan')


if __name__ == '__main__':
    main()
