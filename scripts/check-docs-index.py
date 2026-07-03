#!/usr/bin/env python3
# check-docs-index.py - per docs-discoverability spec (openspec/specs/)
#
# Usage:
#   python3 scripts/check-docs-index.py              # check main project
#   python3 scripts/check-docs-index.py --mock-root /tmp/test   # check temp fixture
#   python3 scripts/check-docs-index.py --verbose    # show details
#
# Exit: 0 if all checks pass, 1 otherwise.
#
# Test fixture support: --mock-root=/path makes all paths relative to /path
# instead of the current working directory. This allows tests to create
# isolated temp directories with synthetic docs/, openspec/changes/archive/,
# .opencode/skills/ for each check without polluting the real project.

import re
import os
import subprocess
import sys

VERBOSE = '--verbose' in sys.argv
MOCK_ROOT = None
for i, arg in enumerate(sys.argv):
    if arg.startswith('--mock-root='):
        MOCK_ROOT = arg.split('=', 1)[1]
        break
    if arg == '--mock-root' and i + 1 < len(sys.argv):
        MOCK_ROOT = sys.argv[i + 1]
        break

ROOT = MOCK_ROOT if MOCK_ROOT else os.getcwd()
os.chdir(ROOT) if MOCK_ROOT else None

FAIL_COUNT = 0


def log_pass(msg):
    print('  PASS ' + msg)


def log_fail(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print('  FAIL ' + msg)


def log_warn(msg):
    print('  WARN ' + msg)


def check_1_subdirs():
    print('=== Check 1: docs/ subdirs vs docs/README.md index ===')
    actual = sorted(d for d in os.listdir('docs') if os.path.isdir(os.path.join('docs', d)))
    with open('docs/README.md', encoding='utf-8') as f:
        md_content = f.read()

    INDEX_RE = re.compile(r'\[`([A-Za-z0-9_-]+)/`\]\(\./([A-Za-z0-9_-]+)/\)')
    indexed = sorted(set(m.group(1) for m in INDEX_RE.finditer(md_content)))

    if len(actual) == len(indexed):
        log_pass('actual subdirs (' + str(len(actual)) + ') match indexed (' + str(len(indexed)) + ')')
    else:
        log_fail('mismatch: ' + str(len(actual)) + ' actual vs ' + str(len(indexed)) + ' indexed')
        for a in actual:
            if a not in indexed:
                print('      NOT_INDEXED: ' + a)
        for i in indexed:
            if i not in actual:
                print('      STALE_INDEX: ' + i)

    if VERBOSE:
        print('    Actual: ' + str(actual))
        print('    Indexed: ' + str(indexed))


def check_2_links():
    print()
    print('=== Check 2: docs/README.md internal links ===')
    with open('docs/README.md', encoding='utf-8') as f:
        md_content = f.read()

    clean = re.sub(r'```[\s\S]*?```', '', md_content)
    links_pat = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
    all_links = links_pat.findall(clean)
    internal = [l for l in all_links if not l.startswith(('http://', 'https://', '#'))]
    broken = []
    for link in internal:
        target = link.split('#')[0]
        if not target:
            continue
        p = os.path.normpath(os.path.join('docs', target))
        if not os.path.exists(p):
            broken.append(link)
    if not broken:
        log_pass('all ' + str(len(internal)) + ' internal links resolve')
    else:
        log_fail(str(len(broken)) + ' of ' + str(len(internal)) + ' links broken')


def check_3_stats():
    print()
    print('=== Check 3: no hand-edited statistics ===')
    with open('docs/README.md', encoding='utf-8') as f:
        md_content = f.read()

    st_pat = re.compile(r'^\|[^|]*\b\d+(\.\d+)?\b[^|]*(测试|行|个|commit|LOC|tests)', re.M)
    he = st_pat.findall(md_content)
    if not he:
        log_pass('no hand-edited statistics in markdown tables')
    else:
        log_fail(str(len(he)) + ' hand-edited stat rows (use scripts/check-docs-index.sh to verify)')


def check_4_orphans():
    print()
    print('=== Check 4: orphan archive changes have README.md + verifiable commit ===')
    orphan_found = 0
    orphan_ok = 0
    orphan_missing = []
    orphan_invalid_hash = []
    HASH_RE = re.compile(r'\*\*Implementation(?:\s+commits?|\s+commit|\s+commits)\*\*\s*:\s*`([0-9a-f]{7,40})`', re.I)
    if os.path.isdir('openspec/changes/archive'):
        for d in sorted(os.listdir('openspec/changes/archive')):
            if d.startswith('.'):
                continue
            dp = os.path.join('openspec/changes/archive', d)
            if not os.path.isdir(dp):
                continue
            if os.path.isfile(os.path.join(dp, 'proposal.md')) and not os.path.isfile(os.path.join(dp, 'design.md')):
                orphan_found += 1
                readme_path = os.path.join(dp, 'README.md')
                if not os.path.isfile(readme_path):
                    orphan_missing.append(d)
                    continue
                with open(readme_path, encoding='utf-8') as rf:
                    readme_content = rf.read()
                hash_match = HASH_RE.search(readme_content)
                if hash_match:
                    hash_val = hash_match.group(1)
                    rc = os.system('git cat-file -t ' + hash_val + ' >/dev/null 2>&1')
                    if rc == 0:
                        orphan_ok += 1
                    else:
                        orphan_invalid_hash.append((d, hash_val))
                else:
                    multi_commit_pattern = re.compile(r'\*\*Implementation commits?\*\*', re.I)
                    if multi_commit_pattern.search(readme_content):
                        BACKTICK_HASH = re.compile(r'`([0-9a-f]{7,40})`')
                        hashes = BACKTICK_HASH.findall(readme_content)
                        verified = False
                        for hash_val in hashes:
                            rc = os.system('git cat-file -t ' + hash_val + ' >/dev/null 2>&1')
                            if rc == 0:
                                verified = True
                                break
                        if verified:
                            orphan_ok += 1
                        else:
                            orphan_invalid_hash.append((d, 'no verifiable hash in multi-commit'))
                    else:
                        orphan_invalid_hash.append((d, 'no hash found'))
    if orphan_found == 0:
        log_pass('no orphan changes detected')
    elif orphan_missing:
        log_fail(str(len(orphan_missing)) + ' of ' + str(orphan_found) + ' orphans lack README.md')
    elif orphan_invalid_hash:
        for name, hash_val in orphan_invalid_hash:
            log_fail('INVALID_COMMIT: ' + name + ' -> ' + hash_val)
    else:
        log_pass('all ' + str(orphan_found) + ' orphan changes have valid README + commit hash')


def check_5_banners():
    print()
    print('=== Check 5: stale documents have required banners ===')
    EXPECTED_BANNERED = {
        'docs/audits/HEALTH-AUDIT-2026-06-21.md': '8 个事实错误已修正',
        'docs/PROJECT-COMPLETION-SUMMARY.md': '标记为过期',
    }
    BANNER_PATTERN = re.compile(r'^\s*>\s+\*\*⚠️')
    existing = [(p, s) for p, s in EXPECTED_BANNERED.items() if os.path.isfile(p)]
    if not existing:
        log_pass('no stale documents present (Check 5 not applicable)')
        return
    missing = []
    for relpath, expected_substr in existing:
        with open(relpath, encoding='utf-8') as f:
            lines = f.readlines()
        banner_found = False
        for line in lines[:15]:
            if BANNER_PATTERN.match(line) and expected_substr in line:
                banner_found = True
                break
        if not banner_found:
            missing.append((relpath, 'expected banner containing "' + expected_substr + '"'))
    if not missing:
        log_pass('all ' + str(len(existing)) + ' stale documents have banners')
    else:
        for path, reason in missing:
            log_fail('MISSING_BANNER: ' + path + ' (' + reason + ')')


def check_6_skills_sync():
    print()
    print('=== Check 6: docs/skills mirrors .opencode/skills ===')
    if not os.path.isdir('.opencode/skills'):
        log_warn('.opencode/skills/ does not exist — skipping Check 6')
        return
    if not os.path.isdir('.opencode/skills.disable'):
        os.makedirs('.opencode/skills.disable', exist_ok=True)
    active_skills = sorted(
        d for d in os.listdir('.opencode/skills')
        if os.path.isdir(os.path.join('.opencode/skills', d)) and d != 'README.md'
    )
    disabled_skills = sorted(
        d for d in os.listdir('.opencode/skills.disable')
        if os.path.isdir(os.path.join('.opencode/skills.disable', d))
    )
    docs_readme = 'docs/skills/README.md'
    if not os.path.isfile(docs_readme):
        log_fail('MISSING: ' + docs_readme)
        return
    with open(docs_readme, encoding='utf-8') as f:
        content = f.read()
    doc_active = set()
    doc_disabled = set()
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('|'):
            continue
        # Skip table separator rows like "|---|---|"
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        cells = [c.strip() for c in line.split('|')]
        if len(cells) < 2:
            continue
        name_cell = cells[1]
        m = re.match(r'`?([a-z0-9_-]+)`?', name_cell)
        if m:
            name = m.group(1)
        else:
            continue
        is_disabled = '[disabled]' in line
        if is_disabled:
            doc_disabled.add(name)
        else:
            doc_active.add(name)
    active_set = set(active_skills)
    disabled_set = set(disabled_skills)
    doc_active_set = doc_active
    doc_disabled_set = doc_disabled
    issues = []
    for s in active_set - doc_active_set:
        issues.append('MISSING_IN_DOCS (active): ' + s)
    for s in doc_active_set - active_set:
        issues.append('STALE_IN_DOCS (active): ' + s)
    for s in disabled_set - doc_disabled_set:
        issues.append('MISSING_IN_DOCS (disabled marker): ' + s)
    for s in doc_disabled_set - disabled_set:
        issues.append('STALE_IN_DOCS (disabled marker): ' + s)
    if not issues:
        log_pass('all skills in sync (' + str(len(active_skills)) + ' active + ' + str(len(disabled_skills)) + ' disabled)')
    else:
        for issue in issues:
            log_fail(issue)


def check_7_banner_body_unchanged():
    print()
    print('=== Check 7: banner commit preserves document body ===')
    EXPECTED_BANNERED = [
        'docs/audits/HEALTH-AUDIT-2026-06-21.md',
        'docs/PROJECT-COMPLETION-SUMMARY.md',
    ]
    if not os.path.isdir('.git'):
        log_warn('no .git/ — skipping Check 7 (mock fixture)')
        return
    issues = []
    for relpath in EXPECTED_BANNERED:
        if not os.path.isfile(relpath):
            continue
        result = subprocess.run(
            ['git', 'log', '--diff-filter=M', '--format=%H', '-S', '⚠️', '--', relpath],
            capture_output=True, text=True, cwd='.'
        )
        commit_hash = result.stdout.split('\n')[0].strip() if result.stdout.strip() else ''
        if not commit_hash:
            issues.append(relpath + ': no banner-adding commit found')
            continue
        parent_result = subprocess.run(
            ['git', 'show', commit_hash + '^:' + relpath],
            capture_output=True, text=True, cwd='.'
        )
        if parent_result.returncode != 0:
            continue
        current_content = open(relpath, encoding='utf-8').read()
        pre_content = parent_result.stdout
        pre_body = pre_content.split('\n', 1)[1] if '\n' in pre_content else pre_content
        cur_lines = current_content.split('\n')
        last_banner_idx = 0
        in_banner = False
        for i, line in enumerate(cur_lines[1:], start=1):
            if line.startswith('> **⚠️') or line.startswith('>**⚠️'):
                in_banner = True
                last_banner_idx = i
            elif line.startswith('>'):
                continue
            elif in_banner and not line.startswith('>'):
                in_banner = False
                break
        cur_after_banner = '\n'.join(cur_lines[last_banner_idx + 1:]) if last_banner_idx else '\n'.join(cur_lines[1:])
        if cur_after_banner.strip() != pre_body.strip():
            issues.append(relpath + ': body changed in banner commit ' + commit_hash[:8])
    if not issues:
        log_pass('all bannered documents preserve body across banner commit')
    else:
        for issue in issues:
            log_fail('BODY_CHANGED: ' + issue)


if __name__ == '__main__':
    check_1_subdirs()
    check_2_links()
    check_3_stats()
    check_4_orphans()
    check_5_banners()
    check_6_skills_sync()
    check_7_banner_body_unchanged()

    print()
    print('=== Summary ===')
    if FAIL_COUNT > 0:
        print('  FAIL ' + str(FAIL_COUNT) + ' check(s) failed')
        sys.exit(1)
    print('  PASS all checks passed')
