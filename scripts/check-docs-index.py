#!/usr/bin/env python3
# check-docs-index.py
import re, os, sys

VERBOSE = '--verbose' in sys.argv
FAIL_COUNT = 0

def log_pass(msg): print('  PASS ' + msg)
def log_fail(msg):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print('  FAIL ' + msg)
def log_warn(msg): print('  WARN ' + msg)

# Check 1
print('=== Check 1: docs/ subdirs vs docs/README.md index ===')
actual = sorted(d for d in os.listdir('docs') if os.path.isdir(os.path.join('docs', d)))
with open('docs/README.md', encoding='utf-8') as f:
    md_content = f.read()

# Pattern: `[\`dir/\`](./dir/)` where dir is word chars + dash
P_INDEX = r'\[[`\]]([a-z0-9_-]+)/[`\[]\]\(\./([a-z0-9_-]+)/\)'
# Actually simpler: \[[`].*?\)\]
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

# Check 2
print()
print('=== Check 2: docs/README.md internal links ===')
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

# Check 3
print()
print('=== Check 3: no hand-edited statistics ===')
st_pat = re.compile(r'^\|[^|]*\b\d+(\.\d+)?\b[^|]*(测试|行|个|commit|LOC|tests)', re.M)
he = st_pat.findall(md_content)
if not he:
    log_pass('no hand-edited statistics in markdown tables')
else:
    log_warn(str(len(he)) + ' possible hand-edited stat rows')

# Check 4
print()
print('=== Check 4: orphan archive changes have README.md ===')
orphan_found = 0
orphan_ok = 0
orphan_missing = []
if os.path.isdir('openspec/changes/archive'):
    for d in sorted(os.listdir('openspec/changes/archive')):
        if d.startswith('.'):
            continue
        dp = os.path.join('openspec/changes/archive', d)
        if not os.path.isdir(dp):
            continue
        if os.path.isfile(os.path.join(dp, 'proposal.md')) and not os.path.isfile(os.path.join(dp, 'design.md')):
            orphan_found += 1
            if os.path.isfile(os.path.join(dp, 'README.md')):
                orphan_ok += 1
            else:
                orphan_missing.append(d)
if orphan_found == 0:
    log_pass('no orphan changes detected')
elif orphan_ok == orphan_found:
    log_pass('all ' + str(orphan_found) + ' orphan changes have README.md')
else:
    log_fail(str(len(orphan_missing)) + ' of ' + str(orphan_found) + ' orphans lack README.md')

# Summary
print()
print('=== Summary ===')
if FAIL_COUNT > 0:
    print('  FAIL ' + str(FAIL_COUNT) + ' check(s) failed')
    sys.exit(1)
print('  PASS all checks passed')
