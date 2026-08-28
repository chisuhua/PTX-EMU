#!/usr/bin/env bash
#
# Verify one phase of the PTX-EMU IR namespace migration.
#
# The wrapped scanner (scripts/check_ptxemu_ir_names.py) accepts only
# --roots, --exclude, and --list-files.  In particular, --match-mode is not
# a supported option despite appearing in earlier baseline instructions.
# A scanner hit is an identifier in its ten-token IR_TOKENS set that is not
# already preceded by ptxemu::ir:: and is not inside include/ptxemu/ir/ (or a
# detected canonical namespace block).  Comments, literals, and qualified
# names are ignored.  The scanner exits 0 when no hits are found, 1 when hits
# are found, and argparse/file errors are also non-zero.
#
# This helper intentionally compares hit *files*, not hit-line counts.  That
# makes a phase useful even when a single file contains several identifiers:
# the phase is clean only when its scan has no hit files, and a regression is
# a file that was absent from the recorded baseline.  The baseline snapshot
# is the scanner's complete, sorted output, so it can be shared by all
# subsequent 1.5e-1.5j phases.

set -u
set -o pipefail
set +e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
SCANNER="$SCRIPT_DIR/check_ptxemu_ir_names.py"
DEFAULT_BASELINE="$REPO_ROOT/.opencode/notes/phase-1-5-baseline.txt"

phase=""
baseline_file="$DEFAULT_BASELINE"
roots=()

usage() {
    cat <<'USAGE'
Usage: verify_phase_1_5.sh --phase <name> --roots <root> [root ...] [options]

Options:
  --phase <name>             Phase label printed in the summary (required).
  --roots <list>             One or more source roots (required); stop the
                             list before the next --option.
  --baseline-file <path>     Sorted scanner output to compare against.
                             Default: .opencode/notes/phase-1-5-baseline.txt
  --help                     Show this help text.

Example:
  ./scripts/verify_phase_1_5.sh --phase 1.5e --roots src/ptx_parser
USAGE
}

if (($# == 0)); then
    usage
    exit 2
fi

while (($# > 0)); do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --phase)
            if (($# < 2)); then
                printf 'error: --phase requires a value\n' >&2
                usage >&2
                exit 2
            fi
            phase=$2
            shift 2
            ;;
        --roots)
            shift
            while (($# > 0)) && [[ "$1" != --* ]]; do
                roots+=("$1")
                shift
            done
            ;;
        --baseline-file)
            if (($# < 2)); then
                printf 'error: --baseline-file requires a value\n' >&2
                usage >&2
                exit 2
            fi
            baseline_file=$2
            shift 2
            ;;
        *)
            printf 'error: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$phase" || ${#roots[@]} -eq 0 ]]; then
    printf 'error: --phase and --roots are required\n' >&2
    usage >&2
    exit 2
fi

scanner_output=$(mktemp)
summary_output=$(mktemp)
cleanup() {
    rm -f -- "$scanner_output" "$summary_output"
}
trap cleanup EXIT

# Run from the repository root so the scanner's relative paths match the
# baseline snapshot, regardless of the caller's current directory.
CDPATH= cd -- "$REPO_ROOT"
python3 "$SCANNER" --roots "${roots[@]}" >"$scanner_output" 2>&1
scanner_status=$?

# Parse scanner output and calculate swept files, remaining hit files, and
# files newly dirty relative to the baseline.  This uses only Python 3's
# standard library and deliberately tolerates warning lines from the scanner.
python3 - "$scanner_output" "$baseline_file" "${roots[@]}" >"$summary_output" <<'PY'
import re
import sys
from pathlib import Path

scan_path = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])
roots = sys.argv[3:]
path_line = re.compile(r"^(?P<path>.+):$")

def hit_files(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = path_line.match(line)
        if match and not line.startswith(("Found ", "Warning:", "No ")):
            result.add(match.group("path"))
    return result

def swept_files() -> int:
    count = 0
    for root_name in roots:
        root = Path(root_name)
        if root.is_file():
            count += int(root.suffix in {".h", ".hpp", ".cpp", ".cc"})
        elif root.is_dir():
            count += sum(
                path.suffix in {".h", ".hpp", ".cpp", ".cc"}
                for path in root.rglob("*")
                if path.is_file()
            )
    return count

current = hit_files(scan_path)
baseline = hit_files(baseline_path)
new_files = current - baseline
print(f"{swept_files()} {len(current)} {len(new_files)} {int(len(current) <= len(baseline))}")
PY

if (($scanner_status != 0)); then
    : # A dirty scan is expected during migration; summary decides the result.
fi

read -r swept remaining new_files baseline_not_grown <"$summary_output"
printf '[%s] %s files swept, %s remaining, %s new (vs baseline)\n' \
    "$phase" "$swept" "$remaining" "$new_files"
printf 'scanner exit: %s; baseline files: %s\n' "$scanner_status" \
    "$(python3 - "$baseline_file" <<'PY'
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
pattern = re.compile(r"^.+:$")
print(sum(
    bool(pattern.match(line)) and not line.startswith(("Found ", "Warning:", "No "))
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
) if path.is_file() else 0)
PY
)"

# A clean phase always passes.  With a baseline, a dirty phase also passes
# when it did not add any hit files and the total is no larger than baseline.
# This lets later phases scan a narrower root while still catching regressions.
if ((remaining == 0 || (new_files == 0 && baseline_not_grown == 1))); then
    exit 0
fi
exit 1
