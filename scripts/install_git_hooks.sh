#!/usr/bin/env bash
# install_git_hooks.sh — install git hooks from scripts/git-hooks/
#
# Both documented and idempotent. Existing hooks are preserved unless
# we are overwriting the same hook name.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# We have two options:
#   A) Append to existing pre-commit hook (safer, preserves other hooks)
#   B) Separate hook `pre-commit-docs` (only triggers on docs/ changes)

MODE="${1:-chained}"

case "$MODE" in
    chained)
        HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"
        if [[ -f "$HOOK_PATH" ]]; then
            if grep -q 'pre-commit-docs' "$HOOK_PATH"; then
                echo "Already installed in $HOOK_PATH"
                exit 0
            fi
            # Append our hook invocation
            cat >> "$HOOK_PATH" <<'EOF'

# docs-index validator (added by scripts/install_git_hooks.sh)
if [[ -f "$(git rev-parse --show-toplevel)/scripts/git-hooks/pre-commit-docs" ]]; then
    "$(git rev-parse --show-toplevel)/scripts/git-hooks/pre-commit-docs" "$@"
fi
EOF
            chmod +x "$HOOK_PATH"
            echo "Appended docs-index validator to existing pre-commit hook"
        else
            echo "No pre-commit hook exists; run with 'separate' mode instead"
            exit 1
        fi
        ;;

    separate)
        # Git 2.9+ supports custom hook paths via core.hooksPath; we
        # instead chain via a separate hook in .git/hooks/pre-commit
        # that calls our validator only on docs/ changes (which is what
        # pre-commit-docs already does).
        HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit-docs"
        cp "$REPO_ROOT/scripts/git-hooks/pre-commit-docs" "$HOOK_PATH"
        chmod +x "$HOOK_PATH"
        echo "Installed $HOOK_PATH"
        echo "NOTE: Git runs hooks named 'pre-commit-X' only if it's invoked"
        echo "      explicitly. To use this, you must also add to your"
        echo "      main pre-commit hook: bash scripts/git-hooks/pre-commit-docs"
        ;;

    *)
        echo "Usage: $0 [chained|separate]"
        echo "  chained  (default) — append to existing .git/hooks/pre-commit"
        echo "  separate           — copy as separate hook"
        exit 1
        ;;
esac

echo ""
echo "Verification: staged docs test"
cd "$REPO_ROOT"
git diff --cached --name-only | grep '^docs/' >/dev/null 2>&1 && {
    echo "Note: you have staged docs files - the hook will trigger on next commit"
} || {
    echo "Note: no docs files currently staged - hook will trigger on next docs commit"
}
