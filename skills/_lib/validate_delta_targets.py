#!/usr/bin/env python3
"""validate_delta_targets.py - Verify spec.md MODIFIED/RENAMED targets exist.

Catches archive aborts caused by MODIFIED or RENAMED sections targeting
capabilities that don't exist in main openspec/specs/.

Exit codes:
  0 = pass (no invalid MODIFIED/RENAMED targets)
  1 = hard fail (at least one target missing)

How it works:
  - Parses spec.md for ## MODIFIED Requirements and ## RENAMED Requirements sections
  - For MODIFIED: each requirement body should target an existing capability
    (v1 default: target = change's own capability name, unless body explicitly
    states "modifies: <other-cap>")
  - For RENAMED: the source capability must exist
"""
import re
import sys
from pathlib import Path
from typing import Optional

import yaml


def find_change_dir(change_name: str, search_root: Path) -> Path:
    changes_root = search_root / "openspec/changes"
    if not changes_root.exists():
        print(f"❌ Change '{change_name}' not found (no openspec/changes/ in {search_root})", file=sys.stderr)
        sys.exit(1)
    cand = changes_root / change_name
    if cand.is_dir():
        return cand
    print(f"❌ Change '{change_name}' not found in {changes_root}/", file=sys.stderr)
    sys.exit(1)


def find_change_capability(change_dir: Path) -> str:
    """Get the capability name from .openspec.yaml name field or dir name."""
    yaml_file = change_dir / ".openspec.yaml"
    if yaml_file.exists():
        try:
            with yaml_file.open() as f:
                data = yaml.safe_load(f) or {}
            name = data.get("name")
            if name:
                return str(name)
        except yaml.YAMLError:
            pass
    return change_dir.name


def find_main_specs_dirs(search_root: Path) -> set:
    """Find all main spec directories under openspec/specs/."""
    specs_root = search_root / "openspec/specs"
    if not specs_root.exists():
        return set()
    return {d.name for d in specs_root.iterdir() if d.is_dir() and (d / "spec.md").exists()}


def parse_delta_sections(spec_md: Path) -> dict:
    """Parse spec.md into sections. Return dict of section_name -> list of requirement body (list of lines).

    A requirement body is the line block from one `### Requirement:` header to the next
    section/requirement header. Empty placeholders like '(none)' or empty bodies are filtered out.
    """
    content = spec_md.read_text()
    sections: dict = {}
    current_section = None
    current_body: list = []
    placeholder_pattern = re.compile(r"^\(none\)\s*$", re.IGNORECASE)

    def is_placeholder(body: list) -> bool:
        non_empty = [ln for ln in body if ln.strip()]
        if not non_empty:
            return True
        return all(placeholder_pattern.match(ln.strip()) for ln in non_empty)

    for line in content.splitlines():
        m = re.match(r"^## (ADDED|MODIFIED|RENAMED|REMOVED) Requirements\s*$", line)
        if m:
            if current_section in ("MODIFIED", "RENAMED") and current_body and not is_placeholder(current_body):
                sections[current_section].append(current_body)
            current_section = m.group(1)
            sections[current_section] = []
            current_body = []
            continue
        m = re.match(r"^### Requirement:", line)
        if m and current_section in ("MODIFIED", "RENAMED"):
            if current_body and not is_placeholder(current_body):
                sections[current_section].append(current_body)
            current_body = [line]
            continue
        if current_section in ("MODIFIED", "RENAMED") and current_body:
            current_body.append(line)
    if current_section in ("MODIFIED", "RENAMED") and current_body and not is_placeholder(current_body):
        sections[current_section].append(current_body)
    return sections


def extract_target_from_body(body_lines: list, change_cap: str) -> str:
    """Extract target capability from a MODIFIED requirement body.

    v1: look for 'modifies: <cap>' or 'target: <cap>' in first 5 lines, else change_cap.
    """
    for line in body_lines[:5]:
        m = re.match(r"\s*(?:modifies|target):\s*(\S+)", line)
        if m:
            return m.group(1)
    return change_cap


def extract_rename_source(body_lines: list) -> str:
    """Extract source capability from a RENAMED requirement header (e.g., 'old-name -> new-name')."""
    if not body_lines:
        return ""
    m = re.search(r"(\S+)\s*->\s*(\S+)", body_lines[0])
    if m:
        return m.group(1)
    return ""


def validate_delta_targets(change_name: str, search_root: Optional[Path] = None) -> int:
    if search_root is None:
        search_root = Path.cwd()
    change_dir = find_change_dir(change_name, search_root)
    specs_dir = change_dir / "specs"
    if not specs_dir.exists():
        print(f"ℹ️  {change_name}: no specs/ directory (pass)", file=sys.stderr)
        return 0

    main_specs = find_main_specs_dirs(search_root)
    change_cap = find_change_capability(change_dir)

    failures = []
    for cap_spec_dir in specs_dir.iterdir():
        if not cap_spec_dir.is_dir():
            continue
        spec_md = cap_spec_dir / "spec.md"
        if not spec_md.exists():
            continue
        sections = parse_delta_sections(spec_md)

        for body_lines in sections.get("MODIFIED", []):
            target = extract_target_from_body(body_lines, change_cap)
            if target not in main_specs:
                available = sorted(main_specs) if main_specs else "(none)"
                failures.append(
                    f"  ❌ MODIFIED target '{target}' not in main openspec/specs/\n"
                    f"     Available: {available}\n"
                    f"     Fix: either create openspec/specs/{target}/spec.md, "
                    f"or move this requirement to ## ADDED Requirements"
                )
        for body_lines in sections.get("RENAMED", []):
            source = extract_rename_source(body_lines)
            if source and source not in main_specs:
                available = sorted(main_specs) if main_specs else "(none)"
                failures.append(
                    f"  ❌ RENAMED source '{source}' not in main openspec/specs/\n"
                    f"     Available: {available}\n"
                    f"     Fix: either create openspec/specs/{source}/spec.md, "
                    f"or remove this requirement"
                )

    if failures:
        print(f"\n❌ {change_name}: {len(failures)} delta target(s) invalid:", file=sys.stderr)
        for f in failures:
            print(f, file=sys.stderr)
        return 1

    print(f"✅ {change_name}: all MODIFIED/RENAMED targets valid")
    return 0


def main():
    if len(sys.argv) != 2:
        print("Usage: validate_delta_targets.py <change-name>", file=sys.stderr)
        sys.exit(2)
    sys.exit(validate_delta_targets(sys.argv[1]))


if __name__ == "__main__":
    main()
