#!/usr/bin/env python3
"""
Token-aware scanner for bare ptxemu::ir type names.

Scans C++ source for unqualified IR type references, ignoring:
- Comments (// and /* */)
- String/char literals (including raw strings)
- ptxemu::ir::-qualified tokens
- Canonical namespace blocks in include/ptxemu/ir/

Usage:
  check_ptxemu_ir_names.py --roots src include tests \\
    --exclude include/ptx_ir/ptx_types.h \\
    --exclude include/ptx_ir/operand_context.h \\
    --exclude include/ptx_ir/statement_context.h \\
    [--list-files]

Exit codes:
  0: no bare tokens found
  1: bare tokens found or error
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Set, List, Tuple

# Minimum IR type token set (per design.md D4 scanner requirements)
IR_TOKENS = {
    'StatementType', 'OperandType', 'InstructionState', 'Qualifier',
    'OperandContext', 'InstrVariant', 'Tcgen05Instr', 'Tcgen05OpKind', 'Tcgen05Dtype',
    'StatementContext',  # Added from spec scenarios
}


def is_canonical_definition_path(path: str) -> bool:
    """Check if path is under include/ptxemu/ir/ canonical definitions."""
    return 'include/ptxemu/ir/' in path.replace('\\', '/')


def tokenize_cpp(content: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize C++ content, stripping comments and literals.
    Returns list of (token, line, col) for identifier tokens only.
    """
    tokens = []
    i = 0
    line = 1
    col = 1
    
    while i < len(content):
        # Track position
        if content[i] == '\n':
            line += 1
            col = 1
            i += 1
            continue
        
        # Skip whitespace
        if content[i].isspace():
            col += 1
            i += 1
            continue
        
        # Skip // comments
        if i + 1 < len(content) and content[i:i+2] == '//':
            while i < len(content) and content[i] != '\n':
                i += 1
            continue
        
        # Skip /* */ comments
        if i + 1 < len(content) and content[i:i+2] == '/*':
            i += 2
            col += 2
            while i + 1 < len(content):
                if content[i:i+2] == '*/':
                    i += 2
                    col += 2
                    break
                if content[i] == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            continue
        
        # Skip raw string literals R"delim(...)delim"
        if i + 2 < len(content) and content[i:i+2] == 'R"':
            # Find delimiter
            i += 2
            col += 2
            delim_end = content.find('(', i)
            if delim_end != -1:
                delim = content[i:delim_end]
                closing = ')' + delim + '"'
                close_pos = content.find(closing, delim_end + 1)
                if close_pos != -1:
                    # Count newlines in skipped region
                    for ch in content[i:close_pos + len(closing)]:
                        if ch == '\n':
                            line += 1
                            col = 1
                        else:
                            col += 1
                    i = close_pos + len(closing)
                    continue
        
        # Skip string literals "..." and char literals '...'
        if content[i] in ('"', "'"):
            quote = content[i]
            i += 1
            col += 1
            while i < len(content):
                if content[i] == '\\' and i + 1 < len(content):
                    # Skip escaped character
                    i += 2
                    col += 2
                elif content[i] == quote:
                    i += 1
                    col += 1
                    break
                else:
                    if content[i] == '\n':
                        line += 1
                        col = 1
                    else:
                        col += 1
                    i += 1
            continue
        
        # Collect identifier tokens (skip following :: for qualified-name detection)
        if content[i].isalpha() or content[i] == '_':
            start_col = col
            token_start = i
            while i < len(content) and (content[i].isalnum() or content[i] == '_'):
                i += 1
                col += 1
            token = content[token_start:i]
            tokens.append((token, line, start_col))

            # Skip a following :: so qualified tokens stay adjacent
            if i + 1 < len(content) and content[i] == ':' and content[i+1] == ':':
                i += 2
                col += 2
            continue
        
        # Skip other characters
        col += 1
        i += 1
    
    return tokens


def is_in_canonical_namespace_block(content: str, line_num: int) -> bool:
    """
    Check if line is inside a 'namespace ptxemu { namespace ir {' block.
    Simple heuristic: count opening/closing braces in preceding namespace declarations.
    """
    lines = content.split('\n')
    if line_num > len(lines):
        return False
    
    # Look for namespace ptxemu { namespace ir { pattern before this line
    in_ptxemu_ir = False
    brace_depth = 0
    
    for i in range(line_num):
        line = lines[i]
        # Check for namespace declarations
        if 'namespace ptxemu' in line or 'namespace ir' in line:
            # Track opening braces
            brace_depth += line.count('{')
            if 'namespace ptxemu' in line and 'namespace ir' in line:
                in_ptxemu_ir = True
            elif 'namespace ptxemu' in line or ('namespace ir' in line and in_ptxemu_ir):
                in_ptxemu_ir = True
        
        # Track all braces
        brace_depth += line.count('{') - line.count('}')
        
        # If we closed all namespace braces, we're out
        if in_ptxemu_ir and brace_depth <= 0:
            in_ptxemu_ir = False
            brace_depth = 0
    
    return in_ptxemu_ir and brace_depth > 0


def scan_file(filepath: str, ir_tokens: Set[str]) -> List[Tuple[str, int, int]]:
    """
    Scan a single file for bare IR tokens.
    Returns list of (token, line, col) matches.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return []
    
    # Skip if in canonical definition path
    if is_canonical_definition_path(filepath):
        return []
    
    tokens = tokenize_cpp(content)
    matches = []
    
    for i, (token, line, col) in enumerate(tokens):
        if token not in ir_tokens:
            continue

        # Walk back through identifier tokens (skipping :: which was consumed in lex)
        # and check if the immediately preceding qualifier is ptxemu::ir::
        qualified = False
        j = i - 1
        prev_ids = []
        while j >= 0 and len(prev_ids) < 3:
            if tokens[j][0] in ('ptxemu', 'ir'):
                prev_ids.append(tokens[j][0])
                j -= 1
            else:
                break
        if len(prev_ids) == 2 and prev_ids == ['ir', 'ptxemu']:
            qualified = True

        if qualified:
            continue

        # Check if inside canonical namespace block
        if is_in_canonical_namespace_block(content, line - 1):
            continue

        matches.append((token, line, col))

    return matches


def collect_files(roots: List[str], exclude_paths: List[str]) -> List[str]:
    """Collect all .h, .hpp, .cpp, .cc files from roots, excluding specific paths."""
    files = []
    exclude_set = set(os.path.normpath(p) for p in exclude_paths)
    
    for root in roots:
        if not os.path.exists(root):
            print(f"Warning: Root path {root} does not exist", file=sys.stderr)
            continue
        
        # Direct file root
        candidates = []
        if os.path.isfile(root):
            candidates = [root]
        else:
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if filename.endswith(('.h', '.hpp', '.cpp', '.cc')):
                        candidates.append(os.path.join(dirpath, filename))
        
        for filepath in candidates:
            norm_path = os.path.normpath(filepath)
            
            # Check exclusions
            excluded = False
            for excl in exclude_set:
                if norm_path == excl or norm_path.endswith(excl):
                    excluded = True
                    break
            
            if not excluded:
                files.append(filepath)
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description='Check for bare ptxemu::ir type names')
    parser.add_argument('--roots', nargs='+', required=True,
                        help='Root directories to scan (e.g., src include tests)')
    parser.add_argument('--exclude', action='append', default=[],
                        help='Paths to exclude (can be repeated)')
    parser.add_argument('--list-files', action='store_true',
                        help='List all scanned files and exit')
    
    args = parser.parse_args()
    
    # Collect files
    files = collect_files(args.roots, args.exclude)
    
    if args.list_files:
        for f in files:
            print(f)
        print(f"\nTotal files: {len(files)}", file=sys.stderr)
        # Count by root
        for root in args.roots:
            count = sum(1 for f in files if f.startswith(root + os.sep) or f.startswith(root + '/'))
            print(f"  {root}: {count}", file=sys.stderr)
        return 0
    
    # Scan files
    all_matches = []
    for filepath in files:
        matches = scan_file(filepath, IR_TOKENS)
        if matches:
            all_matches.append((filepath, matches))
    
    # Report
    if all_matches:
        print("Found bare IR type tokens:", file=sys.stderr)
        for filepath, matches in all_matches:
            print(f"\n{filepath}:", file=sys.stderr)
            for token, line, col in matches:
                print(f"  {line}:{col}: {token}", file=sys.stderr)
        return 1
    else:
        print("No bare IR tokens found.", file=sys.stderr)
        return 0


if __name__ == '__main__':
    sys.exit(main())
