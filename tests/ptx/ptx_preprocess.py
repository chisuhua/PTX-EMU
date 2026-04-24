#!/usr/bin/env python3
"""
PTX Preprocessor - Normalizes problematic PTX syntax for parsing.
Handles multi-line entry params and inline param blocks.
"""

import re
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class ParamBlock:
    opening_brace: str
    content: List[str]
    closing_brace: str
    indent: str


def remove_comments(content: str) -> str:
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'//[^\n]*', '', content)
    return content


def collapse_multiline_entry_params(content: str) -> str:
    lines = content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        entry_match = re.match(r'^(\s*)(\.visible\s+)?\.entry\s+(\S+)\s*\(\s*$', line)

        if entry_match and i + 1 < len(lines):
            indent = entry_match.group(1)
            visible = entry_match.group(2) or ''
            name = entry_match.group(3)
            param_lines = []
            j = i + 1
            paren_depth = 1

            while j < len(lines) and paren_depth > 0:
                next_line = lines[j]
                paren_depth += next_line.count('(') - next_line.count(')')
                if paren_depth > 0:
                    stripped = next_line.strip()
                    if stripped:
                        param_lines.append(stripped)
                j += 1

            if param_lines:
                params = ' '.join(param_lines)
                if visible:
                    result_lines.append(f'{indent}{visible}.entry {name}({params})')
                else:
                    result_lines.append(f'{indent}.entry {name}({params})')
            else:
                result_lines.append(f'{indent}{visible}.entry {name}()')

            i = j
        else:
            result_lines.append(line)
            i += 1

    return '\n'.join(result_lines)


def extract_inline_param_blocks(content: str) -> str:
    """
    Extract and normalize inline param blocks.
    Pattern: { .param ... statements ... }
    Pattern: { // comment .reg .b32 temp_param_reg; .param ... statements ... }
    Transforms to: Hoisted param decls + statements without braces
    Only processes inner blocks (with brace_depth > 1), not the outer function body.
    """
    lines = content.split('\n')
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        brace_match = re.match(r'^(\s*)\{\s*$', line)

        inline_conv_match = re.match(r'^(\s*)\{\s*(.+);\s*\}\s*$', line)

        if inline_conv_match:
            indent = inline_conv_match.group(1)
            instruction = inline_conv_match.group(2).strip()
            if instruction:
                result_lines.append(indent + instruction + ';')
            i += 1
        elif brace_match and i + 1 < len(lines):
            indent = brace_match.group(1)
            next_line = lines[i + 1].strip()

            block_lines = []
            j = i + 1
            brace_depth = 1

            # Collect all lines in this block
            while j < len(lines) and brace_depth > 0:
                block_lines.append(lines[j])
                brace_depth += lines[j].count('{') - lines[j].count('}')
                j += 1

            # Check if this block contains inner braces (nested blocks)
            has_nested_blocks = any('{' in lines[k] for k in range(i + 1, j - 1))

            has_params = any(lines[k].strip().startswith('.param') for k in range(i + 1, j - 1))

            if has_nested_blocks:
                # Outer block with inner blocks - add opening brace, then process inner blocks
                result_lines.append(line)
                i = i + 1  # Reprocess from line after opening brace to handle inner blocks
            elif has_params:
                for block_line in block_lines[:-1]:
                    stripped = block_line.strip()
                    if stripped:
                        result_lines.append(indent + stripped)
                i = j
            elif len(block_lines) == 1 and block_lines[0].strip().endswith(';'):
                stripped = block_lines[0].strip()
                if stripped.endswith(';'):
                    instruction = stripped[:-1].strip()
                    if instruction:
                        result_lines.append(indent + instruction + ';')
                else:
                    result_lines.append(line)
                i = j
            else:
                result_lines.append(line)
                i += 1
        else:
            result_lines.append(line)
            i += 1

    return '\n'.join(result_lines)


def preprocess_ptx(content: str) -> str:
    content = remove_comments(content)
    content = collapse_multiline_entry_params(content)
    content = extract_inline_param_blocks(content)
    return content


def main():
    if len(sys.argv) < 2:
        print("Usage: ptx_preprocess.py input.ptx [output.ptx]", file=sys.stderr)
        print("       cat input.ptx | ptx_preprocess.py", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if input_file == '-':
        content = sys.stdin.read()
    else:
        with open(input_file, 'r') as f:
            content = f.read()

    result = preprocess_ptx(content)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
