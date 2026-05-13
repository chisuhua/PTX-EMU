#!/usr/bin/env python3
import re
import sys
import json
import argparse
import os
from typing import Dict, List, Tuple, Optional, Any

def parse_ptx_file(filepath: str) -> Dict[int, str]:
    instructions = {}
    with open(filepath, 'r') as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('//'):
                continue
            if stripped.startswith('.'):
                continue
            instructions[line_no] = stripped
    return instructions

def find_bb_start_line(ptx_lines: Dict[int, str], bb_num: int) -> Optional[int]:
    for line_no in sorted(ptx_lines.keys()):
        text = ptx_lines[line_no]
        if f'$L__BB0_{bb_num}:' in text:
            return line_no
    return None

def get_next_line(ptx_lines: Dict[int, str], current_line: int) -> Optional[int]:
    sorted_lines = sorted(ptx_lines.keys())
    for i, line_no in enumerate(sorted_lines):
        if line_no == current_line and i + 1 < len(sorted_lines):
            return sorted_lines[i + 1]
    return None

def is_label_line(instr: str) -> bool:
    return instr.endswith(':')

class LaneSimulator:
    def __init__(self, ptx_lines: Dict[int, str], analysis: Dict[str, Any]):
        self.ptx_lines = ptx_lines
        self.analysis = analysis
        self.predicates = analysis.get('predicates', {})
        self.branches = analysis.get('branches', [])
        self.loop_info = analysis.get('loop', {})
        self.register_values = analysis.get('register_values', {})

    def should_branch(self, instr: str, tid_x: int, loop_iteration: int, loop_iterations: int) -> Tuple[bool, Optional[int]]:
        for branch in self.branches:
            pattern = branch.get('pattern', '')
            if pattern in instr:
                condition = branch.get('condition', '')

                if 'lane_id < 16' in condition:
                    lane_id = tid_x & 31
                    if lane_id < 16:
                        target = find_bb_start_line(self.ptx_lines, branch['target_bb'])
                        return True, target
                    return False, None

                elif 'tid_x != 0' in condition:
                    if tid_x != 0:
                        target = find_bb_start_line(self.ptx_lines, branch['target_bb'])
                        return True, target
                    return False, None

                elif 'loop_iteration' in condition:
                    if loop_iteration < loop_iterations:
                        target = find_bb_start_line(self.ptx_lines, branch['target_bb'])
                        return True, target
                    return False, None

        if 'bra.uni' in instr:
            match = re.search(r'\$L__BB(\d+)_(\d+)', instr)
            if match:
                bb_num = int(match.group(2))
                target = find_bb_start_line(self.ptx_lines, bb_num)
                return True, target

        return False, None

    def trace_lane(self, start_line: int, tid_x: int) -> List[Tuple[int, int, str]]:
        lane_id = tid_x & 31
        sequence = []
        current_line = start_line
        pc = 0

        loop_iterations_val = self.loop_info.get('iterations_func', lambda lid: max(0, lid - 15))(lane_id)
        loop_iteration = 0

        while current_line:
            instr = self.ptx_lines.get(current_line)
            if not instr:
                break

            is_label = is_label_line(instr)

            if not is_label:
                pc += 1
                sequence.append((current_line, pc, instr))

            if instr == 'ret;':
                break

            should_br, target = self.should_branch(instr, tid_x, loop_iteration, loop_iterations_val)

            if should_br and target:
                current_line = target
                if 'loop_iteration' in str(self.branches):
                    loop_iteration += 1
            else:
                current_line = get_next_line(self.ptx_lines, current_line)

        return sequence

def trace_lane_from_analysis(ptx_lines: Dict[int, str], start_line: int, tid_x: int,
                               predicates: Dict[str, str], branches: List[Dict],
                               loop_info: Dict) -> List[Tuple[int, int, str]]:
    lane_id = tid_x & 31
    sequence = []
    current_line = start_line
    pc = 0

    loop_iterations_val = max(0, lane_id - 15) if loop_info.get('has_loop') else 0
    loop_iteration = 0

    while current_line:
        instr = ptx_lines.get(current_line)
        if not instr:
            break

        is_label = is_label_line(instr)

        if not is_label:
            pc += 1
            sequence.append((current_line, pc, instr))

        if instr == 'ret;':
            break

        taken = False
        target = None

        for branch in branches:
            pattern = branch.get('pattern', '')
            if pattern not in instr:
                continue

            cond = branch.get('condition', '')

            if 'lane_id < 16' in cond:
                if lane_id < 16:
                    target_bb = branch.get('target_bb')
                    if target_bb:
                        target = find_bb_start_line(ptx_lines, target_bb)
                        taken = True
                break

            elif 'tid.x != 0' in cond:
                if tid_x != 0:
                    target_bb = branch.get('target_bb')
                    if target_bb:
                        target = find_bb_start_line(ptx_lines, target_bb)
                        taken = True
                break

            elif 'loop_iteration' in cond:
                if loop_iteration < loop_iterations_val:
                    target_bb = branch.get('target_bb')
                    if target_bb:
                        target = find_bb_start_line(ptx_lines, target_bb)
                        taken = True
                        loop_iteration += 1
                break

        if not taken:
            if instr.startswith('bra.uni'):
                match = re.search(r'\$L__BB(\d+)_(\d+)', instr)
                if match:
                    bb_num = int(match.group(2))
                    target = find_bb_start_line(ptx_lines, bb_num)
                    if target:
                        current_line = target
                        continue

            if target is None:
                target = get_next_line(ptx_lines, current_line)
            if target is not None:
                current_line = target
            else:
                break
        else:
            if target is not None:
                current_line = target
            else:
                current_line = get_next_line(ptx_lines, current_line)

    return sequence

def get_path_signature(seq: List[Tuple[int, int, str]]) -> str:
    return "|".join(f"{instr[:40]}" for _, _, instr in seq)

def get_normalized_signature(seq: List[Tuple[int, int, str]]) -> str:
    result = []
    i = 0
    while i < len(seq):
        line_no, pc, instr = seq[i]

        if '@%p2 bra $L__BB0_2;' in instr:
            i += 1
            loop_body_lines = []
            loop_jumpbacks = 0
            loop_exited = False

            while i < len(seq):
                line_no2, pc2, instr2 = seq[i]
                if '@%p2 bra $L__BB0_2;' in instr2:
                    loop_jumpbacks += 1
                    i += 1
                    continue
                if 'bra $L__BB0_2' in instr2 and '@%p2 bra' not in instr2:
                    i += 1
                    continue
                if 'bra' in instr2 and '@%p2 bra' not in instr2 and 'L__BB0_2' not in instr2:
                    loop_exited = True
                    break
                if '@%p2 bra' in instr2:
                    loop_exited = True
                    break
                loop_body_lines.append(instr2[:40])
                i += 1

            result.append("LOOP")
            # Skip loop body content entirely - only count LOOP(j=N) markers
            # This normalizes paths regardless of iteration count
            if loop_exited:
                result.append("LOOP_EXIT")
            continue

        if 'bra $L__BB0_2' in instr and '@%p2 bra' not in instr:
            result.append("UNCOND_LOOP_EXIT")
            i += 1
            continue

        result.append(instr[:40])
        i += 1

    return "|".join(result)

def build_pc_to_line_map(all_sequences: List[Tuple[int, int, List[Tuple[int, int, str]]]]) -> Dict[int, Dict[int, Tuple[int, str]]]:
    pc_map = {}
    for tid, lane_id, seq in all_sequences:
        for line_no, pc, instr in seq:
            if pc not in pc_map:
                pc_map[pc] = {}
            pc_map[pc][tid] = (line_no, instr)
    return pc_map

def load_analysis(analysis_file: str) -> Dict[str, Any]:
    with open(analysis_file, 'r') as f:
        return json.load(f)

def generate_report(ptx_file: str, kernel_name: str, start_line: int,
                    all_sequences: List, path_list: List, pc_map: Dict,
                    max_pc: int, analysis: Dict) -> str:
    output = []
    output.append("# PTX Lane Execution Path Review Report")
    output.append("")
    output.append("## Metadata")
    output.append("")
    output.append(f"| Item | Value |")
    output.append(f"|------|-------|")
    output.append(f"| File | `{ptx_file}` |")
    output.append(f"| Kernel | `{kernel_name}` |")
    output.append(f"| Start Line | {start_line} |")
    output.append(f"| Analyzed Lanes | 32 (tid.x 0-31) |")
    output.append(f"| Unique Paths | {len(path_list)} |")
    output.append("")

    if 'predicates' in analysis and analysis['predicates']:
        output.append("## Register & Predicate Analysis (LLM)")
        output.append("")
        for pred, desc in analysis['predicates'].items():
            output.append(f"- **{pred}**: {desc}")
        output.append("")

    output.append("## Path Explanation")
    output.append("")
    for i, (sig, tids) in enumerate(path_list):
        first_tid = tids[0]
        seq = all_sequences[first_tid][2]
        lane_id = first_tid & 31

        if len(tids) == 1:
            output.append(f"### Path {i+1}: tid.x = [{first_tid}] (lane_id = {lane_id})")
        else:
            output.append(f"### Path {i+1}: tid.x = [{min(tids)}-{max(tids)}]")

        output.append("")
        output.append(f"**Total Instructions**: {len(seq)}")
        output.append("")

        branch_instrs = [(pc, line_no, instr) for line_no, pc, instr in seq if '@%p' in instr]

        if branch_instrs:
            output.append(f"**Branch points in this path**:")
            output.append("")
            for pc, line_no, instr in branch_instrs:
                cond = "unknown"
                if '@%p1' in instr:
                    cond = "lane_id < 16"
                elif '@%p2' in instr:
                    cond = "loop_iteration < loop_iterations"
                elif '@%p3' in instr:
                    cond = "tid.x != 0"
                output.append(f"- PC={pc} (Line {line_no}): `{instr}` → condition: **{cond}**")
            output.append("")

        if lane_id >= 16:
            iterations = lane_id - 15
            output.append(f"**Loop iterations**: {iterations} (lane_id - 15 = {lane_id} - 15)")
            output.append("")

    output.append("## Path Summary")
    output.append("")
    output.append(f"| Path | tid.x Range | Lanes | Total PC |")
    output.append(f"|------|-------------|-------|----------|")
    for i, (sig, tids) in enumerate(path_list):
        seq = all_sequences[tids[0]][2]
        output.append(f"| Path {i+1} | {min(tids)}-{max(tids)} | {len(tids)} | {len(seq)} |")
    output.append("")

    output.append("## Execution Matrix (Lane × PC)")
    output.append("")
    output.append(f"*Only showing PC ranges that differ between paths.*")
    output.append("")

    header = "| PC | " + " | ".join([f"Path{i+1}" for i in range(len(path_list))]) + " |"
    output.append(header)
    output.append("|" + "|".join(["---"] * (len(path_list) + 1)) + "|")

    for pc in range(1, max_pc + 1):
        if pc not in pc_map:
            continue

        cells = []
        for i, (sig, tids) in enumerate(path_list):
            lane = tids[0]
            if pc in pc_map and lane in pc_map[pc]:
                line_no, instr = pc_map[pc][lane]
                short_instr = instr[:35] + "..." if len(instr) > 35 else instr
                cells.append(f"L{line_no}: {short_instr}")
            else:
                cells.append("-")

        if len(set(cells)) > 1:
            output.append(f"| {pc} | " + " | ".join(cells) + " |")
    output.append("")

    for i, (sig, tids) in enumerate(path_list):
        seq = all_sequences[tids[0]][2]

        output.append(f"## Path {i+1} Detail")
        output.append("")
        output.append(f"**Lanes**: tid.x = [{', '.join(str(t) for t in tids)}]")
        output.append("")
        output.append(f"**Total Instructions**: {len(seq)}")
        output.append("")
        output.append(f"| PC | Line | Instruction |")
        output.append(f"|----|------|-------------|")
        for line_no, pc, instr in seq:
            output.append(f"| {pc} | {line_no} | `{instr}` |")
        output.append("")

    output.append("## Divergence Analysis")
    output.append("")

    for i, (sig, tids) in enumerate(path_list):
        seq = all_sequences[tids[0]][2]
        branch_instrs = [(pc, line_no, instr) for line_no, pc, instr in seq if '@%p' in instr]

        output.append(f"### Path {i+1} Branches")
        output.append("")
        if branch_instrs:
            output.append(f"| PC | Line | Branch Instruction | Condition |")
            output.append(f"|----|------|--------------------|-----------|")
            for pc, line_no, instr in branch_instrs:
                cond = "unknown"
                if '@%p1' in instr:
                    cond = "lane_id < 16"
                elif '@%p2' in instr:
                    cond = "loop_iteration < loop_iterations"
                elif '@%p3' in instr:
                    cond = "tid.x != 0"
                output.append(f"| {pc} | {line_no} | `{instr}` | {cond} |")
        else:
            output.append("*No branches in this path.*")
        output.append("")

    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='PTX Lane Tracer')
    parser.add_argument('ptx_file', help='PTX file path')
    parser.add_argument('kernel_name', help='Kernel name')
    parser.add_argument('start_line', type=int, nargs='?', default=25, help='Start line (default: 25)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-a', '--analysis', help='LLM analysis JSON file (optional)')
    parser.add_argument('--generate-analysis', action='store_true', help='Output analysis template')

    args = parser.parse_args()

    if args.generate_analysis:
        template = {
            "predicates": {
                "%p1": "lane_id < 16 (from setp.lt.u32 %r2, 16 at PC where %r2 = lane_id)",
                "%p2": "loop_iteration < loop_iterations (from setp.ne.s32 %r88, -15 in loop)",
                "%p3": "tid.x != 0 (from setp.ne.s32 %r1, 0)"
            },
            "branches": [
                {"pattern": "@%p1 bra", "condition": "lane_id < 16", "target_bb": 7, "fallback_bb": None},
                {"pattern": "@%p2 bra", "condition": "loop_iteration < loop_iterations", "target_bb": 2, "fallback_bb": None},
                {"pattern": "@%p3 bra", "condition": "tid.x != 0", "target_bb": 5, "fallback_bb": None}
            ],
            "loop": {
                "has_loop": True,
                "iterations_func": "max(0, lane_id - 15)",
                "header_bb": 2,
                "exit_bb": 3
            },
            "register_values": {
                "%r1": "%tid.x (thread ID)",
                "%r2": "lane_id = %r1 & 31"
            }
        }
        print(json.dumps(template, indent=2))
        return

    ptx_lines = parse_ptx_file(args.ptx_file)

    if args.analysis:
        analysis = load_analysis(args.analysis)
    else:
        analysis = {}

    all_sequences = []
    for tid in range(32):
        if args.analysis:
            seq = trace_lane_from_analysis(
                ptx_lines, args.start_line, tid,
                analysis.get('predicates', {}),
                analysis.get('branches', []),
                analysis.get('loop', {})
            )
        else:
            seq = trace_lane_from_analysis(
                ptx_lines, args.start_line, tid, {}, [
                    {"pattern": "@%p1 bra", "condition": "lane_id < 16", "target_bb": 7},
                    {"pattern": "@%p2 bra", "condition": "loop_iteration < loop_iterations", "target_bb": 2},
                    {"pattern": "@%p3 bra", "condition": "tid.x != 0", "target_bb": 5}
                ],
                {"has_loop": True}
            )
        all_sequences.append((tid, tid & 31, seq))

    paths = {}
    normalized_paths = {}
    for tid, lane_id, seq in all_sequences:
        sig = get_path_signature(seq)
        norm_sig = get_normalized_signature(seq)
        if norm_sig not in normalized_paths:
            normalized_paths[norm_sig] = {'tids': [], 'raw_sig': sig}
        normalized_paths[norm_sig]['tids'].append(tid)

    path_list = [(data['raw_sig'], data['tids']) for data in normalized_paths.values()]
    pc_map = build_pc_to_line_map(all_sequences)
    max_pc = max(pc_map.keys())

    report = generate_report(args.ptx_file, args.kernel_name, args.start_line,
                            all_sequences, path_list, pc_map, max_pc, analysis)

    if args.output:
        output_path = args.output
    else:
        ptx_dir = os.path.dirname(os.path.abspath(args.ptx_file))
        ptx_basename = os.path.splitext(os.path.basename(args.ptx_file))[0]
        output_path = os.path.join(ptx_dir, f"{ptx_basename}_lane_report.md")

    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {output_path}")

if __name__ == '__main__':
    main()