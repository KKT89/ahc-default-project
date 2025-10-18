import os
import sys
import re
import config_util as config_util
from typing import List, Set


def read_file_content(file_path: str) -> str:
    """Read and return the entire content of a file."""
    with open(file_path, 'r') as f:
        return f.read()


def inline_includes(
        file_path: str,
        stack: List[str],
        added: Set[str]
) -> List[str]:
    base_dir = os.path.dirname(file_path)
    lines = read_file_content(file_path).splitlines()
    output: List[str] = []

    for line in lines:
        stripped = line.strip()
        # Skip header guards
        if stripped.startswith(('#pragma once')):
            continue

        # Preserve only bits/stdc++.h system include
        sys_match = re.match(r'^\s*#include\s+<(.+?)>\s*$', line)
        if sys_match:
            if sys_match.group(1) == 'bits/stdc++.h':
                output.append(line)
            continue

        # Handle local includes
        inc_match = re.match(r'^\s*#include\s+"(.+?)"\s*$', line)
        if inc_match:
            inc_name = inc_match.group(1)
            inc_path = os.path.normpath(os.path.join(base_dir, inc_name))

            if inc_path in stack:
                print(f"Circular dependency detected: {' > '.join(stack + [inc_path])}")
                sys.exit(1)

            if inc_path not in added:
                # Separator comment
                output.append(f"// ── {os.path.basename(inc_path)} ──")
                stack.append(inc_path)
                inlined = inline_includes(inc_path, stack, added)
                stack.pop()
                added.add(inc_path)
                # Strip blank lines and append one blank line
                for sub in inlined:
                    if sub.strip():
                        output.append(sub)
                output.append('')
            continue

        # Default: copy line as-is
        output.append(line)

    return output


def main():
    config = config_util.load_config()
    work_dir = config_util.work_dir()

    src = os.path.join(work_dir, config['files']['cpp_file'])
    dst = os.path.join(work_dir, config['files']['combined_file'])

    combined_lines = inline_includes(src, [src], set())
    with open(dst, 'w') as f:
        f.write('\n'.join(combined_lines))

    print(f"Combined file generated: {dst}")


if __name__ == '__main__':
    main()
