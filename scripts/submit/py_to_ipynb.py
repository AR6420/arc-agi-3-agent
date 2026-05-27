"""Convert a `# %% Cell N — ...` annotated .py file into Kaggle notebook JSON.

Output: stdout, single JSON string (the .ipynb format) suitable for the
`text` parameter of mcp__kaggle__save_notebook.
"""

import json
import re
import sys
from pathlib import Path


def convert(src_path: Path) -> dict:
    src = src_path.read_text(encoding="utf-8")
    # Split on lines that start with `# %%`. The first chunk before the first
    # marker is the module docstring; emit as a markdown cell.
    parts = re.split(r"^# %%.*$", src, flags=re.MULTILINE)
    headers = re.findall(r"^# %%.*$", src, flags=re.MULTILINE)

    cells = []
    # Module docstring → markdown cell (if present)
    if parts and parts[0].strip():
        md_text = parts[0].strip()
        # Strip triple-quoted docstring markers
        if md_text.startswith('"""'):
            md_text = md_text[3:]
        if md_text.endswith('"""'):
            md_text = md_text[:-3]
        cells.append({
            "cell_type": "markdown",
            "source": md_text.strip(),
            "metadata": {},
        })

    # Each subsequent part is a code cell; the corresponding header becomes a
    # comment at the top of the cell so the user can still see the structure.
    for header, body in zip(headers, parts[1:]):
        body = body.strip("\n")
        cells.append({
            "cell_type": "code",
            "source": f"{header}\n{body}",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        })

    return {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells,
    }


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    nb = convert(src)
    s = json.dumps(nb, indent=1)
    if out:
        out.write_text(s, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(s)


if __name__ == "__main__":
    main()
