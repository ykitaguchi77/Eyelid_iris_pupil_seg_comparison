"""Split crossvalidation.ipynb Cell 21 into two separate cells:

    Cell 21a  —  Evaluation utilities + function definitions (fast, no GPU work)
    Cell 21b  —  Fold evaluation loop (heavy, 2-3 hour re-run)

Rationale: so the user can execute Cell 21a + the downstream cells (τ sweep,
summary) without re-running the full fold evaluation loop.

Split point: "# ===== 各Foldを評価 =====" marker inside the current Cell 21.

Idempotent: if Cell 21 no longer contains the marker (already split) this
is a no-op.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path("crossvalidation.ipynb")
SPLIT_MARKER = "# ===== 各Foldを評価 ====="


def main():
    with NB.open(encoding="utf-8") as f:
        nb = json.load(f)

    cell21 = nb["cells"][21]
    if cell21["cell_type"] != "code":
        raise RuntimeError("Cell 21 is not code")

    src = "".join(cell21["source"])
    if SPLIT_MARKER not in src:
        print("Split marker not found — maybe already split. Exiting.")
        return

    idx = src.index(SPLIT_MARKER)
    part_a = src[:idx].rstrip() + "\n"
    part_b = src[idx:]

    def lines_with_nl(s: str) -> list[str]:
        lines = s.split("\n")
        out = [l + "\n" for l in lines[:-1]]
        if lines[-1]:
            out.append(lines[-1])
        return out

    cell_a = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines_with_nl(part_a),
    }
    cell_b = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines_with_nl(part_b),
    }

    nb["cells"][21] = cell_a
    nb["cells"].insert(22, cell_b)

    with NB.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    # Report
    a_chars = sum(len(s) for s in cell_a["source"])
    b_chars = sum(len(s) for s in cell_b["source"])
    print(f"Split Cell 21 into 21a ({a_chars} chars, function defs) "
          f"and 21b ({b_chars} chars, fold loop).")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
