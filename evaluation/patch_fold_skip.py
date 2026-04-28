"""Add auto-skip logic to Cell 22 (fold evaluation loop) so that Run All
does not re-execute the 2-3 hour evaluation if per-image CSVs already exist.

Logic:
    - Check for the latest timestamped (non-partial) per-image CSV for each
      enabled method.
    - If all are present and FORCE_REEVALUATE is not set, load them from
      disk and populate evaluation_results[method_id] from fold-wise means,
      then skip the heavy loop.
    - If any CSV is missing, or FORCE_REEVALUATE=True, run the full loop.

Idempotent: detects the marker `# === fold-loop skip guard ===`.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path("crossvalidation.ipynb")
MARKER = "# === fold-loop skip guard ==="

# Anchor: where the original for-loop begins
ANCHOR = "evaluation_results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}\n\nfor fold_idx in range(NUM_FOLDS):"


NEW_PREFIX = f"""evaluation_results = {{1: [], 2: [], 3: [], 4: [], 5: [], 6: []}}

{MARKER}
# Skip the heavy fold loop if all per-image CSVs already exist.
# To force a fresh re-evaluation, set FORCE_REEVALUATE = True in a
# preceding cell before running this one.
FORCE_REEVALUATE = globals().get('FORCE_REEVALUATE', False)
_EXPECTED_CSVS = {{
    1: 'cv_method1_reloaded_perimage_*.csv',
    2: 'cv_method2_reloaded_perimage_*.csv',
    3: 'cv_method3_full_vs_exposed_perimage_*.csv',
    4: 'cv_method4_full_vs_exposed_perimage_*.csv',
    5: 'cv_method5_amodal_perimage_*.csv',
    6: 'cv_method6_visible_boundary_perimage_*.csv',
}}
_PRIMARY_MODE = {{
    1: 'ellipse_regression', 2: 'edge_ellipse_fit',
    3: 'fullmax', 4: 'fullmax', 5: 'fullmax', 6: 'boundary',
}}

def _latest_final(pattern):
    xs = sorted(p for p in result_dir.glob(pattern) if '_partial' not in p.name)
    return xs[-1] if xs else None

_methods_to_check = [m for m in TRAIN_METHODS if m in _EXPECTED_CSVS]
_existing = {{m: _latest_final(_EXPECTED_CSVS[m]) for m in _methods_to_check}}
_all_present = all(_existing.get(m) is not None for m in _methods_to_check)

if _all_present and not FORCE_REEVALUATE:
    print("All per-image evaluation CSVs already exist — skipping fold loop.")
    print("(Set FORCE_REEVALUATE = True in a preceding cell to force re-run.)")
    for _m, _path in _existing.items():
        _df = pd.read_csv(_path)
        _pmode = _PRIMARY_MODE[_m]
        for _fold in range(NUM_FOLDS):
            _fdf = _df[(_df['fold'] == _fold) & (_df['mode'] == _pmode)]
            if len(_fdf) == 0:
                continue
            evaluation_results[_m].append({{
                'lid':   float(_fdf['eyelid'].mean()),
                'iris':  float(_fdf['iris'].mean()),
                'pupil': float(_fdf['pupil'].mean()),
                'mean':  float(_fdf[['eyelid', 'iris', 'pupil']].mean(axis=1).mean()),
                'fold':  _fold,
                'method': _m,
            }})
        print(f"  M{{_m}}: loaded {{_path.name}} ({{len(_df)}} rows)")
else:
    if FORCE_REEVALUATE:
        print("FORCE_REEVALUATE=True — running full fold evaluation loop.")
    else:
        _missing = [m for m in _methods_to_check if _existing.get(m) is None]
        print(f"Missing per-image CSVs for methods {{_missing}} — running fold loop.")

    for fold_idx in range(NUM_FOLDS):"""


def main():
    with NB.open(encoding="utf-8") as f:
        nb = json.load(f)

    # Find cell 22 (fold loop) — originally cell 21 before splitting
    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if "for fold_idx in range(NUM_FOLDS):" in src and "評価中" in src:
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError("Fold-loop cell not found.")

    cell = nb["cells"][target_idx]
    src = "".join(cell["source"])
    if MARKER in src:
        print(f"Cell {target_idx}: skip guard already present — no-op.")
        return

    if ANCHOR not in src:
        raise RuntimeError(f"Anchor not found in cell {target_idx}.")

    new_src = src.replace(ANCHOR, NEW_PREFIX, 1)

    # Indent the rest of the for-loop body by 4 spaces since it's now inside `else:`
    # The original ANCHOR ended with `for fold_idx in range(NUM_FOLDS):` (not indented).
    # Everything after that marker in the source should be indented by 4 more spaces,
    # up to the next top-level block (the CSV save block after the loop).

    # Find where we inserted and where the CSV save block starts
    insert_end = new_src.index(NEW_PREFIX) + len(NEW_PREFIX)
    # The fold-loop body spans from insert_end until the post-loop code
    # (which begins with "# per-image CSV保存" at column 0)
    post_marker = "\n# per-image CSV保存"
    if post_marker not in new_src:
        raise RuntimeError("Post-loop save-block marker not found.")
    post_idx = new_src.index(post_marker, insert_end)

    body = new_src[insert_end:post_idx]
    # Indent every non-empty line by 4 additional spaces
    body_indented_lines = []
    for line in body.split("\n"):
        if line.strip():
            body_indented_lines.append("    " + line)
        else:
            body_indented_lines.append(line)
    body_indented = "\n".join(body_indented_lines)

    new_src = new_src[:insert_end] + body_indented + new_src[post_idx:]

    # Now write back as source list-of-lines
    lines = new_src.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    cell["outputs"] = []
    cell["execution_count"] = None

    with NB.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Cell {target_idx} patched with skip guard.")
    print(f"  size: {len(src)} -> {len(new_src)} chars")


if __name__ == "__main__":
    main()
