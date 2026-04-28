"""Add fold-by-fold incremental CSV saving to the evaluation loop.

If the notebook is interrupted mid-run, the partial CSVs will still exist so
the user does not lose hours of work. Final timestamped CSV is still written
at the end.

Idempotent: detects the marker and skips if already applied.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("crossvalidation.ipynb")

MARKER = "# === incremental fold save ==="

# The anchor is the last two lines inside the for-fold loop (just before the
# post-loop CSV save block).
ANCHOR_OLD = """    del val_loader, val_ds
    torch.cuda.empty_cache()

# per-image CSV保存"""

ANCHOR_NEW = f"""    del val_loader, val_ds
    torch.cuda.empty_cache()

    {MARKER}
    # Save per-image rows so far as "*_partial.csv" after each fold completes.
    # The partial files are overwritten each fold with the cumulative contents;
    # if the run is interrupted, the last-written partial CSV is the state at
    # the end of the previous fold.
    for _m_id, _rows, _stem in [
        (1, method1_rows_all, 'cv_method1_reloaded_perimage'),
        (2, method2_rows_all, 'cv_method2_reloaded_perimage'),
        (3, method3_rows_all, 'cv_method3_full_vs_exposed_perimage'),
        (4, method4_rows_all, 'cv_method4_full_vs_exposed_perimage'),
        (5, method5_rows_all, 'cv_method5_amodal_perimage'),
        (6, method6_rows_all, 'cv_method6_visible_boundary_perimage'),
    ]:
        if len(_rows) > 0:
            pd.DataFrame(_rows).to_csv(result_dir / f'{{_stem}}_partial.csv', index=False)
    print(f"  📝 Partial CSVs saved through fold {{fold_idx}} (n_rows: "
          f"M1={{len(method1_rows_all)}} M2={{len(method2_rows_all)}} M3={{len(method3_rows_all)}} "
          f"M4={{len(method4_rows_all)}} M5={{len(method5_rows_all)}} M6={{len(method6_rows_all)}})")

# per-image CSV保存"""


def main():
    with NB_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    cell = nb["cells"][21]
    src = "".join(cell["source"])

    if MARKER in src:
        print("Already patched — skipping")
        return

    if ANCHOR_OLD not in src:
        # diagnostics
        print("Anchor not found. Looking for partial matches...")
        for probe in ["    del val_loader, val_ds", "    torch.cuda.empty_cache()", "# per-image CSV保存"]:
            print(f"  '{probe[:40]}...': count={src.count(probe)}")
        raise RuntimeError("anchor missing")

    if src.count(ANCHOR_OLD) != 1:
        raise RuntimeError(f"anchor not unique: {src.count(ANCHOR_OLD)}")

    src = src.replace(ANCHOR_OLD, ANCHOR_NEW)

    lines = src.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    cell["outputs"] = []
    cell["execution_count"] = None

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("Incremental fold save patch applied.")


if __name__ == "__main__":
    main()
