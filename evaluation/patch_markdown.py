"""Update markdown cells in crossvalidation.ipynb to reflect HD95+NSD metrics
and fix stale 3-method / 4-method references (now 5 methods).

Idempotent: re-running detects already-patched strings and skips.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("crossvalidation.ipynb")

# md[0] — main intro
MD0_REPLACEMENTS = [
    (
        "このノートブックは、4つの異なるアプローチ（Method1/2/3/4）で5-fold cross-validationを実行し、性能を比較します。",
        "このノートブックは、5つの異なるアプローチ（Method1/2/3/4/5）で5-fold cross-validationを実行し、性能を比較します。",
    ),
    ("## 3つの手法", "## 5つの手法"),
    (
        "- **評価指標**: Eyelid/Iris/Pupilの Dice係数, 平均 Dice",
        """- **評価指標**: 3層評価（Metrics Reloaded, Maier-Hein et al. *Nat Methods* 2024 準拠）
  - **Overlap**: Dice (DSC) — 従来通り
  - **Boundary**: HD95 (Hausdorff 95%ile), NSD (Normalized Surface Distance; τ=eyelid:2px, iris:2px, pupil:1px)
  - 全て `MetricsReloaded` 公式リファレンス実装を使用""",
    ),
    ("5. モデル定義（UNet Method1/2/3）", "5. モデル定義（UNet Method1/2/3/4/5）"),
    ("11. 各Foldの評価（3手法すべて）", "11. 各Foldの評価（5手法すべて）+ HD95/NSD"),
    ("13. 可視化（3手法比較）", "13. 可視化（5手法比較）"),
    (
        "- **Method1**: 40%高速化（60分 → 36分/epoch）\n- **Method2**: 20-30%高速化（60分 → 42-48分/epoch）\n- **Method3**: 30-35%高速化（60分 → 39-42分/epoch）",
        """- **Method1**: 40%高速化（60分 → 36分/epoch）
- **Method2**: 20-30%高速化（60分 → 42-48分/epoch）
- **Method3**: 30-35%高速化（60分 → 39-42分/epoch）

## 📊 評価指標の拡張（2026-04-18）

Metrics Reloaded / Metrics Pitfalls (Nat Methods 2024) に準拠し、従来の Dice 単独評価を **DSC + HD95 + NSD** の 2 層構成に拡張：

| 層 | 指標 | 目的 |
|---|---|---|
| Overlap | Dice (DSC) | 体積的一致度（既存） |
| Boundary | HD95 | 境界の最大誤差（外れ値耐性） |
| Boundary | NSD | GT 楕円フィッティング不正確性に頑健な境界一致度 |

### NSD の許容距離 τ
- Eyelid: 2 px（大構造）
- Iris: 2 px（中構造）
- Pupil: 1 px（小構造・サブ画素精度）

### 出力 CSV
- `results/cv_method1_reloaded_perimage_*.csv` — Method 1（ellipse regression）
- `results/cv_method2_reloaded_perimage_*.csv` — Method 2（edge → ellipse fit）
- `results/cv_method{3,4,5}_*_perimage_*.csv` — Method 3/4/5（既存の per-image CSV に 6 列追加）

### カラム
`eyelid/iris/pupil`（DSC）+ `eyelid_hd95/iris_hd95/pupil_hd95` + `eyelid_nsd/iris_nsd/pupil_nsd`""",
    ),
]

# md[20] — evaluation section
MD20_OLD = """## 11. 評価（各Foldごと）

学習済みモデルを読み込んで、各Foldの評価を実行します。"""

MD20_NEW = """## 11. 評価（各Foldごと）

学習済みモデルを読み込んで、各Foldの評価を実行します。

### 計算される指標（MetricsReloaded 公式実装）

各画像について、eyelid / iris / pupil それぞれに対して：

- **DSC** (Dice Similarity Coefficient) — 既存
- **HD95** (Hausdorff Distance 95%ile) — 境界の外れ値耐性指標
- **NSD** (Normalized Surface Distance) — τ 以内に境界がある割合

### 出力 CSV

| Method | ファイル | モード列の値 |
|---|---|---|
| 1 | `results/cv_method1_reloaded_perimage_{ts}.csv` | `ellipse_regression` |
| 2 | `results/cv_method2_reloaded_perimage_{ts}.csv` | `edge_ellipse_fit` |
| 3 | `results/cv_method3_full_vs_exposed_perimage_{ts}.csv` | `raw` / `outerarc` / `fullmax` / `ransac_whole` |
| 4 | `results/cv_method4_full_vs_exposed_perimage_{ts}.csv` | 同上 |
| 5 | `results/cv_method5_amodal_perimage_{ts}.csv` | `raw` / `fullmax` / `ransac_whole` |
| 6 | `results/cv_method6_visible_boundary_perimage_{ts}.csv` | `raw` / `boundary` / `fullmax` |

### スキーマ

```
filename, subject_id, mode, fold,
eyelid, iris, pupil, mean,                    # DSC（Dice）
eyelid_hd95, iris_hd95, pupil_hd95,           # HD95 (px)
eyelid_nsd, iris_nsd, pupil_nsd               # NSD (τ=2/2/1 px)
```

### 引用
> Maier-Hein L, Reinke A, et al. Metrics reloaded: recommendations for image analysis validation. *Nat Methods* 2024;21:195–212."""


def main():
    with NB_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    changed = 0

    # md[0]
    cell0 = nb["cells"][0]
    assert cell0["cell_type"] == "markdown"
    src0 = "".join(cell0["source"])
    for old, new in MD0_REPLACEMENTS:
        if old in src0:
            src0 = src0.replace(old, new)
            changed += 1
        elif new in src0:
            print(f"  md[0]: already patched — {old[:40]!r}")
        else:
            print(f"  md[0]: WARNING anchor not found — {old[:60]!r}")

    # md[20]
    cell20 = nb["cells"][20]
    assert cell20["cell_type"] == "markdown"
    src20 = "".join(cell20["source"])
    if MD20_OLD in src20:
        src20 = src20.replace(MD20_OLD, MD20_NEW)
        changed += 1
    elif "MetricsReloaded 公式実装" in src20:
        print("  md[20]: already patched")
    else:
        print(f"  md[20]: WARNING anchor not found")

    # Write back as list-of-lines with trailing \n (except possibly the last)
    def split_to_lines(s):
        lines = s.split("\n")
        return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    cell0["source"] = split_to_lines(src0)
    cell20["source"] = split_to_lines(src20)

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Markdown cells updated: {changed} replacements")


if __name__ == "__main__":
    main()
