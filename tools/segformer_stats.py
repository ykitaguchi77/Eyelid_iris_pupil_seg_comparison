from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _latest(results_dir: Path, pattern: str) -> Optional[Path]:
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _subject_id_from_filename(filename: str) -> str:
    return str(filename).split("-", 1)[0]


def _paired_perm_p(diff: np.ndarray, n_perm: int = 20000, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    obs = float(diff.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, diff.size), replace=True)
    perm = (signs * diff[None, :]).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs)).mean())
    return obs, p


def _paired_boot_ci(diff: np.ndarray, n_boot: int = 20000, seed: int = 1) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(lo), float(hi)


@dataclass
class FoldAvg:
    csv_path: Path
    mean_dice_mean: float
    mean_dice_std: float


@dataclass
class SubjectSummary:
    csv_path: Path
    subject_mean_dice: float
    n_subjects: int


@dataclass
class Comparison:
    a: str
    b: str
    n_subjects: int
    mean_diff: float
    ci95_low: float
    ci95_high: float
    win_rate: float
    p_perm: float


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--n_boot", type=int, default=20000)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    modes = ["raw", "outerarc", "fullmax", "ransac_whole", "ransac_arc"]

    fold_avg: Dict[str, FoldAvg] = {}
    for m in modes:
        p = _latest(results_dir, f"segformer_eval_{m}_*.csv")
        if p is None:
            continue
        df = pd.read_csv(p)
        fold_avg[m] = FoldAvg(
            csv_path=p,
            mean_dice_mean=float(df["mean"].mean()),
            mean_dice_std=float(df["mean"].std(ddof=1)),
        )

    subj: Dict[str, pd.DataFrame] = {}
    subj_summary: Dict[str, SubjectSummary] = {}
    for m in modes:
        p = _latest(results_dir, f"segformer_eval_perimage_{m}_*.csv")
        if p is None:
            continue
        df = pd.read_csv(p)
        if "subject_id" not in df.columns:
            df["subject_id"] = df["filename"].astype(str).map(_subject_id_from_filename)
        subj_mean = df.groupby("subject_id")[["mean", "eyelid", "iris", "pupil"]].mean()
        subj[m] = subj_mean
        subj_summary[m] = SubjectSummary(
            csv_path=p,
            subject_mean_dice=float(subj_mean["mean"].mean()),
            n_subjects=int(subj_mean.shape[0]),
        )

    # Ranking (subject-level)
    rank_rows = [
        (m, subj_summary[m].subject_mean_dice, subj_summary[m].n_subjects)
        for m in subj_summary.keys()
    ]
    rank_rows.sort(key=lambda x: x[1], reverse=True)

    # Comparisons (subject-level)
    comps: List[Comparison] = []
    if "fullmax" in subj and "outerarc" in subj:
        common = subj["fullmax"].index.intersection(subj["outerarc"].index)
        diff = (
            subj["fullmax"].loc[common, "mean"].to_numpy()
            - subj["outerarc"].loc[common, "mean"].to_numpy()
        ).astype(float)
        obs, p = _paired_perm_p(diff, n_perm=args.n_perm, seed=10)
        lo, hi = _paired_boot_ci(diff, n_boot=args.n_boot, seed=11)
        comps.append(
            Comparison(
                a="FullMax",
                b="OuterArc",
                n_subjects=int(len(common)),
                mean_diff=float(obs),
                ci95_low=float(lo),
                ci95_high=float(hi),
                win_rate=float((diff > 0).mean()),
                p_perm=float(p),
            )
        )

    if "ransac_whole" in subj and "ransac_arc" in subj:
        common = subj["ransac_whole"].index.intersection(subj["ransac_arc"].index)
        diff = (
            subj["ransac_whole"].loc[common, "mean"].to_numpy()
            - subj["ransac_arc"].loc[common, "mean"].to_numpy()
        ).astype(float)
        obs, p = _paired_perm_p(diff, n_perm=args.n_perm, seed=12)
        lo, hi = _paired_boot_ci(diff, n_boot=args.n_boot, seed=13)
        comps.append(
            Comparison(
                a="RANSAC(whole)",
                b="RANSAC(arc)",
                n_subjects=int(len(common)),
                mean_diff=float(obs),
                ci95_low=float(lo),
                ci95_high=float(hi),
                win_rate=float((diff > 0).mean()),
                p_perm=float(p),
            )
        )

    # Print report (markdown-friendly)
    print("== SegFormer summary (from results/*.csv) ==")
    print("FOLD_AVG_FILES:")
    for m in modes:
        if m in fold_avg:
            print(f"  - {m}: {fold_avg[m].csv_path.as_posix()}")
    print("SUBJECT_FILES:")
    for m in modes:
        if m in subj_summary:
            print(f"  - {m}: {subj_summary[m].csv_path.as_posix()}")

    print("\nFOLD_AVG_MEAN_DICE:")
    for m in modes:
        if m in fold_avg:
            fa = fold_avg[m]
            print(f"  {m}: mean={fa.mean_dice_mean:.6f}, std={fa.mean_dice_std:.6f}")

    print("\nSUBJECT_RANKING_MEAN_DICE:")
    for m, v, n in rank_rows:
        print(f"  {m}: subject_mean={v:.6f}, n_subjects={n}")

    print("\nDIRECT_COMPARISONS (A-B on subject mean Dice):")
    for c in comps:
        print(
            f"  {c.a} - {c.b}: n={c.n_subjects}, mean_diff={c.mean_diff:+.6f}, "
            f"CI95=[{c.ci95_low:+.6f}, {c.ci95_high:+.6f}], win_rate={c.win_rate:.3f}, p_perm={c.p_perm:.6g}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())






