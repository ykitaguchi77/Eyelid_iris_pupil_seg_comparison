"""
Build patient demography tables for Detection and Segmentation cohorts.

Inputs:
  - D:/眼位写真2/DiseaseInfo_all_connected_all.csv
      * BOM-UTF-8, no header, 25 columns
      * col 0: patient id hash / col 1: hash / col 2: patient number
      * col 3: age / col 4: sex (Male/Female) / col 5..24: disease names
  - scripts/out/detection_patient_ids.json
  - scripts/out/segmentation_patient_ids.json

Outputs:
  - Article/demography/demography_detection.csv   (per-patient)
  - Article/demography/demography_segmentation.csv (per-patient)
  - Article/demography/demography_summary.md       (paper-ready tables)

Disease selection priority (highest wins):
  1. Periocular disorders visible on external photos
     (eyelid / strabismus / conjunctiva / cornea / lacrimal / iris / pupil / eyelash)
  2. Common ophthalmology disorders (dry eye / cataract / glaucoma / refractive)
  3. Other ophthalmology (retina / macula / uveitis / optic nerve)
  4. Non-ophthalmic (fallback)
Exclusions: receipt-style qualifiers (疑い, 既往, 経過観察, 未確定) are filtered out first.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:/Users/CorneAI/Eyelid_Iris_pupil_seg_comparison")
CSV_PATH = Path(r"D:/眼位写真2/DiseaseInfo_all_connected_all.csv")
DET_IDS_JSON = ROOT / "scripts" / "out" / "detection_patient_ids.json"
SEG_IDS_JSON = ROOT / "scripts" / "out" / "segmentation_patient_ids.json"
RESULTS_DIR = ROOT / "Article" / "demography"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


EXCLUDE_PATTERNS = [
    r"疑い", r"の疑", r"既往", r"経過観察", r"未確定", r"\(R/O\)", r"R/O", r"ルールアウト",
]

PRIORITY_RULES: list[tuple[int, str, list[str]]] = [
    (
        0,
        "Periocular - Thyroid eye disease",
        [
            "甲状腺眼症", "バセドウ病眼症",
        ],
    ),
    (
        1,
        "Periocular - Eyelid",
        [
            "眼瞼下垂", "眼瞼痙攣", "眼瞼けいれん", "眼瞼内反", "眼瞼外反",
            "眼瞼腫瘍", "眼瞼腫瘤", "眼瞼炎", "眼瞼裂傷", "眼瞼皮膚弛緩",
            "霰粒腫", "麦粒腫", "兎眼", "睫毛内反", "睫毛乱生", "逆睫",
            "マイボーム", "チョコレート嚢胞", "眼瞼",
            "眼窩腫瘤", "眼窩蜂窩織炎", "眼窩脂肪ヘルニア", "眼窩", "眼球突出",
        ],
    ),
    (
        2,
        "Periocular - Strabismus / Eye position",
        [
            "内斜視", "外斜視", "上斜視", "下斜視", "間欠性外斜視", "恒常性",
            "斜視", "眼位異常", "複視", "眼球運動障害", "眼振", "斜位",
        ],
    ),
    (
        3,
        "Periocular - Conjunctiva / Cornea",
        [
            "翼状片", "結膜炎", "結膜下出血", "結膜弛緩", "結膜嚢胞", "結膜腫瘍",
            "角膜炎", "角膜潰瘍", "角膜ヘルペス", "角膜びらん", "角膜混濁",
            "角膜瘢痕", "角膜ジストロフィー", "結膜", "角膜",
        ],
    ),
    (
        4,
        "Periocular - Lacrimal / Iris / Pupil",
        [
            "涙嚢炎", "鼻涙管", "流涙", "涙道", "涙器", "涙点",
            "虹彩炎", "虹彩", "瞳孔", "縮瞳", "散瞳", "ホルネル", "アディー",
        ],
    ),
    (
        5,
        "Ophthalmology - Common",
        [
            "ドライアイ", "乾性角結膜炎", "白内障", "緑内障", "近視",
            "遠視", "乱視", "老視", "屈折異常", "眼精疲労", "弱視",
        ],
    ),
    (
        6,
        "Ophthalmology - Posterior",
        [
            "網膜", "黄斑", "ぶどう膜炎", "硝子体", "視神経", "脈絡膜",
            "糖尿病網膜症", "加齢黄斑変性",
        ],
    ),
    (
        7,
        "Systemic / Other",
        [
            "甲状腺", "糖尿病", "高血圧", "アレルギー", "花粉症", "腎",
        ],
    ),
]


def is_excluded(name: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, name):
            return True
    return False


def score_disease(name: str) -> tuple[int, int, str]:
    """Return (priority, keyword_index, matched_category). Lower priority = better."""
    for priority, category, keywords in PRIORITY_RULES:
        for idx, kw in enumerate(keywords):
            if kw in name:
                return (priority, idx, category)
    return (99, 0, "Unclassified")


def select_primary(diseases: list[str]) -> tuple[str, str, list[str]]:
    """
    Return (primary_disease, category, kept_diseases_after_exclusion).
    """
    kept = [d for d in diseases if d and not is_excluded(d)]
    if not kept:
        kept_for_fallback = [d for d in diseases if d]
        if kept_for_fallback:
            d0 = kept_for_fallback[0]
            _, _, cat = score_disease(d0)
            return (d0, f"fallback (only excluded items); {cat}", kept_for_fallback)
        return ("該当なし", "no disease entries", [])
    scored = [(score_disease(d), d) for d in kept]
    scored.sort(key=lambda x: (x[0][0], x[0][1]))
    (pri, _, cat), chosen = scored[0]
    return (chosen, cat, kept)


def load_demography_csv() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, header=None, encoding="utf-8-sig", dtype=str)
    # rename columns
    disease_cols = [f"d{i+1}" for i in range(df.shape[1] - 5)]
    df.columns = ["pid_hash", "hash2", "patient_no", "age", "sex"] + disease_cols
    df["patient_no"] = pd.to_numeric(df["patient_no"], errors="coerce").astype("Int64")
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def build_cohort_table(
    df: pd.DataFrame, ids: list[int], disease_cols: list[str]
) -> tuple[pd.DataFrame, list[int]]:
    sub = df[df["patient_no"].isin(ids)].copy()
    sub = sub.drop_duplicates(subset=["patient_no"], keep="first")
    matched_ids = set(sub["patient_no"].dropna().astype(int).tolist())
    not_found = sorted(set(ids) - matched_ids)

    primary, category, all_kept = [], [], []
    for _, row in sub.iterrows():
        diseases = [row[c] for c in disease_cols if pd.notna(row[c]) and str(row[c]).strip()]
        p, cat, kept = select_primary([str(d).strip() for d in diseases])
        primary.append(p)
        category.append(cat)
        all_kept.append(" | ".join(kept))

    out = pd.DataFrame({
        "patient_no": sub["patient_no"].astype(int).values,
        "age": sub["age_num"].values,
        "sex": sub["sex"].values,
        "primary_disease": primary,
        "disease_category": category,
        "all_diseases_kept": all_kept,
    }).sort_values("patient_no").reset_index(drop=True)
    return out, not_found


def summarize(tbl: pd.DataFrame) -> dict:
    ages = tbl["age"].dropna().astype(float).tolist()
    sex_counts = tbl["sex"].value_counts(dropna=False).to_dict()
    disease_counts = Counter(tbl["primary_disease"].tolist())
    category_counts = Counter(tbl["disease_category"].tolist())
    return {
        "n": len(tbl),
        "age_n": len(ages),
        "age_mean": statistics.mean(ages) if ages else None,
        "age_sd": statistics.stdev(ages) if len(ages) > 1 else None,
        "age_median": statistics.median(ages) if ages else None,
        "age_min": min(ages) if ages else None,
        "age_max": max(ages) if ages else None,
        "sex_counts": sex_counts,
        "top_diseases": disease_counts.most_common(15),
        "category_counts": category_counts.most_common(),
    }


def fmt_sex(sex_counts: dict, total: int) -> str:
    male = int(sex_counts.get("Male", 0) or 0)
    female = int(sex_counts.get("Female", 0) or 0)
    other = total - male - female
    parts = [
        f"Male {male} ({male/total*100:.1f}%)",
        f"Female {female} ({female/total*100:.1f}%)",
    ]
    if other > 0:
        parts.append(f"Other/Missing {other} ({other/total*100:.1f}%)")
    return ", ".join(parts)


def fmt_age(s: dict) -> str:
    if s["age_n"] == 0:
        return "N/A"
    mean = f"{s['age_mean']:.1f}"
    sd = f"{s['age_sd']:.1f}" if s["age_sd"] is not None else "NA"
    median = f"{s['age_median']:.0f}"
    rng = f"{s['age_min']:.0f}-{s['age_max']:.0f}"
    return f"{mean} ± {sd} (median {median}, range {rng})"


def render_markdown(det_s: dict, seg_s: dict, det_notfound: list[int], seg_notfound: list[int]) -> str:
    lines: list[str] = []
    lines.append("# Patient Demographics\n")
    lines.append(
        "Demography extracted from `D:/眼位写真2/DiseaseInfo_all_connected_all.csv` "
        "for patients actually used in each training task.\n"
    )

    lines.append("## Table 1. Overview\n")
    lines.append("| Characteristic | Detection cohort | Segmentation cohort |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Patients (n) | {det_s['n']} | {seg_s['n']} |"
    )
    lines.append(
        f"| Age, years (mean ± SD) | {fmt_age(det_s)} | {fmt_age(seg_s)} |"
    )
    lines.append(
        f"| Sex | {fmt_sex(det_s['sex_counts'], det_s['n'])} | "
        f"{fmt_sex(seg_s['sex_counts'], seg_s['n'])} |"
    )
    lines.append("")

    for label, s in [("Detection", det_s), ("Segmentation", seg_s)]:
        lines.append(f"## {label} cohort - top primary diseases\n")
        lines.append("| Rank | Primary disease | Count | % |")
        lines.append("|---|---|---|---|")
        for i, (name, cnt) in enumerate(s["top_diseases"], start=1):
            pct = cnt / s["n"] * 100
            lines.append(f"| {i} | {name} | {cnt} | {pct:.1f} |")
        lines.append("")

        lines.append(f"### {label} cohort - disease category distribution\n")
        lines.append("| Category | Count | % |")
        lines.append("|---|---|---|")
        for name, cnt in s["category_counts"]:
            pct = cnt / s["n"] * 100
            lines.append(f"| {name} | {cnt} | {pct:.1f} |")
        lines.append("")

    lines.append("## Notes on matching\n")
    lines.append(
        f"- Detection: {len(det_notfound)} patient ID(s) not found in demography CSV: "
        f"{det_notfound if det_notfound else 'none'}"
    )
    lines.append(
        f"- Segmentation: {len(seg_notfound)} patient ID(s) not found in demography CSV: "
        f"{seg_notfound if seg_notfound else 'none'}"
    )
    lines.append(
        "- Primary disease was chosen by a rule-based priority (periocular > common > posterior > "
        "systemic), excluding receipt-style qualifiers such as 疑い / 既往 / 経過観察."
    )
    lines.append(
        "- Please review `demography_detection.csv` and `demography_segmentation.csv` for "
        "per-patient decisions; the `all_diseases_kept` column preserves the full list for audit."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    det_ids = json.loads(DET_IDS_JSON.read_text())["patient_ids"]
    seg_ids = json.loads(SEG_IDS_JSON.read_text())["patient_ids"]
    print(f"Detection IDs: {len(det_ids)}")
    print(f"Segmentation IDs: {len(seg_ids)}")

    df = load_demography_csv()
    disease_cols = [c for c in df.columns if c.startswith("d")]
    print(f"CSV rows: {len(df)}, disease cols: {len(disease_cols)}")

    det_tbl, det_notfound = build_cohort_table(df, det_ids, disease_cols)
    seg_tbl, seg_notfound = build_cohort_table(df, seg_ids, disease_cols)

    det_out = RESULTS_DIR / "demography_detection.csv"
    seg_out = RESULTS_DIR / "demography_segmentation.csv"
    det_tbl.to_csv(det_out, index=False, encoding="utf-8-sig")
    seg_tbl.to_csv(seg_out, index=False, encoding="utf-8-sig")
    print(f"Wrote: {det_out} ({len(det_tbl)} rows)")
    print(f"Wrote: {seg_out} ({len(seg_tbl)} rows)")

    det_s = summarize(det_tbl)
    seg_s = summarize(seg_tbl)
    md = render_markdown(det_s, seg_s, det_notfound, seg_notfound)
    md_out = RESULTS_DIR / "demography_summary.md"
    md_out.write_text(md, encoding="utf-8")
    print(f"Wrote: {md_out}")

    print("\n--- Detection summary ---")
    print(f"n={det_s['n']}, age={fmt_age(det_s)}, sex={fmt_sex(det_s['sex_counts'], det_s['n'])}")
    print(f"not-found: {det_notfound}")
    print("\n--- Segmentation summary ---")
    print(f"n={seg_s['n']}, age={fmt_age(seg_s)}, sex={fmt_sex(seg_s['sex_counts'], seg_s['n'])}")
    print(f"not-found: {seg_notfound}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
