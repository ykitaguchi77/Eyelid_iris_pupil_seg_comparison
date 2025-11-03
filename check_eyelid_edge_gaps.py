import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize


def list_eyelid_mask_paths(label_seg_dir: Path, limit: int = 0) -> List[Path]:
    """Return a list of paths to '*_mask_lid.png' in the given directory (recursive).
    If limit > 0, return only the first 'limit' paths (sorted for determinism).
    """
    paths = sorted(label_seg_dir.glob("**/*_mask_lid.png"))
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


def mask_to_edge(mask_bin: np.ndarray, thickness: int) -> np.ndarray:
    """Rasterize external contours of a binary mask into an edge map with a given thickness.
    Returns uint8 image with values {0, 255}.
    """
    if mask_bin.dtype != np.uint8:
        mask_bin = mask_bin.astype(np.uint8)
    bw = (mask_bin > 0).astype(np.uint8)
    if bw.ndim != 2:
        raise ValueError("mask_bin must be 2D")

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(bw, dtype=np.uint8)
    if contours:
        cv2.drawContours(edge, contours, -1, 255, thickness=thickness)
    return edge


def count_skeleton_endpoints(edge_bin: np.ndarray) -> int:
    """Return the number of endpoints in the skeleton of the edge map.
    Endpoints are skeleton pixels with exactly one 8-neighbor.
    """
    # Normalize to boolean for skeletonization
    skel = skeletonize((edge_bin > 0))
    skel_u8 = skel.astype(np.uint8)

    # 3x3 neighbor count via convolution (including self)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count_incl_self = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)

    # endpoints: pixels that are 1 in skel and have exactly one neighbor besides itself
    # i.e., neighbor_count_incl_self == 2 (1 self + 1 neighbor)
    endpoints = (skel_u8 == 1) & (neighbor_count_incl_self == 2)
    return int(endpoints.sum())


def has_gap(edge_bin: np.ndarray) -> bool:
    """Heuristic: consider 'gap' if the skeleton has 2+ endpoints.
    Closed contours have zero endpoints; open contours have >=2 endpoints.
    (1 endpoint can occur due to noise/artifacts, so we require >=2)
    """
    return count_skeleton_endpoints(edge_bin) >= 2


def analyze_gaps_for_mask(mask_path: Path, thicknesses: List[int]) -> Dict[int, bool]:
    """Load eyelid mask and compute whether a gap exists for each thickness.
    Returns dict thickness -> has_gap_flag.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    result: Dict[int, bool] = {}
    for t in thicknesses:
        edge = mask_to_edge(mask, thickness=t)
        result[t] = has_gap(edge)
    return result


def run(label_seg_dir: Path, limit: int = 0, save_csv: Path = None) -> None:
    thicknesses = [1, 3, 5, 7]
    paths = list_eyelid_mask_paths(label_seg_dir, limit=limit)
    if not paths:
        print(f"No eyelid mask files found in: {label_seg_dir}")
        return

    counts: Dict[int, int] = {t: 0 for t in thicknesses}
    total = len(paths)

    # Optional CSV rows
    csv_rows: List[Tuple[str, int, int, int, int]] = []  # (filename, gap@1, gap@3, gap@5, gap@7)

    for idx, p in enumerate(paths, 1):
        try:
            flags = analyze_gaps_for_mask(p, thicknesses)
        except Exception as e:
            # Count as gaps for safety? Better to skip and warn.
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        for t in thicknesses:
            if flags[t]:
                counts[t] += 1

        if save_csv is not None:
            csv_rows.append((p.name, int(flags[1]), int(flags[3]), int(flags[5]), int(flags[7])))

        if idx % 200 == 0 or idx == total:
            print(f"Processed {idx}/{total} files...")

    print("\n=== Gap Summary (eyelid, by edge thickness) ===")
    for t in thicknesses:
        n = counts[t]
        pct = (100.0 * n / total) if total else 0.0
        print(f" thickness={t:>2}px : {n:>6d} / {total:<6d} ({pct:5.1f}%)")

    if save_csv is not None:
        import csv
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "gap_t1", "gap_t3", "gap_t5", "gap_t7"])
            for row in csv_rows:
                writer.writerow(row)
        print(f"Saved per-file gap flags to: {save_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count eyelid edge gap occurrences for multiple thicknesses.")
    parser.add_argument("--label-seg-dir", type=str, default=str(Path("Images/labels_seg")), help="Path to labels_seg directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files (0 = all)")
    parser.add_argument("--csv", type=str, default="", help="Optional CSV output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_seg_dir = Path(args.label_seg_dir)
    csv_path = Path(args.csv) if args.csv else None
    run(label_seg_dir=label_seg_dir, limit=args.limit, save_csv=csv_path)
