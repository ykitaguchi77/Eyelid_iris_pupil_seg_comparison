"""Proper gap detection without skeletonization artifacts."""
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np


def mask_to_edge(mask_bin: np.ndarray, thickness: int) -> np.ndarray:
    bw = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(bw, dtype=np.uint8)
    if contours:
        cv2.drawContours(edge, contours, -1, 255, thickness=thickness)
    return edge


def has_gap_proper(edge_bin: np.ndarray, margin: int = 5) -> bool:
    """Proper gap detection based on edge connectivity, not skeletonization.
    
    A closed contour should have all edge pixels with 2 neighbors (forming a ring).
    Gaps are detected when edge pixels have only 1 neighbor (endpoints).
    We exclude boundary pixels to avoid false positives from image edges.
    """
    h, w = edge_bin.shape
    edge_u8 = (edge_bin > 0).astype(np.uint8)
    
    # Count 8-connected neighbors
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(edge_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count = neighbor_count - edge_u8  # Subtract self
    
    # Endpoint: edge pixel with exactly 1 neighbor
    endpoints = (edge_u8 > 0) & (neighbor_count == 1)
    
    # Exclude boundary pixels (image edges)
    boundary_mask = np.zeros((h, w), dtype=bool)
    boundary_mask[:margin, :] = True
    boundary_mask[-margin:, :] = True
    boundary_mask[:, :margin] = True
    boundary_mask[:, -margin:] = True
    
    interior_endpoints = endpoints & ~boundary_mask
    
    return np.count_nonzero(interior_endpoints) >= 2


def analyze_gaps_for_mask_proper(mask_path: Path, thicknesses: List[int]) -> Dict[int, bool]:
    """Proper gap analysis without skeletonization."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read: {mask_path}")
    
    result = {}
    for t in thicknesses:
        edge = mask_to_edge(mask, thickness=t)
        result[t] = has_gap_proper(edge, margin=5)
    
    return result


def run_proper_gap_analysis(label_seg_dir: Path, limit: int = 0):
    """Re-run gap analysis with proper detection method."""
    thicknesses = [1, 3, 5, 7]
    paths = sorted(label_seg_dir.glob("**/*_mask_lid.png"))
    
    if limit > 0:
        paths = paths[:limit]
    
    if not paths:
        print(f"No files found in: {label_seg_dir}")
        return
    
    counts = {t: 0 for t in thicknesses}
    total = len(paths)
    
    for idx, p in enumerate(paths, 1):
        try:
            flags = analyze_gaps_for_mask_proper(p, thicknesses)
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue
        
        for t in thicknesses:
            if flags[t]:
                counts[t] += 1
        
        if idx % 200 == 0 or idx == total:
            print(f"Processed {idx}/{total} files...")
    
    print("\n=== Gap Summary (Proper Detection: Edge connectivity without skeleton) ===")
    for t in thicknesses:
        n = counts[t]
        pct = (100.0 * n / total) if total else 0.0
        print(f" thickness={t:>2}px : {n:>6d} / {total:<6d} ({pct:5.1f}%)")
    
    # Compare with original results
    print("\n=== Comparison with Original Method (Skeleton-based) ===")
    print("Original (skeleton):  1px=0.3%, 3px=1.1%, 5px=2.2%, 7px=2.5%")
    print(f"New (edge):           1px={100.0*counts[1]/total:.1f}%, 3px={100.0*counts[3]/total:.1f}%, 5px={100.0*counts[5]/total:.1f}%, 7px={100.0*counts[7]/total:.1f}%")


if __name__ == "__main__":
    label_seg_dir = Path("Images/labels_seg")
    
    # Test on the problematic example first
    print("=== Testing on the problematic example ===")
    test_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
    result = analyze_gaps_for_mask_proper(test_path, [1, 3, 5, 7])
    for t, has_gap in result.items():
        print(f"  thickness={t}px: {'GAP' if has_gap else 'OK'}")
    
    print("\n" + "="*70)
    print("Running full analysis on all masks...")
    print("="*70 + "\n")
    
    run_proper_gap_analysis(label_seg_dir)

