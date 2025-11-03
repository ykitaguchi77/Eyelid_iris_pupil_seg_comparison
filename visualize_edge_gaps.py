"""
Visualize edge gap examples for different thicknesses.
Shows original mask, edges at 1/3/5/7px, and their skeletons side-by-side.
"""
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


def mask_to_edge(mask_bin: np.ndarray, thickness: int) -> np.ndarray:
    """Extract contours with specified thickness."""
    if mask_bin.dtype != np.uint8:
        mask_bin = mask_bin.astype(np.uint8)
    bw = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(bw, dtype=np.uint8)
    if contours:
        cv2.drawContours(edge, contours, -1, 255, thickness=thickness)
    return edge


def count_skeleton_endpoints(edge_bin: np.ndarray) -> int:
    """Count skeleton endpoints (pixels with exactly 1 neighbor)."""
    skel = skeletonize((edge_bin > 0))
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel_u8 == 1) & (neighbor_count == 2)
    return int(endpoints.sum())


def get_skeleton_with_endpoints(edge_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return skeleton image and endpoint coordinates."""
    skel = skeletonize((edge_bin > 0))
    skel_u8 = skel.astype(np.uint8) * 255
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D((skel > 0).astype(np.uint8), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel > 0) & (neighbor_count == 2)
    
    return skel_u8, endpoints


def visualize_sample(mask_path: Path, thicknesses: List[int] = [1, 3, 5, 7]) -> None:
    """Visualize one mask with edges and skeletons for different thicknesses."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to read: {mask_path}")
        return
    
    fig, axes = plt.subplots(3, len(thicknesses) + 1, figsize=(4 * (len(thicknesses) + 1), 12))
    
    # Column 0: Original mask
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Original Mask', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    
    # Columns 1-4: Each thickness
    for col_idx, t in enumerate(thicknesses, start=1):
        edge = mask_to_edge(mask, thickness=t)
        skel, endpoints = get_skeleton_with_endpoints(edge)
        n_endpoints = int(endpoints.sum())
        
        # Row 0: Edge
        axes[0, col_idx].imshow(edge, cmap='gray')
        axes[0, col_idx].set_title(f'Edge (t={t}px)', fontsize=10)
        axes[0, col_idx].axis('off')
        
        # Row 1: Skeleton
        axes[1, col_idx].imshow(skel, cmap='gray')
        axes[1, col_idx].set_title(f'Skeleton (t={t}px)', fontsize=10)
        axes[1, col_idx].axis('off')
        
        # Row 2: Skeleton + Endpoints marked in red
        skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
        if n_endpoints > 0:
            ys, xs = np.where(endpoints)
            for y, x in zip(ys, xs):
                cv2.circle(skel_rgb, (x, y), 3, (255, 0, 0), -1)
        
        axes[2, col_idx].imshow(skel_rgb)
        axes[2, col_idx].set_title(f'Endpoints={n_endpoints}', fontsize=10, 
                                     color='red' if n_endpoints > 0 else 'green')
        axes[2, col_idx].axis('off')
    
    plt.suptitle(f'{mask_path.name}', fontsize=12, y=0.995)
    plt.tight_layout()
    plt.show()


def find_gap_examples(label_seg_dir: Path, thicknesses: List[int], 
                      n_examples: int = 3) -> List[Path]:
    """Find examples that have gaps at thickness=1 but not at higher thicknesses."""
    paths = sorted(label_seg_dir.glob("**/*_mask_lid.png"))
    
    gap_at_1_only = []
    gap_increasing = []
    
    for p in paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        gaps = {}
        for t in thicknesses:
            edge = mask_to_edge(mask, thickness=t)
            gaps[t] = count_skeleton_endpoints(edge) > 0
        
        # Case 1: Gap at 1px but closed at 3px (expected behavior)
        if gaps[1] and not gaps[3]:
            gap_at_1_only.append(p)
        
        # Case 2: No gap at 1px but gaps at 5/7px (unexpected)
        if not gaps[1] and (gaps[5] or gaps[7]):
            gap_increasing.append(p)
        
        if len(gap_at_1_only) >= n_examples and len(gap_increasing) >= n_examples:
            break
    
    return gap_at_1_only[:n_examples] + gap_increasing[:n_examples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize edge gaps for eyelid masks")
    parser.add_argument("--label-seg-dir", type=str, default="Images/labels_seg")
    parser.add_argument("--examples", type=int, default=3, help="Number of examples per category")
    parser.add_argument("--specific", type=str, default="", help="Specific mask file to visualize")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_seg_dir = Path(args.label_seg_dir)
    
    if args.specific:
        # Visualize specific file
        mask_path = Path(args.specific)
        if not mask_path.exists():
            mask_path = label_seg_dir / args.specific
        visualize_sample(mask_path)
    else:
        # Find and visualize examples
        print("Finding gap examples...")
        examples = find_gap_examples(label_seg_dir, [1, 3, 5, 7], n_examples=args.examples)
        
        if not examples:
            print("No examples found matching the criteria.")
        else:
            print(f"Found {len(examples)} examples. Visualizing...")
            for p in examples:
                visualize_sample(p)

