"""Find eyelid masks with large gaps (not just image boundary endpoints)."""
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
from typing import Tuple, List


def mask_to_edge(mask_bin: np.ndarray, thickness: int) -> np.ndarray:
    bw = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(bw, dtype=np.uint8)
    if contours:
        cv2.drawContours(edge, contours, -1, 255, thickness=thickness)
    return edge


def get_endpoints(edge_bin: np.ndarray) -> np.ndarray:
    """Return endpoint coordinates (N, 2) as (y, x)."""
    skel = skeletonize((edge_bin > 0))
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel_u8 == 1) & (neighbor_count == 2)
    return np.column_stack(np.where(endpoints))  # (y, x)


def is_boundary_endpoint(y: int, x: int, h: int, w: int, margin: int = 5) -> bool:
    """Check if endpoint is near image boundary."""
    return (y < margin or y >= h - margin or x < margin or x >= w - margin)


def has_interior_gap(edge_bin: np.ndarray, boundary_margin: int = 5) -> Tuple[bool, int, List]:
    """Check if there are endpoints away from image boundary.
    Returns: (has_gap, n_interior_endpoints, interior_coords)
    """
    h, w = edge_bin.shape
    endpoints = get_endpoints(edge_bin)
    
    if len(endpoints) < 2:
        return False, 0, []
    
    # Filter out boundary endpoints
    interior_endpoints = []
    for y, x in endpoints:
        if not is_boundary_endpoint(y, x, h, w, boundary_margin):
            interior_endpoints.append((y, x))
    
    n_interior = len(interior_endpoints)
    has_gap = n_interior >= 2
    
    return has_gap, n_interior, interior_endpoints


def find_large_gap_examples(label_seg_dir: Path, thickness: int, 
                            n_examples: int = 10, max_search: int = 2000) -> List[Tuple[Path, int, List]]:
    """Find examples with interior gaps at given thickness.
    Returns: [(path, n_interior_endpoints, coords), ...]
    """
    paths = sorted(label_seg_dir.glob("**/*_mask_lid.png"))[:max_search]
    
    examples = []
    for p in paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        edge = mask_to_edge(mask, thickness=thickness)
        has_gap, n_interior, coords = has_interior_gap(edge)
        
        if has_gap:
            examples.append((p, n_interior, coords))
        
        if len(examples) >= n_examples:
            break
    
    return examples


if __name__ == "__main__":
    label_seg_dir = Path("Images/labels_seg")
    
    print("Searching for interior gaps (away from image boundary)...\n")
    
    for thickness in [1, 3, 5, 7]:
        print(f"=== Thickness = {thickness}px ===")
        examples = find_large_gap_examples(label_seg_dir, thickness, n_examples=5, max_search=2000)
        
        if not examples:
            print(f"  No interior gaps found in first 2000 images.\n")
        else:
            print(f"  Found {len(examples)} examples:")
            for path, n_interior, coords in examples:
                print(f"    {path.name}: {n_interior} interior endpoints")
                if len(coords) <= 4:
                    print(f"      Positions: {coords}")
            print()
    
    # Detailed analysis for thickness=1
    print("\n=== Detailed Analysis: thickness=1px ===")
    examples_1px = find_large_gap_examples(label_seg_dir, thickness=1, n_examples=3, max_search=2000)
    
    if examples_1px:
        print(f"Found {len(examples_1px)} cases with interior gaps at 1px:")
        for path, n_interior, coords in examples_1px:
            print(f"\n{path.name}")
            print(f"  Interior endpoints: {n_interior}")
            print(f"  Positions (y, x): {coords}")
            
            # Calculate gap distances
            if len(coords) >= 2:
                distances = []
                for i in range(len(coords) - 1):
                    y1, x1 = coords[i]
                    y2, x2 = coords[i + 1]
                    dist = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                    distances.append(dist)
                print(f"  Gap distances: {[f'{d:.1f}px' for d in distances]}")
    else:
        print("No interior gaps found at 1px in first 2000 images.")
        print("This suggests original GT quality is very high!")

