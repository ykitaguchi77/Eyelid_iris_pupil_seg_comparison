"""Visualize a large interior gap example."""
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


def mask_to_edge(mask_bin: np.ndarray, thickness: int) -> np.ndarray:
    bw = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(bw, dtype=np.uint8)
    if contours:
        cv2.drawContours(edge, contours, -1, 255, thickness=thickness)
    return edge


def get_endpoints_and_skeleton(edge_bin: np.ndarray):
    skel = skeletonize((edge_bin > 0))
    skel_u8 = (skel * 255).astype(np.uint8)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel.astype(np.uint8), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel > 0) & (neighbor_count == 2)
    coords = np.column_stack(np.where(endpoints))
    
    return skel_u8, coords


if __name__ == "__main__":
    # Example with 357px gap at 1px
    mask_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load: {mask_path}")
        exit(1)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    thicknesses = [1, 3, 5, 7]
    
    for col, t in enumerate(thicknesses):
        edge = mask_to_edge(mask, thickness=t)
        skel, coords = get_endpoints_and_skeleton(edge)
        
        # Row 0: Edge
        axes[0, col].imshow(edge, cmap='gray')
        axes[0, col].set_title(f'Edge t={t}px', fontsize=12, weight='bold')
        axes[0, col].axis('off')
        
        # Row 1: Skeleton
        axes[1, col].imshow(skel, cmap='gray')
        axes[1, col].set_title(f'Skeleton', fontsize=12)
        axes[1, col].axis('off')
        
        # Row 2: Skeleton + endpoints + gap line
        skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
        
        # Mark endpoints in red
        for y, x in coords:
            cv2.circle(skel_rgb, (x, y), 8, (255, 0, 0), -1)
        
        # Draw line between endpoints if there are exactly 2
        if len(coords) == 2:
            y1, x1 = coords[0]
            y2, x2 = coords[1]
            cv2.line(skel_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            gap_dist = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            title_text = f'Gap: {gap_dist:.0f}px'
            color = 'red'
        else:
            title_text = f'{len(coords)} endpoints'
            color = 'orange' if len(coords) > 0 else 'green'
        
        axes[2, col].imshow(skel_rgb)
        axes[2, col].set_title(title_text, fontsize=12, color=color, weight='bold')
        axes[2, col].axis('off')
    
    plt.suptitle(f'{mask_path.name}\n(Large interior gap example)', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path = Path('large_gap_example.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    
    # Print gap distances
    print("\nGap distances:")
    for t in thicknesses:
        edge = mask_to_edge(mask, thickness=t)
        _, coords = get_endpoints_and_skeleton(edge)
        if len(coords) == 2:
            y1, x1 = coords[0]
            y2, x2 = coords[1]
            dist = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            print(f"  t={t}px: {dist:.1f}px gap between ({y1},{x1}) and ({y2},{x2})")
        else:
            print(f"  t={t}px: {len(coords)} endpoints")

