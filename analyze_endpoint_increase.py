"""Analyze why endpoints increase from 1px to 5px."""
import sys
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


def analyze_endpoints(edge_bin: np.ndarray):
    """Return skeleton, endpoint mask, and endpoint coordinates."""
    skel = skeletonize((edge_bin > 0))
    skel_u8 = skel.astype(np.uint8)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel_u8 == 1) & (neighbor_count == 2)
    
    coords = np.column_stack(np.where(endpoints))  # (y, x)
    return skel_u8 * 255, endpoints, coords


if __name__ == "__main__":
    # Case: 1px=1端点, 5px=2端点
    mask_path = Path("Images/labels_seg/10-20171109-85-110054_9576e4e2c7a6d22ecf0329988798daf8e9adfcd0fa38d82bdcd8cfc88db72700_L_mask_lid.png")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load: {mask_path}")
        sys.exit(1)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 0: Original + edges
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(mask, cmap='gray')
    ax_orig.set_title('Original Mask', fontsize=14, weight='bold')
    ax_orig.axis('off')
    
    thicknesses = [1, 3, 5, 7]
    for i, t in enumerate(thicknesses):
        edge = mask_to_edge(mask, thickness=t)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(edge, cmap='gray')
        ax.set_title(f'Edge t={t}px', fontsize=14)
        ax.axis('off')
    
    # Row 1: Skeletons
    for i, t in enumerate(thicknesses):
        edge = mask_to_edge(mask, thickness=t)
        skel, endpoints_mask, coords = analyze_endpoints(edge)
        
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(skel, cmap='gray')
        ax.set_title(f'Skeleton t={t}px', fontsize=14)
        ax.axis('off')
    
    # Row 2: Skeleton + endpoints marked
    for i, t in enumerate(thicknesses):
        edge = mask_to_edge(mask, thickness=t)
        skel, endpoints_mask, coords = analyze_endpoints(edge)
        
        # Convert to RGB and mark endpoints
        skel_rgb = cv2.cvtColor(skel, cv2.COLOR_GRAY2RGB)
        for y, x in coords:
            cv2.circle(skel_rgb, (x, y), 5, (255, 0, 0), -1)
        
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(skel_rgb)
        n_ep = len(coords)
        color = 'red' if n_ep >= 2 else ('orange' if n_ep == 1 else 'green')
        ax.set_title(f'{n_ep} endpoint(s)', fontsize=14, color=color, weight='bold')
        ax.axis('off')
        
        # Print endpoint locations
        if len(coords) > 0:
            print(f"t={t}px: {n_ep} endpoint(s) at {coords.tolist()}")
    
    plt.suptitle(f'{mask_path.name}', fontsize=16, weight='bold')
    
    output_path = Path('endpoint_increase_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    plt.close()

