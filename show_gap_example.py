"""Quick visualization of a specific mask with all thickness variations."""
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


def count_endpoints(edge_bin: np.ndarray) -> int:
    skel = skeletonize((edge_bin > 0))
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel_u8 == 1) & (neighbor_count == 2)
    return int(endpoints.sum())


if __name__ == "__main__":
    # Example: 1px=False, 3px=False, 5px=True, 7px=True
    mask_path = Path("Images/labels_seg/10-20170817-85-103358_71bb768b3c26a5e1ccc462a5cc088f063879b38018c737e30ea9275fb3780f6f_L_mask_lid.png")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load: {mask_path}")
        sys.exit(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Original
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Original Mask', fontsize=12, weight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    thicknesses = [1, 3, 5, 7]
    for i, t in enumerate(thicknesses, 1):
        edge = mask_to_edge(mask, thickness=t)
        n_ep = count_endpoints(edge)
        
        # Edge
        axes[0, i].imshow(edge, cmap='gray')
        axes[0, i].set_title(f'Edge t={t}px', fontsize=12)
        axes[0, i].axis('off')
        
        # Skeleton with endpoints marked
        skel = skeletonize((edge > 0))
        skel_u8 = (skel * 255).astype(np.uint8)
        skel_rgb = cv2.cvtColor(skel_u8, cv2.COLOR_GRAY2RGB)
        
        # Mark endpoints in red
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel.astype(np.uint8), ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
        endpoints = (skel > 0) & (neighbor_count == 2)
        ys, xs = np.where(endpoints)
        for y, x in zip(ys, xs):
            cv2.circle(skel_rgb, (x, y), 4, (255, 0, 0), -1)
        
        axes[1, i].imshow(skel_rgb)
        color = 'red' if n_ep > 0 else 'green'
        axes[1, i].set_title(f'Skeleton: {n_ep} endpoints', fontsize=12, color=color, weight='bold')
        axes[1, i].axis('off')
    
    plt.suptitle(f'{mask_path.name}', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path = Path('gap_analysis_example.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

