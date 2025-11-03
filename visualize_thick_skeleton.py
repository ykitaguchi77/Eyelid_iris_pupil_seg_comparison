"""Visualize that thick edges have no gaps but skeletons do."""
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


def get_endpoints(binary_img):
    kernel = np.ones((3, 3), dtype=np.uint8)
    img_u8 = binary_img.astype(np.uint8)
    neighbor_count = cv2.filter2D(img_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count = neighbor_count - img_u8
    endpoints = (img_u8 > 0) & (neighbor_count == 1)
    return endpoints, np.column_stack(np.where(endpoints))


mask_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
bw = (mask > 0).astype(np.uint8)
contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

thicknesses = [1, 3, 5]

for col, t in enumerate(thicknesses):
    # Draw edge
    edge = np.zeros_like(bw, dtype=np.uint8)
    cv2.drawContours(edge, contours, -1, 255, thickness=t)
    
    # Edge-level endpoints
    edge_endpoints, edge_ep_coords = get_endpoints(edge)
    
    # Skeletonize
    skel = skeletonize((edge > 0))
    skel_u8 = (skel * 255).astype(np.uint8)
    
    # Skeleton-level endpoints
    skel_endpoints, skel_ep_coords = get_endpoints(skel.astype(np.uint8))
    
    # Row 0: Edge
    axes[0, col].imshow(edge, cmap='gray')
    axes[0, col].set_title(f'Edge t={t}px\n{np.count_nonzero(edge)} pixels', fontsize=12, weight='bold')
    axes[0, col].axis('off')
    
    # Row 1: Edge with endpoints
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
    for y, x in edge_ep_coords:
        cv2.circle(edge_rgb, (x, y), 5, (255, 0, 0), -1)
    
    n_edge_ep = len(edge_ep_coords)
    color = 'red' if n_edge_ep > 0 else 'green'
    axes[1, col].imshow(edge_rgb)
    axes[1, col].set_title(f'Edge endpoints: {n_edge_ep}', fontsize=12, color=color, weight='bold')
    axes[1, col].axis('off')
    
    # Row 2: Skeleton with endpoints
    skel_rgb = cv2.cvtColor(skel_u8, cv2.COLOR_GRAY2RGB)
    for y, x in skel_ep_coords:
        cv2.circle(skel_rgb, (x, y), 5, (255, 0, 0), -1)
    
    # Draw line between skeleton endpoints if there are exactly 2
    if len(skel_ep_coords) == 2:
        y1, x1 = skel_ep_coords[0]
        y2, x2 = skel_ep_coords[1]
        cv2.line(skel_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        gap_dist = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        title_text = f'Skeleton endpoints: 2\nGap: {gap_dist:.0f}px'
    else:
        title_text = f'Skeleton endpoints: {len(skel_ep_coords)}'
    
    color = 'red' if len(skel_ep_coords) > 0 else 'green'
    axes[2, col].imshow(skel_rgb)
    axes[2, col].set_title(title_text, fontsize=12, color=color, weight='bold')
    axes[2, col].axis('off')
    
    print(f"t={t}px: Edge endpoints={n_edge_ep}, Skeleton endpoints={len(skel_ep_coords)}")

plt.suptitle(f'{mask_path.name}\nProof: Thick edges close gaps but skeletons still show endpoints', 
             fontsize=14, weight='bold')
plt.tight_layout()

output_path = Path('thick_edge_skeleton_comparison.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.close()

