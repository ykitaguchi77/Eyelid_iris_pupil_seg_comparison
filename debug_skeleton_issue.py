"""Debug why skeletonization creates endpoints in a closed contour."""
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


mask_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

# Get edge with thickness=1
edge = mask_to_edge(mask, thickness=1)

print("=== Edge Analysis ===")
print(f"Edge pixels: {np.count_nonzero(edge)}")

# Check if edge forms a closed ring
# A closed ring should have all pixels with 2 neighbors
kernel = np.ones((3, 3), dtype=np.uint8)
edge_bin = (edge > 0).astype(np.uint8)
neighbor_count = cv2.filter2D(edge_bin, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
neighbor_count = neighbor_count - edge_bin  # Subtract self

print(f"\nNeighbor count statistics on edge:")
print(f"  Min neighbors: {neighbor_count[edge_bin > 0].min()}")
print(f"  Max neighbors: {neighbor_count[edge_bin > 0].max()}")
unique, counts = np.unique(neighbor_count[edge_bin > 0], return_counts=True)
for n, c in zip(unique, counts):
    print(f"  {n} neighbors: {c} pixels ({100*c/np.count_nonzero(edge):.1f}%)")

# Points with 1 neighbor are potential endpoints
potential_endpoints = (edge_bin > 0) & (neighbor_count == 1)
print(f"\nPotential endpoints (1 neighbor): {np.count_nonzero(potential_endpoints)}")
if np.count_nonzero(potential_endpoints) > 0:
    ys, xs = np.where(potential_endpoints)
    print(f"  Locations: {list(zip(ys[:5], xs[:5]))}")

# Now check skeletonization
print("\n=== Skeletonization Analysis ===")
skel = skeletonize((edge > 0))
skel_u8 = skel.astype(np.uint8)

print(f"Skeleton pixels: {np.count_nonzero(skel)}")

# Skeleton endpoints
skel_neighbor_count = cv2.filter2D(skel_u8, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
skel_neighbor_count = skel_neighbor_count - skel_u8
skeleton_endpoints = (skel_u8 > 0) & (skel_neighbor_count == 1)
print(f"Skeleton endpoints: {np.count_nonzero(skeleton_endpoints)}")
if np.count_nonzero(skeleton_endpoints) > 0:
    ys, xs = np.where(skeleton_endpoints)
    print(f"  Locations: {list(zip(ys, xs))}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Edge
axes[0, 0].imshow(edge, cmap='gray')
axes[0, 0].set_title('Edge (t=1)', fontsize=12, weight='bold')
axes[0, 0].axis('off')

# Edge with neighbor count visualization
edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)
if np.count_nonzero(potential_endpoints) > 0:
    ys, xs = np.where(potential_endpoints)
    for y, x in zip(ys, xs):
        cv2.circle(edge_rgb, (x, y), 3, (255, 0, 0), -1)

axes[0, 1].imshow(edge_rgb)
axes[0, 1].set_title(f'Edge with potential breaks\n({np.count_nonzero(potential_endpoints)} pixels with 1 neighbor)', 
                     fontsize=12, color='red' if np.count_nonzero(potential_endpoints) > 0 else 'green')
axes[0, 1].axis('off')

# Neighbor count heatmap on edge
neighbor_map = np.zeros_like(edge, dtype=np.uint8)
neighbor_map[edge > 0] = neighbor_count[edge > 0] * 50
axes[0, 2].imshow(neighbor_map, cmap='hot')
axes[0, 2].set_title('Neighbor count heatmap\n(brighter = more neighbors)', fontsize=12)
axes[0, 2].axis('off')

# Skeleton
skel_vis = (skel * 255).astype(np.uint8)
axes[1, 0].imshow(skel_vis, cmap='gray')
axes[1, 0].set_title('Skeleton', fontsize=12, weight='bold')
axes[1, 0].axis('off')

# Skeleton with endpoints
skel_rgb = cv2.cvtColor(skel_vis, cv2.COLOR_GRAY2RGB)
if np.count_nonzero(skeleton_endpoints) > 0:
    ys, xs = np.where(skeleton_endpoints)
    for y, x in zip(ys, xs):
        cv2.circle(skel_rgb, (x, y), 5, (255, 0, 0), -1)

axes[1, 1].imshow(skel_rgb)
axes[1, 1].set_title(f'Skeleton with endpoints\n({np.count_nonzero(skeleton_endpoints)} endpoints)', 
                     fontsize=12, color='red' if np.count_nonzero(skeleton_endpoints) > 0 else 'green')
axes[1, 1].axis('off')

# Skeleton neighbor count
skel_neighbor_map = np.zeros_like(edge, dtype=np.uint8)
skel_neighbor_map[skel > 0] = skel_neighbor_count[skel > 0] * 50
axes[1, 2].imshow(skel_neighbor_map, cmap='hot')
axes[1, 2].set_title('Skeleton neighbor count', fontsize=12)
axes[1, 2].axis('off')

plt.suptitle(f'{mask_path.name}\nDebugging skeleton endpoint detection', fontsize=14, weight='bold')
plt.tight_layout()

output_path = Path('skeleton_debug.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.close()

