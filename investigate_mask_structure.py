"""Investigate the actual structure of the mask."""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


mask_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

print(f"Mask info:")
print(f"  Shape: {mask.shape}")
print(f"  Non-zero pixels: {np.count_nonzero(mask)}")
print(f"  Unique values: {np.unique(mask)}")

# Original mask
bw = (mask > 0).astype(np.uint8)

# Find contours
contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"\nContour info:")
print(f"  Number of contours: {len(contours)}")
if len(contours) > 0:
    print(f"  Contour 0 points: {len(contours[0])}")
    first_pt = contours[0][0][0]
    last_pt = contours[0][-1][0]
    dist = np.linalg.norm(first_pt - last_pt)
    print(f"  First point: {first_pt}")
    print(f"  Last point: {last_pt}")
    print(f"  Distance: {dist:.2f}px")

# Draw contour on edge (thickness=1)
edge_t1 = np.zeros_like(bw, dtype=np.uint8)
cv2.drawContours(edge_t1, contours, -1, 255, thickness=1)

# Check if edge differs from original mask boundary
mask_boundary = cv2.Canny(mask, 50, 150)

print(f"\nEdge comparison:")
print(f"  Edge (t=1) non-zero: {np.count_nonzero(edge_t1)}")
print(f"  Canny boundary non-zero: {np.count_nonzero(mask_boundary)}")

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 0: Original and edges
axes[0, 0].imshow(mask, cmap='gray')
axes[0, 0].set_title('Original Mask (filled)', fontsize=12, weight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(edge_t1, cmap='gray')
axes[0, 1].set_title(f'Contour drawn (t=1)\n{np.count_nonzero(edge_t1)} pixels', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(mask_boundary, cmap='gray')
axes[0, 2].set_title(f'Canny edge\n{np.count_nonzero(mask_boundary)} pixels', fontsize=12)
axes[0, 2].axis('off')

# Row 1: Zoomed regions around the gap
# Based on endpoint coords: (297, 111) and (345, 465)
y1, x1 = 297, 111
y2, x2 = 345, 465

margin = 50
for idx, (y, x, label) in enumerate([(y1, x1, 'Left endpoint'), (y2, x2, 'Right endpoint')]):
    y_min, y_max = max(0, y-margin), min(mask.shape[0], y+margin)
    x_min, x_max = max(0, x-margin), min(mask.shape[1], x+margin)
    
    zoomed = mask[y_min:y_max, x_min:x_max]
    axes[1, idx].imshow(zoomed, cmap='gray')
    axes[1, idx].set_title(f'{label}\n(y={y}, x={x})', fontsize=12, color='red', weight='bold')
    axes[1, idx].axhline(y - y_min, color='red', linewidth=1, alpha=0.5)
    axes[1, idx].axvline(x - x_min, color='red', linewidth=1, alpha=0.5)
    axes[1, idx].axis('on')

# Full mask with gap points marked
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
cv2.circle(mask_rgb, (x1, y1), 10, (255, 0, 0), -1)
cv2.circle(mask_rgb, (x2, y2), 10, (255, 0, 0), -1)
cv2.line(mask_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

axes[1, 2].imshow(mask_rgb)
axes[1, 2].set_title('Gap location on mask', fontsize=12, weight='bold')
axes[1, 2].axis('off')

plt.suptitle(f'{mask_path.name}\nOriginal mask structure investigation', fontsize=14, weight='bold')
plt.tight_layout()

output_path = Path('mask_structure_investigation.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.close()

