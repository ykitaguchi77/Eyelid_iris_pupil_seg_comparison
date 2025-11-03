"""Analyze if the gap corresponds to horizontally thin regions (eye corners)."""
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt


mask_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
bw = (mask > 0)

# Compute medial axis (skeleton with distance transform)
skel, distance = medial_axis(bw, return_distance=True)
dist_on_skel = distance * skel

# Endpoint locations from previous analysis
endpoint_coords = [(297, 111), (345, 465)]

print("=== Analyzing thin regions ===\n")

# For each y-coordinate, find the horizontal width of the mask
y_min, y_max = 250, 400
widths = []
y_coords = []

for y in range(y_min, y_max):
    row = bw[y, :]
    if np.any(row):
        x_coords = np.where(row)[0]
        width = x_coords.max() - x_coords.min() + 1
        widths.append(width)
        y_coords.append(y)
    else:
        widths.append(0)
        y_coords.append(y)

widths = np.array(widths)
y_coords = np.array(y_coords)

print(f"Width statistics (y={y_min} to {y_max}):")
print(f"  Min width: {widths[widths > 0].min()} px at y={y_coords[np.argmin(widths + 1000*(widths==0))]}")
print(f"  Max width: {widths.max()} px at y={y_coords[np.argmax(widths)]}")
print(f"  Mean width: {widths[widths > 0].mean():.1f} px")

# Find y-coordinates with very thin width
thin_threshold = 10
thin_ys = y_coords[widths < thin_threshold]
if len(thin_ys) > 0:
    print(f"\nRows with width < {thin_threshold}px: {len(thin_ys)}")
    print(f"  Y-coordinates: {thin_ys[:10].tolist()}...")

# Check endpoints
print("\n=== Endpoint Analysis ===")
for i, (y, x) in enumerate(endpoint_coords):
    if y_min <= y < y_max:
        idx = y - y_min
        width = widths[idx]
        print(f"Endpoint {i+1} at ({y}, {x}): horizontal width = {width} px")
        
        # Distance from skeleton
        if skel[y, x]:
            print(f"  On skeleton, local width = {distance[y, x]:.1f} px")
        else:
            # Find nearest skeleton point
            ys, xs = np.where(skel)
            dists = np.sqrt((ys - y)**2 + (xs - x)**2)
            nearest_idx = np.argmin(dists)
            nearest_y, nearest_x = ys[nearest_idx], xs[nearest_idx]
            print(f"  Nearest skeleton at ({nearest_y}, {nearest_x}), width = {distance[nearest_y, nearest_x]:.1f} px")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original mask
axes[0, 0].imshow(mask, cmap='gray')
axes[0, 0].set_title('Original Mask', fontsize=12, weight='bold')
for y, x in endpoint_coords:
    axes[0, 0].plot(x, y, 'ro', markersize=10)
    axes[0, 0].text(x + 10, y, f'({y},{x})', color='red', fontsize=9)
axes[0, 0].axis('off')

# Distance transform (width heatmap)
dist_vis = distance * bw
axes[0, 1].imshow(dist_vis, cmap='jet')
axes[0, 1].set_title('Distance Transform\n(brighter = wider)', fontsize=12, weight='bold')
for y, x in endpoint_coords:
    axes[0, 1].plot(x, y, 'wo', markersize=10, markeredgecolor='black', markeredgewidth=2)
axes[0, 1].axis('off')
plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, label='Width (px)')

# Horizontal width profile
axes[1, 0].plot(y_coords, widths, 'b-', linewidth=2)
axes[1, 0].axhline(thin_threshold, color='r', linestyle='--', label=f'Thin threshold ({thin_threshold}px)')
for y, x in endpoint_coords:
    if y_min <= y < y_max:
        idx = y - y_min
        axes[1, 0].plot(y, widths[idx], 'ro', markersize=10)
        axes[1, 0].text(y + 5, widths[idx] + 20, f'Endpoint\n({y},{x})', fontsize=9, color='red')
axes[1, 0].set_xlabel('Y coordinate', fontsize=11)
axes[1, 0].set_ylabel('Horizontal width (px)', fontsize=11)
axes[1, 0].set_title('Horizontal Width Profile', fontsize=12, weight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Zoomed view around endpoints
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
for y, x in endpoint_coords:
    cv2.circle(mask_rgb, (x, y), 8, (255, 0, 0), -1)
    cv2.line(mask_rgb, (endpoint_coords[0][1], endpoint_coords[0][0]), 
             (endpoint_coords[1][1], endpoint_coords[1][0]), (0, 255, 0), 2)

axes[1, 1].imshow(mask_rgb)
axes[1, 1].set_title('Endpoints on Mask', fontsize=12, weight='bold')
axes[1, 1].axis('off')

plt.suptitle(f'{mask_path.name}\nHypothesis: Gaps occur at horizontally thin regions (eye corners)', 
             fontsize=14, weight='bold')
plt.tight_layout()

output_path = Path('horizontal_thin_region_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to: {output_path}")
plt.close()

