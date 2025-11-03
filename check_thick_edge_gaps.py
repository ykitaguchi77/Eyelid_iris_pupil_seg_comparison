"""Check if thicker edges still have gaps."""
from pathlib import Path
import cv2
import numpy as np


mask_path = Path("Images/labels_seg/10-20170517-84-125849_4d2044d03d7d74093f5e898fa59d8bae18cc2ab4e3a5cd9f015a982ac798852d_L_mask_lid.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
bw = (mask > 0).astype(np.uint8)

contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Original contour: {len(contours[0])} points")

first_pt = contours[0][0][0]
last_pt = contours[0][-1][0]
dist = np.linalg.norm(first_pt - last_pt)
print(f"Contour closure: First={first_pt}, Last={last_pt}, Distance={dist:.2f}px\n")

# Check edge connectivity for different thicknesses
for t in [1, 3, 5, 7]:
    edge = np.zeros_like(bw, dtype=np.uint8)
    cv2.drawContours(edge, contours, -1, 255, thickness=t)
    
    # Count edge pixels with only 1 neighbor (potential breaks)
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_bin = (edge > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(edge_bin, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count = neighbor_count - edge_bin  # Subtract self
    
    endpoints = (edge_bin > 0) & (neighbor_count == 1)
    n_endpoints = np.count_nonzero(endpoints)
    
    print(f"thickness={t}px:")
    print(f"  Total edge pixels: {np.count_nonzero(edge)}")
    print(f"  Pixels with 1 neighbor (endpoints): {n_endpoints}")
    
    if n_endpoints > 0:
        ys, xs = np.where(endpoints)
        print(f"  Endpoint positions: {list(zip(ys[:5], xs[:5]))}")
    print()

