"""
Generate publication-quality figures for the paper.
===================================================
Figures:
  Fig1: Ground truth label pipeline (original + masks + 6cls + 4cls)
  Fig2: 6-method comparison on failure case (M1 pupil=0)
  Fig3: 6-method comparison on typical case
  Fig4: Method5 vs Method6 detail (amodal vs visible boundary)

Usage:
  python generate_paper_figures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ===== Config =====
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = Path("model/cv_300ep")
IMAGE_DIR = Path("Images/images")
LABEL_SEG_DIR = Path("Images/labels_seg")
LABEL_OBB_DIR = Path("Images/labels_obb")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Selected images (identified from per-image CSVs)
# Failure case: M1 pupil=0.21, iris=0.74, lid=0.98 (typical frontal image, M5 recovers to 0.96)
FAILURE_IMG = "129-20160727-72-093155_c56eb6f06690e6e44bd1c1814750171c8cca6d84a5afe7c3c9d707e4956784a3_R.jpg"
FAILURE_FOLD = 3

# Typical case: M5 mean near median (~0.97)
TYPICAL_IMG = "133-20021210-3-120859_05f0c3ba1b9c74e0790b69a37511b44d8a97ac45b254cfc21c82b91fdac5b106_L.jpg"
TYPICAL_FOLD = 3

# Success case: M5 best (0.9938)
SUCCESS_IMG = "195-20120627-37-105245_d6399ac7ca14fe9ea80c170f4de8f6e8ff9427d24e93486fdc08cc2dbd73dc9c_L.jpg"
SUCCESS_FOLD = 4

# 6-class color map
SIXCLS_COLORS = {
    0: (0, 0, 0),        # background - black
    1: (255, 0, 0),      # conjunctiva - red
    2: (0, 255, 0),      # iris_vis - green
    3: (0, 0, 255),      # iris_occ - blue
    4: (0, 255, 255),    # pupil_vis - cyan
    5: (255, 0, 255),    # pupil_occ - magenta
}

SIXCLS_NAMES = {
    0: 'Background', 1: 'Conjunctiva', 2: 'Iris (vis)',
    3: 'Iris (occ)', 4: 'Pupil (vis)', 5: 'Pupil (occ)'
}

FOURCLS_COLORS = {
    0: (0, 0, 0),        # background
    1: (255, 0, 0),      # conjunctiva
    2: (0, 255, 0),      # visible iris
    3: (0, 255, 255),    # visible pupil
}

FOURCLS_NAMES = {
    0: 'Background', 1: 'Conjunctiva', 2: 'Vis. Iris', 3: 'Vis. Pupil'
}


# ===== Model Definitions (same as crossvalidation.ipynb) =====

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16_bn(weights='DEFAULT')
        self.features = vgg.features
    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 5:  feats['0'] = x
            if i == 12: feats['1'] = x
            if i == 22: feats['2'] = x
            if i == 32: feats['3'] = x
        feats['4'] = x
        return feats

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec4 = self._blk(512+512, 256)
        self.dec3 = self._blk(256+256, 128)
        self.dec2 = self._blk(128+128, 64)
        self.dec1 = self._blk(64+64,   64)
    def _blk(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU())
    def forward(self, f):
        x = f['4']
        x = F.interpolate(x, size=f['3'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec4(torch.cat([x, f['3']], 1))
        x = F.interpolate(x, size=f['2'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, f['2']], 1))
        x = F.interpolate(x, size=f['1'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, f['1']], 1))
        x = F.interpolate(x, size=f['0'].shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, f['0']], 1))
        x = F.interpolate(x, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode='bilinear', align_corners=False)
        return x

class UNetMethod1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(); self.decoder = UNetDecoder()
        self.head_lid = nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,1,1))
        self.head_iris = nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,5))
        self.head_pupil = nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,5))
    def forward(self, x):
        d = self.decoder(self.encoder(x))
        return dict(eyelid_seg=self.head_lid(d), iris_ellipse=self.head_iris(d), pupil_ellipse=self.head_pupil(d))

class UNetMethod2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(); self.decoder = UNetDecoder()
        self.head_edge = nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,3,1))
    def forward(self, x):
        return dict(edge_logits=self.head_edge(self.decoder(self.encoder(x))))

class UNetMethod3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(); self.decoder = UNetDecoder()
        self.head_seg6 = nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,6,1))
    def forward(self, x):
        return dict(five_class_seg=self.head_seg6(self.decoder(self.encoder(x))))

class UNetMethod4(UNetMethod3):
    pass

class UNetMethod5(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(); self.decoder = UNetDecoder()
        self.head_seg3 = nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,3,1))
    def forward(self, x):
        return dict(amodal_logits=self.head_seg3(self.decoder(self.encoder(x))))

class UNetMethod6(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(); self.decoder = UNetDecoder()
        self.head_seg4 = nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,4,1))
    def forward(self, x):
        return dict(four_class_seg=self.head_seg4(self.decoder(self.encoder(x))))

METHOD_CLASSES = {1: UNetMethod1, 2: UNetMethod2, 3: UNetMethod3,
                  4: UNetMethod4, 5: UNetMethod5, 6: UNetMethod6}


# ===== Helper functions =====

def load_image_tensor(filename):
    """Load and preprocess image for model input."""
    path = IMAGE_DIR / filename
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_display = img.copy()
    img_t = img.astype(np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_t = (img_t - mean) / std
    tensor = torch.from_numpy(img_t).float().unsqueeze(0).to(DEVICE)
    return tensor, img_display


def load_gt_masks(filename):
    """Load ground truth masks."""
    stem = Path(filename).stem
    lid = cv2.imread(str(LABEL_SEG_DIR / f"{stem}_mask_lid.png"), 0)
    iris = cv2.imread(str(LABEL_OBB_DIR / f"{stem}_mask_iris.png"), 0)
    pupil = cv2.imread(str(LABEL_OBB_DIR / f"{stem}_mask_pupil.png"), 0)
    lid = cv2.resize(lid, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    iris = cv2.resize(iris, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    pupil = cv2.resize(pupil, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    return lid, iris, pupil


def load_gt_sixcls(filename):
    """Load 6-class ground truth."""
    stem = Path(filename).stem
    sixcls_img = cv2.imread(str(LABEL_SEG_DIR / f"{stem}_sixcls.png"))
    if sixcls_img is None:
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    sixcls_img = cv2.resize(sixcls_img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    sixcls_img = cv2.cvtColor(sixcls_img, cv2.COLOR_BGR2RGB)
    gt = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    gt[(sixcls_img[:,:,0] > 128) & (sixcls_img[:,:,1] < 128) & (sixcls_img[:,:,2] < 128)] = 1  # red=conj
    gt[(sixcls_img[:,:,0] < 128) & (sixcls_img[:,:,1] > 128) & (sixcls_img[:,:,2] < 128)] = 2  # green=iris_vis
    gt[(sixcls_img[:,:,0] < 128) & (sixcls_img[:,:,1] < 128) & (sixcls_img[:,:,2] > 128)] = 3  # blue=iris_occ
    gt[(sixcls_img[:,:,0] < 128) & (sixcls_img[:,:,1] > 128) & (sixcls_img[:,:,2] > 128)] = 4  # cyan=pupil_vis
    gt[(sixcls_img[:,:,0] > 128) & (sixcls_img[:,:,1] < 128) & (sixcls_img[:,:,2] > 128)] = 5  # magenta=pupil_occ
    return gt


def load_model(method_id, fold):
    """Load a trained model."""
    model = METHOD_CLASSES[method_id]().to(DEVICE)
    path = MODEL_DIR / f"method{method_id}_fold{fold}_best.pth"
    if path.exists():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model'])
    model.eval()
    return model


def ellipse_params_to_mask(params, H, W):
    cx, cy = params[0]*W, params[1]*H
    w, h = params[2]*W, params[3]*H
    angle = params[4]*180.0
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(1,int(w/2)), max(1,int(h/2))),
                angle, 0, 360, 255, thickness=-1)
    return mask


def bin_edge_to_filled(edge_bin):
    edge_uint8 = (edge_bin > 0).astype(np.uint8)
    kernel = np.ones((25, 25), np.uint8)
    closed = cv2.morphologyEx(edge_uint8, cv2.MORPH_CLOSE, kernel, iterations=6)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edge_bin, dtype=np.uint8)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
    return filled


def fit_ellipse_from_binary(mask_bin):
    m = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5: return None
    try: return cv2.fitEllipse(cnt)
    except: return None


def ellipse_to_mask(ellipse, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    if ellipse is None: return mask
    (cx, cy), (w, h), angle = ellipse
    if w <= 0 or h <= 0 or w > W*2 or h > H*2: return mask
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (max(1,int(round(w/2))), max(1,int(round(h/2)))),
                float(angle), 0, 360, 255, thickness=-1)
    return mask


def boundary_ellipse_fit(pred_labels, region_class, neighbor_class):
    H, W = pred_labels.shape
    region_mask = (pred_labels == region_class).astype(np.uint8)
    neighbor_mask = (pred_labels == neighbor_class).astype(np.uint8)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return np.zeros((H,W), np.uint8), None
    cnt = max(contours, key=cv2.contourArea)
    kernel = np.ones((5,5), np.uint8)
    neighbor_dilated = cv2.dilate(neighbor_mask, kernel, iterations=1)
    boundary_pts = []
    for pt in cnt:
        x, y = pt[0]
        if 0 <= y < H and 0 <= x < W and neighbor_dilated[y, x] > 0:
            boundary_pts.append([x, y])
    if len(boundary_pts) < 5: return np.zeros((H,W), np.uint8), None
    boundary_pts = np.array(boundary_pts, dtype=np.float32)
    try:
        ellipse = cv2.fitEllipse(boundary_pts)
        (cx, cy), (w, h), angle = ellipse
        if w <= 0 or h <= 0 or w > W*2 or h > H*2:
            return np.zeros((H,W), np.uint8), None
        mask = np.zeros((H,W), np.uint8)
        cv2.ellipse(mask, (int(round(cx)),int(round(cy))),
                    (max(1,int(round(w/2))),max(1,int(round(h/2)))),
                    float(angle), 0, 360, 255, thickness=-1)
        return mask, ellipse
    except: return np.zeros((H,W), np.uint8), None


# ===== Predict functions per method =====

@torch.no_grad()
def predict_method(method_id, model, img_tensor):
    """Run inference and return (lid_mask, iris_mask, pupil_mask) as 0/255."""
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH
    with autocast():
        out = model(img_tensor)

    if method_id == 1:
        lid = (torch.sigmoid(out['eyelid_seg']).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
        iris_p = torch.sigmoid(out['iris_ellipse']).cpu().numpy()[0]
        pupil_p = torch.sigmoid(out['pupil_ellipse']).cpu().numpy()[0]
        iris = ellipse_params_to_mask(iris_p, H, W)
        pupil = ellipse_params_to_mask(pupil_p, H, W)

    elif method_id == 2:
        edge = out['edge_logits']
        lid_e = (torch.sigmoid(edge[0,0]).cpu().numpy() >= 0.5).astype(np.uint8)*255
        iris_e = (torch.sigmoid(edge[0,1]).cpu().numpy() >= 0.5).astype(np.uint8)*255
        pupil_e = (torch.sigmoid(edge[0,2]).cpu().numpy() >= 0.5).astype(np.uint8)*255
        lid = bin_edge_to_filled(lid_e)
        iris_pts = np.column_stack(np.where(iris_e > 0))
        iris = ellipse_to_mask(fit_ellipse_from_binary(iris_e), H, W) if len(iris_pts) >= 5 else np.zeros((H,W), np.uint8)
        pupil_pts = np.column_stack(np.where(pupil_e > 0))
        pupil = ellipse_to_mask(fit_ellipse_from_binary(pupil_e), H, W) if len(pupil_pts) >= 5 else np.zeros((H,W), np.uint8)

    elif method_id in (3, 4):
        pred = torch.argmax(out['five_class_seg'], dim=1).cpu().numpy()[0]
        lid = ((pred==1)|(pred==2)|(pred==4)).astype(np.uint8)*255
        iris_raw = ((pred==2)|(pred==3)).astype(np.uint8)*255
        pupil_raw = ((pred==4)|(pred==5)).astype(np.uint8)*255
        iris = ellipse_to_mask(fit_ellipse_from_binary(iris_raw), H, W)
        pupil = ellipse_to_mask(fit_ellipse_from_binary(pupil_raw), H, W)

    elif method_id == 5:
        probs = torch.sigmoid(out['amodal_logits']).cpu().numpy()[0]
        lid = (probs[0] >= 0.5).astype(np.uint8)*255
        iris_raw = (probs[1] >= 0.5).astype(np.uint8)*255
        pupil_raw = (probs[2] >= 0.5).astype(np.uint8)*255
        iris = ellipse_to_mask(fit_ellipse_from_binary(iris_raw), H, W)
        pupil = ellipse_to_mask(fit_ellipse_from_binary(pupil_raw), H, W)

    elif method_id == 6:
        pred = torch.argmax(out['four_class_seg'], dim=1).cpu().numpy()[0]
        lid = ((pred==1)|(pred==2)|(pred==3)).astype(np.uint8)*255
        iris, _ = boundary_ellipse_fit(pred, region_class=2, neighbor_class=1)
        pupil, _ = boundary_ellipse_fit(pred, region_class=3, neighbor_class=2)

    return lid, iris, pupil


def dice_binary(pred, gt):
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    inter = (p & g).sum()
    union = p.sum() + g.sum()
    return (2*inter + 1e-6) / (union + 1e-6)


# ===== Visualization helpers =====

def overlay_masks(img_rgb, lid, iris, pupil, alpha=0.35):
    """Create overlay with colored masks on image."""
    overlay = img_rgb.copy().astype(np.float32)
    # Eyelid: semi-transparent yellow contour area
    lid_color = np.array([255, 255, 0], dtype=np.float32)
    iris_color = np.array([0, 200, 0], dtype=np.float32)
    pupil_color = np.array([0, 200, 255], dtype=np.float32)

    for mask, color in [(lid, lid_color), (iris, iris_color), (pupil, pupil_color)]:
        m = (mask > 0).astype(np.float32)[:, :, None]
        overlay = overlay * (1 - m * alpha) + m * alpha * color[None, None, :]

    # Draw contours
    for mask, color_bgr in [(lid, (0,255,255)), (iris, (0,200,0)), (pupil, (0,200,255))]:
        contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay.astype(np.uint8), contours, -1, color_bgr, 2)
        overlay = overlay.astype(np.float32)  # keep float for next iteration

    return np.clip(overlay, 0, 255).astype(np.uint8)


def sixcls_to_rgb(labels):
    """Convert 6-class label map to RGB image."""
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in SIXCLS_COLORS.items():
        rgb[labels == cls_id] = color
    return rgb


def fourcls_to_rgb(labels):
    """Convert 4-class label map to RGB image."""
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, color in FOURCLS_COLORS.items():
        rgb[labels == cls_id] = color
    return rgb


# ===== Figure generation =====

def generate_fig1_label_pipeline(filename):
    """Fig1: Ground truth label pipeline visualization."""
    print("Generating Fig1: Label pipeline...")
    img_t, img_rgb = load_image_tensor(filename)
    gt_lid, gt_iris, gt_pupil = load_gt_masks(filename)
    gt_sixcls = load_gt_sixcls(filename)

    # Derive 4-class from 6-class
    gt_fourcls = np.zeros_like(gt_sixcls)
    gt_fourcls[gt_sixcls == 1] = 1
    gt_fourcls[gt_sixcls == 2] = 2
    gt_fourcls[gt_sixcls == 4] = 3

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Original, Eyelid, Iris, Pupil
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('(a) Original Image', fontsize=14)

    axes[0, 1].imshow(gt_lid, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('(b) Eyelid Mask', fontsize=14)

    axes[0, 2].imshow(gt_iris, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('(c) Iris Mask (amodal)', fontsize=14)

    axes[0, 3].imshow(gt_pupil, cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title('(d) Pupil Mask (amodal)', fontsize=14)

    # Row 2: 6-class, 4-class, overlay GT, legend
    axes[1, 0].imshow(sixcls_to_rgb(gt_sixcls))
    axes[1, 0].set_title('(e) 6-Class Label (Method3-5)', fontsize=14)

    axes[1, 1].imshow(fourcls_to_rgb(gt_fourcls))
    axes[1, 1].set_title('(f) 4-Class Label (Method6)', fontsize=14)

    gt_overlay = overlay_masks(img_rgb, gt_lid, gt_iris, gt_pupil)
    axes[1, 2].imshow(gt_overlay)
    axes[1, 2].set_title('(g) Ground Truth Overlay', fontsize=14)

    # Legend panel
    axes[1, 3].axis('off')
    legend_items_6cls = [mpatches.Patch(facecolor=np.array(c)/255, label=SIXCLS_NAMES[k])
                         for k, c in SIXCLS_COLORS.items() if k > 0]
    axes[1, 3].legend(handles=legend_items_6cls, loc='center', fontsize=12,
                      title='6-Class Labels', title_fontsize=13, frameon=True)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_label_pipeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_fig2_method_comparison(filename, fold, fig_name, title_prefix):
    """Fig2/3: 6-method comparison on a single image."""
    print(f"Generating {fig_name}: {title_prefix}...")
    img_t, img_rgb = load_image_tensor(filename)
    gt_lid, gt_iris, gt_pupil = load_gt_masks(filename)

    method_names = {
        1: 'M1: Ellipse Reg.',
        2: 'M2: Edge Seg.',
        3: 'M3: 6cls (Dice)',
        4: 'M4: 6cls (CE+Dice)',
        5: 'M5: Amodal (BCE+Dice)',
        6: 'M6: Visible+Boundary',
    }

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # (0,0): Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original', fontsize=13)

    # (0,1): Ground Truth
    gt_overlay = overlay_masks(img_rgb, gt_lid, gt_iris, gt_pupil)
    axes[0, 1].imshow(gt_overlay)
    axes[0, 1].set_title('Ground Truth', fontsize=13)

    # Methods 1-6
    positions = [(0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]
    for method_id, (r, c) in zip(range(1, 7), positions):
        model = load_model(method_id, fold)
        lid, iris, pupil = predict_method(method_id, model, img_t)
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

        pred_overlay = overlay_masks(img_rgb, lid, iris, pupil)
        d_lid = dice_binary(lid, gt_lid)
        d_iris = dice_binary(iris, gt_iris)
        d_pupil = dice_binary(pupil, gt_pupil)
        d_mean = (d_lid + d_iris + d_pupil) / 3

        axes[r, c].imshow(pred_overlay)
        subtitle = f'{method_names[method_id]}\nDice: L={d_lid:.3f} I={d_iris:.3f} P={d_pupil:.3f} (Mean={d_mean:.3f})'
        axes[r, c].set_title(subtitle, fontsize=11)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title_prefix, fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{fig_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_fig2_pipeline(filename, fold):
    """Fig2: Segmentation & fitting pipeline for all methods (M1, M2, M4, M5, M6).
    5 rows (one per method) x 3 columns (raw output, post-processing, final overlay)."""
    print("Generating Fig2: Segmentation pipeline for all methods...")
    img_t, img_rgb = load_image_tensor(filename)
    gt_lid, gt_iris, gt_pupil = load_gt_masks(filename)
    gt_sixcls = load_gt_sixcls(filename)
    H, W = IMAGE_HEIGHT, IMAGE_WIDTH

    # --- Run all models ---
    results = {}

    # M1: Ellipse regression
    model1 = load_model(1, fold)
    with torch.no_grad():
        with autocast():
            out1 = model1(img_t)
    lid1 = (torch.sigmoid(out1['eyelid_seg']).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
    iris1_p = torch.sigmoid(out1['iris_ellipse']).cpu().numpy()[0]
    pupil1_p = torch.sigmoid(out1['pupil_ellipse']).cpu().numpy()[0]
    iris1 = ellipse_params_to_mask(iris1_p, H, W)
    pupil1 = ellipse_params_to_mask(pupil1_p, H, W)
    # Raw output: eyelid mask + ellipse param text
    m1_raw = np.zeros((H, W, 3), dtype=np.uint8)
    m1_raw[lid1 > 0] = [255, 255, 0]
    # Draw predicted ellipses as outlines
    cx_i, cy_i = iris1_p[0]*W, iris1_p[1]*H
    ax_i, bx_i = max(1,int(iris1_p[2]*W/2)), max(1,int(iris1_p[3]*H/2))
    cv2.ellipse(m1_raw, (int(cx_i),int(cy_i)), (ax_i,bx_i), iris1_p[4]*180, 0, 360, (0,255,0), 2)
    cx_p, cy_p = pupil1_p[0]*W, pupil1_p[1]*H
    ax_p, bx_p = max(1,int(pupil1_p[2]*W/2)), max(1,int(pupil1_p[3]*H/2))
    cv2.ellipse(m1_raw, (int(cx_p),int(cy_p)), (ax_p,bx_p), pupil1_p[4]*180, 0, 360, (0,255,255), 2)
    # Intermediate: rendered ellipses
    m1_inter = np.zeros((H, W, 3), dtype=np.uint8)
    m1_inter[lid1 > 0] = [255, 255, 0]
    m1_inter[iris1 > 0] = [0, 200, 0]
    m1_inter[pupil1 > 0] = [0, 200, 255]
    results['M1'] = (m1_raw, m1_inter, lid1, iris1, pupil1)
    del model1

    # M2: Edge segmentation
    model2 = load_model(2, fold)
    with torch.no_grad():
        with autocast():
            out2 = model2(img_t)
    edge = out2['edge_logits']
    lid2_e = (torch.sigmoid(edge[0,0]).cpu().numpy() >= 0.5).astype(np.uint8)*255
    iris2_e = (torch.sigmoid(edge[0,1]).cpu().numpy() >= 0.5).astype(np.uint8)*255
    pupil2_e = (torch.sigmoid(edge[0,2]).cpu().numpy() >= 0.5).astype(np.uint8)*255
    lid2 = bin_edge_to_filled(lid2_e)
    iris2 = ellipse_to_mask(fit_ellipse_from_binary(iris2_e), H, W)
    pupil2 = ellipse_to_mask(fit_ellipse_from_binary(pupil2_e), H, W)
    # Raw: 3ch edge map (color-coded)
    m2_raw = np.zeros((H, W, 3), dtype=np.uint8)
    m2_raw[lid2_e > 0] = [255, 255, 0]
    m2_raw[iris2_e > 0] = [0, 255, 0]
    m2_raw[pupil2_e > 0] = [0, 255, 255]
    # Intermediate: filled + ellipse outlines
    m2_inter = np.zeros((H, W, 3), dtype=np.uint8)
    m2_inter[lid2 > 0] = [255, 255, 0]
    ell_i = fit_ellipse_from_binary(iris2_e)
    ell_p = fit_ellipse_from_binary(pupil2_e)
    if ell_i:
        (cx,cy),(w,h),a = ell_i
        cv2.ellipse(m2_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (0,255,0), -1)
    if ell_p:
        (cx,cy),(w,h),a = ell_p
        cv2.ellipse(m2_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (0,255,255), -1)
    results['M2'] = (m2_raw, m2_inter, lid2, iris2, pupil2)
    del model2

    # M4: 6-class (CE+Dice) - representative of M3/M4
    model4 = load_model(4, fold)
    with torch.no_grad():
        with autocast():
            out4 = model4(img_t)
    pred4 = torch.argmax(out4['five_class_seg'], dim=1).cpu().numpy()[0]
    lid4 = ((pred4==1)|(pred4==2)|(pred4==4)).astype(np.uint8)*255
    iris4_raw = ((pred4==2)|(pred4==3)).astype(np.uint8)*255
    pupil4_raw = ((pred4==4)|(pred4==5)).astype(np.uint8)*255
    iris4 = ellipse_to_mask(fit_ellipse_from_binary(iris4_raw), H, W)
    pupil4 = ellipse_to_mask(fit_ellipse_from_binary(pupil4_raw), H, W)
    # Raw: 6-class map
    m4_raw = sixcls_to_rgb(pred4)
    # Intermediate: merged masks + ellipse contours
    m4_inter = np.zeros((H, W, 3), dtype=np.uint8)
    m4_inter[lid4 > 0] = [255, 255, 0]
    m4_inter[iris4_raw > 0] = [0, 150, 0]
    m4_inter[pupil4_raw > 0] = [0, 150, 200]
    # Draw fitted ellipse outlines
    ell_i4 = fit_ellipse_from_binary(iris4_raw)
    ell_p4 = fit_ellipse_from_binary(pupil4_raw)
    if ell_i4:
        (cx,cy),(w,h),a = ell_i4
        cv2.ellipse(m4_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (0,255,0), 2)
    if ell_p4:
        (cx,cy),(w,h),a = ell_p4
        cv2.ellipse(m4_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (0,255,255), 2)
    results['M4'] = (m4_raw, m4_inter, lid4, iris4, pupil4)
    del model4

    # M5: Amodal 3ch sigmoid
    model5 = load_model(5, fold)
    with torch.no_grad():
        with autocast():
            out5 = model5(img_t)
    probs5 = torch.sigmoid(out5['amodal_logits']).cpu().numpy()[0]
    lid5 = (probs5[0] >= 0.5).astype(np.uint8)*255
    iris5_raw = (probs5[1] >= 0.5).astype(np.uint8)*255
    pupil5_raw = (probs5[2] >= 0.5).astype(np.uint8)*255
    iris5 = ellipse_to_mask(fit_ellipse_from_binary(iris5_raw), H, W)
    pupil5 = ellipse_to_mask(fit_ellipse_from_binary(pupil5_raw), H, W)
    # Raw: 3ch amodal masks (overlapping allowed)
    m5_raw = np.zeros((H, W, 3), dtype=np.uint8)
    m5_raw[lid5 > 0] = [255, 255, 0]
    m5_raw[iris5_raw > 0] = [0, 200, 0]
    m5_raw[pupil5_raw > 0] = [0, 200, 255]
    # Intermediate: ellipse contours on raw masks
    m5_inter = m5_raw.copy()
    ell_i5 = fit_ellipse_from_binary(iris5_raw)
    ell_p5 = fit_ellipse_from_binary(pupil5_raw)
    if ell_i5:
        (cx,cy),(w,h),a = ell_i5
        cv2.ellipse(m5_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (255,255,255), 2)
    if ell_p5:
        (cx,cy),(w,h),a = ell_p5
        cv2.ellipse(m5_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (255,255,255), 2)
    results['M5'] = (m5_raw, m5_inter, lid5, iris5, pupil5)
    del model5

    # M6: 4-class visible + boundary fitting
    model6 = load_model(6, fold)
    with torch.no_grad():
        with autocast():
            out6 = model6(img_t)
    pred6 = torch.argmax(out6['four_class_seg'], dim=1).cpu().numpy()[0]
    lid6 = ((pred6==1)|(pred6==2)|(pred6==3)).astype(np.uint8)*255
    iris6, iris6_ell = boundary_ellipse_fit(pred6, 2, 1)
    pupil6, pupil6_ell = boundary_ellipse_fit(pred6, 3, 2)
    # Raw: 4-class map
    m6_raw = fourcls_to_rgb(pred6)
    # Intermediate: boundary points visualization
    m6_inter = img_rgb.copy()
    # Show visible iris region
    vis_iris = (pred6 == 2).astype(np.uint8)
    vis_pupil = (pred6 == 3).astype(np.uint8)
    m6_inter[vis_iris > 0] = np.clip(
        m6_inter[vis_iris > 0].astype(int) * 0.5 + np.array([0, 128, 0]), 0, 255).astype(np.uint8)
    m6_inter[vis_pupil > 0] = np.clip(
        m6_inter[vis_pupil > 0].astype(int) * 0.5 + np.array([0, 128, 128]), 0, 255).astype(np.uint8)
    # Draw boundary points for iris
    conj_mask = (pred6 == 1).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    conj_dilated = cv2.dilate(conj_mask, kernel, iterations=1)
    iris_mask6 = (pred6 == 2).astype(np.uint8)
    contours_i, _ = cv2.findContours(iris_mask6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_i:
        cnt = max(contours_i, key=cv2.contourArea)
        for pt in cnt:
            x, y = pt[0]
            if 0 <= y < H and 0 <= x < W:
                if conj_dilated[y, x] > 0:
                    cv2.circle(m6_inter, (x, y), 2, (255, 0, 0), -1)  # boundary=red
                else:
                    cv2.circle(m6_inter, (x, y), 1, (128, 128, 128), -1)  # non-boundary=gray
    # Draw boundary points for pupil
    iris_dilated = cv2.dilate(iris_mask6, kernel, iterations=1)
    pupil_mask6 = (pred6 == 3).astype(np.uint8)
    contours_p, _ = cv2.findContours(pupil_mask6, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_p:
        cnt = max(contours_p, key=cv2.contourArea)
        for pt in cnt:
            x, y = pt[0]
            if 0 <= y < H and 0 <= x < W:
                if iris_dilated[y, x] > 0:
                    cv2.circle(m6_inter, (x, y), 2, (0, 0, 255), -1)  # boundary=blue
    # Draw fitted ellipses
    if iris6_ell:
        (cx,cy),(w,h),a = iris6_ell
        cv2.ellipse(m6_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (255,0,0), 2)
    if pupil6_ell:
        (cx,cy),(w,h),a = pupil6_ell
        cv2.ellipse(m6_inter, (int(cx),int(cy)), (max(1,int(w/2)),max(1,int(h/2))), a, 0, 360, (0,0,255), 2)
    results['M6'] = (m6_raw, m6_inter, lid6, iris6, pupil6)
    del model6
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    # --- Create figure: 5 rows x 4 cols ---
    # Col0: Original/GT, Col1: Raw output, Col2: Post-processing, Col3: Final overlay
    method_order = ['M1', 'M2', 'M6', 'M4', 'M5']
    method_labels = {
        'M1': 'M1: Ellipse\nRegression',
        'M2': 'M2: Edge\nSegmentation',
        'M4': 'M4: 6-Class\nRegion Seg.',
        'M5': 'M5: 3-Ch Amodal\n(BCE+Dice)',
        'M6': 'M6: 4-Class\nVisible+Boundary',
    }
    col_titles = ['Raw Model Output', 'Post-Processing', 'Final Result']

    fig, axes = plt.subplots(5, 4, figsize=(22, 26))

    for row, mkey in enumerate(method_order):
        raw_img, inter_img, lid, iris, pupil = results[mkey]
        d_lid = dice_binary(lid, gt_lid)
        d_iris = dice_binary(iris, gt_iris)
        d_pupil = dice_binary(pupil, gt_pupil)
        d_mean = (d_lid + d_iris + d_pupil) / 3

        # Col 0: method label + original or GT
        if row == 0:
            axes[row, 0].imshow(img_rgb)
            axes[row, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        else:
            gt_ov = overlay_masks(img_rgb, gt_lid, gt_iris, gt_pupil)
            axes[row, 0].imshow(gt_ov)
            if row == 1:
                axes[row, 0].set_title('Ground Truth', fontsize=12, fontweight='bold')

        # Method label on left
        axes[row, 0].set_ylabel(method_labels[mkey], fontsize=13, fontweight='bold',
                                rotation=0, labelpad=100, ha='center', va='center')

        # Col 1: Raw output
        axes[row, 1].imshow(raw_img)
        if row == 0:
            axes[row, 1].set_title(col_titles[0], fontsize=12, fontweight='bold')

        # Col 2: Post-processing
        axes[row, 2].imshow(inter_img)
        if row == 0:
            axes[row, 2].set_title(col_titles[1], fontsize=12, fontweight='bold')

        # Col 3: Final overlay
        final_ov = overlay_masks(img_rgb, lid, iris, pupil)
        axes[row, 3].imshow(final_ov)
        dice_text = f'L={d_lid:.3f} I={d_iris:.3f} P={d_pupil:.3f}\nMean={d_mean:.3f}'
        axes[row, 3].text(0.02, 0.98, dice_text, transform=axes[row, 3].transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        if row == 0:
            axes[row, 3].set_title(col_titles[2], fontsize=12, fontweight='bold')

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    out_path = OUTPUT_DIR / "fig2_pipeline.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_fig4_m5_vs_m6(filename, fold):
    """Fig4: Detailed comparison of Method5 (amodal) vs Method6 (visible+boundary)."""
    print("Generating Fig4: Method5 vs Method6 detail...")
    img_t, img_rgb = load_image_tensor(filename)
    gt_lid, gt_iris, gt_pupil = load_gt_masks(filename)
    gt_sixcls = load_gt_sixcls(filename)

    # Method5 prediction
    model5 = load_model(5, fold)
    with torch.no_grad():
        with autocast():
            out5 = model5(img_t)
    probs5 = torch.sigmoid(out5['amodal_logits']).cpu().numpy()[0]
    lid5 = (probs5[0] >= 0.5).astype(np.uint8) * 255
    iris5_raw = (probs5[1] >= 0.5).astype(np.uint8) * 255
    pupil5_raw = (probs5[2] >= 0.5).astype(np.uint8) * 255
    iris5 = ellipse_to_mask(fit_ellipse_from_binary(iris5_raw), IMAGE_HEIGHT, IMAGE_WIDTH)
    pupil5 = ellipse_to_mask(fit_ellipse_from_binary(pupil5_raw), IMAGE_HEIGHT, IMAGE_WIDTH)
    del model5

    # Method6 prediction
    model6 = load_model(6, fold)
    with torch.no_grad():
        with autocast():
            out6 = model6(img_t)
    pred6 = torch.argmax(out6['four_class_seg'], dim=1).cpu().numpy()[0]
    lid6 = ((pred6==1)|(pred6==2)|(pred6==3)).astype(np.uint8)*255
    iris6_vis = (pred6 == 2).astype(np.uint8) * 255  # visible iris only
    iris6, iris6_ell = boundary_ellipse_fit(pred6, 2, 1)
    pupil6, pupil6_ell = boundary_ellipse_fit(pred6, 3, 2)
    del model6
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Row 1: Method5
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original', fontsize=13)

    axes[0, 1].imshow(overlay_masks(img_rgb, gt_lid, gt_iris, gt_pupil))
    axes[0, 1].set_title('Ground Truth', fontsize=13)

    # M5 raw amodal prediction
    m5_raw_vis = np.zeros_like(img_rgb)
    m5_raw_vis[iris5_raw > 0] = [0, 200, 0]
    m5_raw_vis[pupil5_raw > 0] = [0, 200, 255]
    m5_raw_vis[lid5 > 0] = np.clip(m5_raw_vis[lid5 > 0].astype(int) + [80, 80, 0], 0, 255).astype(np.uint8)
    axes[0, 2].imshow(m5_raw_vis)
    axes[0, 2].set_title('M5: Amodal Prediction (raw)', fontsize=13)

    d5 = (dice_binary(lid5, gt_lid) + dice_binary(iris5, gt_iris) + dice_binary(pupil5, gt_pupil)) / 3
    axes[0, 3].imshow(overlay_masks(img_rgb, lid5, iris5, pupil5))
    axes[0, 3].set_title(f'M5: After Ellipse Fit (Mean={d5:.3f})', fontsize=13)

    # Row 2: Method6
    # M6 4-class prediction
    axes[1, 0].imshow(fourcls_to_rgb(pred6))
    axes[1, 0].set_title('M6: 4-Class Prediction', fontsize=13)

    # M6 visible iris with boundary points
    m6_boundary_vis = img_rgb.copy()
    m6_boundary_vis[iris6_vis > 0] = np.clip(
        m6_boundary_vis[iris6_vis > 0].astype(int) * 0.5 + np.array([0, 128, 0]), 0, 255).astype(np.uint8)
    # Draw boundary ellipse
    if iris6_ell is not None:
        (cx, cy), (w, h), angle = iris6_ell
        cv2.ellipse(m6_boundary_vis, (int(cx), int(cy)), (max(1,int(w/2)), max(1,int(h/2))),
                    angle, 0, 360, (255, 0, 0), 2)
    if pupil6_ell is not None:
        (cx, cy), (w, h), angle = pupil6_ell
        cv2.ellipse(m6_boundary_vis, (int(cx), int(cy)), (max(1,int(w/2)), max(1,int(h/2))),
                    angle, 0, 360, (0, 0, 255), 2)
    axes[1, 1].imshow(m6_boundary_vis)
    axes[1, 1].set_title('M6: Visible Region + Boundary Ellipse', fontsize=13)

    # M6 iris detail: show boundary points
    axes[1, 2].imshow(iris6_vis, cmap='gray')
    # Overlay boundary region
    conj_mask = (pred6 == 1).astype(np.uint8)
    kernel = np.ones((5,5), np.uint8)
    conj_dilated = cv2.dilate(conj_mask, kernel, iterations=1)
    iris_mask = (pred6 == 2).astype(np.uint8)
    contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary_img = cv2.cvtColor(iris6_vis, cv2.COLOR_GRAY2RGB)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        for pt in cnt:
            x, y = pt[0]
            if 0 <= y < IMAGE_HEIGHT and 0 <= x < IMAGE_WIDTH:
                if conj_dilated[y, x] > 0:
                    cv2.circle(boundary_img, (x, y), 1, (255, 0, 0), -1)  # boundary=red
                else:
                    cv2.circle(boundary_img, (x, y), 1, (100, 100, 100), -1)  # non-boundary=gray
    axes[1, 2].imshow(boundary_img)
    axes[1, 2].set_title('M6: Iris Boundary Points (red)', fontsize=13)

    d6 = (dice_binary(lid6, gt_lid) + dice_binary(iris6, gt_iris) + dice_binary(pupil6, gt_pupil)) / 3
    axes[1, 3].imshow(overlay_masks(img_rgb, lid6, iris6, pupil6))
    axes[1, 3].set_title(f'M6: After Boundary Fit (Mean={d6:.3f})', fontsize=13)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle('Method5 (Amodal) vs Method6 (Visible + Boundary Fitting)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_m5_vs_m6.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===== Main =====

def main():
    print("="*60)
    print("  PAPER FIGURE GENERATION")
    print(f"  Device: {DEVICE}")
    print("="*60)

    # Fig1: Label pipeline
    generate_fig1_label_pipeline(TYPICAL_IMG)

    # Fig2: Failure case (M1 pupil failure)
    generate_fig2_method_comparison(
        FAILURE_IMG, FAILURE_FOLD,
        "fig2_failure_case",
        "Method Comparison: Failure Case (Method1 Pupil Dice = 0.00)"
    )

    # Fig3: Typical case
    generate_fig2_method_comparison(
        TYPICAL_IMG, TYPICAL_FOLD,
        "fig3_typical_case",
        "Method Comparison: Typical Case"
    )

    # Fig4: M5 vs M6 detail
    generate_fig4_m5_vs_m6(TYPICAL_IMG, TYPICAL_FOLD)

    print("\n" + "="*60)
    print("  All figures saved to:", OUTPUT_DIR)
    print("="*60)


if __name__ == '__main__':
    main()
