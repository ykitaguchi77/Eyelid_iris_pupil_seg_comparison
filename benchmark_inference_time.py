"""
Inference Time & Model Size Benchmark for Methods 1-6
======================================================
Measures:
  - Model parameter count
  - Per-image inference time (forward pass only)
  - Per-image inference + post-processing time (including RANSAC, morphological ops, etc.)
  - GPU memory usage
  - FPS

Uses REAL validation images (fold 0) to ensure post-processing costs
(RANSAC, morphological closing, boundary fitting) are realistic.

Usage:
  python benchmark_inference_time.py
"""

import time
import json
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

# ===== Config =====
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
BATCH_SIZE = 1  # Per-image timing requires batch_size=1
NUM_WARMUP = 20  # GPU warmup iterations
NUM_BENCHMARK = 200  # Number of REAL images to benchmark
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = Path("model/cv_300ep")
FOLD = 0  # Use fold 0 for benchmarking
IMAGE_DIR = Path("Images/images")
LABEL_SEG_DIR = Path("Images/labels_seg")
LABEL_OBB_DIR = Path("Images/labels_obb")

print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

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
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()
        )

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
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_lid = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )
        self.head_iris = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 5)
        )
        self.head_pupil = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 5)
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(
            eyelid_seg=self.head_lid(d),
            iris_ellipse=self.head_iris(d),
            pupil_ellipse=self.head_pupil(d)
        )


class UNetMethod2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_edge = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(edge_logits=self.head_edge(d))


class UNetMethod3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_seg6 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 6, 1)
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(five_class_seg=self.head_seg6(d))


class UNetMethod4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_seg6 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 6, 1)
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(five_class_seg=self.head_seg6(d))


class UNetMethod5(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_seg3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(amodal_logits=self.head_seg3(d))


class UNetMethod6(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_seg4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 4, 1)
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(four_class_seg=self.head_seg4(d))


# ===== Post-processing functions =====

def ellipse_params_to_mask(params, H, W):
    cx, cy = params[0]*W, params[1]*H
    w, h = params[2]*W, params[3]*H
    angle = params[4]*180.0
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(1, int(w/2)), max(1, int(h/2))),
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


def fit_ellipse_from_points(pts_yx):
    """Fit ellipse from (y,x) points. Returns mask or zeros."""
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
    if pts_yx is None or len(pts_yx) < 5:
        return mask
    pts_xy = pts_yx[:, ::-1].astype(np.float32)
    try:
        ellipse = cv2.fitEllipse(pts_xy)
        (cx, cy), (w, h), angle = ellipse
        if w > 0 and h > 0 and w < IMAGE_WIDTH*2 and h < IMAGE_HEIGHT*2:
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                        (max(1, int(round(w/2))), max(1, int(round(h/2)))),
                        float(angle), 0, 360, 255, thickness=-1)
    except Exception:
        pass
    return mask


def boundary_ellipse_fit(pred_labels, region_class, neighbor_class):
    H, W = pred_labels.shape
    region_mask = (pred_labels == region_class).astype(np.uint8)
    neighbor_mask = (pred_labels == neighbor_class).astype(np.uint8)
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((H, W), dtype=np.uint8)
    cnt = max(contours, key=cv2.contourArea)
    kernel = np.ones((5, 5), np.uint8)
    neighbor_dilated = cv2.dilate(neighbor_mask, kernel, iterations=1)
    boundary_pts = []
    for pt in cnt:
        x, y = pt[0]
        if 0 <= y < H and 0 <= x < W and neighbor_dilated[y, x] > 0:
            boundary_pts.append([x, y])
    if len(boundary_pts) < 5:
        return np.zeros((H, W), dtype=np.uint8)
    boundary_pts = np.array(boundary_pts, dtype=np.float32)
    try:
        ellipse = cv2.fitEllipse(boundary_pts)
        (cx, cy), (w, h), angle = ellipse
        if w <= 0 or h <= 0 or w > W*2 or h > H*2:
            return np.zeros((H, W), dtype=np.uint8)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                    (max(1, int(round(w/2))), max(1, int(round(h/2)))),
                    float(angle), 0, 360, 255, thickness=-1)
        return mask
    except Exception:
        return np.zeros((H, W), dtype=np.uint8)


def mask_to_edge(mask_bin, thickness=3):
    contours, _ = cv2.findContours((mask_bin > 0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(mask_bin, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(edge, [cnt], -1, 255, thickness=thickness)
    return edge


# ===== Post-processing per method =====

def postprocess_method1(out):
    lid_pred = (torch.sigmoid(out['eyelid_seg']).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
    iris_params = torch.sigmoid(out['iris_ellipse']).cpu().numpy()[0]
    pupil_params = torch.sigmoid(out['pupil_ellipse']).cpu().numpy()[0]
    iris_mask = ellipse_params_to_mask(iris_params, IMAGE_HEIGHT, IMAGE_WIDTH)
    pupil_mask = ellipse_params_to_mask(pupil_params, IMAGE_HEIGHT, IMAGE_WIDTH)
    return lid_pred, iris_mask, pupil_mask


def postprocess_method2(out):
    edge_logits = out['edge_logits']
    lid_edge = (torch.sigmoid(edge_logits[0, 0:1]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
    iris_edge = (torch.sigmoid(edge_logits[0, 1:2]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
    pupil_edge = (torch.sigmoid(edge_logits[0, 2:3]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
    lid_fill = bin_edge_to_filled(lid_edge)
    iris_pts = np.column_stack(np.where(iris_edge > 0))
    iris_mask = fit_ellipse_from_points(iris_pts)
    pupil_pts = np.column_stack(np.where(pupil_edge > 0))
    pupil_mask = fit_ellipse_from_points(pupil_pts)
    return lid_fill, iris_mask, pupil_mask


def postprocess_method3(out):
    """Method3: 6-class argmax + RANSAC ellipse fitting"""
    from skimage.measure import EllipseModel, ransac
    logits = out['five_class_seg']
    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    lid = ((pred == 1) | (pred == 2) | (pred == 4)).astype(np.uint8) * 255
    iris_raw = ((pred == 2) | (pred == 3)).astype(np.uint8) * 255
    pupil_raw = ((pred == 4) | (pred == 5)).astype(np.uint8) * 255
    # RANSAC ellipse
    iris_edge = mask_to_edge(iris_raw, thickness=3)
    pupil_edge = mask_to_edge(pupil_raw, thickness=3)
    iris_pts = np.column_stack(np.where(iris_edge > 0))
    pupil_pts = np.column_stack(np.where(pupil_edge > 0))
    # Use RANSAC for Method3
    iris_mask = _ransac_ellipse_mask(iris_pts)
    pupil_mask = _ransac_ellipse_mask(pupil_pts)
    return lid, iris_mask, pupil_mask


def _ransac_ellipse_mask(pts_yx):
    """RANSAC ellipse fitting from (y,x) points."""
    from skimage.measure import EllipseModel, ransac
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
    if pts_yx is None or len(pts_yx) < 5:
        return mask
    points = pts_yx[:, ::-1].astype(np.float64)
    try:
        model, inliers = ransac(points, EllipseModel, min_samples=5,
                                 residual_threshold=2.0, max_trials=100)
        if inliers is not None and np.sum(inliers) >= 5:
            final = EllipseModel()
            final.estimate(points[inliers])
            xc, yc, a, b, theta = final.params
            cx_n = np.clip(xc / IMAGE_WIDTH, 0, 1)
            cy_n = np.clip(yc / IMAGE_HEIGHT, 0, 1)
            a_n = np.clip(2*a / IMAGE_WIDTH, 1e-6, 1.0)
            b_n = np.clip(2*b / IMAGE_HEIGHT, 1e-6, 1.0)
            theta_n = (np.degrees(theta) % 180) / 180.0
            params = np.array([cx_n, cy_n, a_n, b_n, theta_n])
            mask = ellipse_params_to_mask(params, IMAGE_HEIGHT, IMAGE_WIDTH)
    except Exception:
        pass
    return mask


def postprocess_method4(out):
    """Method4: same as Method3 but with fullmax contour fitting"""
    logits = out['five_class_seg']
    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    lid = ((pred == 1) | (pred == 2) | (pred == 4)).astype(np.uint8) * 255
    iris_raw = ((pred == 2) | (pred == 3)).astype(np.uint8) * 255
    pupil_raw = ((pred == 4) | (pred == 5)).astype(np.uint8) * 255
    iris_pts = np.column_stack(np.where(iris_raw > 0))
    pupil_pts = np.column_stack(np.where(pupil_raw > 0))
    iris_mask = fit_ellipse_from_points(iris_pts[:, ::-1].reshape(-1, 2)[:, ::-1] if len(iris_pts) > 0 else None)
    pupil_mask = fit_ellipse_from_points(pupil_pts[:, ::-1].reshape(-1, 2)[:, ::-1] if len(pupil_pts) > 0 else None)
    return lid, iris_mask, pupil_mask


def _fit_ellipse_from_binary(mask_bin):
    m = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    try:
        return cv2.fitEllipse(cnt)
    except Exception:
        return None


def postprocess_method5(out):
    """Method5: 3-class amodal sigmoid + fullmax ellipse"""
    probs = torch.sigmoid(out['amodal_logits']).cpu().numpy()[0]
    lid_raw = (probs[0] >= 0.5).astype(np.uint8) * 255
    iris_raw = (probs[1] >= 0.5).astype(np.uint8) * 255
    pupil_raw = (probs[2] >= 0.5).astype(np.uint8) * 255
    # fullmax for iris/pupil
    iris_mask = np.zeros_like(iris_raw)
    ell = _fit_ellipse_from_binary(iris_raw)
    if ell is not None:
        (cx, cy), (w, h), angle = ell
        cv2.ellipse(iris_mask, (int(round(cx)), int(round(cy))),
                    (max(1, int(round(w/2))), max(1, int(round(h/2)))),
                    float(angle), 0, 360, 255, thickness=-1)
    pupil_mask = np.zeros_like(pupil_raw)
    ell = _fit_ellipse_from_binary(pupil_raw)
    if ell is not None:
        (cx, cy), (w, h), angle = ell
        cv2.ellipse(pupil_mask, (int(round(cx)), int(round(cy))),
                    (max(1, int(round(w/2))), max(1, int(round(h/2)))),
                    float(angle), 0, 360, 255, thickness=-1)
    return lid_raw, iris_mask, pupil_mask


def postprocess_method6(out):
    """Method6: 4-class visible + boundary ellipse fitting"""
    logits = out['four_class_seg']
    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    lid = ((pred == 1) | (pred == 2) | (pred == 3)).astype(np.uint8) * 255
    iris_mask = boundary_ellipse_fit(pred, region_class=2, neighbor_class=1)
    pupil_mask = boundary_ellipse_fit(pred, region_class=3, neighbor_class=2)
    return lid, iris_mask, pupil_mask


# ===== Utility =====

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


METHOD_CLASSES = {
    1: UNetMethod1,
    2: UNetMethod2,
    3: UNetMethod3,
    4: UNetMethod4,
    5: UNetMethod5,
    6: UNetMethod6,
}

POSTPROCESS_FNS = {
    1: postprocess_method1,
    2: postprocess_method2,
    3: postprocess_method3,
    4: postprocess_method4,
    5: postprocess_method5,
    6: postprocess_method6,
}


# ===== Main benchmark =====

def benchmark_method(method_id, real_images):
    """Benchmark a single method using real images."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking Method {method_id}")
    print(f"{'='*60}")

    # Load model
    model_cls = METHOD_CLASSES[method_id]
    model = model_cls().to(DEVICE)

    model_path = MODEL_DIR / f"method{method_id}_fold{FOLD}_best.pth"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"  Loaded: {model_path}")
    else:
        print(f"  WARNING: {model_path} not found, using random weights")

    model.eval()

    # Parameter count
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # GPU memory before
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

    postprocess_fn = POSTPROCESS_FNS[method_id]
    n_images = len(real_images)

    # Warmup with real images
    print(f"  Warming up ({NUM_WARMUP} iters with real images)...")
    with torch.no_grad():
        for i in range(NUM_WARMUP):
            img = real_images[i % n_images].to(DEVICE)
            with autocast():
                out = model(img)
            _ = postprocess_fn(out)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    # Benchmark: forward pass only (real images)
    print(f"  Benchmarking forward pass ({n_images} real images)...")
    forward_times = []
    with torch.no_grad():
        for i in range(n_images):
            img = real_images[i].to(DEVICE)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast():
                out = model(img)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            forward_times.append(t1 - t0)

    # Benchmark: forward + postprocess (real images)
    print(f"  Benchmarking forward+postprocess ({n_images} real images)...")
    total_times = []
    postproc_times = []
    with torch.no_grad():
        for i in range(n_images):
            img = real_images[i].to(DEVICE)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast():
                out = model(img)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            # Post-processing (CPU) - this is where RANSAC/morphological ops happen
            _ = postprocess_fn(out)
            t1 = time.perf_counter()
            total_times.append(t1 - t0)
            postproc_times.append(t1 - t_fwd)

    # GPU memory
    gpu_mem_mb = 0
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)

    # Compute stats
    fwd_mean = np.mean(forward_times) * 1000  # ms
    fwd_std = np.std(forward_times) * 1000
    fwd_median = np.median(forward_times) * 1000
    total_mean = np.mean(total_times) * 1000
    total_std = np.std(total_times) * 1000
    total_median = np.median(total_times) * 1000
    fps_fwd = 1000.0 / fwd_mean if fwd_mean > 0 else 0
    fps_total = 1000.0 / total_mean if total_mean > 0 else 0
    postproc_mean = np.mean(postproc_times) * 1000
    postproc_std = np.std(postproc_times) * 1000
    postproc_median = np.median(postproc_times) * 1000

    result = {
        'method': f'Method{method_id}',
        'params_total': total_params,
        'params_M': round(total_params / 1e6, 2),
        'forward_mean_ms': round(fwd_mean, 2),
        'forward_std_ms': round(fwd_std, 2),
        'forward_median_ms': round(fwd_median, 2),
        'postprocess_mean_ms': round(postproc_mean, 2),
        'postprocess_std_ms': round(postproc_std, 2),
        'postprocess_median_ms': round(postproc_median, 2),
        'total_mean_ms': round(total_mean, 2),
        'total_std_ms': round(total_std, 2),
        'total_median_ms': round(total_median, 2),
        'fps_forward': round(fps_fwd, 1),
        'fps_total': round(fps_total, 1),
        'gpu_mem_MB': round(gpu_mem_mb, 1),
    }

    print(f"\n  Results:")
    print(f"    Forward:     {fwd_mean:.2f} +/- {fwd_std:.2f} ms (median {fwd_median:.2f})")
    print(f"    PostProcess: {postproc_mean:.2f} +/- {postproc_std:.2f} ms (median {postproc_median:.2f})")
    print(f"    Total:       {total_mean:.2f} +/- {total_std:.2f} ms (median {total_median:.2f})")
    print(f"    FPS (fwd):   {fps_fwd:.1f}")
    print(f"    FPS (total): {fps_total:.1f}")
    print(f"    GPU Memory:  {gpu_mem_mb:.1f} MB")

    # Cleanup
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return result


def load_real_images():
    """Load real validation images from fold 0."""
    with open("fold_indices.json") as f:
        folds = json.load(f)
    val_indices = folds[str(FOLD)]["val"]

    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    val_paths = [all_images[i] for i in val_indices if i < len(all_images)]

    # Select subset for benchmarking
    selected = val_paths[:NUM_BENCHMARK]
    print(f"  Loading {len(selected)} real validation images...")

    tensors = []
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    for p in selected:
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        img_t = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_t = (img_t - mean) / std
        tensors.append(torch.from_numpy(img_t).float().unsqueeze(0))

    print(f"  Loaded {len(tensors)} images")
    return tensors


def main():
    print("="*60)
    print("  INFERENCE TIME BENCHMARK (Real Images)")
    print(f"  Image size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    print(f"  Warmup: {NUM_WARMUP}, Benchmark: {NUM_BENCHMARK} real images")
    print(f"  Device: {DEVICE}")
    print("="*60)

    # Load real images
    real_images = load_real_images()

    results = []
    for method_id in range(1, 7):
        try:
            r = benchmark_method(method_id, real_images)
            results.append(r)
        except Exception as e:
            print(f"  ERROR benchmarking Method{method_id}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path("results") / f"benchmark_inference_time_{timestamp}.csv"

    fieldnames = ['method', 'params_total', 'params_M',
                  'forward_mean_ms', 'forward_std_ms', 'forward_median_ms',
                  'postprocess_mean_ms', 'postprocess_std_ms', 'postprocess_median_ms',
                  'total_mean_ms', 'total_std_ms', 'total_median_ms',
                  'fps_forward', 'fps_total', 'gpu_mem_MB']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Method':<10} {'Params':>8} {'Fwd(ms)':>10} {'Post(ms)':>10} {'Total(ms)':>10} {'FPS':>8} {'GPU(MB)':>10}")
    print(f"  {'-'*68}")
    for r in results:
        print(f"  {r['method']:<10} {r['params_M']:>7.2f}M {r['forward_mean_ms']:>9.2f} {r['postprocess_mean_ms']:>9.2f} {r['total_mean_ms']:>9.2f} {r['fps_total']:>7.1f} {r['gpu_mem_MB']:>9.1f}")

    print(f"\n  Results saved to: {csv_path}")
    print(f"  Benchmark completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
