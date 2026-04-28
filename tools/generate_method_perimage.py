"""Generate per-image Dice CSVs for Method1, Method2, Method3.

Loads trained models from model/cv_300ep/ and evaluates on each fold's
validation set, saving per-image results to results/.

Usage:
    python tools/generate_method_perimage.py
    python tools/generate_method_perimage.py --methods 1 2   # skip Method3
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
BATCH_SIZE = 16
NUM_WORKERS = 0
PIN_MEMORY = True

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "Images" / "images"
LABEL_SEG_DIR = ROOT / "Images" / "labels_seg"
LABEL_OBB_DIR = ROOT / "Images" / "labels_obb"
MODEL_DIR = ROOT / "model" / "cv_300ep"
RESULTS_DIR = ROOT / "results"
FOLD_INDICES_PATH = ROOT / "fold_indices.json"

# ---------------------------------------------------------------------------
# Model architectures  (from crossvalidation.ipynb)
# ---------------------------------------------------------------------------

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16_bn(weights="DEFAULT")
        self.features = vgg.features

    def forward(self, x):
        feats = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 5:
                feats["0"] = x
            if i == 12:
                feats["1"] = x
            if i == 22:
                feats["2"] = x
            if i == 32:
                feats["3"] = x
        feats["4"] = x
        return feats


class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec4 = self._blk(512 + 512, 256)
        self.dec3 = self._blk(256 + 256, 128)
        self.dec2 = self._blk(128 + 128, 64)
        self.dec1 = self._blk(64 + 64, 64)

    def _blk(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        )

    def forward(self, f):
        x = f["4"]
        x = F.interpolate(x, size=f["3"].shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec4(torch.cat([x, f["3"]], 1))
        x = F.interpolate(x, size=f["2"].shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, f["2"]], 1))
        x = F.interpolate(x, size=f["1"].shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, f["1"]], 1))
        x = F.interpolate(x, size=f["0"].shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, f["0"]], 1))
        x = F.interpolate(x, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False)
        return x


class UNetMethod1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_lid = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )
        self.head_iris = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 5),
        )
        self.head_pupil = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 5),
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(
            eyelid_seg=self.head_lid(d),
            iris_ellipse=self.head_iris(d),
            pupil_ellipse=self.head_pupil(d),
        )


class UNetMethod2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
        self.head_edge = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 3, 1),
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
            nn.Conv2d(32, 6, 1),
        )

    def forward(self, x):
        f = self.encoder(x)
        d = self.decoder(f)
        return dict(five_class_seg=self.head_seg6(d))


MODEL_CLS = {1: UNetMethod1, 2: UNetMethod2, 3: UNetMethod3}

# ---------------------------------------------------------------------------
# Dataset (eval-only, simplified)
# ---------------------------------------------------------------------------

def _resize_mask(mask, H=IMAGE_HEIGHT, W=IMAGE_WIDTH):
    if mask is None:
        return np.zeros((H, W), dtype=np.uint8)
    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)


class EvalDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        stem = p.stem
        mask_lid = cv2.imread(str(LABEL_SEG_DIR / f"{stem}_mask_lid.png"), 0)
        mask_iris = cv2.imread(str(LABEL_OBB_DIR / f"{stem}_mask_iris.png"), 0)
        mask_pupil = cv2.imread(str(LABEL_OBB_DIR / f"{stem}_mask_pupil.png"), 0)
        mask_lid = _resize_mask(mask_lid)
        mask_iris = _resize_mask(mask_iris)
        mask_pupil = _resize_mask(mask_pupil)

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        img_t = (img_t - mean) / std

        return dict(
            image=img_t,
            mask_lid=torch.from_numpy(mask_lid).long(),
            mask_iris=torch.from_numpy(mask_iris).long(),
            mask_pupil=torch.from_numpy(mask_pupil).long(),
            filename=p.name,
        )


# ---------------------------------------------------------------------------
# Evaluation utilities  (from crossvalidation.ipynb)
# ---------------------------------------------------------------------------

def dice_binary_np(pred_bin_255, gt_bin_255, smooth=1e-6):
    p = (pred_bin_255 > 0).astype(np.uint8)
    g = (gt_bin_255 > 0).astype(np.uint8)
    inter = (p & g).sum()
    union = p.sum() + g.sum()
    return float((2 * inter + smooth) / (union + smooth))


def ellipse_params_to_mask(params, H, W):
    cx = params[0] * W
    cy = params[1] * H
    w = params[2] * W
    h = params[3] * H
    angle = params[4] * 180.0
    mask = np.zeros((H, W), dtype=np.uint8)
    center = (int(cx), int(cy))
    axes = (max(1, int(w / 2)), max(1, int(h / 2)))
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, thickness=-1)
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


def mask_to_edge(mask_bin, thickness=3):
    contours, _ = cv2.findContours(
        (mask_bin > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    edge = np.zeros_like(mask_bin, dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(edge, [cnt], -1, 255, thickness=thickness)
    return edge


def fit_ellipse_ransac(edge_points, min_samples=5, residual_threshold=2.0, max_trials=100):
    from skimage.measure import EllipseModel, ransac as sk_ransac

    if len(edge_points) < 5:
        return None
    points = edge_points[:, ::-1].astype(np.float64)
    try:
        model, inliers = sk_ransac(
            points, EllipseModel,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
        if inliers is not None and np.sum(inliers) >= 5:
            final = EllipseModel()
            final.estimate(points[inliers])
            xc, yc, a, b, theta = final.params
            return ((float(xc), float(yc)), (float(2 * a), float(2 * b)), float(np.degrees(theta)))
    except Exception:
        pass
    return None


def binary_to_ellipse_params(mask_bin, residual_threshold=2.0):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) < 5:
        return None
    edge_points = np.column_stack([ys, xs])
    ellipse = fit_ellipse_ransac(edge_points, residual_threshold=residual_threshold, max_trials=100)
    if ellipse is None:
        return None
    (cx, cy), (w, h), angle = ellipse
    cx_n = np.clip(cx / mask_bin.shape[1], 0, 1)
    cy_n = np.clip(cy / mask_bin.shape[0], 0, 1)
    a_n = np.clip(w / mask_bin.shape[1], 1e-6, 1.0)
    b_n = np.clip(h / mask_bin.shape[0], 1e-6, 1.0)
    theta_n = (angle % 180) / 180.0
    return np.array([cx_n, cy_n, a_n, b_n, theta_n], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-image evaluation functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_method1_perimage(model, val_loader, device, fold_idx):
    model.eval()
    rows = []
    for batch in tqdm(val_loader, desc=f"  M1 fold{fold_idx}", leave=False):
        img = batch["image"].to(device)
        gt_lid = batch["mask_lid"].cpu().numpy()
        gt_iris = batch["mask_iris"].cpu().numpy()
        gt_pupil = batch["mask_pupil"].cpu().numpy()
        filenames = batch["filename"]

        with autocast():
            out = model(img)

        for b in range(img.shape[0]):
            lid_pred = (torch.sigmoid(out["eyelid_seg"][b : b + 1]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
            iris_params = torch.sigmoid(out["iris_ellipse"]).cpu().numpy()[b]
            pupil_params = torch.sigmoid(out["pupil_ellipse"]).cpu().numpy()[b]
            iris_mask = ellipse_params_to_mask(iris_params, IMAGE_HEIGHT, IMAGE_WIDTH)
            pupil_mask = ellipse_params_to_mask(pupil_params, IMAGE_HEIGHT, IMAGE_WIDTH)

            lid_d = dice_binary_np(lid_pred, gt_lid[b])
            iris_d = dice_binary_np(iris_mask, gt_iris[b])
            pupil_d = dice_binary_np(pupil_mask, gt_pupil[b])
            rows.append(dict(
                filename=str(filenames[b]),
                subject_id=str(filenames[b]).split("-", 1)[0],
                eyelid=lid_d, iris=iris_d, pupil=pupil_d,
                mean=float(np.mean([lid_d, iris_d, pupil_d])),
                fold=fold_idx,
            ))
    return rows


@torch.no_grad()
def evaluate_method2_perimage(model, val_loader, device, fold_idx):
    model.eval()
    rows = []
    for batch in tqdm(val_loader, desc=f"  M2 fold{fold_idx}", leave=False):
        img = batch["image"].to(device)
        gt_lid = batch["mask_lid"].cpu().numpy()
        gt_iris = batch["mask_iris"].cpu().numpy()
        gt_pupil = batch["mask_pupil"].cpu().numpy()
        filenames = batch["filename"]

        with autocast():
            out = model(img)
            edge_logits = out["edge_logits"]

        for b in range(img.shape[0]):
            lid_edge = (torch.sigmoid(edge_logits[b, 0:1]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
            lid_fill = bin_edge_to_filled(lid_edge)

            iris_edge = (torch.sigmoid(edge_logits[b, 1:2]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255
            pupil_edge = (torch.sigmoid(edge_logits[b, 2:3]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8) * 255

            iris_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
            iris_pts = np.column_stack(np.where(iris_edge > 0))
            if len(iris_pts) >= 5:
                try:
                    ell = cv2.fitEllipse(iris_pts[:, ::-1].astype(np.int32))
                    cv2.ellipse(iris_mask, ell, 255, thickness=-1)
                except Exception:
                    pass

            pupil_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
            pupil_pts = np.column_stack(np.where(pupil_edge > 0))
            if len(pupil_pts) >= 5:
                try:
                    ell = cv2.fitEllipse(pupil_pts[:, ::-1].astype(np.int32))
                    cv2.ellipse(pupil_mask, ell, 255, thickness=-1)
                except Exception:
                    pass

            lid_d = dice_binary_np(lid_fill, gt_lid[b])
            iris_d = dice_binary_np(iris_mask, gt_iris[b])
            pupil_d = dice_binary_np(pupil_mask, gt_pupil[b])
            rows.append(dict(
                filename=str(filenames[b]),
                subject_id=str(filenames[b]).split("-", 1)[0],
                eyelid=lid_d, iris=iris_d, pupil=pupil_d,
                mean=float(np.mean([lid_d, iris_d, pupil_d])),
                fold=fold_idx,
            ))
    return rows


@torch.no_grad()
def evaluate_method3_perimage(model, val_loader, device, fold_idx):
    """Method3: 6-class segmentation with RANSAC whole-mask ellipse fit (default)."""
    model.eval()
    rows = []
    for batch in tqdm(val_loader, desc=f"  M3 fold{fold_idx}", leave=False):
        img = batch["image"].to(device)
        gt_lid = batch["mask_lid"].cpu().numpy()
        gt_iris = batch["mask_iris"].cpu().numpy()
        gt_pupil = batch["mask_pupil"].cpu().numpy()
        filenames = batch["filename"]

        with autocast():
            out = model(img)
            logits = out["five_class_seg"]

        pred_labels = torch.argmax(logits, dim=1).cpu().numpy()

        for b in range(pred_labels.shape[0]):
            pred = pred_labels[b]
            filename = filenames[b]

            # Eyelid: classes 1|2|4
            lid_bin = ((pred == 1) | (pred == 2) | (pred == 4)).astype(np.uint8) * 255
            lid_d = dice_binary_np(lid_bin, gt_lid[b])

            # Iris/Pupil: RANSAC whole-mask (default evaluation)
            iris_raw = ((pred == 2) | (pred == 3)).astype(np.uint8) * 255
            pupil_raw = ((pred == 4) | (pred == 5)).astype(np.uint8) * 255

            iris_edge = mask_to_edge(iris_raw, thickness=3)
            pupil_edge = mask_to_edge(pupil_raw, thickness=3)
            iris_params = binary_to_ellipse_params(iris_edge)
            pupil_params = binary_to_ellipse_params(pupil_edge)

            iris_mask = (
                ellipse_params_to_mask(iris_params, IMAGE_HEIGHT, IMAGE_WIDTH)
                if iris_params is not None
                else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
            )
            pupil_mask = (
                ellipse_params_to_mask(pupil_params, IMAGE_HEIGHT, IMAGE_WIDTH)
                if pupil_params is not None
                else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
            )

            iris_d = dice_binary_np(iris_mask, gt_iris[b])
            pupil_d = dice_binary_np(pupil_mask, gt_pupil[b])
            rows.append(dict(
                filename=str(filename),
                subject_id=str(filename).split("-", 1)[0],
                eyelid=lid_d, iris=iris_d, pupil=pupil_d,
                mean=float(np.mean([lid_d, iris_d, pupil_d])),
                fold=fold_idx,
            ))
    return rows


EVAL_FN = {
    1: evaluate_method1_perimage,
    2: evaluate_method2_perimage,
    3: evaluate_method3_perimage,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate per-image Dice CSVs for Method1/2/3")
    ap.add_argument("--methods", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load fold indices
    with open(FOLD_INDICES_PATH) as f:
        fold_data = json.load(f)

    # Collect all image paths (sorted for reproducibility)
    all_images = sorted(IMAGE_DIR.glob("*.jpg"))
    print(f"Total images found: {len(all_images)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(exist_ok=True)

    for method_id in args.methods:
        print(f"\n{'='*60}")
        print(f"Method {method_id}")
        print(f"{'='*60}")

        all_rows = []
        for fold_idx in range(5):
            fold_key = str(fold_idx)
            val_indices = fold_data[fold_key]["val"]
            val_paths = [all_images[i] for i in val_indices]
            print(f"  Fold {fold_idx}: {len(val_paths)} val images")

            val_ds = EvalDataset(val_paths)
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size,
                shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            )

            # Load model
            model_path = MODEL_DIR / f"method{method_id}_fold{fold_idx}_best.pth"
            if not model_path.exists():
                print(f"  WARNING: {model_path} not found, skipping")
                continue

            model = MODEL_CLS[method_id]()
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.to(device)
            model.eval()

            rows = EVAL_FN[method_id](model, val_loader, device, fold_idx)
            all_rows.extend(rows)

            # Verify: fold-level average
            fold_df = pd.DataFrame(rows)
            print(f"    Eyelid={fold_df['eyelid'].mean():.4f}  "
                  f"Iris={fold_df['iris'].mean():.4f}  "
                  f"Pupil={fold_df['pupil'].mean():.4f}  "
                  f"Mean={fold_df['mean'].mean():.4f}")

            del model
            torch.cuda.empty_cache()

        # Save per-image CSV
        if all_rows:
            out_path = RESULTS_DIR / f"cv_method{method_id}_perimage_{timestamp}.csv"
            df = pd.DataFrame(all_rows)
            df.to_csv(out_path, index=False)
            print(f"\nSaved: {out_path}  ({len(df)} rows)")

            # Summary
            print(f"  Overall mean Dice: {df['mean'].mean():.6f}")
            for fold_idx in range(5):
                fdf = df[df["fold"] == fold_idx]
                if len(fdf):
                    print(f"    Fold {fold_idx}: mean={fdf['mean'].mean():.6f}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
