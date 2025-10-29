import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EyeSegmentationDataset(Dataset):
    def __init__(self, image_paths, label_seg_dir, label_obb_dir, image_size=512, is_train=True):
        self.image_paths = image_paths
        self.label_seg_dir = Path(label_seg_dir)
        self.label_obb_dir = Path(label_obb_dir)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def _resize_mask(self, mask):
        if mask is None:
            return np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        return cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

    def _augment(self, image, masks):
        if self.is_train:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                masks = [np.fliplr(m).copy() for m in masks]
            if random.random() < 0.3:
                angle = random.uniform(-7, 7)
                image = rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode="reflect")
                masks = [rotate(m, angle, axes=(0, 1), reshape=False, order=0, mode="reflect") for m in masks]
        return image, masks

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        filename_base = image_path.stem

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        mask_lid = cv2.imread(str(self.label_seg_dir / f"{filename_base}_mask_lid.png"), 0)
        mask_iris = cv2.imread(str(self.label_obb_dir / f"{filename_base}_mask_iris.png"), 0)
        mask_pupil = cv2.imread(str(self.label_obb_dir / f"{filename_base}_mask_pupil.png"), 0)

        mask_lid = self._resize_mask(mask_lid)
        mask_iris = self._resize_mask(mask_iris)
        mask_pupil = self._resize_mask(mask_pupil)

        image, (mask_lid, mask_iris, mask_pupil) = self._augment(image, [mask_lid, mask_iris, mask_pupil])

        image = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        sample = {
            "image": image,
            "mask_lid": torch.from_numpy(mask_lid.copy()).long(),
            "mask_iris": torch.from_numpy(mask_iris.copy()).long(),
            "mask_pupil": torch.from_numpy(mask_pupil.copy()).long(),
            "filename": image_path.name,
        }
        return sample


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetMethod1(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.eyelid_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.iris_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 5),
        )
        self.pupil_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return {
            "eyelid_seg": self.eyelid_head(d1),
            "iris_ellipse": self.iris_head(d1),
            "pupil_ellipse": self.pupil_head(d1),
        }


def dice_coeff(pred, target, smooth=1e-5):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


def dice_loss(pred, target):
    return 1 - dice_coeff(pred, target)


def render_ellipse_mask(ellipse_params, H, W, device):
    B = ellipse_params.shape[0]
    cx = ellipse_params[:, 0] * W
    cy = ellipse_params[:, 1] * H
    a = ellipse_params[:, 2] * W / 2
    b = ellipse_params[:, 3] * H / 2
    theta = ellipse_params[:, 4] * 2 * np.pi - np.pi
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij"
    )
    x_coords = x_coords.unsqueeze(0).expand(B, -1, -1)
    y_coords = y_coords.unsqueeze(0).expand(B, -1, -1)
    dx = x_coords - cx.view(B, 1, 1)
    dy = y_coords - cy.view(B, 1, 1)
    cos_t = torch.cos(theta).view(B, 1, 1)
    sin_t = torch.sin(theta).view(B, 1, 1)
    dx_r = dx * cos_t + dy * sin_t
    dy_r = -dx * sin_t + dy * cos_t
    ellipse_eq = (dx_r / (a.view(B, 1, 1) + 1e-6))**2 + (dy_r / (b.view(B, 1, 1) + 1e-6))**2
    return torch.sigmoid(10 * (1 - ellipse_eq))


class LossFunction1(nn.Module):
    def __init__(self, lambda_ellipse=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.lambda_ellipse = float(lambda_ellipse)

    def forward(self, pred, target):
        eyelid_pred = pred["eyelid_seg"]
        gt_lid = (target["mask_lid"].float() / 255.0)
        gt_iris = (target["mask_iris"].float() / 255.0)
        gt_pupil = (target["mask_pupil"].float() / 255.0)
        H, W = gt_lid.shape[-2], gt_lid.shape[-1]
        if tuple(eyelid_pred.shape[-2:]) != (H, W):
            eyelid_pred = F.interpolate(eyelid_pred, size=(H, W), mode="bilinear", align_corners=False)
        eyelid_pred = eyelid_pred.squeeze(1)
        with torch.amp.autocast(device_type=eyelid_pred.device.type, enabled=False):
            loss_lid_bce = self.bce(eyelid_pred.float(), gt_lid.float())
        loss_lid = loss_lid_bce + dice_loss(eyelid_pred, gt_lid)
        iris_mask_pred = render_ellipse_mask(torch.sigmoid(pred["iris_ellipse"]), H, W, gt_lid.device)
        pupil_mask_pred = render_ellipse_mask(torch.sigmoid(pred["pupil_ellipse"]), H, W, gt_lid.device)
        with torch.amp.autocast(device_type=eyelid_pred.device.type, enabled=False):
            loss_ellipse = self.bce(iris_mask_pred.float(), gt_iris.float()) + self.bce(pupil_mask_pred.float(), gt_pupil.float())
        return loss_lid + self.lambda_ellipse * loss_ellipse


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        images = batch["image"].to(device)
        target = {
            "mask_lid": batch["mask_lid"].to(device),
            "mask_iris": batch["mask_iris"].to(device),
            "mask_pupil": batch["mask_pupil"].to(device),
        }
        with torch.amp.autocast(device_type="cuda"):
            pred = model(images)
            loss = criterion(pred, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = {
                "mask_lid": batch["mask_lid"].to(device),
                "mask_iris": batch["mask_iris"].to(device),
                "mask_pupil": batch["mask_pupil"].to(device),
            }
            pred = model(images)
            loss = criterion(pred, target)
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    images_dir = Path("Images/images")
    label_seg_dir = Path("Images/labels_seg")
    label_obb_dir = Path("Images/labels_obb")
    fold_file = Path("fold_indices.json")

    df = pd.read_csv("image_metadata.csv")
    image_paths = [images_dir / row["filename"] for _, row in df.iterrows()]

    folds = json.loads(fold_file.read_text())
    fold_idx = 0
    train_indices = folds[str(fold_idx)]["train"]
    val_indices = folds[str(fold_idx)]["val"]

    train_paths = [image_paths[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]

    train_dataset = EyeSegmentationDataset(train_paths, label_seg_dir, label_obb_dir, is_train=True)
    val_dataset = EyeSegmentationDataset(val_paths, label_seg_dir, label_obb_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    model = UNetMethod1().to(device)
    criterion = LossFunction1(lambda_ellipse=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler("cpu")

    best_loss = float("inf")
    patience = 30
    patience_counter = 0

    for epoch in range(50):
        start = datetime.now()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = datetime.now() - start
        print(f"Epoch {epoch+1:02d}: train={train_loss:.4f} val={val_loss:.4f} ({elapsed})")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Path("model") / "method1_fold0_best.pth")
            print("  -> best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    print(f"Training done. Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
