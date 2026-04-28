"""Patch crossvalidation.ipynb Cell 21 to add HD95 + NSD via MetricsReloaded.

Changes:
  1. Prepend imports and a _augment_hd95_nsd helper.
  2. Replace evaluate_method1 and evaluate_method2 with per_rows-enabled versions.
  3. Augment per_rows.append sites in evaluate_method3/5/6 with HD95+NSD per
     (structure, mode).
  4. Update the fold loop to unpack tuples from method1/2 and save their CSVs.

Idempotent: re-running detects the `# === MetricsReloaded patch ===` marker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NB_PATH = Path("crossvalidation.ipynb")

PATCH_MARKER = "# === MetricsReloaded patch ==="

HEADER_INJECT = f"""{PATCH_MARKER}
import sys, os
_repo_root = os.path.abspath('.')
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from evaluation.metrics_reloaded_eval import compute_hd95_nsd, DEFAULT_NSD_TAU

def _augment_hd95_nsd(row, lid_pred, iris_pred, pupil_pred, gt_lid, gt_iris, gt_pupil):
    \"\"\"Add hd95 / nsd columns for each structure to a per-image row dict.\"\"\"
    h, n = compute_hd95_nsd(lid_pred, gt_lid, DEFAULT_NSD_TAU['eyelid'])
    row['eyelid_hd95'] = h; row['eyelid_nsd'] = n
    h, n = compute_hd95_nsd(iris_pred, gt_iris, DEFAULT_NSD_TAU['iris'])
    row['iris_hd95'] = h; row['iris_nsd'] = n
    h, n = compute_hd95_nsd(pupil_pred, gt_pupil, DEFAULT_NSD_TAU['pupil'])
    row['pupil_hd95'] = h; row['pupil_nsd'] = n
# === end MetricsReloaded patch ===

"""

# Replacement for evaluate_method1
METHOD1_OLD = """@torch.no_grad()
def evaluate_method1(model, val_loader, device):
    \"\"\"Method1評価: Eyelid, Iris, Pupilの3つのDice\"\"\"
    model.eval()
    lid_scores, iris_scores, pupil_scores = [], [], []

    for batch in tqdm(val_loader, desc="M1評価", leave=False):
        img = batch['image'].to(device)
        gt_lid   = batch['mask_lid'].cpu().numpy()
        gt_iris  = batch['mask_iris'].cpu().numpy()
        gt_pupil = batch['mask_pupil'].cpu().numpy()

        with autocast():
            out = model(img)

        for b in range(img.shape[0]):
            # Eyelid
            lid_logits = out['eyelid_seg'][b:b+1]
            lid_pred = (torch.sigmoid(lid_logits).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
            lid_scores.append(dice_binary_np(lid_pred, gt_lid[b]))

            # Iris/Pupil ellipse
            iris_params  = torch.sigmoid(out['iris_ellipse']).cpu().numpy()[b]
            pupil_params = torch.sigmoid(out['pupil_ellipse']).cpu().numpy()[b]
            iris_mask  = ellipse_params_to_mask(iris_params,  IMAGE_HEIGHT, IMAGE_WIDTH)
            pupil_mask = ellipse_params_to_mask(pupil_params, IMAGE_HEIGHT, IMAGE_WIDTH)
            iris_scores.append(dice_binary_np(iris_mask, gt_iris[b]))
            pupil_scores.append(dice_binary_np(pupil_mask, gt_pupil[b]))

    return {
        'lid': np.mean(lid_scores),
        'iris': np.mean(iris_scores),
        'pupil': np.mean(pupil_scores),
        'mean': np.mean([np.mean(lid_scores), np.mean(iris_scores), np.mean(pupil_scores)])
    }"""

METHOD1_NEW = """@torch.no_grad()
def evaluate_method1(model, val_loader, device):
    \"\"\"Method1評価: Eyelid, Iris, Pupilの3つのDice + HD95 + NSD (per-image)\"\"\"
    model.eval()
    lid_scores, iris_scores, pupil_scores = [], [], []
    per_rows = []

    for batch in tqdm(val_loader, desc="M1評価", leave=False):
        img = batch['image'].to(device)
        gt_lid   = batch['mask_lid'].cpu().numpy()
        gt_iris  = batch['mask_iris'].cpu().numpy()
        gt_pupil = batch['mask_pupil'].cpu().numpy()
        filenames = batch.get('filename', None)

        with autocast():
            out = model(img)

        for b in range(img.shape[0]):
            # Eyelid
            lid_logits = out['eyelid_seg'][b:b+1]
            lid_pred = (torch.sigmoid(lid_logits).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
            lid_d = float(dice_binary_np(lid_pred, gt_lid[b]))
            lid_scores.append(lid_d)

            # Iris/Pupil ellipse
            iris_params  = torch.sigmoid(out['iris_ellipse']).cpu().numpy()[b]
            pupil_params = torch.sigmoid(out['pupil_ellipse']).cpu().numpy()[b]
            iris_mask  = ellipse_params_to_mask(iris_params,  IMAGE_HEIGHT, IMAGE_WIDTH)
            pupil_mask = ellipse_params_to_mask(pupil_params, IMAGE_HEIGHT, IMAGE_WIDTH)
            iris_d = float(dice_binary_np(iris_mask, gt_iris[b]))
            pupil_d = float(dice_binary_np(pupil_mask, gt_pupil[b]))
            iris_scores.append(iris_d); pupil_scores.append(pupil_d)

            filename = filenames[b] if filenames is not None else str(b)
            subject_id = str(filename).split('-', 1)[0]
            row = {
                'filename': str(filename), 'subject_id': str(subject_id),
                'mode': 'ellipse_regression',
                'eyelid': lid_d, 'iris': iris_d, 'pupil': pupil_d,
                'mean': float(np.mean([lid_d, iris_d, pupil_d])),
            }
            _augment_hd95_nsd(row, lid_pred, iris_mask, pupil_mask,
                              gt_lid[b], gt_iris[b], gt_pupil[b])
            per_rows.append(row)

    result = {
        'lid': float(np.mean(lid_scores)),
        'iris': float(np.mean(iris_scores)),
        'pupil': float(np.mean(pupil_scores)),
        'mean': float(np.mean([np.mean(lid_scores), np.mean(iris_scores), np.mean(pupil_scores)])),
    }
    return result, per_rows"""

# Replacement for evaluate_method2
METHOD2_OLD = """@torch.no_grad()
def evaluate_method2(model, val_loader, device):
    \"\"\"Method2評価: Eyelid, Iris, Pupilの3つのDice（楕円近似前の参考値も計算）\"\"\"
    model.eval()
    lid_scores, iris_scores, pupil_scores = [], [], []
    # 楕円近似前の参考値（エッジ→塗りつぶし）
    iris_scores_before_ellipse, pupil_scores_before_ellipse = [], []

    for batch in tqdm(val_loader, desc="M2評価", leave=False):
        img = batch['image'].to(device)
        gt_lid   = batch['mask_lid'].cpu().numpy()
        gt_iris  = batch['mask_iris'].cpu().numpy()
        gt_pupil = batch['mask_pupil'].cpu().numpy()

        with autocast():
            out = model(img)
            edge_logits = out['edge_logits']

        for b in range(img.shape[0]):
            # Eyelid: edge -> fill
            lid_edge = (torch.sigmoid(edge_logits[b,0:1]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
            lid_fill = bin_edge_to_filled(lid_edge)
            lid_scores.append(dice_binary_np(lid_fill, gt_lid[b]))

            # Iris/Pupil: edge -> ellipse fit
            iris_edge  = (torch.sigmoid(edge_logits[b,1:2]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
            pupil_edge = (torch.sigmoid(edge_logits[b,2:3]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255

            # 楕円近似前の参考値：エッジ→塗りつぶし
            iris_fill_before = bin_edge_to_filled(iris_edge)
            pupil_fill_before = bin_edge_to_filled(pupil_edge)
            iris_scores_before_ellipse.append(dice_binary_np(iris_fill_before, gt_iris[b]))
            pupil_scores_before_ellipse.append(dice_binary_np(pupil_fill_before, gt_pupil[b]))

            # 楕円フィット（本評価）
            iris_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
            pupil_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)

            iris_pts = np.column_stack(np.where(iris_edge > 0))
            if len(iris_pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(iris_pts[:, ::-1].astype(np.int32))
                    cv2.ellipse(iris_mask, ellipse, 255, thickness=-1)
                except: pass

            pupil_pts = np.column_stack(np.where(pupil_edge > 0))
            if len(pupil_pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(pupil_pts[:, ::-1].astype(np.int32))
                    cv2.ellipse(pupil_mask, ellipse, 255, thickness=-1)
                except: pass

            iris_scores.append(dice_binary_np(iris_mask, gt_iris[b]))
            pupil_scores.append(dice_binary_np(pupil_mask, gt_pupil[b]))

    return {
        'lid': np.mean(lid_scores),
        'iris': np.mean(iris_scores),
        'pupil': np.mean(pupil_scores),
        'mean': np.mean([np.mean(lid_scores), np.mean(iris_scores), np.mean(pupil_scores)]),
        # 参考値：楕円近似前のDice
        'iris_before_ellipse': np.mean(iris_scores_before_ellipse),
        'pupil_before_ellipse': np.mean(pupil_scores_before_ellipse)
    }"""

METHOD2_NEW = """@torch.no_grad()
def evaluate_method2(model, val_loader, device):
    \"\"\"Method2評価: Eyelid, Iris, Pupilの3つのDice + HD95 + NSD (per-image)\"\"\"
    model.eval()
    lid_scores, iris_scores, pupil_scores = [], [], []
    iris_scores_before_ellipse, pupil_scores_before_ellipse = [], []
    per_rows = []

    for batch in tqdm(val_loader, desc="M2評価", leave=False):
        img = batch['image'].to(device)
        gt_lid   = batch['mask_lid'].cpu().numpy()
        gt_iris  = batch['mask_iris'].cpu().numpy()
        gt_pupil = batch['mask_pupil'].cpu().numpy()
        filenames = batch.get('filename', None)

        with autocast():
            out = model(img)
            edge_logits = out['edge_logits']

        for b in range(img.shape[0]):
            # Eyelid: edge -> fill
            lid_edge = (torch.sigmoid(edge_logits[b,0:1]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
            lid_fill = bin_edge_to_filled(lid_edge)
            lid_d = float(dice_binary_np(lid_fill, gt_lid[b]))
            lid_scores.append(lid_d)

            # Iris/Pupil: edge -> ellipse fit
            iris_edge  = (torch.sigmoid(edge_logits[b,1:2]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255
            pupil_edge = (torch.sigmoid(edge_logits[b,2:3]).cpu().squeeze().numpy() >= 0.5).astype(np.uint8)*255

            iris_fill_before = bin_edge_to_filled(iris_edge)
            pupil_fill_before = bin_edge_to_filled(pupil_edge)
            iris_scores_before_ellipse.append(float(dice_binary_np(iris_fill_before, gt_iris[b])))
            pupil_scores_before_ellipse.append(float(dice_binary_np(pupil_fill_before, gt_pupil[b])))

            iris_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)
            pupil_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8)

            iris_pts = np.column_stack(np.where(iris_edge > 0))
            if len(iris_pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(iris_pts[:, ::-1].astype(np.int32))
                    cv2.ellipse(iris_mask, ellipse, 255, thickness=-1)
                except: pass

            pupil_pts = np.column_stack(np.where(pupil_edge > 0))
            if len(pupil_pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(pupil_pts[:, ::-1].astype(np.int32))
                    cv2.ellipse(pupil_mask, ellipse, 255, thickness=-1)
                except: pass

            iris_d = float(dice_binary_np(iris_mask, gt_iris[b]))
            pupil_d = float(dice_binary_np(pupil_mask, gt_pupil[b]))
            iris_scores.append(iris_d); pupil_scores.append(pupil_d)

            filename = filenames[b] if filenames is not None else str(b)
            subject_id = str(filename).split('-', 1)[0]
            row = {
                'filename': str(filename), 'subject_id': str(subject_id),
                'mode': 'edge_ellipse_fit',
                'eyelid': lid_d, 'iris': iris_d, 'pupil': pupil_d,
                'mean': float(np.mean([lid_d, iris_d, pupil_d])),
            }
            _augment_hd95_nsd(row, lid_fill, iris_mask, pupil_mask,
                              gt_lid[b], gt_iris[b], gt_pupil[b])
            per_rows.append(row)

    result = {
        'lid': float(np.mean(lid_scores)),
        'iris': float(np.mean(iris_scores)),
        'pupil': float(np.mean(pupil_scores)),
        'mean': float(np.mean([np.mean(lid_scores), np.mean(iris_scores), np.mean(pupil_scores)])),
        'iris_before_ellipse': float(np.mean(iris_scores_before_ellipse)),
        'pupil_before_ellipse': float(np.mean(pupil_scores_before_ellipse)),
    }
    return result, per_rows"""


# For Method3/5/6 we replace the per-mode append loop. Unique anchors used.

METHOD3_LOOP_OLD = """            # per-image rows（long format）
            for mode, i_d, p_d, m_d in [
                ('raw', iris_raw_d, pupil_raw_d, mean_raw),
                ('outerarc', iris_outer_d, pupil_outer_d, mean_outer),
                ('fullmax', iris_fullmax_d, pupil_fullmax_d, mean_fullmax),
                ('ransac_whole', iris_ransac_d, pupil_ransac_d, mean_ransac),
            ]:
                per_rows.append({
                    'filename': str(filename),
                    'subject_id': str(subject_id),
                    'mode': mode,
                    'eyelid': lid_d,
                    'iris': i_d,
                    'pupil': p_d,
                    'mean': m_d,
                })"""

METHOD3_LOOP_NEW = """            # per-image rows (long format) with HD95 + NSD
            for mode, i_d, p_d, m_d, i_mask, p_mask in [
                ('raw', iris_raw_d, pupil_raw_d, mean_raw, iris_raw, pupil_raw),
                ('outerarc', iris_outer_d, pupil_outer_d, mean_outer, iris_outer, pupil_outer),
                ('fullmax', iris_fullmax_d, pupil_fullmax_d, mean_fullmax, iris_fullmax, pupil_fullmax),
                ('ransac_whole', iris_ransac_d, pupil_ransac_d, mean_ransac, iris_ransac, pupil_ransac),
            ]:
                row = {
                    'filename': str(filename), 'subject_id': str(subject_id),
                    'mode': mode,
                    'eyelid': lid_d, 'iris': i_d, 'pupil': p_d, 'mean': m_d,
                }
                _augment_hd95_nsd(row, lid_m3_bin, i_mask, p_mask,
                                  gt_lid[b], gt_iris[b], gt_pupil[b])
                per_rows.append(row)"""

METHOD5_LOOP_OLD = """            # per-image rows (long format)
            for mode, i_d, p_d, m_d in [
                ('raw', iris_raw_d, pupil_raw_d, mean_raw),
                ('fullmax', iris_fullmax_d, pupil_fullmax_d, mean_fullmax),
                ('ransac_whole', iris_ransac_d, pupil_ransac_d, mean_ransac),
            ]:
                per_rows.append({
                    'filename': str(filename),
                    'subject_id': str(subject_id),
                    'mode': mode,
                    'eyelid': lid_d,
                    'iris': i_d,
                    'pupil': p_d,
                    'mean': m_d,
                })"""

METHOD5_LOOP_NEW = """            # per-image rows (long format) with HD95 + NSD
            for mode, i_d, p_d, m_d, i_mask, p_mask in [
                ('raw', iris_raw_d, pupil_raw_d, mean_raw, iris_raw, pupil_raw),
                ('fullmax', iris_fullmax_d, pupil_fullmax_d, mean_fullmax, iris_fullmax_mask, pupil_fullmax_mask),
                ('ransac_whole', iris_ransac_d, pupil_ransac_d, mean_ransac, iris_ransac, pupil_ransac),
            ]:
                row = {
                    'filename': str(filename), 'subject_id': str(subject_id),
                    'mode': mode,
                    'eyelid': lid_d, 'iris': i_d, 'pupil': p_d, 'mean': m_d,
                }
                _augment_hd95_nsd(row, lid_raw, i_mask, p_mask,
                                  gt_lid[b], gt_iris[b], gt_pupil[b])
                per_rows.append(row)"""

METHOD6_LOOP_OLD = """            # per-image rows (long format)
            for mode, i_d, p_d, m_d in [
                ('raw', iris_raw_d, pupil_raw_d, mean_raw),
                ('boundary', iris_boundary_d, pupil_boundary_d, mean_boundary),
                ('fullmax', iris_fullmax_d, pupil_fullmax_d, mean_fullmax),
            ]:
                per_rows.append({
                    'filename': str(filename),
                    'subject_id': str(subject_id),
                    'mode': mode,
                    'eyelid': lid_d,
                    'iris': i_d,
                    'pupil': p_d,
                    'mean': m_d,
                })"""

METHOD6_LOOP_NEW = """            # per-image rows (long format) with HD95 + NSD
            for mode, i_d, p_d, m_d, i_mask, p_mask in [
                ('raw', iris_raw_d, pupil_raw_d, mean_raw, iris_raw, pupil_raw),
                ('boundary', iris_boundary_d, pupil_boundary_d, mean_boundary, iris_boundary_mask, pupil_boundary_mask),
                ('fullmax', iris_fullmax_d, pupil_fullmax_d, mean_fullmax, iris_fullmax_mask, pupil_fullmax_mask),
            ]:
                row = {
                    'filename': str(filename), 'subject_id': str(subject_id),
                    'mode': mode,
                    'eyelid': lid_d, 'iris': i_d, 'pupil': p_d, 'mean': m_d,
                }
                _augment_hd95_nsd(row, lid_pred, i_mask, p_mask,
                                  gt_lid[b], gt_iris[b], gt_pupil[b])
                per_rows.append(row)"""

# Fold evaluation loop: method1/2 now return tuples; save their per_rows too.
FOLD_LOOP_OLD = """    # 各メソッドを評価
    for method_id in TRAIN_METHODS:
        model_path = MODEL_DIR / f"method{method_id}_fold{fold_idx}_best.pth"
        if not model_path.exists():
            print(f"  ⚠️ Method{method_id} モデルが見つかりません: {model_path}")
            continue

        # モデルロード
        if method_id == 1:
            model = UNetMethod1().to(device)
        elif method_id == 2:
            model = UNetMethod2().to(device)
        elif method_id == 3:
            model = UNetMethod3().to(device)
        elif method_id == 4:
            model = UNetMethod4().to(device)
        elif method_id == 5:
            model = UNetMethod5().to(device)
        else:  # method_id == 6
            model = UNetMethod6().to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # 評価実行
        if method_id == 1:
            result = evaluate_method1(model, val_loader, device)
        elif method_id == 2:
            result = evaluate_method2(model, val_loader, device)
        elif method_id == 3:"""

FOLD_LOOP_NEW = """    # 各メソッドを評価
    for method_id in TRAIN_METHODS:
        model_path = MODEL_DIR / f"method{method_id}_fold{fold_idx}_best.pth"
        if not model_path.exists():
            print(f"  ⚠️ Method{method_id} モデルが見つかりません: {model_path}")
            continue

        # モデルロード
        if method_id == 1:
            model = UNetMethod1().to(device)
        elif method_id == 2:
            model = UNetMethod2().to(device)
        elif method_id == 3:
            model = UNetMethod3().to(device)
        elif method_id == 4:
            model = UNetMethod4().to(device)
        elif method_id == 5:
            model = UNetMethod5().to(device)
        else:  # method_id == 6
            model = UNetMethod6().to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # 評価実行
        if method_id == 1:
            result, per_rows = evaluate_method1(model, val_loader, device)
            for r in per_rows: r['fold'] = fold_idx
            method1_rows_all.extend(per_rows)
        elif method_id == 2:
            result, per_rows = evaluate_method2(model, val_loader, device)
            for r in per_rows: r['fold'] = fold_idx
            method2_rows_all.extend(per_rows)
        elif method_id == 3:"""

# Initialize method1_rows_all / method2_rows_all alongside the others
INIT_ROWS_OLD = """method3_rows_all = []
method4_rows_all = []
method5_rows_all = []
method6_rows_all = []
method6_rows_all = []"""

INIT_ROWS_NEW = """method1_rows_all = []
method2_rows_all = []
method3_rows_all = []
method4_rows_all = []
method5_rows_all = []
method6_rows_all = []"""

# Save method1/method2 per-image CSVs (insert just before method3 save)
SAVE_CSV_OLD = """# per-image CSV保存
if len(method3_rows_all) > 0:"""

SAVE_CSV_NEW = """# per-image CSV保存
if len(method1_rows_all) > 0:
    _ts_m1 = datetime.now().strftime("%Y%m%d_%H%M%S")
    method1_perimage_csv = result_dir / f"cv_method1_reloaded_perimage_{_ts_m1}.csv"
    pd.DataFrame(method1_rows_all).to_csv(method1_perimage_csv, index=False)
    print(f"\\n✅ Method1 per-image (DSC+HD95+NSD) saved: {method1_perimage_csv}")

if len(method2_rows_all) > 0:
    _ts_m2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    method2_perimage_csv = result_dir / f"cv_method2_reloaded_perimage_{_ts_m2}.csv"
    pd.DataFrame(method2_rows_all).to_csv(method2_perimage_csv, index=False)
    print(f"\\n✅ Method2 per-image (DSC+HD95+NSD) saved: {method2_perimage_csv}")

if len(method3_rows_all) > 0:"""


def apply_patches(src: str) -> tuple[str, list[str]]:
    """Apply all patches. Returns (new_src, list_of_applied_names)."""
    applied = []

    if PATCH_MARKER in src:
        raise RuntimeError("Patch marker already present — aborting to avoid double-patch.")

    # 1. Prepend header
    anchor = "from skimage.measure import EllipseModel, ransac\n"
    if anchor not in src:
        raise RuntimeError("Header anchor not found.")
    src = src.replace(anchor, anchor + "\n" + HEADER_INJECT, 1)
    applied.append("header")

    # 2. Method1/2 rewrites
    for name, old, new in [
        ("method1", METHOD1_OLD, METHOD1_NEW),
        ("method2", METHOD2_OLD, METHOD2_NEW),
    ]:
        if old not in src:
            raise RuntimeError(f"{name} anchor not found.")
        if src.count(old) != 1:
            raise RuntimeError(f"{name} anchor not unique: {src.count(old)}")
        src = src.replace(old, new)
        applied.append(name)

    # 3. Method3/5/6 loop augmentation
    # NOTE: method6 has two duplicate def blocks in the notebook (historical
    # artifact — the second one shadows the first). We replace all occurrences.
    for name, old, new, allow_multi in [
        ("method3_loop", METHOD3_LOOP_OLD, METHOD3_LOOP_NEW, False),
        ("method5_loop", METHOD5_LOOP_OLD, METHOD5_LOOP_NEW, False),
        ("method6_loop", METHOD6_LOOP_OLD, METHOD6_LOOP_NEW, True),
    ]:
        if old not in src:
            raise RuntimeError(f"{name} anchor not found.")
        count = src.count(old)
        if not allow_multi and count != 1:
            raise RuntimeError(f"{name} anchor not unique: {count}")
        src = src.replace(old, new)
        applied.append(f"{name}×{count}")

    # 4. Init extension
    if INIT_ROWS_OLD not in src:
        raise RuntimeError("init_rows anchor not found.")
    src = src.replace(INIT_ROWS_OLD, INIT_ROWS_NEW, 1)
    applied.append("init_rows")

    # 5. Fold loop unpack
    if FOLD_LOOP_OLD not in src:
        raise RuntimeError("fold_loop anchor not found.")
    src = src.replace(FOLD_LOOP_OLD, FOLD_LOOP_NEW, 1)
    applied.append("fold_loop")

    # 6. CSV save
    if SAVE_CSV_OLD not in src:
        raise RuntimeError("save_csv anchor not found.")
    src = src.replace(SAVE_CSV_OLD, SAVE_CSV_NEW, 1)
    applied.append("save_csv")

    return src, applied


def normalize_src(src: str) -> str:
    """Strip trailing whitespace from each line (Python-safe)."""
    return "\n".join(line.rstrip() for line in src.split("\n"))


def main():
    with NB_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    cell = nb["cells"][21]
    if cell["cell_type"] != "code":
        raise RuntimeError("Cell 21 is not code")

    src = normalize_src("".join(cell["source"]))
    new_src, applied = apply_patches(src)

    # Convert back to lines with trailing \n except the last
    lines = new_src.split("\n")
    new_source = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        new_source.append(lines[-1])

    cell["source"] = new_source

    # Clear outputs to keep notebook small
    if cell.get("outputs"):
        cell["outputs"] = []
    cell["execution_count"] = None

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Applied patches: {applied}")
    print(f"Cell 21 source size: {len(src)} -> {len(new_src)} chars")


if __name__ == "__main__":
    main()
