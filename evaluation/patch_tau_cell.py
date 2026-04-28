"""Append a new code cell to crossvalidation.ipynb that performs the NSD
τ-sensitivity sweep for Methods 4 and 5 on the primary mode (fullmax).

Reuses utilities already defined in earlier cells (model classes, dataset,
dataloader, post-processing functions, compute_hd95_nsd helper).

Idempotent: detects an existing marker `# === tau sensitivity sweep ===`
and skips if present.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("crossvalidation.ipynb")
MARKER = "# === tau sensitivity sweep ==="

CELL_CODE = f'''{MARKER}
# Set RUN_TAU_SENSITIVITY=True to run. Estimated time: ~30-60 min on GPU.
# Evaluates Methods 4 and 5 (primary mode = fullmax) at τ ∈ TAU_VALUES and
# stores per-image NSD in results/cv_tau_sensitivity_*.csv.

RUN_TAU_SENSITIVITY = True

if RUN_TAU_SENSITIVITY:
    from datetime import datetime
    import pandas as pd
    import numpy as np
    import torch
    from pathlib import Path

    from evaluation.metrics_reloaded_eval import compute_hd95_nsd

    TAU_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
    TARGET_METHODS = [4, 5]

    if 'fold_indices' not in globals():
        with open('fold_indices.json', 'r') as f:
            fold_indices = json.load(f)
    if 'image_paths' not in globals():
        IMAGES_DIR = Path("Images/images")
        df_meta = pd.read_csv('image_metadata.csv')
        image_paths = [IMAGES_DIR / row['filename'] for _, row in df_meta.iterrows()]
    MODEL_DIR = Path("model/cv_300ep") if 'MODEL_DIR' not in globals() else MODEL_DIR

    tau_rows = []
    for fold_idx in range(NUM_FOLDS):
        val_indices = fold_indices[str(fold_idx)]['val']
        val_paths = [image_paths[i] for i in val_indices]
        val_ds = EyeSegmentationDataset(
            val_paths, LABEL_SEG_DIR, LABEL_OBB_DIR,
            transform=False, use_ellipse_cache=False, use_sixcls_direct=True,
        )
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        for method_id in TARGET_METHODS:
            model_path = MODEL_DIR / f"method{{method_id}}_fold{{fold_idx}}_best.pth"
            if not model_path.exists():
                print(f"Missing {{model_path}} — skipping")
                continue
            if method_id == 4:
                model = UNetMethod4().to(device)
            else:
                model = UNetMethod5().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            model.eval()

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"τ-sweep M{{method_id}} fold{{fold_idx}}", leave=False):
                    img = batch['image'].to(device)
                    gt_lid   = batch['mask_lid'].cpu().numpy()
                    gt_iris  = batch['mask_iris'].cpu().numpy()
                    gt_pupil = batch['mask_pupil'].cpu().numpy()
                    filenames = batch.get('filename', None)

                    with autocast():
                        out = model(img)

                    if method_id == 4:
                        logits = out['five_class_seg']
                        pred_labels = torch.argmax(logits, dim=1).cpu().numpy()
                    else:
                        probs = torch.sigmoid(out['amodal_logits']).cpu().numpy()

                    for b in range(img.shape[0]):
                        filename = filenames[b] if filenames is not None else str(b)
                        subject_id = str(filename).split('-', 1)[0]

                        if method_id == 4:
                            pred = pred_labels[b]
                            lid_bin = (((pred == 1) | (pred == 2) | (pred == 4)).astype(np.uint8) * 255)
                            iris_vis = ((pred == 2).astype(np.uint8) * 255)
                            iris_occ = ((pred == 3).astype(np.uint8) * 255)
                            pupil_vis = ((pred == 4).astype(np.uint8) * 255)
                            pupil_occ = ((pred == 5).astype(np.uint8) * 255)
                            iris_mask, _ = ellipse_mask_from_fullmax_contour(iris_vis, iris_occ)
                            pupil_mask, _ = ellipse_mask_from_fullmax_contour(pupil_vis, pupil_occ)
                        else:
                            lid_bin = (probs[b, 0] >= 0.5).astype(np.uint8) * 255
                            iris_raw = (probs[b, 1] >= 0.5).astype(np.uint8) * 255
                            pupil_raw = (probs[b, 2] >= 0.5).astype(np.uint8) * 255
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

                        for tau in TAU_VALUES:
                            _, nsd_lid   = compute_hd95_nsd(lid_bin,   gt_lid[b],   tau=tau)
                            _, nsd_iris  = compute_hd95_nsd(iris_mask, gt_iris[b],  tau=tau)
                            _, nsd_pupil = compute_hd95_nsd(pupil_mask, gt_pupil[b], tau=tau)
                            tau_rows.append({{
                                'method': method_id, 'fold': fold_idx, 'tau': tau,
                                'filename': str(filename), 'subject_id': str(subject_id),
                                'nsd_eyelid': nsd_lid, 'nsd_iris': nsd_iris, 'nsd_pupil': nsd_pupil,
                            }})

            del model
            torch.cuda.empty_cache()

        del val_loader, val_ds
        torch.cuda.empty_cache()
        print(f"Fold {{fold_idx}} done, rows so far: {{len(tau_rows)}}")

    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tau_csv = Path('results') / f"cv_tau_sensitivity_{{_ts}}.csv"
    pd.DataFrame(tau_rows).to_csv(tau_csv, index=False)
    print(f"\\n✅ τ sensitivity CSV saved: {{tau_csv}}  (rows = {{len(tau_rows)}})")
else:
    print("Set RUN_TAU_SENSITIVITY = True and re-run this cell to execute.")
'''


def main():
    with NB_PATH.open(encoding="utf-8") as f:
        nb = json.load(f)

    # Check idempotency
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and MARKER in "".join(cell["source"]):
            print("Tau-sweep cell already present — skipping.")
            return

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CELL_CODE.splitlines(keepends=True),
    }
    # Append after the last code cell
    nb["cells"].append(new_cell)

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Appended tau-sweep cell. Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
