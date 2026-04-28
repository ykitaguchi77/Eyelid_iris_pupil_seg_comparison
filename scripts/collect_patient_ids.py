"""
Collect patient IDs actually used in Detection and Segmentation training.

Detection:
  Images in C:/Users/CorneAI/FacePhoto_instance/images-labels/1-295/images/
  Patient ID = first token when filename is split on '-'.

Segmentation:
  Images listed in CVAT XML (ID 0-2999) AND present in Images/images/.
  Reproduces Cell 5 logic of train_SegFormerB0_amodal_blur_ver3_3000mai.ipynb.
  Patient ID = first token when filename is split on '-'.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DETECTION_IMG_DIR = Path(r"C:/Users/CorneAI/FacePhoto_instance/images-labels/1-295/images")
SEG_ROOT = Path(r"C:/Users/CorneAI/Eyelid_Iris_pupil_seg_comparison")
SEG_IMG_DIR = SEG_ROOT / "Images" / "images"
XML_EYELID = SEG_ROOT / "Images" / "eyelid_caruncle_seg_0-3000.xml"
XML_IRIS_PUPIL = SEG_ROOT / "Images" / "obb_iris_pupil_1-3000.xml"

OUT_DIR = SEG_ROOT / "scripts" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_IMAGE_ID = 0
MAX_IMAGE_ID = 2999


def patient_id_from_name(name: str) -> str:
    return str(name).split("-", 1)[0]


def collect_detection_ids() -> list[int]:
    if not DETECTION_IMG_DIR.exists():
        raise FileNotFoundError(DETECTION_IMG_DIR)
    ids: set[int] = set()
    for p in DETECTION_IMG_DIR.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        tok = patient_id_from_name(p.name)
        if re.fullmatch(r"\d+", tok):
            ids.add(int(tok))
    return sorted(ids)


def load_cvat_image_names(xml_path: Path, min_id: int, max_id: int) -> dict[int, str]:
    if not xml_path.exists():
        return {}
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    result: dict[int, str] = {}
    for img_el in root.findall(".//image"):
        img_id_str = img_el.get("id")
        name = img_el.get("name")
        if img_id_str is None or not name:
            continue
        try:
            img_id = int(img_id_str)
        except ValueError:
            continue
        if min_id <= img_id <= max_id:
            result[img_id] = Path(name).name
    return result


def collect_segmentation_ids() -> list[int]:
    eyelid = load_cvat_image_names(XML_EYELID, MIN_IMAGE_ID, MAX_IMAGE_ID)
    iris = load_cvat_image_names(XML_IRIS_PUPIL, MIN_IMAGE_ID, MAX_IMAGE_ID)
    all_dict: dict[int, str] = {**eyelid, **iris}
    ids: set[int] = set()
    missing = 0
    for name in all_dict.values():
        if not (SEG_IMG_DIR / name).exists():
            missing += 1
            continue
        tok = patient_id_from_name(name)
        if re.fullmatch(r"\d+", tok):
            ids.add(int(tok))
    print(f"[seg] XML eyelid entries: {len(eyelid)}")
    print(f"[seg] XML iris/pupil entries: {len(iris)}")
    print(f"[seg] union entries: {len(all_dict)}")
    print(f"[seg] missing on disk: {missing}")
    return sorted(ids)


def main() -> int:
    print("=== Detection cohort ===")
    det_ids = collect_detection_ids()
    print(f"Unique patients: {len(det_ids)}")
    print(f"ID range: {det_ids[0]} - {det_ids[-1]}")
    expected = set(range(1, 296))
    missing_det = sorted(expected - set(det_ids))
    print(f"Missing from 1-295: {missing_det}")

    print("\n=== Segmentation cohort ===")
    seg_ids = collect_segmentation_ids()
    print(f"Unique patients: {len(seg_ids)}")
    if seg_ids:
        print(f"ID range: {seg_ids[0]} - {seg_ids[-1]}")

    det_out = OUT_DIR / "detection_patient_ids.json"
    seg_out = OUT_DIR / "segmentation_patient_ids.json"
    det_out.write_text(json.dumps({"patient_ids": det_ids, "count": len(det_ids)}, indent=2))
    seg_out.write_text(json.dumps({"patient_ids": seg_ids, "count": len(seg_ids)}, indent=2))
    print(f"\nWrote: {det_out}")
    print(f"Wrote: {seg_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
