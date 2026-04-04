import os, sys, random
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))

from configs.config import (
    DATA_RAW, DATA_PROC, CENTERS,
    PATCH_SEG, PATCH_CLS,
    EXPAND_VOXEL, RANDOM_OFFSET, RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_nii(path):
    try:
        return np.asarray(nib.load(path).dataobj, dtype=np.float32)
    except:
        return None


def normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def roi_crop(img, mask, expand):
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return img, mask

    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)

    mins = np.maximum(mins - expand, 0)
    maxs = np.minimum(maxs + expand, np.array(img.shape) - 1)

    return img[
        mins[0]:maxs[0],
        mins[1]:maxs[1],
        mins[2]:maxs[2]
    ], mask[
        mins[0]:maxs[0],
        mins[1]:maxs[1],
        mins[2]:maxs[2]
    ]


def resize_crop(img, size, interp):
    h, w = img.shape

    if h < size or w < size:
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")

    return cv2.resize(img, (size, size), interpolation=interp)


def find_annotation(ann_dir, case_id, filename):
    for p in [
        Path(ann_dir) / filename,
        Path(ann_dir) / f"{case_id}_1.nii.gz",
        Path(ann_dir) / f"{case_id}_2.nii.gz",
    ]:
        if p.exists():
            return str(p)
    return None


def run_preprocessing():
    seg_img_dir = Path(DATA_PROC) / "images"
    seg_msk_dir = Path(DATA_PROC) / "masks"
    cls_img_dir = Path(DATA_PROC) / "cls_images"

    seg_img_dir.mkdir(parents=True, exist_ok=True)
    seg_msk_dir.mkdir(parents=True, exist_ok=True)
    cls_img_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print("=" * 55)
    print("PREPROCESSING STARTED")
    print("=" * 55)

    for center in CENTERS:
        center_dir = Path(DATA_RAW) / center
        img_dir = center_dir / "Image"
        ann_dir = center_dir / "Annotation"

        label_file = list(center_dir.glob("*.xlsx"))
        if not label_file:
            continue

        df_lbl = pd.read_excel(label_file[0])

        label_col = next(
            (c for c in df_lbl.columns if "MIBC" in c.upper() or "LABEL" in c.upper()),
            df_lbl.columns[-1]
        )

        id_col = [c for c in df_lbl.columns if c != label_col][0]

        label_map = dict(zip(df_lbl[id_col].astype(str), df_lbl[label_col].astype(int)))

        for img_file in tqdm(list(img_dir.glob("*.nii.gz")), desc=center):
            case_id = img_file.stem.replace(".nii", "")

            img_vol = load_nii(str(img_file))
            if img_vol is None:
                continue

            ann_path = find_annotation(ann_dir, case_id, img_file.name)
            if ann_path is None:
                continue

            mask_vol = load_nii(ann_path)
            if mask_vol is None:
                continue

            mask_vol = (mask_vol > 0).astype(np.uint8)

            label = label_map.get(case_id)
            if label is None:
                continue

            img_vol, mask_vol = roi_crop(img_vol, mask_vol, EXPAND_VOXEL)

            for i in range(1, img_vol.shape[2] - 1):

                mask_slice = mask_vol[:, :, i]
                if mask_slice.sum() == 0:
                    continue

                prev = img_vol[:, :, i - 1]
                curr = img_vol[:, :, i]
                next_ = img_vol[:, :, i + 1]

                img_stack = np.stack([prev, curr, next_], axis=2)

                img_stack = normalize(img_stack)
                curr_mask = mask_slice

                seg_img = resize_crop(img_stack[:, :, 1], PATCH_SEG, cv2.INTER_LINEAR)
                seg_msk = resize_crop(curr_mask, PATCH_SEG, cv2.INTER_NEAREST)

                cls_img = resize_crop(img_stack[:, :, 1], PATCH_CLS, cv2.INTER_LINEAR)

                fname = f"{center}_{case_id}_sl{i:03d}"

                cv2.imwrite(str(seg_img_dir / f"{fname}.png"), (seg_img * 255).astype(np.uint8))
                cv2.imwrite(str(seg_msk_dir / f"{fname}.png"), (seg_msk * 255).astype(np.uint8))
                cv2.imwrite(str(cls_img_dir / f"{fname}.png"), (cls_img * 255).astype(np.uint8))

                records.append({
                    "filename": fname,
                    "center": center,
                    "patient": case_id,
                    "slice": i,
                    "label": int(label)
                })

    df = pd.DataFrame(records)
    df.to_csv(Path(DATA_PROC) / "labels.csv", index=False)

    print("\nDONE")
    print(f"Total: {len(df)}")
    print(f"MIBC: {df['label'].sum()}")
    print(f"NMIBC: {(df['label']==0).sum()}")

    return df


if __name__ == "__main__":
    run_preprocessing()