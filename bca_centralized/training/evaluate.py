import sys, os
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
import torch.nn.functional as F

try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    BASE_DIR = os.getcwd()

sys.path.insert(0, BASE_DIR)

from models.unet import MultiTaskUNet
from datasets.bca_dataset import BcaMultiTaskDataset
from utils.metrics import compute_seg_metrics, compute_cls_metrics, MetricStore
from configs.config import (
    SPLITS, DATA_PROC, CKPT_MULTI,
    SEG_THRESHOLD
)


def get_ckpt_path():
    for path in [
        os.path.join(CKPT_MULTI, "fold0", "best_model.pth"),
        os.path.join(CKPT_MULTI, "full", "best_model.pth"),
        os.path.join(CKPT_MULTI, "best_model.pth"),
    ]:
        if os.path.exists(path):
            return path
    return None


def evaluate_multitask():
    print("\n" + "=" * 60)
    print("MULTITASK EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        print("No checkpoint found")
        return

    model = MultiTaskUNet().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print(f"Loaded: {ckpt_path}")

    test_csv = f"{SPLITS}/seg_test.csv"
    test_ds = BcaMultiTaskDataset(test_csv, DATA_PROC, is_train=False)
    test_df = pd.read_csv(test_csv)

    for c_id in [1, 2, 3, 4]:
        mask = test_df["center"].astype(str).str.endswith(str(c_id))
        idxs = test_df.index[mask].tolist()

        if len(idxs) == 0:
            continue

        loader = DataLoader(
            Subset(test_ds, idxs),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        seg_store = MetricStore()
        probs_all = []
        labels_all = []

        with torch.no_grad():
            for imgs, masks, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                seg_out, cls_out = model(imgs)

                if seg_out.shape != masks.shape:
                    seg_out = F.interpolate(seg_out, size=masks.shape[2:], mode="bilinear", align_corners=False)

                seg_probs = torch.sigmoid(seg_out)

                for p, m in zip(seg_probs.cpu().numpy(), masks.cpu().numpy()):
                    seg_store.update(compute_seg_metrics(p[0], m[0], SEG_THRESHOLD))

                probs = torch.softmax(cls_out, dim=1)[:, 1]

                probs_all.extend(probs.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        seg_metrics = seg_store.mean()
        cls_metrics = compute_cls_metrics(np.array(probs_all), np.array(labels_all))

        print(f"\nCenter {c_id}")
        print("-" * 40)
        print(f"DSC:  {seg_metrics.get('DSC', 0):.4f}")
        print(f"IoU:  {seg_metrics.get('IoU', 0):.4f}")
        print(f"HD95: {seg_metrics.get('HD95', None)}")
        print(f"AUC:  {cls_metrics['AUC']:.4f}")
        print(f"Acc:  {cls_metrics['Accuracy']:.4f}")
        print(f"Sens: {cls_metrics['Sensitivity']:.4f}")
        print(f"Spec: {cls_metrics['Specificity']:.4f}")
        print(f"F1:   {cls_metrics['F1']:.4f}")


if __name__ == "__main__":
    evaluate_multitask()