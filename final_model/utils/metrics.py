"""
utils/metrics.py
=================
Evaluation metrics for FedBCa.

FIXES:
- combined = 0.55 * tumour_DSC + 0.45 * AUC (was 0.80/0.20 — suppressed AUC)
- patient aggregation: pure mean (was top1/top3 — inflated false positives)
- evaluate() accepts amp flag from trainer
"""

import numpy as np
import torch
from torch.cuda.amp import autocast
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
)
from collections import defaultdict


@torch.no_grad()
def dice_score(pred_logit, target, threshold=0.5, smooth=1e-5):
    pred   = (torch.sigmoid(pred_logit) > threshold).float()
    pred   = pred.reshape(pred.size(0), -1)
    target = target.float().reshape(target.size(0), -1)
    inter  = (pred * target).sum(1)
    union  = pred.sum(1) + target.sum(1)
    valid  = union > 0
    d      = torch.ones_like(union)
    d[valid] = (2.0 * inter[valid] + smooth) / (union[valid] + smooth)
    return d.clamp(0, 1).mean().item()


@torch.no_grad()
def evaluate(model, loader, device, amp=False):
    """
    Full validation loop.

    amp=True  for EfficientNet and MedSAM (stable in FP16)
    amp=False for ResNet50 (FP32 required)

    Returns dict with: dsc, tumour_dsc, auc, accuracy,
                       balanced_accuracy, f1, combined_score
    """
    model.eval()

    all_dsc    = []
    tumor_dsc  = []
    pid_probs  = defaultdict(list)
    pid_labels = {}

    for batch in loader:
        images     = batch['image'].to(device, non_blocking=True)
        masks      = batch['mask'].to(device, non_blocking=True)
        labels     = batch['label']
        has_tumor  = batch['has_tumor'].cpu().numpy()
        pids       = batch['patient_id']

        with autocast(enabled=amp):
            outputs = model(images)

        seg = torch.nan_to_num(outputs['seg'], nan=0.0, posinf=0.0, neginf=0.0)
        cls = torch.nan_to_num(outputs['cls'], nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(images.size(0)):
            if masks[i].sum() <= 0:
                continue
            d = dice_score(seg[i:i+1], masks[i:i+1])
            all_dsc.append(d)
            if has_tumor[i]:
                tumor_dsc.append(d)

        probs = torch.softmax(cls, dim=1)[:, 1].detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5)

        for i, pid in enumerate(pids):
            pid_probs[pid].append(float(probs[i]))
            pid_labels[pid] = int(labels[i].item())

    # Patient-level aggregation: PURE MEAN
    # (top1/top3 weighting inflated false positives from noisy artifact slices)
    pt_ids   = sorted(pid_probs.keys())
    pt_prob  = np.array([float(np.mean(pid_probs[p])) for p in pt_ids])
    pt_label = np.array([pid_labels[p] for p in pt_ids])
    pt_pred  = (pt_prob >= 0.5).astype(np.int32)

    try:
        auc = roc_auc_score(pt_label, pt_prob) if len(np.unique(pt_label)) >= 2 else 0.5
    except Exception:
        auc = 0.5

    mean_dsc       = float(np.mean(all_dsc))   if all_dsc   else 0.0
    mean_tumor_dsc = float(np.mean(tumor_dsc)) if tumor_dsc else 0.0

    # Combined score: 0.55 * DSC + 0.45 * AUC
    # (was 0.80/0.20 — structural suppression of AUC checkpointing)
    combined = 0.55 * mean_tumor_dsc + 0.45 * auc

    return {
        'dsc':               round(mean_dsc, 4),
        'tumour_dsc':        round(mean_tumor_dsc, 4),
        'auc':               round(float(auc), 4),
        'accuracy':          round(float(accuracy_score(pt_label, pt_pred)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(pt_label, pt_pred)), 4),
        'precision':         round(float(precision_score(pt_label, pt_pred, zero_division=0)), 4),
        'recall':            round(float(recall_score(pt_label, pt_pred, zero_division=0)), 4),
        'f1':                round(float(f1_score(pt_label, pt_pred, zero_division=0)), 4),
        'combined_score':    round(combined, 4),
        'n_patients':        len(pt_ids),
        'n_tumor_slices':    len(tumor_dsc),
    }
