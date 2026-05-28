"""
utils/losses.py
================
Multi-task loss functions for FedBCa pipeline.

FIXES:
- JointLoss now uses REAL focal loss (focal_gamma was silently ignored before)
- tversky_weight=0.20 (was 0.35 — caused over-prediction of tumor regions)
- FocalTversky alpha=0.40, beta=0.60 (was 0.30/0.70 — over-penalised FN)
- mibc_alpha class weighting actually applied (was null in all configs)
- StableFocalLoss replaces StableCrossEntropy in classification branch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# DICE LOSS
# ─────────────────────────────────────────────

class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5, sigmoid: bool = True):
        super().__init__()
        self.smooth  = smooth
        self.sigmoid = sigmoid

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        pred   = pred.contiguous().view(pred.size(0), -1)
        target = target.float().contiguous().view(target.size(0), -1)
        inter  = (pred * target).sum(1)
        union  = pred.sum(1) + target.sum(1)
        dice   = (2.0 * inter + self.smooth) / (union + self.smooth)
        return (1.0 - dice.clamp(0, 1)).mean()


# ─────────────────────────────────────────────
# BCE + DICE COMBINED
# ─────────────────────────────────────────────

class DiceCELoss(nn.Module):

    def __init__(self, lambda_dice: float = 0.7, lambda_ce: float = 0.3):
        super().__init__()
        self.ld   = lambda_dice
        self.lc   = lambda_ce
        self.dice = DiceLoss(sigmoid=True)
        self.ce   = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ld * self.dice(pred, target) + self.lc * self.ce(pred, target.float())


# ─────────────────────────────────────────────
# FOCAL TVERSKY LOSS
# ─────────────────────────────────────────────

class FocalTverskyLoss(nn.Module):
    """
    alpha=0.40, beta=0.60: moderate FN penalisation.
    (Was alpha=0.30, beta=0.70 — over-penalised FN causing over-prediction.)
    gamma=0.75: focal weighting on hard samples.
    """

    def __init__(self, alpha: float = 0.40, beta: float = 0.60,
                 gamma: float = 0.75, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred)
        pred   = pred.contiguous().view(pred.size(0), -1)
        target = target.float().contiguous().view(target.size(0), -1)

        tp = (pred * target).sum(1)
        fp = (pred * (1 - target)).sum(1)
        fn = ((1 - pred) * target).sum(1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        focal_t = (1.0 - tversky.clamp(0, 1)) ** self.gamma
        return focal_t.mean()


# ─────────────────────────────────────────────
# BOUNDARY LOSS
# ─────────────────────────────────────────────

class BoundaryLoss(nn.Module):

    @staticmethod
    def gradient(x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx + dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(
            self.gradient(torch.sigmoid(pred)),
            self.gradient(target.float())
        )


# ─────────────────────────────────────────────
# SEGMENTATION LOSS
# ─────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    region_loss + tversky_weight * FocalTversky + boundary_weight * Boundary

    tversky_weight=0.20 (was 0.35): reduces FN over-penalisation.
    When tversky_weight was 0.35, models predicted large blobs to
    avoid missing any tumor → high recall but poor precision → low DSC.
    """

    def __init__(
        self,
        lambda_dice:     float = 0.70,
        lambda_ce:       float = 0.30,
        tversky_weight:  float = 0.20,
        boundary_weight: float = 0.03,
    ):
        super().__init__()
        self.region   = DiceCELoss(lambda_dice, lambda_ce)
        self.tversky  = FocalTverskyLoss(alpha=0.40, beta=0.60, gamma=0.75)
        self.boundary = BoundaryLoss()
        self.tw = tversky_weight
        self.bw = boundary_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.region(pred, target)
            + self.tw * self.tversky(pred, target)
            + self.bw * self.boundary(pred, target)
        )


# ─────────────────────────────────────────────
# DEEP SUPERVISION LOSS
# ─────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):

    def __init__(self, ds_weights=(0.50, 0.25), tversky_weight=0.20):
        super().__init__()
        self.ds_weights = ds_weights
        self.seg_loss   = SegmentationLoss(tversky_weight=tversky_weight)

    def forward(self, main_pred, ds_preds, target):
        loss = self.seg_loss(main_pred, target)
        for w, ds in zip(self.ds_weights, ds_preds):
            if ds.shape[2:] != target.shape[2:]:
                ds = F.interpolate(
                    ds, size=target.shape[2:],
                    mode='bilinear', align_corners=False
                )
            loss = loss + w * self.seg_loss(ds, target)
        return loss


# ─────────────────────────────────────────────
# STABLE FOCAL LOSS  (replaces StableCrossEntropy)
# FIXES: focal_gamma now actually used
# ─────────────────────────────────────────────

class StableFocalLoss(nn.Module):
    """
    Focal cross-entropy with optional class weighting.

    Previously JointLoss used StableCrossEntropy and focal_gamma
    was silently ignored — this is the fix.

    gamma=2.0: gives hard MIBC examples ~4x more gradient weight
    than easy correctly-classified NMIBC examples.

    mibc_alpha: class weight for MIBC class (index 1).
    With 36.4% MIBC in training, alpha=1.4 compensates imbalance
    without over-correcting.
    """

    def __init__(
        self,
        gamma:           float = 2.0,
        mibc_alpha:      float = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma           = gamma
        self.mibc_alpha      = mibc_alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, -15.0, 15.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        weight = None
        if self.mibc_alpha is not None:
            weight = torch.tensor(
                [1.0, self.mibc_alpha],
                dtype=logits.dtype,
                device=logits.device
            )

        ce = F.cross_entropy(
            logits, targets,
            weight=weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        ce = torch.nan_to_num(ce, nan=0.0, posinf=0.0, neginf=0.0)

        pt     = torch.exp(-ce)
        focal  = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# ─────────────────────────────────────────────
# JOINT MULTI-TASK LOSS
# ─────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    seg_weight * SegmentationLoss + cls_weight * StableFocalLoss

    cls_weight is updated each epoch by trainer (staged ramp).
    cls_mask: apply classification loss only on tumor + best slices.
    """

    def __init__(
        self,
        seg_weight:     float = 0.85,
        cls_weight:     float = 0.15,
        focal_gamma:    float = 2.0,
        mibc_alpha:     float = None,
        ds_weights:     tuple = (0.50, 0.25),
        tversky_weight: float = 0.20,
    ):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight   # updated each epoch by trainer

        self.seg_loss_fn = DeepSupervisionLoss(
            ds_weights=ds_weights,
            tversky_weight=tversky_weight,
        )
        self.cls_loss_fn = StableFocalLoss(
            gamma=focal_gamma,
            mibc_alpha=mibc_alpha,
            label_smoothing=0.05,
        )

    def forward(
        self,
        outputs:    dict,
        seg_target: torch.Tensor,
        cls_target: torch.Tensor,
        cls_mask:   torch.Tensor = None,
    ) -> dict:

        seg_pred = outputs['seg']
        cls_pred = outputs['cls']
        ds_preds = outputs.get('ds', [])

        # Segmentation loss
        seg_loss = self.seg_loss_fn(seg_pred, ds_preds, seg_target)
        seg_loss = torch.nan_to_num(seg_loss, nan=1.0, posinf=1.0, neginf=0.0)

        # Classification loss (on tumor + best slices only)
        if self.cls_weight > 0:
            if cls_mask is not None:
                valid = cls_mask > 0
                if valid.sum() > 0:
                    cls_loss = self.cls_loss_fn(
                        cls_pred[valid], cls_target[valid]
                    )
                else:
                    cls_loss = torch.zeros(1, device=seg_pred.device).squeeze()
            else:
                cls_loss = self.cls_loss_fn(cls_pred, cls_target)
            cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            cls_loss = torch.zeros(1, device=seg_pred.device).squeeze()

        total = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        total = torch.nan_to_num(total, nan=1.0, posinf=1.0, neginf=0.0)

        return {
            'loss':     total,
            'seg_loss': seg_loss.detach(),
            'cls_loss': cls_loss.detach(),
        }
