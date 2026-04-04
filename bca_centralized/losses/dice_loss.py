import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        inter = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, pred, target):
        return self.loss(pred, target)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, smooth=1.0, gamma=2.0, alpha=0.25):
        super().__init__()
        self.dice = DiceLoss(smooth)
        self.focal = FocalLoss(gamma, alpha)

    def forward(self, pred, target):
        return 0.7 * self.dice(pred, target) + 0.3 * self.focal(pred, target)


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        tp = (pred * target).sum(dim=1)
        fp = ((1 - target) * pred).sum(dim=1)
        fn = (target * (1 - pred)).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss()

    def forward(self, pred, target):
        return (
            0.5 * self.dice(pred, target) +
            0.3 * self.bce(pred, target) +
            0.2 * self.tversky(pred, target)
        )