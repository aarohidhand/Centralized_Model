"""
utils/blocks.py
================
Shared neural network building blocks for FedBCa pipeline.

FIXES in this version:
- SimpleClassifier: BatchNorm1d momentum=0.05 (was 0.1), Dropout(0.15/0.10)
- InputAdapter: mean-initialised weights (1/in_ch per channel)
- freeze_batchnorm: correctly handles both BN2d and BN1d
- MixStyle: p=0.35 default (was 0.05 — effectively disabled)
- All attention gate dims verified
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# GROUPNORM HELPER
# ─────────────────────────────────────────────

def gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    g = min(groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


# ─────────────────────────────────────────────
# BN FREEZE (called in __init__ AND train() override)
# ─────────────────────────────────────────────

def freeze_batchnorm(module: nn.Module):
    """
    Freeze all BatchNorm layers: set eval mode + disable gradients.
    Must be called BOTH in __init__ AND in train() override,
    because model.train() resets BN to training mode every epoch.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


# ─────────────────────────────────────────────
# INPUT ADAPTER
# ─────────────────────────────────────────────

class InputAdapter(nn.Module):
    """
    Projects N-channel MRI input to 3-channel for pretrained backbones.

    Initialised with equal weights (1/in_ch per output channel) so that
    immediately after init, the pretrained backbone receives a reasonable
    weighted average of all input channels — not random noise.

    For 5-channel 2.5D stack, this means each output channel sees the
    average of all 5 MRI slices at init, which looks like a valid MRI
    image to the pretrained encoder from the first forward pass.
    """

    def __init__(self, in_ch: int = 5, out_ch: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        # Mean-initialise: each output sees equal average of all inputs
        with torch.no_grad():
            self.proj.weight.fill_(1.0 / in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────
# CONV BLOCK
# ─────────────────────────────────────────────

class ConvGNReLU(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s,
                      padding=p, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# RESIDUAL BLOCK
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            gn(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                gn(out_ch),
            ) if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.shortcut(x))


# ─────────────────────────────────────────────
# ATTENTION GATE
# ─────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Soft attention gate (Attention U-Net, Oktay et al. 2018).
    Filters irrelevant low-level skip features before decoder fusion.

    Args:
        F_g   : channels of gating signal (decoder feature)
        F_l   : channels of skip connection (encoder feature)
        F_int : intermediate channels (typically F_l // 2)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False), gn(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False), gn(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False), nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1  = F.interpolate(self.W_g(g), size=x.shape[2:],
                            mode='bilinear', align_corners=False)
        x1  = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi


# ─────────────────────────────────────────────
# DECODER BLOCK
# ─────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsample × 2 → concat skip → ResidualBlock × 2.

    Args:
        in_ch   : input channels from previous decoder stage
        skip_ch : skip connection channels (0 = no skip)
        out_ch  : output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        total      = in_ch + skip_ch
        self.conv1 = ResidualBlock(total,  out_ch, dropout=dropout)
        self.conv2 = ResidualBlock(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


# ─────────────────────────────────────────────
# CBAM
# ─────────────────────────────────────────────

class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp   = nn.Sequential(
            nn.Conv2d(channels, reduced,  1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(reduced,  channels, 1, bias=False),
        )
        self.ch_sig  = nn.Sigmoid()
        self.sp_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sp_sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch = self.ch_sig(self.ch_mlp(self.avg_pool(x)) + self.ch_mlp(self.max_pool(x)))
        x  = x * ch
        sp = self.sp_sig(self.sp_conv(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        ))
        return x * sp


# ─────────────────────────────────────────────
# MIXSTYLE
# ─────────────────────────────────────────────

class MixStyle(nn.Module):
    """
    Feature-level domain generalisation (Zhou et al., ICLR 2021).
    Mixes feature statistics between samples in a batch to force
    the encoder to learn scanner-invariant representations.

    p=0.35: fires in ~1/3 of batches — sufficient for consistent
    domain invariance pressure on a 4-center dataset.
    (Previous p=0.05 was effectively disabled.)
    """

    def __init__(self, p: float = 0.35, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p     = p
        self.alpha = alpha
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.random() > self.p:
            return x
        B = x.size(0)
        if B < 2:
            return x
        mu  = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3],  keepdim=True)
        sig = (var + self.eps).sqrt()
        x_n = (x - mu) / sig
        perm  = torch.randperm(B, device=x.device)
        lmda  = (torch.distributions.Beta(self.alpha, self.alpha)
                 .sample((B, 1, 1, 1)).to(x.device))
        mu_m  = lmda * mu  + (1 - lmda) * mu[perm]
        sig_m = lmda * sig + (1 - lmda) * sig[perm]
        return x_n * sig_m + mu_m


# ─────────────────────────────────────────────
# DEEP SUPERVISION HEAD
# ─────────────────────────────────────────────

class DeepSupervisionHead(nn.Module):

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x: torch.Tensor, out_size: tuple) -> torch.Tensor:
        return F.interpolate(
            self.head(x), size=out_size,
            mode='bilinear', align_corners=False
        )


# ─────────────────────────────────────────────
# SIMPLE CLASSIFIER
# ─────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    """
    Global average pool → 2-layer MLP → class logits.

    FIXES:
    - BatchNorm1d momentum=0.05 (was 0.1): slower stat update
      prevents corrupt stats during cls_rampup zero-gradient phase
    - Dropout(0.15) first layer (was 0.25 → then 0.05): balanced
    - Dropout(0.10) second layer: enough regularisation
    - Final Linear init std=0.01: prevents large initial logits
      that caused cls=nan in early epochs
    """

    def __init__(self, feat_ch: int, num_classes: int = 2, dropout: float = 0.15):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_ch, 256, bias=False),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.05),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.05),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )
        # Small init for final layer → prevents early logit overflow
        with torch.no_grad():
            final_linear = [m for m in self.fc.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.trunc_normal_(final_linear.weight, std=0.01)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.fc(x)


# ─────────────────────────────────────────────
# WEIGHT INITIALISATION
# ─────────────────────────────────────────────

def initialize_weights(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
