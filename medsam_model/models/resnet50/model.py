"""
models/resnet50/model.py
=========================
ResNet50 + CBAM + MixStyle Multi-Task Model.

FIXES:
- train() override re-freezes BN every epoch (BN unfreeze was the biggest bug)
- MixStyle p=0.35 (was 0.05 — effectively disabled)
- SimpleClassifier(feat_ch=2048, dropout=0.15)
- use_amp: true in config (safe with BN1d in classifier)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.blocks import (
    gn, freeze_batchnorm, InputAdapter,
    AttentionGate, DecoderBlock, CBAM,
    DeepSupervisionHead, SimpleClassifier,
    MixStyle, initialize_weights,
)


class ResNet50MultiTask(nn.Module):

    def __init__(
        self,
        in_channels:  int  = 5,
        pretrained:   bool = True,
        num_classes:  int  = 2,
        use_mixstyle: bool = True,
        dropout:      float = 0.15,
    ):
        super().__init__()

        self.input_adapter = InputAdapter(in_ch=in_channels, out_ch=3)

        bb = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # 256ch
        self.layer2 = bb.layer2   # 512ch
        self.layer3 = bb.layer3   # 1024ch
        self.layer4 = bb.layer4   # 2048ch

        # Freeze pretrained BN in encoder
        freeze_batchnorm(self.stem)
        freeze_batchnorm(self.layer1)
        freeze_batchnorm(self.layer2)
        freeze_batchnorm(self.layer3)
        freeze_batchnorm(self.layer4)

        # MixStyle after layer2 — rich enough for style, not yet over-specialised
        self.mix_l2 = MixStyle(p=0.15, alpha=0.1) if use_mixstyle else nn.Identity()

        # CBAM attention at deep layers
        self.cbam3 = CBAM(1024, reduction=16)
        self.cbam4 = CBAM(2048, reduction=16)

        # ── Decoder (5-stage Attention U-Net) ──────────
        # Verified channel dimensions:
        # s4=2048, s3=1024, s2=512, s1=256, s0=64(stem)
        self.ag4 = AttentionGate(2048, 1024, 512)
        self.ag3 = AttentionGate(384,  512,  256)
        self.ag2 = AttentionGate(192,  256,  128)
        self.ag1 = AttentionGate(96,   64,    32)

        self.dec4 = DecoderBlock(2048, 1024, 384, dropout=dropout)
        self.dec3 = DecoderBlock(384,   512, 192, dropout=dropout)
        self.dec2 = DecoderBlock(192,   256,  96, dropout=max(0, dropout - 0.05))
        self.dec1 = DecoderBlock(96,     64,  48, dropout=0.0)
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            gn(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            gn(32), nn.GELU(),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            gn(16), nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

        self.ds_head1 = DeepSupervisionHead(384)
        self.ds_head2 = DeepSupervisionHead(192)

        self.cls_head = SimpleClassifier(
            feat_ch=2048, num_classes=num_classes, dropout=0.15
        )

        initialize_weights(self.seg_head)
        initialize_weights(self.dec0)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[ResNet50-V5] in_ch={in_channels} | mixstyle={use_mixstyle} | params={n:,}')

    def train(self, mode=True):
        """
        CRITICAL FIX: Re-freeze backbone BN after every model.train() call.

        PyTorch's model.train() recursively resets ALL submodules including
        BatchNorm layers back to training mode. Without this override, the
        freeze_batchnorm() called in __init__ is undone at the start of
        every training epoch. This was the single biggest bug causing
        scanner-specific domain shift to corrupt encoder features.
        """
        super().train(mode)
        freeze_batchnorm(self.stem)
        freeze_batchnorm(self.layer1)
        freeze_batchnorm(self.layer2)
        freeze_batchnorm(self.layer3)
        freeze_batchnorm(self.layer4)
        return self

    def forward(self, x: torch.Tensor) -> dict:
        B, C, H, W = x.shape

        x3 = self.input_adapter(x)

        s0 = self.stem(x3)                   # 64ch,  H/4
        s1 = self.layer1(s0)                 # 256ch, H/4
        s2 = self.mix_l2(self.layer2(s1))    # 512ch, H/8  + MixStyle
        s3 = self.cbam3(self.layer3(s2))     # 1024ch,H/16 + CBAM
        s4 = self.cbam4(self.layer4(s3))     # 2048ch,H/32 + CBAM

        d4 = self.dec4(s4, self.ag4(s4, s3))
        d3 = self.dec3(d4, self.ag3(d4, s2))
        d2 = self.dec2(d3, self.ag2(d3, s1))
        d1 = self.dec1(d2, self.ag1(d2, s0))
        d0 = self.dec0(d1)

        seg = self.seg_head(d0)
        if seg.shape[2:] != (H, W):
            seg = F.interpolate(seg, size=(H, W), mode='bilinear', align_corners=False)
        seg = torch.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        ds_outs = []
        if self.training:
            ds_outs = [self.ds_head1(d4, (H, W)), self.ds_head2(d3, (H, W))]

        cls = self.cls_head(s4)

        return {'seg': seg, 'cls': cls, 'ds': ds_outs}


def build_model(pretrained=True, in_channels=5, use_mixstyle=True, **kwargs):
    return ResNet50MultiTask(
        in_channels=in_channels,
        pretrained=pretrained,
        use_mixstyle=use_mixstyle,
    )
