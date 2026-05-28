"""
models/swin_hybrid/model.py
============================
Swin-Tiny Hybrid Multi-Task Model for FedBCa.

FIXED VERSION:
- Correct Swin tensor layout handling
- timm Swin features_only=True returns BHWC tensors
- Converted properly to BCHW using permute()
- Prevents decoder channel mismatch crash

Architecture:
    5ch MRI → InputAdapter(5→3)
        ↓
    HybridEncoder:
        CNN stream:  ResNet-34 via timm
        Swin stream: Swin-Tiny via timm
        ↓
    Attention U-Net Decoder
        ↓
    Segmentation + Classification

Expected:
    DSC ~0.66-0.76
    AUC ~0.82-0.88
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.blocks import (
    gn,
    freeze_batchnorm,
    InputAdapter,
    AttentionGate,
    DecoderBlock,
    DeepSupervisionHead,
    MixStyle,
    initialize_weights,
)


class HybridEncoder(nn.Module):

    def __init__(self, use_mixstyle: bool = True):
        super().__init__()

        # ─────────────────────────────────────
        # CNN STREAM
        # ─────────────────────────────────────
        self.cnn = timm.create_model(
            'resnet34',
            pretrained=True,
            features_only=True,
            out_indices=(1, 2),
        )

        freeze_batchnorm(self.cnn)

        self.mix1 = (
            MixStyle(p=0.35, alpha=0.1)
            if use_mixstyle else nn.Identity()
        )

        self.mix2 = (
            MixStyle(p=0.35, alpha=0.1)
            if use_mixstyle else nn.Identity()
        )

        # ─────────────────────────────────────
        # SWIN STREAM
        # ─────────────────────────────────────
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2),
        )

        # Decoder channel registry
        self.out_ch = {
            'l1': 64,
            'l2': 128,
            's1': 96,
            's2': 192,
            's3': 384,
        }

    def train(self, mode=True):
        super().train(mode)
        freeze_batchnorm(self.cnn)
        return self

    def forward(self, x):

        # ─────────────────────────────────────
        # CNN FEATURES
        # ─────────────────────────────────────
        cnn_feats = self.cnn(x)

        l1 = self.mix1(cnn_feats[0])   # 64ch
        l2 = self.mix2(cnn_feats[1])   # 128ch

        # ─────────────────────────────────────
        # SWIN INPUT NORMALIZATION
        # ─────────────────────────────────────
        x_std = x.std(
            dim=[1, 2, 3],
            keepdim=True
        ).clamp(min=1e-6)

        x_swin = x / x_std

        x_swin = torch.nan_to_num(
            x_swin,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ─────────────────────────────────────
        # SWIN FEATURES
        # timm returns BHWC
        # convert → BCHW
        # ─────────────────────────────────────
        swin_feats = self.swin(x_swin)

        s1 = swin_feats[0].permute(0, 3, 1, 2).contiguous()
        s2 = swin_feats[1].permute(0, 3, 1, 2).contiguous()
        s3 = swin_feats[2].permute(0, 3, 1, 2).contiguous()

        s1 = torch.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
        s2 = torch.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
        s3 = torch.nan_to_num(s3, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            'l1': l1,
            'l2': l2,
            's1': s1,
            's2': s2,
            's3': s3,
        }


class SwinHybridMultiTask(nn.Module):

    def __init__(
        self,
        in_channels=5,
        num_classes=2,
        use_mixstyle=True,
        dropout=0.15,
    ):
        super().__init__()

        self.input_adapter = InputAdapter(
            in_ch=in_channels,
            out_ch=3
        )

        self.encoder = HybridEncoder(
            use_mixstyle=use_mixstyle
        )

        ch = self.encoder.out_ch

        # ─────────────────────────────────────
        # SKIP PROJECTION
        # s1(96) + l2(128) = 224
        # ─────────────────────────────────────
        self.skip_proj = nn.Sequential(
            nn.Conv2d(
                ch['s1'] + ch['l2'],
                128,
                1,
                bias=False
            ),
            gn(128),
            nn.GELU(),
        )

        # ─────────────────────────────────────
        # ATTENTION GATES
        # ─────────────────────────────────────
        self.ag3 = AttentionGate(
            ch['s3'],
            ch['s2'],
            96
        )

        self.ag2 = AttentionGate(
            192,
            128,
            64
        )

        self.ag1 = AttentionGate(
            96,
            ch['l1'],
            32
        )

        # ─────────────────────────────────────
        # DECODER
        # ─────────────────────────────────────
        self.dec4 = DecoderBlock(
            ch['s3'],
            ch['s2'],
            192,
            dropout=dropout
        )

        self.dec3 = DecoderBlock(
            192,
            128,
            96,
            dropout=dropout
        )

        self.dec2 = DecoderBlock(
            96,
            ch['l1'],
            48,
            dropout=max(0, dropout - 0.05)
        )

        self.dec1 = DecoderBlock(
            48,
            0,
            24,
            dropout=0.0
        )

        # ─────────────────────────────────────
        # SEGMENTATION HEAD
        # ─────────────────────────────────────
        self.seg_head = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            ),

            nn.Conv2d(
                24,
                16,
                3,
                padding=1,
                bias=False
            ),

            gn(16),
            nn.GELU(),

            nn.Conv2d(
                16,
                1,
                1
            ),
        )

        self.ds1 = DeepSupervisionHead(192)
        self.ds2 = DeepSupervisionHead(96)

        # ─────────────────────────────────────
        # CLASSIFIER
        # ─────────────────────────────────────
        self.cls_proj = nn.Sequential(
            nn.Linear(ch['s3'], 128, bias=False),

            nn.BatchNorm1d(
                128,
                eps=1e-3,
                momentum=0.05
            ),

            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(128, num_classes),
        )

        with torch.no_grad():

            last = [
                m for m in self.cls_proj.modules()
                if isinstance(m, nn.Linear)
            ][-1]

            nn.init.trunc_normal_(
                last.weight,
                std=0.01
            )

            if last.bias is not None:
                nn.init.zeros_(last.bias)

        initialize_weights(self.seg_head)
        initialize_weights(self.dec1)
        initialize_weights(self.skip_proj)

        n = sum(p.numel() for p in self.parameters())
        t = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        print(
            f'[SwinHybrid-V6] '
            f'in_ch={in_channels} | '
            f'mixstyle={use_mixstyle} | '
            f'params={n:,} | '
            f'trainable={t:,}'
        )

    def train(self, mode=True):
        super().train(mode)
        freeze_batchnorm(self.encoder.cnn)
        return self

    def forward(self, x):

        B, C, H, W = x.shape

        # Input projection
        x3 = self.input_adapter(x)

        # Encode
        enc = self.encoder(x3)

        l1 = enc['l1']
        l2 = enc['l2']

        s1 = enc['s1']
        s2 = enc['s2']
        s3 = enc['s3']

        # ─────────────────────────────────────
        # SKIP ALIGNMENT
        # ─────────────────────────────────────
        if l2.shape[2:] != s1.shape[2:]:

            s1 = F.interpolate(
                s1,
                size=l2.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        skip_mid = self.skip_proj(
            torch.cat([s1, l2], dim=1)
        )

        # ─────────────────────────────────────
        # DECODER
        # ─────────────────────────────────────
        d4 = self.dec4(
            s3,
            self.ag3(s3, s2)
        )

        d3 = self.dec3(
            d4,
            self.ag2(d4, skip_mid)
        )

        if l1.shape[2:] != (H // 4, W // 4):

            l1 = F.interpolate(
                l1,
                size=(H // 4, W // 4),
                mode='bilinear',
                align_corners=False
            )

        d2 = self.dec2(
            d3,
            self.ag1(d3, l1)
        )

        d1 = self.dec1(d2)

        # ─────────────────────────────────────
        # SEGMENTATION
        # ─────────────────────────────────────
        seg = self.seg_head(d1)

        if seg.shape[2:] != (H, W):

            seg = F.interpolate(
                seg,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        seg = torch.nan_to_num(
            seg,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Deep supervision
        ds_outs = []

        if self.training:

            ds_outs = [
                self.ds1(d4, (H, W)),
                self.ds2(d3, (H, W)),
            ]

        # ─────────────────────────────────────
        # CLASSIFICATION
        # ─────────────────────────────────────
        seg_attn = torch.sigmoid(seg.detach())

        seg_attn = F.interpolate(
            seg_attn,
            size=s3.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        seg_attn = seg_attn / (
            seg_attn.sum(
                dim=[2, 3],
                keepdim=True
            ).clamp(min=1e-6)
        )

        weighted = (
            s3 * seg_attn
        ).sum(dim=[2, 3])

        weighted = torch.nan_to_num(
            weighted,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        cls = self.cls_proj(weighted)

        return {
            'seg': seg,
            'cls': cls,
            'ds': ds_outs,
        }


def build_model(
    in_channels=5,
    use_mixstyle=True,
    **kwargs
):
    return SwinHybridMultiTask(
        in_channels=in_channels,
        use_mixstyle=use_mixstyle,
    )