"""
models/efficientnet/model.py
=============================

EfficientNet-B0 Multi-Task Model
CORRECTED VERSION

Correct timm EfficientNet-B0 features_only outputs:

    f0 :  16 channels
    f1 :  24 channels
    f2 :  40 channels
    f3 : 112 channels
    f4 : 320 channels

The previous "112 bottleneck" fix was incorrect.
timm features_only=True still outputs 320-channel deepest features.

This file restores the correct encoder dimensions.
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
    SimpleClassifier,
    initialize_weights,
)


class EfficientNetMultiTask(nn.Module):

    # CORRECT timm EfficientNet-B0 feature dimensions
    ENC_CH = [16, 24, 40, 112, 320]

    def __init__(
        self,
        in_channels: int = 5,
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()

        # -------------------------------------------------
        # INPUT ADAPTER
        # -------------------------------------------------

        self.input_adapter = InputAdapter(
            in_ch=in_channels,
            out_ch=3,
        )

        # -------------------------------------------------
        # ENCODER
        # -------------------------------------------------

        self.encoder = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        freeze_batchnorm(self.encoder)

        ch = self.ENC_CH

        # -------------------------------------------------
        # ATTENTION GATES
        # -------------------------------------------------

        self.ag4 = AttentionGate(
            F_g=ch[4],   # 320
            F_l=ch[3],   # 112
            F_int=56,
        )

        self.ag3 = AttentionGate(
            F_g=64,
            F_l=ch[2],   # 40
            F_int=20,
        )

        self.ag2 = AttentionGate(
            F_g=32,
            F_l=ch[1],   # 24
            F_int=12,
        )

        self.ag1 = AttentionGate(
            F_g=16,
            F_l=ch[0],   # 16
            F_int=8,
        )

        # -------------------------------------------------
        # DECODER
        # -------------------------------------------------

        self.dec4 = DecoderBlock(
            in_ch=ch[4],      # 320
            skip_ch=ch[3],    # 112
            out_ch=64,
            dropout=dropout,
        )

        self.dec3 = DecoderBlock(
            in_ch=64,
            skip_ch=ch[2],    # 40
            out_ch=32,
            dropout=dropout,
        )

        self.dec2 = DecoderBlock(
            in_ch=32,
            skip_ch=ch[1],    # 24
            out_ch=16,
            dropout=max(0.0, dropout - 0.05),
        )

        self.dec1 = DecoderBlock(
            in_ch=16,
            skip_ch=ch[0],    # 16
            out_ch=8,
            dropout=0.0,
        )

        self.dec0 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False,
            ),

            nn.Conv2d(
                8,
                8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            gn(8),
            nn.GELU(),
        )

        # -------------------------------------------------
        # SEGMENTATION HEAD
        # -------------------------------------------------

        self.seg_head = nn.Sequential(

            nn.Conv2d(
                8,
                4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            gn(4),
            nn.GELU(),

            nn.Conv2d(
                4,
                1,
                kernel_size=1,
            ),
        )

        # -------------------------------------------------
        # DEEP SUPERVISION
        # -------------------------------------------------

        self.ds_head1 = DeepSupervisionHead(64)
        self.ds_head2 = DeepSupervisionHead(32)

        # -------------------------------------------------
        # CLASSIFICATION HEAD
        # -------------------------------------------------

        self.cls_head = SimpleClassifier(
            feat_ch=ch[4],   # 320
            num_classes=num_classes,
            dropout=0.20,
        )

        # -------------------------------------------------
        # INIT
        # -------------------------------------------------

        initialize_weights(self.seg_head)
        initialize_weights(self.dec0)

        n = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        print(
            f'[EfficientNet-B0-CORRECTED] '
            f'in_ch={in_channels} | params={n:,}'
        )

    # -----------------------------------------------------
    # TRAIN OVERRIDE
    # -----------------------------------------------------

    def train(self, mode=True):

        super().train(mode)

        # Keep BN frozen for DG stability
        freeze_batchnorm(self.encoder)

        return self

    # -----------------------------------------------------
    # FORWARD
    # -----------------------------------------------------

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape

        # ---------------------------------------------
        # INPUT ADAPTATION
        # ---------------------------------------------

        x3 = self.input_adapter(x)

        # ---------------------------------------------
        # ENCODER
        # ---------------------------------------------

        feats = self.encoder(x3)

        f0, f1, f2, f3, f4 = feats

        # ---------------------------------------------
        # DECODER
        # ---------------------------------------------

        d4 = self.dec4(
            f4,
            self.ag4(f4, f3),
        )

        d3 = self.dec3(
            d4,
            self.ag3(d4, f2),
        )

        d2 = self.dec2(
            d3,
            self.ag2(d3, f1),
        )

        d1 = self.dec1(
            d2,
            self.ag1(d2, f0),
        )

        d0 = self.dec0(d1)

        # ---------------------------------------------
        # SEGMENTATION
        # ---------------------------------------------

        seg = self.seg_head(d0)

        if seg.shape[2:] != (H, W):

            seg = F.interpolate(
                seg,
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            )

        seg = torch.nan_to_num(
            seg,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ---------------------------------------------
        # DEEP SUPERVISION
        # ---------------------------------------------

        ds_outs = []

        if self.training:

            ds_outs = [

                self.ds_head1(
                    d4,
                    (H, W),
                ),

                self.ds_head2(
                    d3,
                    (H, W),
                ),
            ]

        # ---------------------------------------------
        # CLASSIFICATION
        # ---------------------------------------------

        cls = self.cls_head(f4)

        return {
            'seg': seg,
            'cls': cls,
            'ds': ds_outs,
        }


def build_model(
    pretrained=True,
    in_channels=5,
    num_classes=2,
    **kwargs
):

    return EfficientNetMultiTask(
        in_channels=in_channels,
        pretrained=pretrained,
        num_classes=num_classes,
    )