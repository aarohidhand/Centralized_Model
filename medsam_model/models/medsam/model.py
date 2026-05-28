"""
models/medsam/model.py
======================
MedSAM ViT-B Multi-Task Model for FedBCa.

Architecture:
    5ch MRI input
        → InputAdapter (5→3, mean-init)
        → MedSAM image encoder (ViT-B, pretrained 1.5M medical pairs)
        → FPNBridge (single 256ch ViT output → 4-scale pyramid)
        → Attention U-Net Decoder (4 stages, GroupNorm)
        → Segmentation head (B, 1, H, W)
        → Deep supervision heads (2 auxiliary)
        → SimpleClassifier on bottleneck (B, 2)

Why MedSAM over ImageNet backbones:
    - Pretrained on 1.5M medical image-mask pairs across 11 modalities
    - Features are boundary-discriminative by training objective
    - ViT self-attention is inherently scanner-agnostic
    - Partially solves domain shift before GIN augmentation even begins

VRAM on RTX 4070 12GB:
    ViT-B at 256×256: ~4.5 GB
    FPN + decoder:    ~2.0 GB
    batch=4, amp=True: ~7.5 GB total  ← fits comfortably

Input: 256×256 (uses CNN preprocessing)
MedSAM was designed for 1024×1024.
We resize to 256 with interpolated positional embeddings.
Performance is slightly lower than 1024 but VRAM-feasible.
"""

 

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    MixStyle,
)


# ──────────────────────────────────────────────
# POSITIONAL EMBEDDING RESIZE
# ──────────────────────────────────────────────

def resize_pos_embed(
    pos_embed: torch.Tensor,
    target_hw: tuple,
):

    pe = pos_embed.permute(0, 3, 1, 2)

    pe = F.interpolate(
        pe,
        size=target_hw,
        mode='bicubic',
        align_corners=False,
    )

    pe = pe.permute(0, 2, 3, 1)

    return pe


# ──────────────────────────────────────────────
# FPN BRIDGE
# ──────────────────────────────────────────────

class FPNBridge(nn.Module):

    def __init__(self, in_ch: int = 256):

        super().__init__()

        self.level3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            ),
            nn.Conv2d(
                in_ch,
                128,
                3,
                padding=1,
                bias=False
            ),
            gn(128),
            nn.GELU(),
        )

        self.level2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            ),
            nn.Conv2d(
                128,
                64,
                3,
                padding=1,
                bias=False
            ),
            gn(64),
            nn.GELU(),
        )

        self.level1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False
            ),
            nn.Conv2d(
                64,
                32,
                3,
                padding=1,
                bias=False
            ),
            gn(32),
            nn.GELU(),
        )

        initialize_weights(self)

    def forward(self, vit_feat: torch.Tensor):

        s4 = vit_feat
        s3 = self.level3(s4)
        s2 = self.level2(s3)
        s1 = self.level1(s2)

        return s1, s2, s3, s4


# ──────────────────────────────────────────────
# MEDSAM MULTI-TASK MODEL
# ──────────────────────────────────────────────

class MedSAMMultiTask(nn.Module):

    def __init__(
        self,
        medsam_checkpoint: str,
        in_channels: int = 5,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        dropout: float = 0.15,
    ):

        super().__init__()

        # ──────────────────────────────────────
        # LOAD MEDSAM
        # ──────────────────────────────────────

        from segment_anything import sam_model_registry

        sam = sam_model_registry['vit_b'](
            checkpoint=medsam_checkpoint
        )

        self.image_encoder = sam.image_encoder

        # ──────────────────────────────────────
        # RESIZE POSITIONAL EMBEDDINGS
        # 1024×1024 → 256×256
        # 64×64 PE  → 16×16 PE
        # ──────────────────────────────────────

        target_hw = (16, 16)

        with torch.no_grad():

            self.image_encoder.pos_embed = nn.Parameter(
                resize_pos_embed(
                    self.image_encoder.pos_embed,
                    target_hw
                )
            )

        print(
            f'[MedSAM] resized pos_embed -> '
            f'{self.image_encoder.pos_embed.shape}'
        )

        # ──────────────────────────────────────
        # FREEZE ENCODER IF REQUESTED
        # ──────────────────────────────────────

        if freeze_encoder:

            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # ──────────────────────────────────────
        # INPUT ADAPTER
        # ──────────────────────────────────────

        self.input_adapter = InputAdapter(
            in_ch=in_channels,
            out_ch=3,
        )

        # ──────────────────────────────────────
        # FPN
        # ──────────────────────────────────────

        self.fpn = FPNBridge(in_ch=256)

        # ──────────────────────────────────────
        # DECODER
        # ──────────────────────────────────────

        self.ag3 = AttentionGate(256, 128, 64)
        self.ag2 = AttentionGate(128,  64, 32)
        self.ag1 = AttentionGate(64,   32, 16)

        self.dec4 = DecoderBlock(
            256,
            128,
            128,
            dropout=dropout
        )

        self.dec3 = DecoderBlock(
            128,
            64,
            64,
            dropout=dropout
        )

        self.dec2 = DecoderBlock(
            64,
            32,
            32,
            dropout=max(0, dropout - 0.05)
        )

        self.dec1 = DecoderBlock(
            32,
            0,
            16,
            dropout=0.0
        )

        # ──────────────────────────────────────
        # SEGMENTATION HEAD
        # ──────────────────────────────────────

        self.seg_head = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                3,
                padding=1,
                bias=False
            ),
            gn(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )

        # ──────────────────────────────────────
        # DEEP SUPERVISION
        # ──────────────────────────────────────

        self.ds_head1 = DeepSupervisionHead(128)
        self.ds_head2 = DeepSupervisionHead(64)

        # ──────────────────────────────────────
        # CLASSIFIER
        # ──────────────────────────────────────

        self.cls_head = SimpleClassifier(
            feat_ch=256,
            num_classes=num_classes,
            dropout=0.20,
        )

        initialize_weights(self.seg_head)
        initialize_weights(self.dec1)

        n_params = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        print(
            f'[MedSAM-MultiTask] '
            f'in_ch={in_channels} | '
            f'freeze_enc={freeze_encoder} | '
            f'params={n_params:,}'
        )

    def train(self, mode=True):

        super().train(mode)

        return self

    def forward(self, x: torch.Tensor) -> dict:

        B, C, H, W = x.shape

        # ──────────────────────────────────────
        # INPUT ADAPTER
        # ──────────────────────────────────────

        x3 = self.input_adapter(x)

        if H != 256 or W != 256:

            x3 = F.interpolate(
                x3,
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            )

        # ──────────────────────────────────────
        # MEDSAM ENCODER
        # ──────────────────────────────────────

        vit_feat = self.image_encoder(x3)

        vit_feat = torch.nan_to_num(
            vit_feat,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ──────────────────────────────────────
        # FPN
        # ──────────────────────────────────────

        s1, s2, s3, s4 = self.fpn(vit_feat)

        # ──────────────────────────────────────
        # DECODER
        # ──────────────────────────────────────

        d4 = self.dec4(
            s4,
            self.ag3(s4, s3)
        )

        d3 = self.dec3(
            d4,
            self.ag2(d4, s2)
        )

        d2 = self.dec2(
            d3,
            self.ag1(d3, s1)
        )

        d1 = self.dec1(d2)

        # ──────────────────────────────────────
        # SEGMENTATION
        # ──────────────────────────────────────

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

        # ──────────────────────────────────────
        # DEEP SUPERVISION
        # ──────────────────────────────────────

        ds_outs = []

        if self.training:

            out_size = (H, W)

            ds_outs = [
                self.ds_head1(d4, out_size),
                self.ds_head2(d3, out_size),
            ]

        # ──────────────────────────────────────
        # CLASSIFICATION
        # ──────────────────────────────────────

        cls = self.cls_head(s4)

        return {
            'seg': seg,
            'cls': cls,
            'ds':  ds_outs,
        }


# ──────────────────────────────────────────────
# BUILD FUNCTION
# ──────────────────────────────────────────────

def build_medsam_model(
    config: dict
) -> MedSAMMultiTask:

    ckpt = config.get(
        'medsam_checkpoint',
        ''
    )

    assert ckpt and Path(ckpt).exists(), (
        f'MedSAM checkpoint not found: {ckpt}\n'
        f'Download from: https://github.com/bowang-lab/MedSAM'
    )

    model = MedSAMMultiTask(
        medsam_checkpoint=ckpt,
        in_channels=config.get('in_channels', 5),
        num_classes=2,
        freeze_encoder=config.get('freeze_encoder', False),
        dropout=config.get('dp_rate', 0.15),
    )

    return model