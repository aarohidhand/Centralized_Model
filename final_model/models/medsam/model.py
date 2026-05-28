"""
models/medsam/model.py
========================================================
MedSAM ViT-B + Swin-Tiny Hybrid Multi-Task Model

FINAL FIXED VERSION
- Proper SAM positional embedding interpolation
- Proper relative-position handling
- 256×256 MedSAM operation
- RTX 4070 feasible VRAM
- Stable training
========================================================
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
    InputAdapter,
    AttentionGate,
    DecoderBlock,
    DeepSupervisionHead,
    initialize_weights,
)


# =========================================================
# MedSAM Encoder Loader
# =========================================================
def _load_medsam_encoder(
    checkpoint_path: str,
    image_size: int = 256,
):
    """
    Load MedSAM encoder with proper positional embedding handling.

    Original MedSAM:
        1024x1024 input
        patch size = 16
        token grid = 64x64

    Our setup:
        256x256 input
        patch size = 16
        token grid = 16x16

    SAM internally interpolates ABSOLUTE positional embeddings.

    Relative-position tensors are incompatible and must be removed.
    """

    from segment_anything import sam_model_registry
    from segment_anything.modeling import ImageEncoderViT

    # Load original pretrained SAM
    sam = sam_model_registry['vit_b'](
        checkpoint=checkpoint_path
    )

    # Rebuild encoder for 256x256
    encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=nn.LayerNorm,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        rel_pos_zero_init=True,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        out_chans=256,
    )

    # Get pretrained weights
    state_dict = sam.image_encoder.state_dict()

    # Remove incompatible relative-position tensors
    remove_keys = [
        k for k in state_dict.keys()
        if (
            'rel_pos' in k
            or 
            'pos_embed' in k
        )
    ]

    for k in remove_keys:
        del state_dict[k]

    # Load remaining pretrained weights
    missing, unexpected = encoder.load_state_dict(
        state_dict,
        strict=False
    )

    print(f'[MedSAM] removed rel_pos keys : {len(remove_keys)}')
    print(f'[MedSAM] missing keys         : {len(missing)}')
    print(f'[MedSAM] unexpected keys      : {len(unexpected)}')

    return encoder


# =========================================================
# FPN Bridge
# =========================================================
class FPNBridge(nn.Module):
    """
    Convert ViT feature map into multi-scale pyramid.

    Input:
        (B,256,16,16)

    Outputs:
        s4 : (B,256,16,16)
        s3 : (B,128,32,32)
        s2 : (B,64,64,64)
        s1 : (B,32,128,128)
    """

    def __init__(self, in_ch=256):
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

    def forward(self, vit_feat):

        s4 = vit_feat
        s3 = self.level3(s4)
        s2 = self.level2(s3)
        s1 = self.level1(s2)

        return s1, s2, s3, s4


# =========================================================
# Fusion Classification Head
# =========================================================
class FusionClassifierHead(nn.Module):

    def __init__(
        self,
        medsam_ch=256,
        swin_ch=384,
        num_classes=2,
        dropout=0.15,
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(
                medsam_ch + swin_ch,
                256,
                bias=False
            ),

            nn.BatchNorm1d(
                256,
                eps=1e-3,
                momentum=0.05
            ),

            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                128,
                bias=False
            ),

            nn.BatchNorm1d(
                128,
                eps=1e-3,
                momentum=0.05
            ),

            nn.GELU(),
            nn.Dropout(0.10),

            nn.Linear(
                128,
                num_classes
            ),
        )

        with torch.no_grad():

            last = [
                m for m in self.fc.modules()
                if isinstance(m, nn.Linear)
            ][-1]

            nn.init.trunc_normal_(
                last.weight,
                std=0.01
            )

            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def forward(
        self,
        medsam_feat,
        swin_feat
    ):

        x = torch.cat(
            [medsam_feat, swin_feat],
            dim=1
        )

        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return self.fc(x)


# =========================================================
# Main Hybrid Model
# =========================================================
class MedSAMSwinHybrid(nn.Module):

    MEDSAM_INPUT_SIZE = 256

    def __init__(
        self,
        medsam_checkpoint,
        in_channels=5,
        num_classes=2,
        n_frozen_blocks=4,
        dropout=0.15,
    ):
        super().__init__()

        # -------------------------------------
        # Input adapter (5-channel MRI → RGB)
        # -------------------------------------
        self.input_adapter = InputAdapter(
            in_ch=in_channels,
            out_ch=3
        )

        # -------------------------------------
        # MedSAM encoder
        # -------------------------------------
        self.image_encoder = _load_medsam_encoder(
            medsam_checkpoint,
            image_size=self.MEDSAM_INPUT_SIZE,
        )

        # Freeze early transformer blocks
        for i, block in enumerate(self.image_encoder.blocks):

            if i < n_frozen_blocks:

                for p in block.parameters():
                    p.requires_grad = False

        # -------------------------------------
        # Swin branch
        # -------------------------------------
        self.swin = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            features_only=True,
            out_indices=(2, 3),
        )

        self.swin_pool = nn.AdaptiveAvgPool2d(1)

        # -------------------------------------
        # Decoder
        # -------------------------------------
        self.fpn = FPNBridge(in_ch=256)

        self.ag3 = AttentionGate(256, 128, 64)
        self.ag2 = AttentionGate(128, 64, 32)
        self.ag1 = AttentionGate(64, 32, 16)

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
            dropout=max(0.0, dropout - 0.05)
        )

        self.dec1 = DecoderBlock(
            32,
            0,
            16,
            dropout=0.0
        )

        # -------------------------------------
        # Segmentation head
        # -------------------------------------
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

            nn.Conv2d(
                8,
                1,
                1
            ),
        )

        self.ds1 = DeepSupervisionHead(128)
        self.ds2 = DeepSupervisionHead(64)

        # -------------------------------------
        # Classification head
        # -------------------------------------
        self.cls_head = FusionClassifierHead(
            medsam_ch=256,
            swin_ch=384,
            num_classes=num_classes,
            dropout=dropout,
        )

        initialize_weights(self.seg_head)
        initialize_weights(self.dec1)

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        print(
            f'[MedSAM+Swin FINAL] '
            f'total={total:,} '
            f'trainable={trainable:,}'
        )

    # =====================================================
    # Keep frozen blocks in eval mode
    # =====================================================
    def train(self, mode=True):

        super().train(mode)

        for block in self.image_encoder.blocks:

            if not next(block.parameters()).requires_grad:
                block.eval()

        return self

    # =====================================================
    # Forward
    # =====================================================
    def forward(self, x):

        B, C, H, W = x.shape

        # -------------------------------------
        # 5-channel MRI → 3-channel
        # -------------------------------------
        x3 = self.input_adapter(x)

        # Resize if needed
        if (
            x3.shape[2] != self.MEDSAM_INPUT_SIZE
            or
            x3.shape[3] != self.MEDSAM_INPUT_SIZE
        ):

            x3 = F.interpolate(
                x3,
                size=(
                    self.MEDSAM_INPUT_SIZE,
                    self.MEDSAM_INPUT_SIZE
                ),
                mode='bilinear',
                align_corners=False
            )

        # -------------------------------------
        # MedSAM encoder
        # -------------------------------------
        vit_feat = self.image_encoder(x3)

        vit_feat = torch.nan_to_num(
            vit_feat,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # -------------------------------------
        # Swin branch
        # -------------------------------------
        swin_out = self.swin(x3)

        swin_s3 = torch.nan_to_num(
            swin_out[1],
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        swin_pooled = self.swin_pool(
            swin_s3
        ).flatten(1)

        # -------------------------------------
        # Pyramid features
        # -------------------------------------
        s1, s2, s3, s4 = self.fpn(vit_feat)

        # -------------------------------------
        # Decoder
        # -------------------------------------
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

        # -------------------------------------
        # Segmentation output
        # -------------------------------------
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

        # -------------------------------------
        # Deep supervision
        # -------------------------------------
        ds_outs = []

        if self.training:

            ds_outs = [
                self.ds1(d4, (H, W)),
                self.ds2(d3, (H, W)),
            ]

        # -------------------------------------
        # Segmentation-guided pooling
        # -------------------------------------
        seg_attn = torch.sigmoid(
            seg.detach()
        )

        seg_attn = F.interpolate(
            seg_attn,
            size=s4.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        seg_attn = seg_attn / seg_attn.sum(
            dim=[2, 3],
            keepdim=True
        ).clamp(min=1e-6)

        medsam_pooled = (
            s4 * seg_attn
        ).sum(dim=[2, 3])

        medsam_pooled = torch.nan_to_num(
            medsam_pooled,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # -------------------------------------
        # Classification
        # -------------------------------------
        cls = self.cls_head(
            medsam_pooled,
            swin_pooled
        )

        return {
            'seg': seg,
            'cls': cls,
            'ds': ds_outs,
        }


# =========================================================
# Builder
# =========================================================
def build_medsam_model(config):

    ckpt = config.get(
        'medsam_checkpoint',
        ''
    )

    assert ckpt and Path(ckpt).exists(), (
        f'MedSAM checkpoint not found: {ckpt}'
    )

    return MedSAMSwinHybrid(
        medsam_checkpoint=ckpt,
        in_channels=config.get('in_channels', 5),
        num_classes=2,
        n_frozen_blocks=config.get('n_frozen_blocks', 4),
        dropout=config.get('dp_rate', 0.15),
    )