"""
utils/trainer.py
=================
Universal training engine for EfficientNet, ResNet50, and MedSAM.

FIXES:
- min_lr_ratio read from config (not hardcoded 0.02)
- rampup_done = cls_rampup * 2 + 5 (classifier warms up before combined scoring)
- use_amp read from config (not fragile exp_name string parsing)
- input_proj included in encoder param group (slow LR)
- evaluate() receives correct amp flag
- staged cls_weight ramp: 0 → target over cls_rampup_epochs
"""

import os
import math
import time
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast


# ─────────────────────────────────────────────
# LR SCHEDULE
# ─────────────────────────────────────────────

def warmup_cosine(step, total_steps, warmup_steps, min_lr_ratio=0.05):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress   = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val


# ─────────────────────────────────────────────
# OPTIMIZER BUILDER
# ─────────────────────────────────────────────

def build_optimizer(model, config):
    """
    Differential learning rates:
      encoder group (slow): pretrained backbone layers
      decoder group (fast): decoder, heads, FPN

    input_proj is in encoder group — adapts slowly with backbone.
    """
    lr_base    = config['lr']
    lr_enc     = lr_base * config.get('enc_lr_ratio', 0.10)
    wd         = config.get('weight_decay', 1e-4)

    enc_keywords = {
        'encoder', 'image_encoder', 'swin', 'cnn',
        'layer1', 'layer2', 'layer3', 'layer4',
        'stem', 'input_proj', 'input_adapter',
    }

    enc_params, dec_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in enc_keywords):
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': enc_params, 'lr': lr_enc},
        {'params': dec_params, 'lr': lr_base},
    ], weight_decay=wd)

    return optimizer


# ─────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                criterion, config, exp_name, save_dir,
                device=None):
    """
    Universal training loop for all three models.

    Args:
        model        : nn.Module (EfficientNet, ResNet50, or MedSAM)
        train_loader : DataLoader for training split
        val_loader   : DataLoader for validation split
        criterion    : JointLoss instance
        config       : dict from yaml config
        exp_name     : experiment name string
        save_dir     : Path to save checkpoints and history
        device       : torch.device (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = Path(save_dir) / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # ── AMP: read from config, NOT from exp_name string ──
    use_amp = config.get('use_amp', False) and device.type == 'cuda'
    scaler  = GradScaler(enabled=use_amp)

    # ── Optimizer + scheduler ────────────────────────
    optimizer    = build_optimizer(model, config)
    epochs       = config.get('epochs', 200)
    accum_steps  = config.get('accum_steps', 6)
    steps_per_ep = max(1, len(train_loader))
    total_steps  = epochs * steps_per_ep // accum_steps
    warmup_steps = config.get('warmup_epochs', 10) * steps_per_ep // accum_steps
    min_lr_ratio = config.get('min_lr_ratio', 0.05)   # FIX: from config

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine(
            step, total_steps, warmup_steps, min_lr_ratio
        ),
    )

    # ── Training params ──────────────────────────────
    patience       = config.get('patience', 80)
    eval_every     = config.get('eval_every', 5)
    grad_clip      = config.get('grad_clip', 1.0)
    cls_rampup     = config.get('cls_rampup_epochs', 10)
    target_cls_w   = config.get('cls_weight', 0.15)

    # rampup_done: switch from loss-based to combined-score checkpointing
    # after classifier has had full gradient for at least 5 epochs
    rampup_done    = cls_rampup * 2 + 5

    # ── State ────────────────────────────────────────
    best_combined  = -1.0
    best_loss      = float('inf')
    patience_count = 0
    global_step    = 0

    history = defaultdict(list)

    from utils.metrics import evaluate

    print(f'\n[{exp_name}] Starting training')
    print(f'  Device    : {device}')
    print(f'  AMP       : {use_amp}')
    print(f'  Epochs    : {epochs}')
    print(f'  Eff batch : {config.get("batch_size",4) * accum_steps}')
    print(f'  Rampup    : cls gradient starts ep {cls_rampup}, '
          f'full at ep {cls_rampup * 2}, combined scoring from ep {rampup_done}')
    print()

    for epoch in range(epochs):

        # ── Staged cls_weight ramp ────────────────────
        if epoch < cls_rampup:
            effective_cls = 0.0
        elif epoch < cls_rampup * 2:
            progress      = (epoch - cls_rampup) / float(cls_rampup)
            effective_cls = target_cls_w * progress
        else:
            effective_cls = target_cls_w
        criterion.cls_weight = effective_cls

        # ── Train ────────────────────────────────────
        model.train()

        ep_loss = ep_seg = ep_cls = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images   = batch['image'].to(device, non_blocking=True)
            masks    = batch['mask'].to(device, non_blocking=True)
            labels   = batch['label'].to(device, non_blocking=True)
            cls_mask = batch['use_cls_loss'].float().to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                outputs   = model(images)
                loss_dict = criterion(outputs, masks, labels, cls_mask=cls_mask)
                loss      = loss_dict['loss'] / accum_steps

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm_val = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                if torch.isfinite(grad_norm_val):
                    scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            ep_loss += loss_dict['loss'].item()
            ep_seg  += loss_dict['seg_loss'].item()
            ep_cls  += loss_dict['cls_loss'].item()

        n_steps  = max(1, len(train_loader))
        avg_loss = ep_loss / n_steps
        avg_seg  = ep_seg  / n_steps
        avg_cls  = ep_cls  / n_steps
        cur_lr   = optimizer.param_groups[1]['lr']

        history['train_loss'].append(avg_loss)
        history['seg_loss'].append(avg_seg)
        history['cls_loss'].append(avg_cls)
        history['lr'].append(cur_lr)
        history['epoch'].append(epoch)

        # ── Evaluate ──────────────────────────────────
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:

            metrics  = evaluate(model, val_loader, device, amp=use_amp)
            combined = metrics['combined_score']
            val_dsc  = metrics['tumour_dsc']
            val_auc  = metrics['auc']

            history['val_dsc'].append(val_dsc)
            history['val_auc'].append(val_auc)
            history['val_combined'].append(combined)

            # Checkpoint selection:
            # Early epochs (< rampup_done): save on training loss
            # Later epochs: save on combined = 0.55*DSC + 0.45*AUC
            if epoch < rampup_done:
                improved = avg_loss < best_loss
                if improved:
                    best_loss = avg_loss
            else:
                improved = combined > best_combined
                if improved:
                    best_combined = combined

            star = '★' if improved else ''

            print(
                f'Ep {epoch+1:3d}/{epochs} | '
                f'Loss={avg_loss:.4f} '
                f'(seg={avg_seg:.3f} cls={avg_cls:.3f}) | '
                f'tDSC={val_dsc:.4f} '
                f'AUC={val_auc:.4f} '
                f'comb={combined:.4f}{star} | '
                f'LR={cur_lr:.2e} | '
                f'cls_w={effective_cls:.3f}'
            )

            if improved:
                torch.save({
                    'epoch':     epoch,
                    'state':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics':   metrics,
                    'config':    config,
                }, save_dir / 'checkpoint_best.pt')
                patience_count = 0
            else:
                patience_count += eval_every
                if patience_count >= patience:
                    print(f'\n[EARLY STOP] No improvement for {patience} epochs.')
                    break

    # Save history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(dict(history), f, indent=2)

    # Load best checkpoint metrics
    ckpt = torch.load(save_dir / 'checkpoint_best.pt', map_location='cpu')
    best_m = ckpt['metrics']

    print(f'\n[{exp_name}] Done.')
    print(f'  Best DSC : {best_m["tumour_dsc"]:.4f}')
    print(f'  Best AUC : {best_m["auc"]:.4f}')
    print(f'  Combined : {best_m["combined_score"]:.4f}')

    return {
        'best_dsc':  best_m['tumour_dsc'],
        'best_auc':  best_m['auc'],
        'combined':  best_m['combined_score'],
        'save_dir':  str(save_dir),
        'history':   dict(history),
    }
