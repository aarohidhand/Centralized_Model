"""
datasets/fedbca_dataset.py
===========================
PyTorch Dataset for FedBCa multi-task pipeline.

FIXES:
- clip range [-5.0, 5.0] (was [-3.0, 3.0] — destroyed tumor hyperintensity)
- use_cls_loss uses has_tumor OR is_best_slice (denser classification signal)
- MIBC tumor slices get weight 4.5 (was 3.0) — minority class oversampling
- nan_to_num uses posinf=5.0, neginf=-5.0 (matches new clip range)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path


class FedBCaDataset(Dataset):

    def __init__(
        self,
        df,
        preprocessed_root: str,
        mode:              str  = 'train',
        model_type:        str  = 'cnn',
        use_augmentation:  bool = False,
        augmentation             = None,
        use_gin_only:      bool = False,
        use_resnet_aug:    bool = False,
    ):
        self.records   = df.to_dict('records')
        self.data_root = Path(preprocessed_root) / model_type
        self.mode      = mode
        self.augment   = use_augmentation and (mode == 'train')

        # Select augmentation pipeline
        if self.augment:
            if augmentation is not None:
                self.aug_fn = augmentation
            elif use_gin_only:
                from preprocessing.augmentations import GINOnlyPipeline
                self.aug_fn = GINOnlyPipeline()
            elif use_resnet_aug:
                from preprocessing.augmentations import ResNet50AugmentationPipeline
                self.aug_fn = ResNet50AugmentationPipeline()
            else:
                from preprocessing.augmentations import AugmentationPipeline
                self.aug_fn = AugmentationPipeline()
        else:
            from preprocessing.augmentations import MinimalAugmentation
            self.aug_fn = MinimalAugmentation()

        # Compute sampler weights
        self.sample_weights = self._compute_weights()

    def _compute_weights(self):
        weights = []
        for r in self.records:
            w = 1.0
            if int(r.get('has_tumor', 0)):
                if int(r.get('label', 0)) == 1:   # MIBC tumor slice
                    w = 4.5                        # highest weight
                else:                              # NMIBC tumor slice
                    w = 3.0
            elif int(r.get('is_best_slice', 0)):
                w = 1.5                            # best slice gets slight boost
            weights.append(w)
        return weights

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        fname = rec['fname']

        img_path  = self.data_root / f'{fname}_img.npy'
        mask_path = self.data_root / f'{fname}_mask.npy'

        img  = np.load(str(img_path)).astype(np.float32)   # (5, H, W)
        mask = np.load(str(mask_path)).astype(np.float32)  # (H, W)

        # Safety: handle single-channel stored files
        if img.ndim == 2:
            img = np.stack([img] * 5, axis=0)

        # Clip to [-5, 5] — preserves tumor hyperintensity range
        # (was [-3, 3] which destroyed discriminative tumor contrast)
        img = np.nan_to_num(img, nan=0.0, posinf=5.0, neginf=-5.0)
        img = np.clip(img, -5.0, 5.0)

        # Augmentation
        if self.augment:
            img, mask_aug = self.aug_fn(img, mask.astype(np.uint8))
            mask = mask_aug.astype(np.float32)

        img_t  = torch.from_numpy(img)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        label      = int(rec.get('label', 0))
        has_tumor  = int(rec.get('has_tumor', 0))
        is_best    = int(rec.get('is_best_slice', 0))

        # use_cls_loss: tumor slices + best slice
        # Denser classification supervision without random empty-slice noise
        use_cls = int(has_tumor or is_best)

        return {
            'image':       img_t,
            'mask':        mask_t,
            'label':       torch.tensor(label,     dtype=torch.long),
            'has_tumor':   torch.tensor(has_tumor, dtype=torch.long),
            'use_cls_loss':torch.tensor(use_cls,   dtype=torch.float32),
            'patient_id':  str(rec.get('patient_id', f'p{idx}')),
            'center':      int(rec.get('center', 0)),
            'slice_idx':   int(rec.get('slice_idx', 0)),
        }


def build_dataloader(
    dataset:        FedBCaDataset,
    batch_size:     int,
    num_workers:    int  = 4,
    use_sampler:    bool = True,
    pin_memory:     bool = True,
) -> DataLoader:

    if use_sampler and dataset.mode == 'train':
        sampler = WeightedRandomSampler(
            weights     = dataset.sample_weights,
            num_samples = len(dataset),
            replacement = True,
        )
        return DataLoader(
            dataset,
            batch_size  = batch_size,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,
        )

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (dataset.mode == 'train'),
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = (dataset.mode == 'train'),
    )
