import random
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2

import sys
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except:
    BASE_DIR = Path.cwd()

sys.path.insert(0, str(BASE_DIR))


class JointTransform:
    def __init__(self, augment=True):
        self.augment = augment

    def elastic(self, img, mask):
        if random.random() > 0.3:
            return img, mask

        img_np = np.array(img)
        mask_np = np.array(mask)

        shape = img_np.shape

        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), 0) * 5
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), 0) * 5

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        img_np = cv2.remap(img_np, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        mask_np = cv2.remap(mask_np, map_x, map_y, interpolation=cv2.INTER_NEAREST)

        return TF.to_pil_image(img_np), TF.to_pil_image(mask_np)

    def __call__(self, img, mask):
        if not self.augment:
            return img, mask

        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        if random.random() < 0.6:
            angle = random.uniform(-25, 25)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            shear = random.uniform(-15, 15)
            tx = random.randint(-10, 10)
            ty = random.randint(-10, 10)

            img = TF.affine(
                img,
                angle=0,
                translate=[tx, ty],
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR
            )

            mask = TF.affine(
                mask,
                angle=0,
                translate=[tx, ty],
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST
            )

        img, mask = self.elastic(img, mask)

        if random.random() < 0.4:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)

        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.5)
            img = TF.adjust_gamma(img, gamma)

        if random.random() < 0.3:
            img_t = TF.to_tensor(img)
            noise = torch.randn_like(img_t) * 0.05
            img_t = torch.clamp(img_t + noise, 0.0, 1.0)
            img = TF.to_pil_image(img_t)

        return img, mask