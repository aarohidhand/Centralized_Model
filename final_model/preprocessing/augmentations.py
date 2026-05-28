"""
preprocessing/augmentations.py
================================
Hybrid augmentation system for FedBCa.

FINAL VERSION:
- Swin / EfficientNet / MedSAM keep current V3 augmentation
- ONLY ResNet uses old stable fedbca_medsam GIN behaviour
- Prevents ResNet collapse while preserving strong Swin results
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


# ──────────────────────────────────────────────
# GEOMETRIC AUGMENTATION
# ──────────────────────────────────────────────

class GeometricAugmentation:

    def __init__(self, p_flip: float = 0.5, max_angle: float = 15.0):
        self.p_flip = p_flip
        self.max_angle = max_angle

    def __call__(self, image: np.ndarray, mask: np.ndarray):

        if np.random.random() < self.p_flip:
            image = image[:, :, ::-1].copy()
            mask = mask[:, ::-1].copy()

        if np.random.random() < 0.70:

            angle = np.random.uniform(
                -self.max_angle,
                self.max_angle
            )

            H, W = mask.shape

            M = cv2.getRotationMatrix2D(
                (W / 2, H / 2),
                angle,
                1.0
            )

            for ch in range(image.shape[0]):

                image[ch] = cv2.warpAffine(
                    image[ch],
                    M,
                    (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT,
                )

            mask = cv2.warpAffine(
                mask.astype(np.uint8),
                M,
                (W, H),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REFLECT,
            )

        return image.astype(np.float32), mask.astype(np.uint8)


# ──────────────────────────────────────────────
# ELASTIC DEFORMATION
# ──────────────────────────────────────────────

class ElasticDeformation:

    def __init__(
        self,
        p: float = 0.35,
        alpha: float = 35.0,
        sigma: float = 5.0
    ):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image: np.ndarray, mask: np.ndarray):

        if np.random.random() > self.p:
            return image, mask

        H, W = mask.shape

        dx = gaussian_filter(
            np.random.randn(H, W),
            self.sigma
        ) * self.alpha

        dy = gaussian_filter(
            np.random.randn(H, W),
            self.sigma
        ) * self.alpha

        x, y = np.meshgrid(
            np.arange(W),
            np.arange(H)
        )

        map_x = np.clip(x + dx, 0, W - 1)
        map_y = np.clip(y + dy, 0, H - 1)

        coords = [map_y.ravel(), map_x.ravel()]

        out = image.copy()

        for ch in range(image.shape[0]):

            out[ch] = map_coordinates(
                image[ch],
                coords,
                order=1,
                mode='reflect'
            ).reshape(H, W)

        out_mask = map_coordinates(
            mask.astype(np.float32),
            coords,
            order=0,
            mode='reflect'
        ).reshape(H, W)

        return out.astype(np.float32), out_mask.astype(np.uint8)


# ──────────────────────────────────────────────
# CURRENT V3 GIN
# (Used by Swin / EfficientNet / MedSAM)
# ──────────────────────────────────────────────

class GINAugmentation:

    def __init__(
        self,
        n_layers: int = 4,
        p: float = 0.70
    ):
        self.n_layers = n_layers
        self.p = p

    def _build_net(self):

        import torch.nn as nn

        layers = []

        in_ch = 1

        for _ in range(self.n_layers):

            out_ch = int(np.random.randint(2, 8))

            layers += [
                nn.Conv2d(in_ch, out_ch, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, 1))

        net = nn.Sequential(*layers)

        for m in net.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1)
                nn.init.zeros_(m.bias)

        return net

    def __call__(self, image: np.ndarray) -> np.ndarray:

        if np.random.random() > self.p:
            return image

        import torch

        alpha = np.random.uniform(0.20, 0.60)

        net = self._build_net()

        result = image.copy()

        with torch.no_grad():

            for ch in range(image.shape[0]):

                x_t = torch.FloatTensor(
                    image[ch]
                ).unsqueeze(0).unsqueeze(0)

                out = net(x_t).squeeze().numpy()

                blended = (
                    alpha * out +
                    (1 - alpha) * image[ch]
                )

                orig_norm = np.linalg.norm(image[ch])

                blend_norm = np.linalg.norm(blended) + 1e-8

                if orig_norm > 1e-6:
                    result[ch] = (
                        blended *
                        orig_norm /
                        blend_norm
                    )
                else:
                    result[ch] = blended

        return result.astype(np.float32)


# ──────────────────────────────────────────────
# OLD STABLE RESNET GIN
# ──────────────────────────────────────────────

class ResNetGINAugmentation:

    def __init__(
        self,
        n_layers: int = 3,
        p: float = 0.30
    ):
        self.n_layers = n_layers
        self.p = p

    def _build_net(self):

        import torch.nn as nn

        layers = []

        in_ch = 1

        for _ in range(self.n_layers):

            out_ch = int(np.random.randint(2, 6))

            layers += [
                nn.Conv2d(in_ch, out_ch, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ]

            in_ch = out_ch

        layers.append(nn.Conv2d(in_ch, 1, 1))

        net = nn.Sequential(*layers)

        for m in net.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.zeros_(m.bias)

        return net

    def __call__(self, image: np.ndarray) -> np.ndarray:

        if np.random.random() > self.p:
            return image

        import torch

        alpha = np.random.uniform(0.08, 0.22)

        net = self._build_net()

        result = image.copy()

        with torch.no_grad():

            for ch in range(image.shape[0]):

                x_t = torch.FloatTensor(
                    image[ch]
                ).unsqueeze(0).unsqueeze(0)

                out = net(x_t).squeeze().numpy()

                blended = (
                    alpha * out +
                    (1 - alpha) * image[ch]
                )

                orig_norm = np.linalg.norm(image[ch])

                blend_norm = np.linalg.norm(blended) + 1e-8

                if orig_norm > 1e-6:
                    result[ch] = (
                        blended *
                        orig_norm /
                        blend_norm
                    )
                else:
                    result[ch] = blended

        return result.astype(np.float32)


# ──────────────────────────────────────────────
# BIAS FIELD
# ──────────────────────────────────────────────

class BiasFieldAugmentation:

    def __init__(
        self,
        p: float = 0.40,
        max_strength: float = 0.20
    ):
        self.p = p
        self.max_strength = max_strength

    def __call__(self, image: np.ndarray) -> np.ndarray:

        if np.random.random() > self.p:
            return image

        H, W = image.shape[1], image.shape[2]

        ctrl = np.random.uniform(
            -1,
            1,
            (4, 4)
        ).astype(np.float32)

        bias = cv2.resize(
            ctrl,
            (W, H),
            interpolation=cv2.INTER_CUBIC
        )

        bias *= self.max_strength

        result = image.copy()

        for ch in range(image.shape[0]):
            result[ch] = image[ch] * (1.0 + bias)

        return result.astype(np.float32)


# ──────────────────────────────────────────────
# GAMMA AUGMENTATION
# ──────────────────────────────────────────────

class GammaAugmentation:

    def __init__(
        self,
        p: float = 0.40,
        gamma_range: tuple = (0.75, 1.35)
    ):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, image: np.ndarray) -> np.ndarray:

        if np.random.random() > self.p:
            return image

        gamma = np.random.uniform(*self.gamma_range)

        result = image.copy()

        for ch in range(image.shape[0]):

            img = image[ch]

            mn, mx = img.min(), img.max()

            if mx - mn < 1e-8:
                continue

            normed = np.clip(
                (img - mn) / (mx - mn),
                0,
                1
            )

            result[ch] = (
                np.power(normed, gamma) *
                (mx - mn) + mn
            )

        return result.astype(np.float32)


# ──────────────────────────────────────────────
# GAUSSIAN NOISE
# ──────────────────────────────────────────────

class GaussianNoise:

    def __init__(
        self,
        p: float = 0.25,
        sigma_range: tuple = (0.0, 0.04)
    ):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, image: np.ndarray) -> np.ndarray:

        if np.random.random() > self.p:
            return image

        sigma = np.random.uniform(*self.sigma_range)

        return (
            image +
            np.random.randn(*image.shape).astype(np.float32) * sigma
        )


# ──────────────────────────────────────────────
# STANDARD FULL AUGMENTATION
# ──────────────────────────────────────────────

class AugmentationPipeline:

    def __init__(self):

        self.geometric = GeometricAugmentation()

        self.elastic = ElasticDeformation(
            p=0.20,
            alpha=18.0,
            sigma=5.0
        )

        self.gin = GINAugmentation(
            n_layers=4,
            p=0.40
        )

        self.bias = BiasFieldAugmentation(
            p=0.30,
            max_strength=0.12
        )

        self.gamma = GammaAugmentation(
            p=0.35,
            gamma_range=(0.85, 1.18)
        )

        self.noise = GaussianNoise(
            p=0.25,
            sigma_range=(0.0, 0.04)
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray):

        image, mask = self.geometric(image, mask)

        image, mask = self.elastic(image, mask)

        image = self.gin(image)

        image = self.bias(image)

        image = self.gamma(image)

        image = self.noise(image)

        return image.astype(np.float32), mask.astype(np.uint8)


# ──────────────────────────────────────────────
# RESNET50 FULL AUGMENTATION
# ──────────────────────────────────────────────

class ResNet50AugmentationPipeline:

    def __init__(self):

        self.geometric = GeometricAugmentation()

        self.elastic = ElasticDeformation(
            p=0.35,
            alpha=35.0,
            sigma=5.0
        )

        self.gin = ResNetGINAugmentation(
            n_layers=3,
            p=0.30
        )

        self.bias = BiasFieldAugmentation(
            p=0.40,
            max_strength=0.20
        )

        self.gamma = GammaAugmentation(
            p=0.40,
            gamma_range=(0.75, 1.35)
        )

        self.noise = GaussianNoise(
            p=0.25,
            sigma_range=(0.0, 0.04)
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray):

        image, mask = self.geometric(image, mask)

        image, mask = self.elastic(image, mask)

        image = self.gin(image)

        image = self.bias(image)

        image = self.gamma(image)

        image = self.noise(image)

        return image.astype(np.float32), mask.astype(np.uint8)


# ──────────────────────────────────────────────
# RESNET GIN ONLY
# ──────────────────────────────────────────────

class GINOnlyPipeline:

    def __init__(self):

        self.gin = ResNetGINAugmentation(
            n_layers=3,
            p=0.30
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray):

        image = self.gin(image)

        return image.astype(np.float32), mask.astype(np.uint8)


# ──────────────────────────────────────────────
# MINIMAL AUGMENTATION
# ──────────────────────────────────────────────

class MinimalAugmentation:

    def __init__(self):

        self.flip = GeometricAugmentation(
            p_flip=0.5,
            max_angle=0.0
        )

    def __call__(self, image: np.ndarray, mask: np.ndarray):

        image, mask = self.flip(image, mask)

        return image.astype(np.float32), mask.astype(np.uint8)