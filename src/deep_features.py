"""Frozen deep features (DINOv2, CLIP, BioCLIP) and a small from-scratch CNN.

The foundation models are RGB-pretrained, so we feed them the thermal crop resized to
their native size with the single gray channel replicated to three. Whether that domain
gap hurts them versus the classical hand features is the point of the experiment. Frozen
features are cached to disk keyed by model + a hash of the crop list so reruns are cheap.

The CNN is the opposite end: one input channel, a few conv blocks, trained on the crops
themselves. It is small on purpose -- a few hundred labelled crops won't feed anything
bigger.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ImageNet stats; open_clip/timm DINOv2 all expect roughly these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def to_three_channel(crop_gray, size=224):
    """Resize a grayscale crop to size x size and replicate to 3 channels, [0,1] float."""
    import cv2

    g = crop_gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    g = np.clip(g / 255.0, 0.0, 1.0)
    return np.stack([g, g, g], axis=0)  # (3, H, W)


def _normalize(batch):
    mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_STD).view(1, 3, 1, 1)
    return (batch - mean) / std


class FrozenBackbone:
    """A frozen image encoder; .embed(crops) returns one feature vector per crop."""

    def __init__(self, kind, device=None):
        self.kind = kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.size = 224
        self._build()

    def _build(self):
        if self.kind == "dinov2":
            import timm
            self.model = timm.create_model(
                "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
            self.size = 518  # this DINOv2 variant wants 518; resize handles it
        elif self.kind in ("clip", "bioclip"):
            import open_clip
            name, pretrained = self._openclip_spec(self.kind)
            self.model, _, _ = open_clip.create_model_and_transforms(
                name, pretrained=pretrained)
        else:
            raise ValueError(f"unknown backbone {self.kind}")
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _openclip_spec(kind):
        if kind == "bioclip":
            return "hf-hub:imageomics/bioclip", None
        return "ViT-B-16-quickgelu", "openai"  # quickgelu matches the openai weights

    @torch.no_grad()
    def embed(self, crops_gray, batch_size=64):
        """Embed a list of grayscale crops. Returns (N, D) float32 array."""
        feats = []
        for i in range(0, len(crops_gray), batch_size):
            chunk = crops_gray[i:i + batch_size]
            arr = np.stack([to_three_channel(c, self.size) for c in chunk])
            x = _normalize(torch.from_numpy(arr).float()).to(self.device)
            if self.kind == "dinov2":
                y = self.model(x)
            else:
                y = self.model.encode_image(x)
            feats.append(y.float().cpu().numpy())
        return np.concatenate(feats, axis=0)


def _cache_key(kind, crop_ids):
    h = hashlib.sha1("|".join(crop_ids).encode()).hexdigest()[:16]
    return f"{kind}_{h}.npy"


def cached_features(kind, crops_gray, crop_ids, cache_dir, device=None):
    """Frozen features for crops, cached on disk. crop_ids identifies the exact crop set."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / _cache_key(kind, crop_ids)
    if path.exists():
        return np.load(path)
    backbone = FrozenBackbone(kind, device=device)
    feats = backbone.embed(crops_gray)
    np.save(path, feats)
    return feats


class SmallCNN(nn.Module):
    """Three conv blocks on a 1-channel 64x64 crop, global-pooled to a small head."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def prep_cnn_batch(crops_gray, size=64):
    """List of grayscale crops -> (N,1,size,size) float tensor in [0,1]."""
    import cv2

    arr = np.empty((len(crops_gray), 1, size, size), np.float32)
    for i, c in enumerate(crops_gray):
        g = c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
        arr[i, 0] = np.clip(g / 255.0, 0.0, 1.0)
    return torch.from_numpy(arr)


def train_cnn(train_crops, train_y, val_crops, val_y, epochs=40, lr=1e-3,
              device=None, seed=0):
    """Train SmallCNN on grayscale crops; return predicted probs for val_crops.

    Class-weighted cross-entropy handles the imbalance. Returns P(moving) per val crop.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    xb = prep_cnn_batch(train_crops).to(device)
    yb = torch.tensor(train_y, dtype=torch.long, device=device)
    xv = prep_cnn_batch(val_crops).to(device)

    counts = np.bincount(train_y, minlength=2).astype(np.float32)
    w = torch.tensor(counts.sum() / (2.0 * np.maximum(counts, 1.0)),
                     dtype=torch.float32, device=device)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(xv), dim=1)[:, 1].cpu().numpy()
    return probs
