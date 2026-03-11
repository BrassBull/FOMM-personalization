"""
dataset/voxceleb.py — VoxCeleb Dataset for Talking-Head Training.

Expected directory layout
--------------------------
  <root>/
    train/
      id00001/
        video_id/
          frames/  *.png  (or *.jpg)
    test/
      ...

Each item is a (source_frame, driving_frame) pair sampled from the SAME clip.
"""

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import DatasetConfig


def _list_video_dirs(root: str) -> List[Path]:
    dirs = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_dir():
            images = list(path.glob("*.png")) + list(path.glob("*.jpg"))
            if len(images) >= 2:
                dirs.append(path)
    return dirs


def _load_frame(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


class VoxCelebDataset(Dataset):
    """
    Samples random (source, driving) pairs from VoxCeleb video clips.

    Args:
        root      : path to split directory (e.g. "data/voxceleb/train")
        cfg       : DatasetConfig
        is_train  : enable data augmentation
    """

    def __init__(self, root: str, cfg: DatasetConfig, is_train: bool = True):
        self.cfg      = cfg
        self.is_train = is_train
        self.size     = cfg.frame_shape

        self.video_dirs = _list_video_dirs(root)
        if not self.video_dirs:
            raise RuntimeError(
                f"No frame directories found under {root}. "
                "Expected sub-folders with at least 2 .png/.jpg files."
            )

        self.frame_lists: List[List[Path]] = []
        for vdir in self.video_dirs:
            frames = sorted(
                list(vdir.glob("*.png")) + list(vdir.glob("*.jpg"))
            )
            if len(frames) >= 2:
                self.frame_lists.append(frames)

        self.pairs: List[Tuple[int, int, int]] = []
        for v_idx, frames in enumerate(self.frame_lists):
            n = len(frames)
            k = min(cfg.pairs_per_video, n * (n - 1))
            for _ in range(k):
                i, j = random.sample(range(n), 2)
                self.pairs.append((v_idx, i, j))

    def __len__(self) -> int:
        return len(self.pairs)

    def _augment(self, src: torch.Tensor,
                 drv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            src = TF.hflip(src)
            drv = TF.hflip(drv)
        if random.random() < 0.5:
            src, drv = drv, src
        jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        )
        state = torch.get_rng_state()
        src   = jitter(src)
        torch.set_rng_state(state)
        drv   = jitter(drv)
        return src, drv

    def __getitem__(self, idx: int) -> dict:
        v_idx, s_idx, d_idx = self.pairs[idx]
        frames = self.frame_lists[v_idx]
        source_frame  = _load_frame(frames[s_idx], self.size)
        driving_frame = _load_frame(frames[d_idx], self.size)
        if self.is_train and self.cfg.augmentation:
            source_frame, driving_frame = self._augment(source_frame, driving_frame)
        return {
            "source":   source_frame,
            "driving":  driving_frame,
            "video_id": v_idx,
        }


def build_dataloader(root: str, cfg: DatasetConfig,
                     batch_size: int, is_train: bool = True) -> DataLoader:
    dataset = VoxCelebDataset(root, cfg, is_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=is_train,
    )


class SyntheticDataset(Dataset):
    """Random image pairs for unit-testing without real data."""

    def __init__(self, size: Tuple[int, int] = (256, 256), length: int = 200):
        self.size   = size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        H, W = self.size
        return {
            "source":   torch.rand(3, H, W),
            "driving":  torch.rand(3, H, W),
            "video_id": idx % 10,
        }


def build_synthetic_dataloader(batch_size: int = 4,
                                length: int = 200,
                                size: Tuple[int, int] = (256, 256)) -> DataLoader:
    return DataLoader(
        SyntheticDataset(size, length),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
