"""
train.py — Main training script for the Hybrid FOMM Talking-Head Model.

Usage
-----
  # Train on VoxCeleb:
  python train.py --data_root data/voxceleb --epochs 100 --batch 4

  # Resume from checkpoint:
  python train.py --resume checkpoints/epoch_050.pth

  # Smoke-test with synthetic random data (no dataset required):
  python train.py --synthetic --epochs 2 --batch 2
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, TrainConfig
from dataset.voxceleb import VoxCelebDataset, build_synthetic_dataloader
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      default="data/voxceleb")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch",          type=int,   default=4)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--resume",         default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--image_size",     type=int,   default=256)
    p.add_argument("--synthetic",      action="store_true")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    model_cfg = ModelConfig()
    model_cfg.image_size = args.image_size

    train_cfg = TrainConfig()
    train_cfg.num_epochs     = args.epochs
    train_cfg.batch_size     = args.batch
    train_cfg.checkpoint_dir = args.checkpoint_dir
    train_cfg.log_dir        = args.log_dir
    train_cfg.optimiser.lr_generator     = args.lr
    train_cfg.optimiser.lr_discriminator = args.lr
    train_cfg.optimiser.lr_keypoint      = args.lr
    train_cfg.dataset.root = args.data_root

    logging.info("Initialising trainer …")
    trainer = Trainer(model_cfg, train_cfg, device=args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.synthetic:
        logging.info("Synthetic mode — generating random data")
        train_loader = build_synthetic_dataloader(
            batch_size=args.batch, length=400,
            size=(args.image_size, args.image_size))
        val_loader = build_synthetic_dataloader(
            batch_size=1, length=50,
            size=(args.image_size, args.image_size))
    else:
        full_ds = VoxCelebDataset(
            root=os.path.join(args.data_root, "train"),
            cfg=train_cfg.dataset, is_train=True)
        n_val   = max(1, int(0.05 * len(full_ds)))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=args.batch,
                                  shuffle=True, num_workers=4, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=1, num_workers=2)

    logging.info(f"Train: {len(train_loader)} batches | "
                 f"Val: {len(val_loader)} batches")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
