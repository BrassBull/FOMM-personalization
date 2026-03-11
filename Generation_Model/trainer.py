"""
training/trainer.py — Training Engine for the Hybrid Talking-Head System.

Handles:
  - Mixed-precision training (torch.cuda.amp)
  - Alternating generator / discriminator updates
  - Learning-rate scheduling
  - Checkpoint save / resume
  - TensorBoard logging (scalars + image grids)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from config import TrainConfig, ModelConfig
from models.hybrid_model import HybridTalkingHeadModel
from models.discriminator import MultiScaleDiscriminator
from training.losses import CompositeLoss

log = logging.getLogger(__name__)


class Trainer:
    """
    Full training loop for the hybrid talking-head model.

    Args:
        model_cfg : ModelConfig
        train_cfg : TrainConfig
        device    : "cuda" | "cpu"
    """

    def __init__(self,
                 model_cfg: ModelConfig,
                 train_cfg: TrainConfig,
                 device: str = "cuda"):
        self.cfg    = train_cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ---- Model ----------------------------------------------------------
        self.model = HybridTalkingHeadModel(model_cfg).to(self.device)
        self.disc  = MultiScaleDiscriminator().to(self.device)

        # ---- Loss -----------------------------------------------------------
        self.loss_fn = CompositeLoss(train_cfg.loss).to(self.device)

        # ---- Optimisers -----------------------------------------------------
        opt_cfg = train_cfg.optimiser
        self.opt_g = torch.optim.Adam(
            [
                {"params": self.model.keypoint_detector.parameters(),
                 "lr": opt_cfg.lr_keypoint},
                {"params": self.model.dense_motion.parameters()},
                {"params": self.model.generator.parameters()},
                {"params": self.model.auxiliary_stream.parameters()},
                {"params": self.model.fusion.parameters()},
            ],
            lr=opt_cfg.lr_generator,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
        )
        self.opt_d = torch.optim.Adam(
            self.disc.parameters(),
            lr=opt_cfg.lr_discriminator,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
        )

        # ---- Schedulers -----------------------------------------------------
        self.sched_g = torch.optim.lr_scheduler.MultiStepLR(
            self.opt_g,
            milestones=opt_cfg.scheduler_milestones,
            gamma=opt_cfg.scheduler_gamma,
        )
        self.sched_d = torch.optim.lr_scheduler.MultiStepLR(
            self.opt_d,
            milestones=opt_cfg.scheduler_milestones,
            gamma=opt_cfg.scheduler_gamma,
        )

        # ---- AMP scalers ----------------------------------------------------
        self.scaler_g = GradScaler(enabled=train_cfg.mixed_precision)
        self.scaler_d = GradScaler(enabled=train_cfg.mixed_precision)

        # ---- Logging --------------------------------------------------------
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(train_cfg.log_dir, exist_ok=True)
        self.writer = SummaryWriter(train_cfg.log_dir)
        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------
    # One training step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict) -> dict:
        source  = batch["source"].to(self.device)
        driving = batch["driving"].to(self.device)
        batch_d = {"source": source, "driving": driving}

        # ---- Generator step ------------------------------------------------
        self.opt_g.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg.mixed_precision):
            output = self.model(source, driving)
            losses = self.loss_fn(
                output, batch_d,
                discriminator=self.disc,
                kp_detector=self.model.keypoint_detector,
            )

        self.scaler_g.scale(losses["total_g"]).backward()
        self.scaler_g.unscale_(self.opt_g)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        # ---- Discriminator step --------------------------------------------
        self.opt_d.zero_grad(set_to_none=True)
        with autocast(enabled=self.cfg.mixed_precision):
            # Re-forward with detached generator output
            output_d = self.model(source, driving)
            losses_d = self.loss_fn(output_d, batch_d,
                                    discriminator=self.disc)

        self.scaler_d.scale(losses_d["total_d"]).backward()
        self.scaler_d.unscale_(self.opt_d)
        nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=1.0)
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()

        return {k: v.item() for k, v in losses.items()}

    # ------------------------------------------------------------------
    # Validation pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in val_loader:
            source  = batch["source"].to(self.device)
            driving = batch["driving"].to(self.device)
            output  = self.model(source, driving, apply_aux_compression=False)
            loss    = torch.nn.functional.l1_loss(output["fused_frame"], driving)
            total_loss += loss.item()
            n += 1
            if n >= 50:
                break
        self.model.train()
        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None):
        """
        Run training for cfg.num_epochs epochs.

        Args:
            train_loader : DataLoader yielding {"source", "driving"}
            val_loader   : optional DataLoader for validation
        """
        log.info(f"Training on {self.device} | "
                 f"{self.cfg.num_epochs} epochs | "
                 f"batch={self.cfg.batch_size}")

        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()
            epoch_losses = {}
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                step_losses = self._train_step(batch)
                self.global_step += 1

                # Accumulate for epoch average
                for k, v in step_losses.items():
                    epoch_losses.setdefault(k, []).append(v)

                # TensorBoard step-level logging
                if self.global_step % self.cfg.log_every == 0:
                    for k, v in step_losses.items():
                        self.writer.add_scalar(f"train_step/{k}", v,
                                               self.global_step)

            # ---- Epoch summaries -------------------------------------------
            for k, vs in epoch_losses.items():
                avg = sum(vs) / len(vs)
                self.writer.add_scalar(f"train_epoch/{k}", avg, epoch)

            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch:03d}/{self.cfg.num_epochs} | "
                f"total_g={sum(epoch_losses.get('total_g', [0])) / max(len(epoch_losses.get('total_g', [1])), 1):.4f} | "
                f"elapsed={elapsed:.1f}s"
            )

            # ---- Validation ------------------------------------------------
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.writer.add_scalar("val/l1_loss", val_loss, epoch)
                log.info(f"  Validation L1: {val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, tag="best")

            # ---- Periodic checkpoint ---------------------------------------
            if epoch % self.cfg.save_every == 0:
                self._save_checkpoint(epoch, tag=f"epoch_{epoch:03d}")

            # ---- LR step ---------------------------------------------------
            self.sched_g.step()
            self.sched_d.step()

        self._save_checkpoint(self.cfg.num_epochs, tag="final")
        self.writer.close()
        log.info("Training complete.")

    # ------------------------------------------------------------------
    # Image grid visualisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def visualise_batch(self, batch: dict, step: int):
        """Write an image grid (source | driving | semantic | fused) to TB."""
        self.model.eval()
        source  = batch["source"][:4].to(self.device)
        driving = batch["driving"][:4].to(self.device)
        out     = self.model(source, driving, apply_aux_compression=False)

        grid = vutils.make_grid(
            torch.cat([source, driving,
                       out["semantic_frame"], out["fused_frame"]], dim=0),
            nrow=4, normalize=True
        )
        self.writer.add_image("vis/source_driving_semantic_fused", grid, step)
        self.model.train()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, tag: str = "latest"):
        path = Path(self.cfg.checkpoint_dir) / f"{tag}.pth"
        torch.save({
            "epoch":        epoch,
            "global_step":  self.global_step,
            "model_state":  self.model.state_dict(),
            "disc_state":   self.disc.state_dict(),
            "opt_g_state":  self.opt_g.state_dict(),
            "opt_d_state":  self.opt_d.state_dict(),
            "sched_g":      self.sched_g.state_dict(),
            "sched_d":      self.sched_d.state_dict(),
            "best_val":     self.best_val_loss,
        }, path)
        log.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.disc.load_state_dict(state["disc_state"])
        self.opt_g.load_state_dict(state["opt_g_state"])
        self.opt_d.load_state_dict(state["opt_d_state"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.sched_d.load_state_dict(state["sched_d"])
        self.global_step   = state.get("global_step", 0)
        self.best_val_loss = state.get("best_val", float("inf"))
        log.info(f"Resumed from {path} (epoch {state['epoch']})")
