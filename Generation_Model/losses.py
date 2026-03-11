"""
training/losses.py — Composite Training Objective.

Loss terms
----------
1. L1 reconstruction (fused vs ground-truth)
2. VGG perceptual loss (feature-matching on multiple layers)
3. Equivariance loss (keypoints consistent under known random TPS warp)
4. Generator adversarial loss (non-saturating GAN)
5. Discriminator feature-matching loss
6. Auxiliary reconstruction loss (aux frame vs driving)
7. Fusion perceptual loss (fused vs driving)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List

from config import LossConfig


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using frozen VGG-19 intermediate activations."""

    def __init__(self, layers: List[int] = (0, 1, 2, 3)):
        super().__init__()
        vgg      = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())
        ends     = [2, 7, 12, 21, 30]
        self.layers = sorted(layers)

        slices = []
        prev = 0
        for end in ends:
            slices.append(nn.Sequential(*features[prev:end]))
            prev = end
        self.slices = nn.ModuleList(slices)
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = torch.tensor(0.0, device=pred.device)
        p, t = pred, target
        for i, slab in enumerate(self.slices):
            p = slab(p)
            t = slab(t)
            if i in self.layers:
                loss = loss + F.l1_loss(p, t.detach())
        return loss


class EquivarianceLoss(nn.Module):
    """
    Enforces keypoint consistency under random TPS warps.
    Reference: Siarohin et al. NeurIPS 2019 Eq.(5)
    """

    def __init__(self, num_keypoints: int = 15, image_size: int = 256,
                 sigma_tps: float = 0.005, num_tps_points: int = 5):
        super().__init__()
        self.K         = num_keypoints
        self.sigma_tps = sigma_tps
        self.P         = num_tps_points
        self.image_size = image_size

    def _random_tps_warp(self, images: torch.Tensor):
        B, _, H, W = images.shape
        device = images.device
        pts   = torch.rand(B, self.P, 2, device=device) * 2 - 1
        disps = torch.randn(B, self.P, 2, device=device) * self.sigma_tps

        ys = torch.linspace(-1, 1, H, device=device)
        xs = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        flow = torch.zeros_like(grid)
        for b in range(B):
            for p in range(self.P):
                dist   = ((grid[b] - pts[b, p]) ** 2).sum(-1)
                weight = torch.exp(-dist / (2 * self.sigma_tps))
                flow[b] += weight.unsqueeze(-1) * disps[b, p]

        warped_grid = (grid + flow).clamp(-1, 1)
        warped = F.grid_sample(images, warped_grid,
                               mode="bilinear", padding_mode="zeros",
                               align_corners=False)
        return warped, flow   # flow: (B, H, W, 2)

    def forward(self, kp_detector: nn.Module,
                images: torch.Tensor) -> dict:
        warped, flow = self._random_tps_warp(images)
        kp_orig   = kp_detector(images)
        kp_warped = kp_detector(warped)

        kp_pts   = kp_orig["keypoints"]          # (B, K, 2)
        B, K, _  = kp_pts.shape
        # Sample flow at keypoint locations
        flow_bchw = flow.permute(0, 3, 1, 2)    # (B, 2, H, W)
        grid_pts  = kp_pts.unsqueeze(1)          # (B, 1, K, 2)
        sampled   = F.grid_sample(flow_bchw, grid_pts,
                                  mode="bilinear", padding_mode="border",
                                  align_corners=False)  # (B, 2, 1, K)
        sampled = sampled.squeeze(2).permute(0, 2, 1)   # (B, K, 2)

        kp_expected = (kp_pts + sampled).clamp(-1, 1)
        loss_value  = F.l1_loss(kp_warped["keypoints"], kp_expected.detach())

        loss_jac = torch.tensor(0.0, device=images.device)
        if (kp_orig.get("jacobians") is not None and
                kp_warped.get("jacobians") is not None):
            loss_jac = F.l1_loss(kp_warped["jacobians"],
                                 kp_orig["jacobians"].detach())
        return {"value": loss_value, "jacobian": loss_jac}


def generator_loss(fake_preds: list) -> torch.Tensor:
    loss = torch.tensor(0.0, device=fake_preds[0][-1].device)
    for scale_preds in fake_preds:
        loss = loss + F.softplus(-scale_preds[-1]).mean()
    return loss / len(fake_preds)


def discriminator_loss(real_preds: list, fake_preds: list) -> torch.Tensor:
    loss = torch.tensor(0.0, device=real_preds[0][-1].device)
    for r_feats, f_feats in zip(real_preds, fake_preds):
        loss = loss + (F.relu(1.0 - r_feats[-1]).mean() +
                       F.relu(1.0 + f_feats[-1]).mean())
    return loss / len(real_preds)


def feature_matching_loss(real_preds: list, fake_preds: list) -> torch.Tensor:
    loss = torch.tensor(0.0, device=real_preds[0][-1].device)
    for r_feats, f_feats in zip(real_preds, fake_preds):
        for r, f in zip(r_feats[:-1], f_feats[:-1]):
            loss = loss + F.l1_loss(f, r.detach())
    return loss / len(real_preds)


class CompositeLoss(nn.Module):
    """
    Collects and weights all loss terms for a training step.
    Returns a dict of named scalars with keys "total_g" and "total_d".
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg        = cfg
        self.perceptual = VGGPerceptualLoss(cfg.perceptual_layers)
        self.equivar    = EquivarianceLoss(num_keypoints=15)

    def forward(self, output: dict, batch: dict,
                discriminator: nn.Module = None,
                kp_detector: nn.Module = None) -> dict:
        losses = {}
        target   = batch["driving"]
        fused    = output["fused_frame"]
        aux      = output["aux_frame"]

        losses["l1"] = F.l1_loss(fused, target) * self.cfg.lambda_l1
        losses["perceptual"] = (
            self.perceptual(fused, target) * self.cfg.lambda_perceptual)
        losses["auxiliary"] = (
            F.l1_loss(aux, target) * self.cfg.lambda_auxiliary)
        losses["fusion_perceptual"] = (
            self.perceptual(fused, target) * self.cfg.lambda_fusion_perceptual)

        if kp_detector is not None:
            eq = self.equivar(kp_detector, batch["source"])
            losses["equivariance_value"]    = eq["value"] * self.cfg.lambda_equivariance_value
            losses["equivariance_jacobian"] = eq["jacobian"] * self.cfg.lambda_equivariance_jacobian
        else:
            losses["equivariance_value"]    = torch.tensor(0.0)
            losses["equivariance_jacobian"] = torch.tensor(0.0)

        if discriminator is not None:
            real_p, fake_p = discriminator(target, fused)
            losses["gen_adv"]   = generator_loss(fake_p)      * self.cfg.lambda_gan
            losses["feat_match"]= feature_matching_loss(real_p, fake_p) * self.cfg.lambda_gan
            real_d, fake_d = discriminator(target, fused.detach())
            losses["disc"] = discriminator_loss(real_d, fake_d)
        else:
            losses["gen_adv"]    = torch.tensor(0.0)
            losses["feat_match"] = torch.tensor(0.0)
            losses["disc"]       = torch.tensor(0.0)

        g_keys = ["l1", "perceptual", "auxiliary", "fusion_perceptual",
                  "equivariance_value", "equivariance_jacobian",
                  "gen_adv", "feat_match"]
        losses["total_g"] = sum(losses[k] for k in g_keys)
        losses["total_d"] = losses["disc"]
        return losses
