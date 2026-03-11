"""
models/keypoint_detector.py — Semantic keypoint detector.

Architecture
------------
  Hourglass encoder-decoder → K Gaussian heatmaps → 2-D keypoints
                                                    → 2×2 Jacobians (local affine)

The heatmaps are sharpened with a temperature-scaled softmax so that the
"argmax" operation (weighted average of a spatial grid) remains differentiable.
Jacobians are predicted as a small MLP on top of the bottleneck features
warped to each keypoint neighbourhood.

Following FOMM (Siarohin et al., NeurIPS 2019) the detector is applied to
BOTH the source and driving frames; the resulting keypoint sets are handed to
the Dense Motion Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KeypointConfig
from models.blocks import DownBlock2d, SameBlock2d, ResBlock2d, AntiAliasInterp2d


# ---------------------------------------------------------------------------
# Hourglass backbone
# ---------------------------------------------------------------------------

class Hourglass(nn.Module):
    """
    Encoder-decoder with skip connections.

    The encoder produces a spatial feature pyramid; the decoder merges them
    back at each scale. The bottleneck features are returned alongside the
    final heatmap prediction.
    """

    def __init__(self, block_expansion: int, in_features: int,
                 num_blocks: int = 5, max_features: int = 256):
        super().__init__()

        # --- Encoder ---
        encoder_blocks = []
        for i in range(num_blocks):
            in_ch  = in_features if i == 0 else min(
                max_features, block_expansion * (2 ** (i - 1)))
            out_ch = min(max_features, block_expansion * (2 ** i))
            encoder_blocks.append(DownBlock2d(in_ch, out_ch))
        self.encoder = nn.ModuleList(encoder_blocks)

        # --- Decoder ---
        decoder_blocks = []
        for i in range(num_blocks):
            # at decoder step i we merge level (num_blocks-1-i) from encoder
            level = num_blocks - 1 - i
            in_ch  = min(max_features, block_expansion * (2 ** level))
            # skip connection doubles channels (except at the bottom)
            if i > 0:
                in_ch += min(max_features, block_expansion * (2 ** (level + 1)))
            out_ch = min(max_features, block_expansion * (2 ** level))
            decoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.decoder = nn.ModuleList(decoder_blocks)
        self.out_channels = min(max_features, block_expansion)

    def forward(self, x: torch.Tensor):
        enc_features = []
        for block in self.encoder:
            x = block(x)
            enc_features.append(x)

        out = enc_features[-1]
        for i, block in enumerate(self.decoder):
            if i > 0:
                skip = enc_features[-(i + 1)]
                out = F.interpolate(out, size=skip.shape[-2:],
                                    mode="bilinear", align_corners=False)
                out = torch.cat([out, skip], dim=1)
            else:
                out = F.interpolate(out, scale_factor=2,
                                    mode="bilinear", align_corners=False)
            out = block(out)
        return out, enc_features


# ---------------------------------------------------------------------------
# Keypoint Detector
# ---------------------------------------------------------------------------

class KeypointDetector(nn.Module):
    """
    Detects K semantic facial keypoints from a single RGB image.

    Outputs
    -------
    keypoints  : (B, K, 2)      normalised coordinates in [-1, 1]
    jacobians  : (B, K, 2, 2)   local affine transformation (if enabled)
    heatmaps   : (B, K, H', W') soft heatmaps (for visualisation / losses)
    """

    def __init__(self, cfg: KeypointConfig, in_channels: int = 3):
        super().__init__()
        self.cfg = cfg
        K = cfg.num_keypoints

        # Optional spatial anti-alias downscale before hourglass
        self.anti_alias = (
            AntiAliasInterp2d(in_channels, cfg.scale_factor)
            if cfg.scale_factor != 1.0 else nn.Identity()
        )

        self.hourglass = Hourglass(
            block_expansion=32,
            in_features=in_channels,
            num_blocks=5,
            max_features=256,
        )

        out_ch = self.hourglass.out_channels
        self.heatmap_head = nn.Conv2d(out_ch, K, kernel_size=7, padding=3)

        if cfg.use_jacobian:
            # Predict the 4 entries of a 2×2 Jacobian per keypoint
            self.jacobian_head = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, K * 4, 1),
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _softargmax(heatmaps: torch.Tensor,
                    temperature: float) -> torch.Tensor:
        """
        Differentiable spatial argmax via temperature-softmax.

        Args:
            heatmaps : (B, K, H, W)
        Returns:
            coords   : (B, K, 2)  in normalised [-1, 1]
        """
        B, K, H, W = heatmaps.shape
        flat = heatmaps.view(B, K, -1) / temperature
        weights = torch.softmax(flat, dim=-1)           # (B, K, H*W)

        # Build meshgrid in [-1, 1]
        ys = torch.linspace(-1, 1, H, device=heatmaps.device)
        xs = torch.linspace(-1, 1, W, device=heatmaps.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (H*W,)
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        # Weighted sum  → (B, K, 2)
        coords = (weights.unsqueeze(-1) * grid.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        return coords

    # ------------------------------------------------------------------

    def _predict_jacobians(self, features: torch.Tensor,
                           coords: torch.Tensor) -> torch.Tensor:
        """
        Sample features at keypoint locations and regress local Jacobians.

        Args:
            features : (B, C, H, W)
            coords   : (B, K, 2)
        Returns:
            jacobians : (B, K, 2, 2)
        """
        B, K, _ = coords.shape
        raw = self.jacobian_head(features)        # (B, K*4, H, W)

        # Sample at each keypoint's (sub-pixel) location
        # coords : (B, K, 2)  →  grid_sample expects (B, 1, K, 2)
        sample_grid = coords.unsqueeze(1)         # (B, 1, K, 2)
        raw = raw.view(B, K, 4,
                       raw.shape[-2],
                       raw.shape[-1])             # (B, K, 4, H, W)

        j_list = []
        for k in range(K):
            feat_k = raw[:, k, :, :, :]          # (B, 4, H, W)
            g = sample_grid[:, :, k:k+1, :]      # (B, 1, 1, 2)
            sampled = F.grid_sample(feat_k, g,
                                    align_corners=False,
                                    mode="bilinear",
                                    padding_mode="border")
            j_list.append(sampled.squeeze(-1).squeeze(-1))  # (B, 4)

        j = torch.stack(j_list, dim=1)           # (B, K, 4)
        # Map to 2×2: add identity for stability
        j = j.view(B, K, 2, 2)
        identity = torch.eye(2, device=j.device).unsqueeze(0).unsqueeze(0)
        return j + identity

    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor) -> dict:
        """
        Args:
            image : (B, 3, H, W)  in [0, 1]
        Returns dict with keys:
            'keypoints'  : (B, K, 2)
            'jacobians'  : (B, K, 2, 2)  or None
            'heatmaps'   : (B, K, H', W')
        """
        scaled = self.anti_alias(image)
        features, _ = self.hourglass(scaled)

        heatmaps = self.heatmap_head(features)    # (B, K, H', W')
        coords   = self._softargmax(heatmaps, self.cfg.temperature)

        out = {
            "keypoints": coords,
            "heatmaps": heatmaps,
            "jacobians": None,
        }

        if self.cfg.use_jacobian:
            out["jacobians"] = self._predict_jacobians(features, coords)

        return out
