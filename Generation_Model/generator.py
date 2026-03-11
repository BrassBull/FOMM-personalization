"""
models/generator.py — Occlusion-Aware Semantic Frame Generator.

Architecture
------------
  U-Net encoder ─┐
                  │  skip connections
  U-Net decoder ←┘
       ↑  at each encoder scale:
           1. Warp encoder feature map with dense flow
           2. Multiply by (1 - occlusion_mask)  [zero out occluded regions]
           3. Concatenate with decoder activation

The generator produces a *semantic* frame, later refined by the FusionModule.
Bottleneck ResBlocks allow deep nonlinear reasoning about the warped content.

Reference: FOMM generator (Siarohin et al., NeurIPS 2019) with additional
           multi-scale occlusion handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GeneratorConfig
from models.blocks import (
    ResBlock2d, DownBlock2d, UpBlock2d, SameBlock2d
)


# ---------------------------------------------------------------------------
# Flow-based feature warper
# ---------------------------------------------------------------------------

def warp_features(features: torch.Tensor,
                  flow: torch.Tensor,
                  occlusion: torch.Tensor = None) -> torch.Tensor:
    """
    Warp a feature map with a dense optical flow field.

    Args:
        features   : (B, C, H, W)
        flow       : (B, 2, H, W)  flow in pixel offset format
        occlusion  : (B, 1, H, W)  soft mask in [0,1]; 1 = occluded

    Returns:
        warped : (B, C, H, W)
    """
    B, C, H, W = features.shape

    # Convert flow to normalised grid
    ys = torch.linspace(-1, 1, H, device=flow.device)
    xs = torch.linspace(-1, 1, W, device=flow.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=0)  # (2,H,W)

    # flow in pixel offsets → normalised offsets
    norm_flow = flow.clone()
    norm_flow[:, 0] = flow[:, 0] / (W / 2)
    norm_flow[:, 1] = flow[:, 1] / (H / 2)

    grid = identity.unsqueeze(0) + norm_flow          # (B,2,H,W)
    grid = grid.permute(0, 2, 3, 1)                   # (B,H,W,2)
    grid = grid.clamp(-1, 1)

    warped = F.grid_sample(features, grid,
                           mode="bilinear",
                           padding_mode="zeros",
                           align_corners=False)

    if occlusion is not None:
        occ = F.interpolate(occlusion, size=(H, W),
                            mode="bilinear", align_corners=False)
        warped = warped * (1.0 - occ)

    return warped


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class OcclusionAwareGenerator(nn.Module):
    """
    Semantic (FOMM-style) generator that warps multi-scale encoder features
    before decoding.

    Inputs
    ------
    source_image  : (B, 3, H, W)
    dense_flow    : (B, 2, H, W)
    occlusion_map : (B, 1, H, W)

    Output
    ------
    semantic_frame : (B, 3, H, W)  in [0, 1]
    """

    def __init__(self, cfg: GeneratorConfig, num_channels: int = 3):
        super().__init__()
        self.cfg = cfg

        # ---- Encoder --------------------------------------------------------
        enc_ch = [num_channels]
        for i in range(cfg.num_down_blocks):
            enc_ch.append(min(cfg.max_features,
                              cfg.block_expansion * (2 ** i)))

        self.encoder = nn.ModuleList()
        for i in range(cfg.num_down_blocks):
            self.encoder.append(DownBlock2d(enc_ch[i], enc_ch[i + 1]))

        # ---- Bottleneck residual blocks -------------------------------------
        bottleneck_ch = enc_ch[-1]
        self.bottleneck = nn.Sequential(*[
            ResBlock2d(bottleneck_ch)
            for _ in range(cfg.num_bottleneck_blocks)
        ])

        # ---- Decoder with skip connections ----------------------------------
        # At step i we up-sample to enc level (num_down_blocks - 1 - i)
        # and concatenate skip from that encoder level.
        self.decoder = nn.ModuleList()
        self.skip_conv = nn.ModuleList()

        dec_ch = bottleneck_ch
        for i in range(cfg.num_down_blocks):
            level = cfg.num_down_blocks - 1 - i
            skip_ch = enc_ch[level]
            in_ch = dec_ch + skip_ch
            out_ch = max(cfg.block_expansion,
                         cfg.block_expansion * (2 ** (level - 1)))
            self.skip_conv.append(
                nn.Conv2d(skip_ch, skip_ch, 1)   # 1×1 for channel alignment
            )
            self.decoder.append(UpBlock2d(in_ch, out_ch))
            dec_ch = out_ch

        # ---- Output head ----------------------------------------------------
        self.final = nn.Sequential(
            nn.Conv2d(dec_ch, num_channels, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------

    def forward(self,
                source_image: torch.Tensor,
                dense_flow: torch.Tensor,
                occlusion_map: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            source_image  : (B, 3, H, W)
            dense_flow    : (B, 2, H, W)
            occlusion_map : (B, 1, H, W)  or None

        Returns:
            semantic_frame : (B, 3, H, W)
        """
        # --- Encode ----------------------------------------------------------
        enc_features = []
        x = source_image
        for block in self.encoder:
            x = block(x)
            enc_features.append(x)

        # --- Bottleneck with warping -----------------------------------------
        x = self.bottleneck(x)

        # Warp bottleneck features with full-resolution flow
        flow_small = F.interpolate(dense_flow,
                                   size=x.shape[-2:],
                                   mode="bilinear",
                                   align_corners=False)
        occ_small = None
        if occlusion_map is not None:
            occ_small = F.interpolate(occlusion_map,
                                      size=x.shape[-2:],
                                      mode="bilinear",
                                      align_corners=False)
        x = warp_features(x, flow_small, occ_small)

        # --- Decode with warped skips ----------------------------------------
        for i, (up_block, skip_block) in enumerate(
                zip(self.decoder, self.skip_conv)):
            level = self.cfg.num_down_blocks - 1 - i
            skip = enc_features[level]

            # Warp skip connection at its native resolution
            flow_skip = F.interpolate(dense_flow,
                                      size=skip.shape[-2:],
                                      mode="bilinear",
                                      align_corners=False)
            occ_skip = None
            if occlusion_map is not None:
                occ_skip = F.interpolate(occlusion_map,
                                         size=skip.shape[-2:],
                                         mode="bilinear",
                                         align_corners=False)
            skip_warped = warp_features(skip, flow_skip, occ_skip)
            skip_warped = skip_block(skip_warped)

            # Upsample decoder activation to match skip resolution
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode="bilinear", align_corners=False)
            x = torch.cat([x, skip_warped], dim=1)
            x = up_block(x)

        # Upsample to original resolution if needed
        x = F.interpolate(x, size=source_image.shape[-2:],
                          mode="bilinear", align_corners=False)
        return self.final(x)
