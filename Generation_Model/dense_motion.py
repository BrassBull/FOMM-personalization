"""
models/dense_motion.py — Dense Motion Network with Recurrent Temporal Blocks.

Pipeline
--------
  1. For each of K keypoints compute a local TPS/affine warp of the source
     image based on (source_kp, driving_kp, Jacobians).
  2. Concatenate K+1 warped images + Gaussian difference maps into a motion
     representation tensor.
  3. Pass through a hourglass with injected ConvGRU cells → dense optical
     flow field + occlusion mask.
  4. ConvGRU hidden states carry temporal context across frames, enabling
     motion smoothing and temporal coherence.

References
----------
  Siarohin et al. "First Order Motion Model for Image Animation" NeurIPS 2019
  Egorova et al.  (recurrent extension for video coding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DenseMotionConfig
from models.blocks import (
    ResBlock2d, DownBlock2d, UpBlock2d, SameBlock2d, ConvGRU
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_coordinate_grid(h: int, w: int,
                           device: torch.device) -> torch.Tensor:
    """Return a (1, H, W, 2) grid with values in [-1, 1]."""
    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)


def _transform_kp(kp_source: torch.Tensor,
                  kp_driving: torch.Tensor,
                  jacobian_source: torch.Tensor = None,
                  jacobian_driving: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the approximate inverse warp for each keypoint using the local
    affine (Jacobian) model.

    Returns a normalised flow (B, H, W, 2) per keypoint embedded in a
    (B, K, H, W, 2) tensor of *absolute* coordinates.
    """
    # kp_source, kp_driving : (B, K, 2)
    return kp_driving - kp_source  # simplest first-order displacement (B,K,2)


def _kp_to_gaussian(kp: torch.Tensor, h: int, w: int,
                    kp_variance: float = 0.01) -> torch.Tensor:
    """
    Convert keypoints to Gaussian heatmaps.

    Args:
        kp : (B, K, 2)  normalised coords in [-1, 1]
    Returns:
        heatmaps : (B, K, H, W)
    """
    B, K, _ = kp.shape
    grid = _make_coordinate_grid(h, w, kp.device)          # (1,H,W,2)
    kp_expanded = kp.unsqueeze(2).unsqueeze(2)              # (B,K,1,1,2)
    dist = ((grid.unsqueeze(1) - kp_expanded) ** 2).sum(-1) # (B,K,H,W)
    return torch.exp(-dist / (2 * kp_variance))


# ---------------------------------------------------------------------------
# Per-keypoint warper
# ---------------------------------------------------------------------------

class KeypointWarp(nn.Module):
    """
    Warp the source image K times using local affine transformations,
    one per keypoint.  Returns K warped images and K Gaussian difference maps
    stacked along the channel dimension.
    """

    def __init__(self, num_keypoints: int, kp_variance: float = 0.01):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kp_variance = kp_variance

    def forward(self,
                source_image: torch.Tensor,    # (B, C, H, W)
                kp_source: dict,               # {"keypoints":(B,K,2), ...}
                kp_driving: dict               # {"keypoints":(B,K,2), ...}
                ) -> dict:
        B, C, H, W = source_image.shape
        K = self.num_keypoints

        src_pts = kp_source["keypoints"]       # (B, K, 2)
        drv_pts = kp_driving["keypoints"]

        # --- Gaussian heatmaps for driving and difference --------------------
        drv_heatmaps = _kp_to_gaussian(drv_pts, H, W, self.kp_variance)
        src_heatmaps = _kp_to_gaussian(src_pts, H, W, self.kp_variance)
        heatmap_diff = drv_heatmaps - src_heatmaps                 # (B,K,H,W)

        # --- Dense sampling grid per keypoint --------------------------------
        # Build a (B, K, H, W, 2) grid where each keypoint's slice describes
        # the sampling location in the source image after local affine warp.
        grid = _make_coordinate_grid(H, W, source_image.device)    # (1,H,W,2)
        grid = grid.unsqueeze(1).repeat(1, K, 1, 1, 1)             # (1,K,H,W,2)

        # Displacement: driving → source (approximate inverse)
        disp = (src_pts - drv_pts)  # (B, K, 2)
        # Apply Jacobian correction if available
        if kp_source.get("jacobians") is not None and \
           kp_driving.get("jacobians") is not None:
            J_src = kp_source["jacobians"]   # (B, K, 2, 2)
            J_drv = kp_driving["jacobians"]
            # J_composed ≈ J_src @ J_drv⁻¹ — approximate local inverse
            try:
                J_inv_drv = torch.linalg.inv(J_drv)
                J_composed = torch.matmul(J_src, J_inv_drv)  # (B,K,2,2)
            except Exception:
                J_composed = torch.eye(2, device=J_src.device)\
                               .unsqueeze(0).unsqueeze(0)\
                               .expand_as(J_src)

            # Warp grid: for each pixel p, new_p = J_composed @ (p - drv_kp) + src_kp
            drv_exp = drv_pts.unsqueeze(2).unsqueeze(2)           # (B,K,1,1,2)
            src_exp = src_pts.unsqueeze(2).unsqueeze(2)
            rel = grid - drv_exp.expand_as(grid)                  # (B,K,H,W,2)
            # matmul: (B,K,H,W,1,2) @ (B,K,1,1,2,2)
            J_exp = J_composed.unsqueeze(2).unsqueeze(2)          # (B,K,1,1,2,2)
            warped_rel = torch.matmul(rel.unsqueeze(-2), J_exp.transpose(-1,-2))
            warped_rel = warped_rel.squeeze(-2)
            sampling_grid = warped_rel + src_exp.expand_as(grid)  # (B,K,H,W,2)
        else:
            disp_exp = disp.unsqueeze(2).unsqueeze(2)             # (B,K,1,1,2)
            sampling_grid = grid + disp_exp                       # (B,K,H,W,2)

        # Clamp to [-1, 1]
        sampling_grid = sampling_grid.clamp(-1, 1)

        # --- Warp source image for each keypoint ----------------------------
        warped_images = []
        for k in range(K):
            grid_k = sampling_grid[:, k, :, :, :]                 # (B,H,W,2)
            warped_k = F.grid_sample(source_image, grid_k,
                                     mode="bilinear",
                                     padding_mode="zeros",
                                     align_corners=False)
            warped_images.append(warped_k)                         # (B,C,H,W)

        warped_images = torch.stack(warped_images, dim=1)          # (B,K,C,H,W)

        # Background motion estimate (identity warp)
        bg_motion = torch.zeros(B, 2, H, W, device=source_image.device)

        return {
            "warped_images": warped_images,      # (B, K, C, H, W)
            "heatmap_diff": heatmap_diff,        # (B, K, H, W)
            "sampling_grid": sampling_grid,      # (B, K, H, W, 2)
            "bg_motion": bg_motion,              # (B, 2, H, W)
        }


# ---------------------------------------------------------------------------
# Dense Motion Network
# ---------------------------------------------------------------------------

class DenseMotionNetwork(nn.Module):
    """
    Converts sparse keypoint displacements into a dense optical-flow field and
    an occlusion mask using a hourglass with recurrent ConvGRU blocks.

    The ConvGRU hidden states are held externally so that the caller can
    propagate them across frames of a video sequence.
    """

    def __init__(self, cfg: DenseMotionConfig,
                 num_keypoints: int, num_channels: int):
        super().__init__()
        self.cfg = cfg
        self.num_keypoints = num_keypoints

        # Input channels: K warped images (K*C) + K heatmap diffs + source img
        in_ch = num_keypoints * (num_channels + 1) + num_channels
        if cfg.use_bg_motion:
            in_ch += 2  # background flow channels

        # ---- Encoder --------------------------------------------------------
        enc_layers = []
        prev_ch = in_ch
        for i in range(cfg.num_blocks):
            out_ch = min(cfg.max_features,
                         cfg.block_expansion * (2 ** i))
            enc_layers.append(DownBlock2d(prev_ch, out_ch))
            prev_ch = out_ch
        self.encoder = nn.ModuleList(enc_layers)

        # ---- ConvGRU recurrent blocks (injected at bottleneck) --------------
        self.conv_gru = ConvGRU(
            input_channels=prev_ch,
            hidden_channels=cfg.hidden_channels,
            kernel_size=cfg.gru_kernel_size,
            num_layers=cfg.num_gru_layers,
        )

        # ---- Decoder --------------------------------------------------------
        dec_in = cfg.hidden_channels
        dec_layers = []
        for i in range(cfg.num_blocks):
            level = cfg.num_blocks - 1 - i
            skip_ch = min(cfg.max_features,
                          cfg.block_expansion * (2 ** level))
            out_ch = max(cfg.block_expansion,
                         cfg.block_expansion * (2 ** (level - 1)))
            dec_layers.append(UpBlock2d(dec_in + skip_ch, out_ch))
            dec_in = out_ch
        self.decoder = nn.ModuleList(dec_layers)

        # ---- Output heads ---------------------------------------------------
        self.flow_head = nn.Conv2d(dec_in, 2, kernel_size=7, padding=3)
        self.occ_head  = nn.Sequential(
            nn.Conv2d(dec_in, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

        # ---- Warper ---------------------------------------------------------
        self.warper = KeypointWarp(num_keypoints)

        # ---- Downscaler to motion resolution --------------------------------
        if cfg.scale_factor != 1.0:
            from models.blocks import AntiAliasInterp2d
            self.down = AntiAliasInterp2d(num_channels, cfg.scale_factor)
        else:
            self.down = nn.Identity()

    # ------------------------------------------------------------------

    def forward(self,
                source_image: torch.Tensor,
                kp_source: dict,
                kp_driving: dict,
                hidden: list = None) -> dict:
        """
        Args:
            source_image : (B, C, H, W)  reference frame
            kp_source    : keypoint dict from KeypointDetector(source_image)
            kp_driving   : keypoint dict from KeypointDetector(driving_frame)
            hidden       : ConvGRU hidden state list (or None for first frame)

        Returns dict:
            'dense_flow'       : (B, 2, H, W)  optical flow in pixel coords
            'occlusion_mask'   : (B, 1, H, W)  in [0,1]
            'warped_images'    : (B, K, C, H, W)
            'hidden'           : updated ConvGRU hidden states
        """
        B, C, H, W = source_image.shape

        # Work at motion resolution
        src_small = self.down(source_image)
        Hm, Wm = src_small.shape[-2:]

        # --- Warp source image at each keypoint ------------------------------
        warp_out = self.warper(src_small, kp_source, kp_driving)
        warped   = warp_out["warped_images"]      # (B, K, C, Hm, Wm)
        hm_diff  = warp_out["heatmap_diff"]       # (B, K, Hm, Wm)
        bg_flow  = warp_out["bg_motion"]          # (B, 2, Hm, Wm) if bg

        # Interpolate heatmap diff to motion resolution
        hm_diff = F.interpolate(hm_diff, size=(Hm, Wm),
                                mode="bilinear", align_corners=False)

        # Flatten warped: (B, K*C, Hm, Wm)
        warped_flat = warped.view(B, -1, Hm, Wm)

        parts = [warped_flat, hm_diff, src_small]
        if self.cfg.use_bg_motion:
            parts.append(bg_flow)
        x = torch.cat(parts, dim=1)               # (B, in_ch, Hm, Wm)

        # --- Encoder ---------------------------------------------------------
        enc_features = []
        for block in self.encoder:
            x = block(x)
            enc_features.append(x)

        # --- ConvGRU at bottleneck -------------------------------------------
        gru_out, new_hidden = self.conv_gru(enc_features[-1], hidden)

        # --- Decoder with skip connections -----------------------------------
        dec = gru_out
        for i, block in enumerate(self.decoder):
            skip = enc_features[-(i + 1)]
            dec = F.interpolate(dec, size=skip.shape[-2:],
                                mode="bilinear", align_corners=False)
            dec = torch.cat([dec, skip], dim=1)
            dec = block(dec)

        # --- Output heads at full resolution ---------------------------------
        dec_full = F.interpolate(dec, size=(H, W),
                                 mode="bilinear", align_corners=False)
        dense_flow     = self.flow_head(dec_full)  # (B, 2, H, W)
        occlusion_mask = self.occ_head(dec_full)   # (B, 1, H, W)

        return {
            "dense_flow":     dense_flow,
            "occlusion_mask": occlusion_mask,
            "warped_images":  warped,
            "hidden":         new_hidden,
        }
