"""
training/metrics.py
===================
Evaluation Metrics
------------------

Implements all metrics required for model evaluation:

1. **PSNR** (Peak Signal-to-Noise Ratio)
   Standard distortion metric; higher is better.

2. **SSIM** (Structural Similarity Index)
   Perceptually motivated similarity that accounts for luminance, contrast,
   and structure; higher is better.

3. **LPIPS** (Learned Perceptual Image Patch Similarity)
   Deep-feature-based metric that correlates well with human judgement;
   lower is better.

4. **Auxiliary Dependency** (custom metric)
   Measures how much the fused output depends on the auxiliary HEVC stream
   vs. the semantic stream alone.  Computed as the normalised L1 distance
   between:
     (a) the final fused frame
     (b) the semantic-only frame
   averaged over the batch.  Higher value ⇒ fusion adds more from the
   auxiliary stream.  Useful for monitoring that the fusion module is
   actually leveraging both streams.

All metrics operate on (B, 3, H, W) float tensors in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import Dict, List, Optional
import math


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor,
                 target: torch.Tensor,
                 max_val: float = 1.0) -> torch.Tensor:
    """
    Per-sample PSNR averaged over the batch.

    Parameters
    ----------
    pred, target : (B, C, H, W) float in [0, max_val]

    Returns
    -------
    Scalar tensor (mean PSNR in dB over the batch).
    """
    assert pred.shape == target.shape
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])  # B
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr.mean()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float,
                     channels: int, device) -> torch.Tensor:
    """Create a 2-D Gaussian kernel for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)   # w×w
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size)
    return kernel_2d.contiguous()


def compute_ssim(pred: torch.Tensor,
                 target: torch.Tensor,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 C1: float = 0.01 ** 2,
                 C2: float = 0.03 ** 2) -> torch.Tensor:
    """
    Differentiable SSIM (mean over batch and channels).

    Parameters
    ----------
    pred, target : (B, C, H, W) float in [0, 1]
    """
    B, C, H, W = pred.shape
    device = pred.device

    kernel = _gaussian_kernel(window_size, sigma, C, device)
    pad    = window_size // 2

    mu_x  = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu_y  = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_x2   = mu_x ** 2
    mu_y2   = mu_y ** 2
    mu_xy   = mu_x * mu_y

    sigma_x2  = F.conv2d(pred   ** 2, kernel, padding=pad, groups=C) - mu_x2
    sigma_y2  = F.conv2d(target ** 2, kernel, padding=pad, groups=C) - mu_y2
    sigma_xy  = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return ssim_map.mean()


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

class LPIPSMetric(nn.Module):
    """
    Learned Perceptual Image Patch Similarity using VGG-16 features.
    Follows the standard LPIPS computation (Zhang et al., 2018).
    """

    def __init__(self, pretrained: bool = True, device: str = 'cpu'):
        super().__init__()
        vgg = tv_models.vgg16(
            weights=tv_models.VGG16_Weights.DEFAULT if pretrained else None
        ).features

        # Extract feature blocks
        slice_ends = [4, 9, 16, 23, 30]   # relu1_2 … relu5_2
        self.slices = nn.ModuleList()
        prev = 0
        for end in slice_ends:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev:end]))
            prev = end

        for p in self.parameters():
            p.requires_grad_(False)

        # Per-layer linear weights (learned in LPIPS; here we use equal weights
        # for the pre-trained-only variant)
        self.weights = [1 / len(slice_ends)] * len(slice_ends)

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])
                             .view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225])
                             .view(1, 3, 1, 1))

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred   = self._normalise(pred)
        target = self._normalise(target)

        loss = torch.tensor(0.0, device=pred.device)
        for sl, w in zip(self.slices, self.weights):
            pred   = sl(pred)
            target = sl(target)
            # Normalise channel-wise
            p_norm = F.normalize(pred,   dim=1)
            t_norm = F.normalize(target, dim=1)
            loss = loss + w * (p_norm - t_norm).pow(2).mean()
        return loss


# ---------------------------------------------------------------------------
# Auxiliary Dependency (custom metric)
# ---------------------------------------------------------------------------

def compute_auxiliary_dependency(fused_frame: torch.Tensor,
                                 semantic_frame: torch.Tensor) -> torch.Tensor:
    """
    Measure how much the fusion output differs from the semantic-only output.

    AD = mean_pixel( |fused - semantic| ) / mean_pixel( semantic )

    Values close to 0 → fusion is dominated by the semantic stream.
    Values far from 0 → fusion heavily leverages the auxiliary stream.

    Parameters
    ----------
    fused_frame, semantic_frame : (B, 3, H, W) in [0, 1]

    Returns
    -------
    Scalar auxiliary-dependency score (higher ⇒ more auxiliary influence).
    """
    diff      = (fused_frame - semantic_frame).abs()
    numerator = diff.mean()
    denominator = semantic_frame.abs().mean() + 1e-8
    return numerator / denominator


# ---------------------------------------------------------------------------
# Metrics accumulator for validation loop
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """
    Running mean tracker for all evaluation metrics.

    Usage::

        acc = MetricsAccumulator(device)
        for batch in val_loader:
            acc.update(pred, target, fused, semantic)
        summary = acc.summary()
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.lpips  = LPIPSMetric(pretrained=True, device=device).to(device)
        self.lpips.eval()

        self._reset()

    def _reset(self):
        self.psnr_sum  = 0.0
        self.ssim_sum  = 0.0
        self.lpips_sum = 0.0
        self.aux_dep_sum = 0.0
        self.count     = 0

    def update(self,
               fused_frame:    torch.Tensor,
               target_frame:   torch.Tensor,
               semantic_frame: torch.Tensor):
        """
        fused_frame, target_frame, semantic_frame : (B, 3, H, W) in [0, 1]
        """
        with torch.no_grad():
            psnr   = compute_psnr(fused_frame, target_frame).item()
            ssim   = compute_ssim(fused_frame, target_frame).item()
            lp     = self.lpips(fused_frame, target_frame).item()
            aux_dep = compute_auxiliary_dependency(
                fused_frame, semantic_frame
            ).item()

        B = fused_frame.shape[0]
        self.psnr_sum   += psnr   * B
        self.ssim_sum   += ssim   * B
        self.lpips_sum  += lp     * B
        self.aux_dep_sum += aux_dep * B
        self.count      += B

    def summary(self) -> Dict[str, float]:
        if self.count == 0:
            return {}
        n = self.count
        metrics = {
            'PSNR':               self.psnr_sum  / n,
            'SSIM':               self.ssim_sum  / n,
            'LPIPS':              self.lpips_sum / n,
            'Auxiliary_Dependency': self.aux_dep_sum / n,
        }
        self._reset()
        return metrics

    def format_summary(self, metrics: Optional[Dict] = None) -> str:
        if metrics is None:
            metrics = self.summary()
        lines = ['Evaluation Metrics:']
        lines.append(f"  PSNR  (↑) : {metrics.get('PSNR',  0):.4f} dB")
        lines.append(f"  SSIM  (↑) : {metrics.get('SSIM',  0):.4f}")
        lines.append(f"  LPIPS (↓) : {metrics.get('LPIPS', 0):.4f}")
        lines.append(f"  Aux.Dep.  : {metrics.get('Auxiliary_Dependency', 0):.4f}")
        return '\n'.join(lines)
