"""
models/fusion.py — Auxiliary HEVC Stream + Neural Fusion Module
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from config import ModelConfig


class DifferentiableQuantise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step):
        return torch.round(x / step) * step
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def simulate_hevc(frame: torch.Tensor, scale: float, qp: int) -> torch.Tensor:
    B, C, H, W = frame.shape
    lq = F.interpolate(frame, scale_factor=scale, mode='bilinear',
                       align_corners=False, recompute_scale_factor=True)
    step = max((2 ** ((qp - 4) / 6)) / 255.0, 1e-4)
    lq_q = DifferentiableQuantise.apply(lq, step)
    blur_radius = max(1, int(qp / 20))
    kernel_size = 2 * blur_radius + 1
    sigma = blur_radius / 3.0
    coords = torch.arange(kernel_size, dtype=lq.dtype, device=lq.device) - blur_radius
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); gauss /= gauss.sum()
    kernel_2d = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
    lq_blur = F.conv2d(lq_q, kernel_2d, padding=blur_radius, groups=C)
    return F.interpolate(lq_blur, size=(H, W), mode='bilinear', align_corners=False).clamp(0, 1)


class AuxiliaryHEVCStream(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.scale = cfg.hevc_scale
        self.base_qp = cfg.hevc_qp
        self.qp_offset = nn.Parameter(torch.zeros(1))

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        qp = int((self.base_qp + self.qp_offset.clamp(-10, 10)).item())
        return simulate_hevc(frame, self.scale, max(0, min(51, qp)))


class SpatialCrossAttention(nn.Module):
    def __init__(self, feat_ch: int, num_heads: int, patch_size: int = 8):
        super().__init__()
        assert feat_ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = feat_ch // num_heads
        self.patch_size = patch_size
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(feat_ch, feat_ch, bias=False)
        self.k_proj = nn.Linear(feat_ch, feat_ch, bias=False)
        self.v_proj = nn.Linear(feat_ch, feat_ch, bias=False)
        self.out_proj = nn.Linear(feat_ch, feat_ch, bias=False)

    def _patchify(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = F.adaptive_avg_pool2d(x, (H // p, W // p))
        return x.flatten(2).permute(0, 2, 1)

    def forward(self, sem: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        B, C, H, W = sem.shape
        q_seq = self._patchify(sem); k_seq = self._patchify(aux); v_seq = self._patchify(aux)
        Q = self.q_proj(q_seq); K = self.k_proj(k_seq); V = self.v_proj(v_seq)
        def split_heads(t):
            B, N, C = t.shape
            return t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, -1, C)
        out = self.out_proj(out)
        ph_ = H // self.patch_size; pw_ = W // self.patch_size
        out_s = out.permute(0, 2, 1).reshape(B, C, ph_, pw_)
        return F.interpolate(out_s, size=(H, W), mode='bilinear', align_corners=True)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b = nn.Sequential(
            nn.GroupNorm(max(1, ch // 8), ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, ch // 8), ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
    def forward(self, x): return x + self.b(x)


class FusionModule(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        F_ = cfg.fusion_features
        self.sem_enc = nn.Sequential(nn.Conv2d(3, F_, 3, padding=1), nn.ReLU(inplace=True), ResBlock(F_), ResBlock(F_))
        self.aux_enc = nn.Sequential(nn.Conv2d(3, F_, 3, padding=1), nn.ReLU(inplace=True), ResBlock(F_), ResBlock(F_))
        self.fusion_attention = SpatialCrossAttention(F_, cfg.num_attention_heads)
        self.merge = nn.Sequential(nn.Conv2d(F_ * 3 + 1, F_, 1), nn.ReLU(inplace=True))
        self.refine = nn.Sequential(*[ResBlock(F_) for _ in range(cfg.fusion_depth)])
        self.head = nn.Sequential(nn.Conv2d(F_, 3, 1), nn.Sigmoid())

    def forward(self, semantic_frame, aux_frame, occlusion) -> Dict:
        H, W = semantic_frame.shape[-2:]
        sem_feat = self.sem_enc(semantic_frame)
        aux_feat = self.aux_enc(aux_frame)
        attn_feat = self.fusion_attention(sem_feat, aux_feat)
        occ = F.interpolate(occlusion, size=(H, W), mode='bilinear', align_corners=True)
        merged = self.merge(torch.cat([sem_feat, aux_feat, attn_feat, occ], dim=1))
        refined = self.refine(merged)
        return {'fused_frame': self.head(refined)}
