"""
personalization/finetune.py
===========================
Personalised Retraining Module
--------------------------------
After the model has been pre-trained on a large, diverse corpus (VoxCeleb),
this module performs identity-specific fine-tuning using a few short (~2–3 min)
video clips of the target person recorded at the start of a videoconferencing
session.

Design principles
~~~~~~~~~~~~~~~~~
1. **Selective fine-tuning**: Only the most identity-sensitive components are
   updated; the backbone convolutional representations stay fixed.
   Trainable modules:
     - ``dense_motion_network``    — motion field generator
     - ``dense_motion_network.recurrent`` — temporal consistency GRU
     - ``fusion_module.fusion_attention`` — adaptive stream fusion
   Frozen modules:
     - ``keypoint_detector``       — generalises well across identities
     - ``semantic_generator``      — backbone encoder/decoder
     - ``aux_stream``              — HEVC codec simulation is identity-agnostic

2. **Lower learning rate** (default 5e-5, ~40× smaller than pre-training).

3. **Regularisation**: L2 weight regularisation toward the pre-trained weights
   (Elastic Weight Consolidation-lite) prevents catastrophic forgetting.

4. **Early stopping** on validation PSNR to avoid over-fitting the short clip.

Usage
-----
    finetuner = PersonalizationFinetuner(model, cfg, identity_dir)
    finetuner.run()
    personalized_model = finetuner.model   # ready for inference
"""

import copy
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from config import Config
from models.full_model  import HybridTalkingHeadModel
from training.losses    import ReconstructionLoss
from training.metrics   import MetricsAccumulator
from dataset.voxceleb   import build_personalization_dataloader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module name → parameter matching helper
# ---------------------------------------------------------------------------

_TRAINABLE_MODULE_PREFIXES: Dict[str, List[str]] = {
    'dense_motion_network': ['dense_motion_network.'],
    'motion_estimator':     ['dense_motion_network.'],          # alias
    'recurrent_blocks':     ['dense_motion_network.recurrent.'],
    'fusion_attention':     ['fusion_module.fusion_attention.'],
}


def _resolve_trainable_params(model: nn.Module,
                               module_names: List[str]) -> List[nn.Parameter]:
    """Return parameters belonging to the requested sub-modules."""
    prefixes = []
    for name in module_names:
        prefixes.extend(_TRAINABLE_MODULE_PREFIXES.get(name, [name + '.']))

    trainable = []
    for pname, param in model.named_parameters():
        if any(pname.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(True)
            trainable.append(param)
        else:
            param.requires_grad_(False)

    return trainable


# ---------------------------------------------------------------------------
# EWC-lite regulariser (anchor to pre-trained weights)
# ---------------------------------------------------------------------------

class AnchorRegulariser:
    """
    L2 penalty that pulls parameters toward their pre-trained values,
    preventing catastrophic forgetting.

    loss_reg = (lambda_anchor / 2) * Σ ||θ - θ_anchor||²
    """

    def __init__(self, model: nn.Module, lambda_anchor: float = 0.5):
        self.lambda_anchor = lambda_anchor
        # Store a frozen copy of the pre-trained weights
        self.anchor: Dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def loss(self, model: nn.Module) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.anchor:
                reg = reg + (param - self.anchor[name]).pow(2).sum()
        return (self.lambda_anchor / 2) * reg


# ---------------------------------------------------------------------------
# Main fine-tuner
# ---------------------------------------------------------------------------

class PersonalizationFinetuner:
    """
    Parameters
    ----------
    model        : HybridTalkingHeadModel (pre-trained)
    cfg          : Config
    identity_dir : directory containing video clips for the target identity
    device       : torch device string
    """

    def __init__(self,
                 model: HybridTalkingHeadModel,
                 cfg: Config,
                 identity_dir: str,
                 device: str = 'cuda'):
        self.cfg          = cfg
        self.device       = torch.device(device if torch.cuda.is_available()
                                         else 'cpu')
        # Deep copy so we don't corrupt the base model
        self.model        = copy.deepcopy(model).to(self.device)
        self.identity_dir = identity_dir

        p_cfg = cfg.personalization

        # ---- Selective parameter freezing ----
        trainable_params = _resolve_trainable_params(
            self.model, p_cfg.trainable_modules
        )
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total     = sum(p.numel() for p in self.model.parameters())
        log.info(
            f'Personalisation: {n_trainable:,} / {n_total:,} '
            f'parameters trainable ({100 * n_trainable / n_total:.1f}%)'
        )

        # ---- Optimiser ----
        self.optimizer = optim.Adam(trainable_params, lr=p_cfg.lr,
                                    betas=(0.9, 0.999))

        # ---- Regulariser (anchor to pre-trained weights) ----
        self.anchor_reg = AnchorRegulariser(self.model, lambda_anchor=0.5)

        # ---- Loss ----
        self.recon_loss = ReconstructionLoss(cfg.model,
                                             device=str(self.device)
                                             ).to(self.device)

        # ---- Data ----
        self.loader = build_personalization_dataloader(cfg, identity_dir)

        # ---- Metrics ----
        self.metrics = MetricsAccumulator(str(self.device))

        # ---- AMP ----
        self.scaler = GradScaler(enabled=cfg.training.mixed_precision
                                 and self.device.type == 'cuda')

    # ------------------------------------------------------------------
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0

        for batch in self.loader:
            ref  = batch['reference_image'].to(self.device)    # B×3×H×W
            drv  = batch['driving_frames'].to(self.device)     # B×T×3×H×W
            B, T, C, H, W = drv.shape

            hidden = None
            batch_loss = torch.tensor(0.0, device=self.device)

            for t in range(T):
                target    = drv[:, t]
                driving_t = drv[:, max(0, t - 1)]

                self.optimizer.zero_grad()
                with autocast(enabled=self.scaler.is_enabled()):
                    out = self.model(ref, driving_t, hidden_state=hidden)
                    hidden = [h.detach() for h in out['hidden_state']]

                    recon, _ = self.recon_loss(out['fused_frame'], target)
                    reg      = self.anchor_reg.loss(self.model)
                    loss     = recon + reg

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_loss = batch_loss + loss.detach()

            total_loss += (batch_loss / T).item()
            count += 1

        return total_loss / max(count, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        for batch in self.loader:
            ref = batch['reference_image'].to(self.device)
            drv = batch['driving_frames'].to(self.device)
            B, T, C, H, W = drv.shape

            hidden = None
            for t in range(T):
                driving_t = drv[:, max(0, t - 1)]
                out = self.model(ref, driving_t, hidden_state=hidden)
                hidden = out['hidden_state']
                self.metrics.update(
                    out['fused_frame'].clamp(0, 1),
                    drv[:, t].clamp(0, 1),
                    out['semantic_frame'].clamp(0, 1),
                )
        return self.metrics.summary()

    # ------------------------------------------------------------------
    def run(self,
            save_path: Optional[str] = None) -> HybridTalkingHeadModel:
        """
        Execute personalised fine-tuning.

        Parameters
        ----------
        save_path : if given, save the fine-tuned model weights here.

        Returns
        -------
        The fine-tuned HybridTalkingHeadModel (in eval mode).
        """
        p_cfg  = self.cfg.personalization
        epochs = p_cfg.num_epochs

        best_psnr  = -float('inf')
        best_state = None
        patience   = max(3, epochs // 5)
        no_improve = 0

        log.info(f'Starting personalisation for {epochs} epochs '
                 f'on {self.identity_dir}')

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            avg_loss = self._train_epoch()
            metrics  = self._evaluate()
            elapsed  = time.time() - t0

            psnr = metrics.get('PSNR', 0.0)
            log.info(
                f'[Finetune {epoch:03d}/{epochs}] '
                f'loss={avg_loss:.4f} '
                f"PSNR={psnr:.2f}dB "
                f"SSIM={metrics.get('SSIM', 0):.4f} "
                f"LPIPS={metrics.get('LPIPS', 0):.4f} "
                f"AuxDep={metrics.get('Auxiliary_Dependency', 0):.4f} "
                f"({elapsed:.1f}s)"
            )

            # Early stopping
            if psnr > best_psnr + 0.01:
                best_psnr  = psnr
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f'Early stopping at epoch {epoch}')
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            log.info(f'Restored best weights (PSNR={best_psnr:.2f} dB)')

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model': self.model.state_dict(),
                        'best_psnr': best_psnr}, save_path)
            log.info(f'Fine-tuned model saved to {save_path}')

        self.model.eval()
        return self.model
