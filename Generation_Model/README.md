# Hybrid FOMM Talking-Head Codec

A PyTorch implementation of a hybrid talking-head video generation system
combining a First-Order Motion Model (FOMM) backbone with an auxiliary
HEVC-encoded stream and a neural fusion module.

---

## Architecture Overview

```
Reference Image ──► KeypointDetector ──► kp_source ──┐
Driving Frame   ──► KeypointDetector ──► kp_driving ─┤
                                                       │
                          DenseMotionNetwork ◄─────────┘
                          + RecurrentBlocks (GRU)
                                 │
                    ┌────────────┴──────────────┐
                    │                           │
          SemanticFrameGenerator      AuxiliaryHEVCStream
          (UNet + multi-scale warp)   (simulated codec)
                    │                           │
                    └───────────┬───────────────┘
                          FusionModule
                    (cross-attention + conv)
                                │
                          Final Output Frame
```

### Modules

| Module | File | Description |
|--------|------|-------------|
| `KeypointDetector` | `models/keypoint_detector.py` | Hourglass network → N semantic facial keypoints + 2×2 Jacobians |
| `DenseMotionNetwork` | `models/dense_motion.py` | Sparse → dense displacement field; ConvGRU temporal memory |
| `SemanticFrameGenerator` | `models/generator.py` | UNet that warps and decodes reference image using dense field |
| `AuxiliaryHEVCStream` | `models/fusion.py` | Differentiable HEVC-like codec simulation for the auxiliary bitstream |
| `FusionModule` | `models/fusion.py` | Multi-head cross-attention merger of semantic + auxiliary streams |
| `HybridTalkingHeadModel` | `models/full_model.py` | Orchestrates all modules end-to-end |

---

## Project Structure

```
talking_head/
├── config.py                        # All hyper-parameters (ModelConfig, TrainingConfig, PersonalizationConfig)
├── train.py                         # Main entry point (train / evaluate / personalize)
├── test_smoke.py                    # Smoke tests (no real data needed)
│
├── models/
│   ├── keypoint_detector.py         # Keypoint Detector
│   ├── dense_motion.py              # Dense Motion Network + ConvGRU
│   ├── generator.py                 # Semantic Frame Generator
│   ├── fusion.py                    # HEVC stream + Fusion Module
│   └── full_model.py                # Full end-to-end model
│
├── dataset/
│   └── voxceleb.py                  # VoxCeleb DataLoader + PersonalizationDataset
│
├── training/
│   ├── losses.py                    # Reconstruction, Equivariance, GAN losses
│   ├── metrics.py                   # PSNR, SSIM, LPIPS, Auxiliary Dependency
│   └── trainer.py                   # Training loop (GAN, AMP, TensorBoard)
│
└── personalization/
    └── finetune.py                  # Personalised fine-tuning with EWC regularisation
```

---

## Loss Functions

| Loss | Term | Purpose |
|------|------|---------|
| **L1 Reconstruction** | `λ_rec · L1(ŷ, y)` | Pixel-level fidelity |
| **Perceptual (VGG-19)** | `λ_perc · Σ_l L1(φ_l(ŷ), φ_l(y))` | Feature-level similarity |
| **Equivariance (value)** | `λ_eq_v · L1(KP(T(x)), T(KP(x)))` | Geometrically consistent keypoints |
| **Equivariance (Jacobian)** | `λ_eq_j · L1(J_T, J_orig)` | Consistent local affine transforms |
| **Adversarial (Hinge GAN)** | `−E[D(ŷ)]` | Sharp, realistic textures |

---

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| **PSNR** | ↑ higher better | Peak Signal-to-Noise Ratio (dB) |
| **SSIM** | ↑ higher better | Structural Similarity Index |
| **LPIPS** | ↓ lower better | Learned Perceptual Image Patch Similarity (VGG-16) |
| **Auxiliary Dependency** | monitor | `‖fused − semantic‖ / ‖semantic‖` — fraction of output coming from the auxiliary stream |

---

## Personalisation

Fine-tuning uses **selective parameter updates** on the most identity-sensitive components:

```
Trainable during personalisation:
  ✓ dense_motion_network      (motion field generator)
  ✓ dense_motion_network.recurrent  (GRU temporal blocks)
  ✓ fusion_module.fusion_attention  (stream-fusion attention)

Frozen during personalisation:
  ✗ keypoint_detector          (generalises across identities)
  ✗ semantic_generator (backbone)
  ✗ aux_stream                 (codec-agnostic)
```

An **EWC-lite anchor regulariser** pulls fine-tuned weights toward their
pre-trained values to prevent catastrophic forgetting.

---

## Installation

```bash
pip install torch torchvision tensorboard
```

---

## Usage

```bash
# Pre-train on VoxCeleb
python train.py --mode train --dataset_root /path/to/voxceleb

# Resume training
python train.py --mode train --resume checkpoints/checkpoint_epoch_0050.pt

# Evaluate on test split
python train.py --mode evaluate --resume checkpoints/checkpoint_epoch_0100.pt

# Personalise for a single identity
python train.py --mode personalize \
    --identity_dir /path/to/voxceleb/id_target \
    --resume checkpoints/checkpoint_epoch_0100.pt \
    --save_personalized checkpoints/personalized.pt

# Smoke-test (no data required)
python train.py --mode train --smoke_test
python test_smoke.py
```

---

## Dataset Layout (VoxCeleb)

```
data/voxceleb/
├── train/
│   ├── id10001/
│   │   ├── clip_001/  0001.png  0002.png  …
│   │   └── clip_002/  …
│   └── id10002/ …
└── test/
    └── …
```

---

## Key Configuration Options (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_keypoints` | 15 | Semantic facial keypoints |
| `image_size` | 256 | Spatial resolution |
| `rnn_hidden` | 256 | ConvGRU hidden channels |
| `hevc_qp` | 40 | Auxiliary stream quality (0–51) |
| `hevc_scale` | 0.25 | Auxiliary stream spatial scale |
| `lambda_perceptual` | 10.0 | Weight of VGG perceptual loss |
| `batch_size` | 8 | Training batch size |
| `num_frames` | 16 | Temporal unrolling steps |
