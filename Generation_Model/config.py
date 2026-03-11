"""
config.py — Centralised configuration for the Hybrid FOMM Talking-Head system.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class KeypointConfig:
    num_keypoints: int = 15
    temperature: float = 0.1
    scale_factor: float = 0.25
    use_jacobian: bool = True


@dataclass
class DenseMotionConfig:
    block_expansion: int = 64
    max_features: int = 1024
    num_blocks: int = 5
    scale_factor: float = 0.25
    hidden_channels: int = 128
    gru_kernel_size: int = 3
    num_gru_layers: int = 2
    use_bg_motion: bool = True


@dataclass
class GeneratorConfig:
    block_expansion: int = 64
    max_features: int = 512
    num_down_blocks: int = 5
    num_bottleneck_blocks: int = 6
    estimate_occlusion_map: bool = True


@dataclass
class FusionConfig:
    num_features: int = 64
    num_blocks: int = 4
    num_attention_heads: int = 4
    use_spectral_norm: bool = True


@dataclass
class AuxiliaryStreamConfig:
    scale_factor: float = 0.25
    jpeg_quality: int = 20
    encode_channels: int = 64


@dataclass
class ModelConfig:
    num_channels: int = 3
    image_size: int = 256
    keypoint: KeypointConfig = field(default_factory=KeypointConfig)
    dense_motion: DenseMotionConfig = field(default_factory=DenseMotionConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    auxiliary: AuxiliaryStreamConfig = field(default_factory=AuxiliaryStreamConfig)


@dataclass
class DatasetConfig:
    root: str = "data/voxceleb"
    frame_shape: Tuple[int, int] = (256, 256)
    num_workers: int = 4
    pairs_per_video: int = 50
    augmentation: bool = True


@dataclass
class LossConfig:
    lambda_l1: float = 10.0
    lambda_perceptual: float = 10.0
    perceptual_layers: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    lambda_equivariance_value: float = 10.0
    lambda_equivariance_jacobian: float = 10.0
    lambda_gan: float = 1.0
    lambda_auxiliary: float = 5.0
    lambda_fusion_perceptual: float = 5.0


@dataclass
class OptimiserConfig:
    lr_generator: float = 2e-4
    lr_discriminator: float = 2e-4
    lr_keypoint: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    scheduler_milestones: List[int] = field(default_factory=lambda: [60, 90])
    scheduler_gamma: float = 0.1


@dataclass
class TrainConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    batch_size: int = 4
    num_epochs: int = 100
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    log_every: int = 100
    save_every: int = 5
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/best.pth"
    output_dir: str = "eval_output"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    metrics: List[str] = field(
        default_factory=lambda: ["psnr", "ssim", "lpips", "auxiliary_dependency"]
    )
    auxiliary_dropout_eval: bool = True
    batch_size: int = 1


@dataclass
class PersonalizeConfig:
    pretrained_checkpoint: str = "checkpoints/pretrained.pth"
    output_checkpoint: str = "checkpoints/personalised.pth"
    video_paths: List[str] = field(default_factory=list)
    num_epochs: int = 15
    batch_size: int = 2
    lr_dense_motion: float = 5e-5
    lr_motion_module: float = 5e-5
    lr_recurrent: float = 5e-5
    lr_attention: float = 5e-5
    beta1: float = 0.5
    beta2: float = 0.999
    freeze_backbone: bool = True
    freeze_keypoint_stem: bool = True
    lambda_l2_reg: float = 1e-3
    log_dir: str = "logs/personalization"
    save_every: int = 3
