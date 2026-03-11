"""
test_smoke.py
=============
Smoke tests for the hybrid FOMM talking-head model.

Runs forward passes through every module using random tensors to catch shape
errors, import issues, and numerical problems *without* requiring real data or
a GPU.  All tests use a miniaturised model configuration.

Run with:
    python test_smoke.py
"""

import sys
import traceback
import torch
import torch.nn as nn

# ---- Miniature config ----
from config import Config, ModelConfig, TrainingConfig, PersonalizationConfig

MINI_CFG = Config()
m = MINI_CFG.model
m.image_size            = 64
m.num_keypoints         = 5
m.kp_detector_features  = 16
m.kp_detector_depth     = 2
m.dense_motion_features = 16
m.dense_motion_depth    = 2
m.generator_features    = 16
m.generator_depth       = 2
m.rnn_hidden            = 32
m.rnn_layers            = 1
m.fusion_features       = 16
m.fusion_depth          = 2
m.num_attention_heads   = 2
m.hevc_scale            = 0.5
m.hevc_qp               = 35
MINI_CFG.training.mixed_precision = False
MINI_CFG.personalization.num_epochs = 1

DEVICE = 'cpu'
B, H, W = 2, 64, 64


def section(name: str):
    print(f'\n{"─"*60}')
    print(f'  {name}')
    print('─'*60)


def ok(msg: str):
    print(f'  ✓  {msg}')


def fail(msg: str, exc: Exception):
    print(f'  ✗  {msg}')
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Individual module tests
# ---------------------------------------------------------------------------

def test_keypoint_detector():
    section('KeypointDetector')
    from models.keypoint_detector import KeypointDetector

    model = KeypointDetector(MINI_CFG.model).to(DEVICE)
    model.eval()

    img = torch.rand(B, 3, H, W)
    with torch.no_grad():
        out = model(img)

    assert out['keypoints'].shape  == (B, m.num_keypoints, 2),         \
        f"Bad kp shape: {out['keypoints'].shape}"
    assert out['jacobians'].shape  == (B, m.num_keypoints, 2, 2),      \
        f"Bad jac shape: {out['jacobians'].shape}"
    assert out['heatmaps'].shape   == (B, m.num_keypoints, H, W),      \
        f"Bad heatmap shape: {out['heatmaps'].shape}"
    ok(f"keypoints={tuple(out['keypoints'].shape)}  "
       f"jacobians={tuple(out['jacobians'].shape)}")


def test_dense_motion_network():
    section('DenseMotionNetwork')
    from models.keypoint_detector import KeypointDetector
    from models.dense_motion      import DenseMotionNetwork

    kp_det = KeypointDetector(MINI_CFG.model).to(DEVICE).eval()
    dm_net = DenseMotionNetwork(MINI_CFG.model).to(DEVICE).eval()

    ref = torch.rand(B, 3, H, W)
    drv = torch.rand(B, 3, H, W)

    with torch.no_grad():
        kp_src = kp_det(ref)
        kp_drv = kp_det(drv)
        out = dm_net(ref, kp_src, kp_drv, hidden_states=None)

    assert out['deformation'].shape == (B, H, W, 2),   \
        f"Bad deformation shape: {out['deformation'].shape}"
    assert out['occlusion'].shape   == (B, 1, H, W),   \
        f"Bad occlusion shape: {out['occlusion'].shape}"
    assert out['hidden_state'] is not None
    ok(f"deformation={tuple(out['deformation'].shape)}  "
       f"occlusion={tuple(out['occlusion'].shape)}")

    # Test with hidden state (second frame)
    with torch.no_grad():
        out2 = dm_net(ref, kp_src, kp_drv, hidden_states=out['hidden_state'])
    ok('Second-frame pass with hidden state: OK')


def test_semantic_generator():
    section('SemanticFrameGenerator')
    from models.generator import SemanticFrameGenerator

    gen = SemanticFrameGenerator(MINI_CFG.model).to(DEVICE).eval()
    ref        = torch.rand(B, 3, H, W)
    deformation = torch.rand(B, H, W, 2) * 2 - 1   # in [-1, 1]
    occlusion   = torch.rand(B, 1, H, W)

    with torch.no_grad():
        out = gen(ref, deformation, occlusion)

    assert out['semantic_frame'].shape   == (B, 3, H, W), \
        f"Bad semantic frame shape: {out['semantic_frame'].shape}"
    assert out['warped_reference'].shape == (B, 3, H, W), \
        f"Bad warped ref shape: {out['warped_reference'].shape}"
    # Values should be in [0, 1] due to Sigmoid
    assert out['semantic_frame'].min() >= 0.0
    assert out['semantic_frame'].max() <= 1.0
    ok(f"semantic_frame={tuple(out['semantic_frame'].shape)} ∈ [0,1]")


def test_aux_stream_and_fusion():
    section('AuxiliaryHEVCStream + FusionModule')
    from models.fusion import AuxiliaryHEVCStream, FusionModule

    aux    = AuxiliaryHEVCStream(MINI_CFG.model).to(DEVICE).eval()
    fusion = FusionModule(MINI_CFG.model).to(DEVICE).eval()

    frame    = torch.rand(B, 3, H, W)
    semantic = torch.rand(B, 3, H, W)
    occlusion = torch.rand(B, 1, H, W)

    with torch.no_grad():
        aux_frame  = aux(frame)
        fused_out  = fusion(semantic, aux_frame, occlusion)

    assert aux_frame.shape           == (B, 3, H, W)
    assert fused_out['fused_frame'].shape == (B, 3, H, W)
    ok(f"aux_frame={tuple(aux_frame.shape)}  "
       f"fused_frame={tuple(fused_out['fused_frame'].shape)}")


def test_full_model():
    section('HybridTalkingHeadModel (end-to-end)')
    from models.full_model import HybridTalkingHeadModel

    model = HybridTalkingHeadModel(MINI_CFG.model).to(DEVICE).eval()

    ref = torch.rand(B, 3, H, W)
    drv = torch.rand(B, 3, H, W)

    with torch.no_grad():
        out = model(ref, drv, hidden_state=None)

    assert out['fused_frame'].shape    == (B, 3, H, W)
    assert out['semantic_frame'].shape == (B, 3, H, W)
    assert out['aux_frame'].shape      == (B, 3, H, W)
    assert out['deformation'].shape    == (B, H, W, 2)
    assert out['occlusion'].shape      == (B, 1, H, W)

    ok('All output tensors have correct shapes')

    # Second frame with hidden state
    with torch.no_grad():
        out2 = model(ref, drv, hidden_state=out['hidden_state'])
    ok('Second-frame pass: OK')

    # Parameter count
    counts = model.count_parameters()
    ok(f"Total parameters: {counts['total']:,}")
    for k, v in counts.items():
        if k != 'total':
            print(f'      {k:30s}: {v:>10,}')


def test_losses():
    section('Loss Functions')
    from models.keypoint_detector import KeypointDetector
    from models.full_model        import HybridTalkingHeadModel
    from training.losses          import (CompositeLoss, PatchDiscriminator,
                                          ReconstructionLoss, EquivarianceLoss)

    model = HybridTalkingHeadModel(MINI_CFG.model).to(DEVICE)
    disc  = PatchDiscriminator().to(DEVICE)

    ref    = torch.rand(B, 3, H, W)
    target = torch.rand(B, 3, H, W)
    drv    = torch.rand(B, 3, H, W)

    out = model(ref, drv, hidden_state=None)

    # Reconstruction loss
    recon = ReconstructionLoss(MINI_CFG.model, device=DEVICE).to(DEVICE)
    r_loss, r_logs = recon(out['fused_frame'], target)
    assert r_loss.item() > 0
    ok(f"Reconstruction loss={r_loss.item():.4f}")

    # Equivariance loss
    eq = EquivarianceLoss(MINI_CFG.model)
    e_loss, e_logs = eq(model.keypoint_detector, ref)
    assert e_loss.item() >= 0
    ok(f"Equivariance loss={e_loss.item():.4f}")

    # Discriminator loss
    real_pred = disc(target)
    fake_pred = disc(out['fused_frame'].detach())
    from training.losses import GANLoss
    gan  = GANLoss()
    d_loss = gan.discriminator_loss(real_pred, fake_pred)
    ok(f"Discriminator loss={d_loss.item():.4f}")


def test_metrics():
    section('Evaluation Metrics (PSNR / SSIM / LPIPS / AuxDep)')
    from training.metrics import (compute_psnr, compute_ssim,
                                   compute_auxiliary_dependency,
                                   LPIPSMetric, MetricsAccumulator)

    pred   = torch.rand(B, 3, H, W)
    target = torch.rand(B, 3, H, W)

    psnr = compute_psnr(pred, target)
    ok(f"PSNR  = {psnr.item():.2f} dB")

    ssim = compute_ssim(pred, target)
    ok(f"SSIM  = {ssim.item():.4f}")

    lpips_fn = LPIPSMetric(pretrained=False).to(DEVICE)
    lp = lpips_fn(pred, target)
    ok(f"LPIPS = {lp.item():.4f}")

    semantic = torch.rand(B, 3, H, W)
    ad = compute_auxiliary_dependency(pred, semantic)
    ok(f"AuxDep= {ad.item():.4f}")

    acc = MetricsAccumulator(DEVICE)
    acc.lpips = LPIPSMetric(pretrained=False).to(DEVICE)   # avoid download
    acc.update(pred, target, semantic)
    summary = acc.summary()
    ok(f"Accumulator summary: {summary}")


def test_personalization_module():
    section('PersonalizationFinetuner (1 synthetic epoch)')
    from models.full_model         import HybridTalkingHeadModel
    from personalization.finetune  import PersonalizationFinetuner

    model = HybridTalkingHeadModel(MINI_CFG.model)

    MINI_CFG.personalization.num_epochs = 1

    finetuner = PersonalizationFinetuner(
        model=model,
        cfg=MINI_CFG,
        identity_dir='./nonexistent_synthetic',  # triggers synthetic mode
        device=DEVICE,
    )
    personalized = finetuner.run(save_path=None)
    assert personalized is not None
    ok('Personalisation fine-tuning completed (synthetic data)')


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ('KeypointDetector',         test_keypoint_detector),
    ('DenseMotionNetwork',       test_dense_motion_network),
    ('SemanticFrameGenerator',   test_semantic_generator),
    ('AuxStream + Fusion',       test_aux_stream_and_fusion),
    ('HybridTalkingHeadModel',   test_full_model),
    ('Loss Functions',           test_losses),
    ('Evaluation Metrics',       test_metrics),
    ('PersonalizationFinetuner', test_personalization_module),
]

if __name__ == '__main__':
    print('=' * 60)
    print('  Hybrid FOMM Talking-Head — Smoke Tests')
    print(f'  Device: {DEVICE}  |  Image: {B}×3×{H}×{W}')
    print('=' * 60)

    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            fail(name, e)
            failed += 1

    print('\n' + '=' * 60)
    print(f'  Results: {passed} passed, {failed} failed')
    print('=' * 60)
    sys.exit(0 if failed == 0 else 1)
