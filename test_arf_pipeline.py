"""
hevc_integration/test_arf_pipeline.py
======================================
Smoke test for the Neural ARF Pipeline (no real codec / GPU required).

Tests:
  1. DPB — neural ARF slot management and RPL construction
  2. SEI — pack/unpack round-trip with quantisation error check
  3. Encoder (simulate=True) — frame encode with ARF injection
  4. Decoder (simulate=True) — frame decode with ARF regeneration
  5. Full pipeline loopback — encode + mux + demux + decode
  6. Mux container — multi-frame round-trip
"""

import sys
import traceback
import struct
import numpy as np
import torch

# ── Minimal model stub ────────────────────────────────────────────────────

class _DummyKPDetector(torch.nn.Module):
    """Returns random but deterministic keypoints for testing."""
    def __init__(self, K=5, H=64, W=64):
        super().__init__()
        self.K = K; self.H = H; self.W = W
    def forward(self, img):
        B = img.shape[0]
        return {
            'keypoints': torch.zeros(B, self.K, 2),
            'jacobians': torch.eye(2).unsqueeze(0).unsqueeze(0).expand(B, self.K, -1, -1),
            'heatmaps':  torch.zeros(B, self.K, self.H, self.W),
        }


class _DummyDenseMotion(torch.nn.Module):
    def forward(self, ref, kp_src, kp_drv, hidden_states=None):
        B, C, H, W = ref.shape
        return {
            'deformation': torch.zeros(B, H, W, 2),
            'occlusion':   torch.zeros(B, 1, H, W),
            'hidden_state': [],
            'flow':        torch.zeros(B, 2, H, W),
        }


class _DummyGenerator(torch.nn.Module):
    def forward(self, ref, deformation, occlusion):
        return {
            'semantic_frame':   torch.rand_like(ref),
            'warped_reference': ref,
        }


class _DummyFusion(torch.nn.Module):
    def forward(self, sem, aux, occ):
        return {'fused_frame': (sem + aux) / 2}


class _DummyAux(torch.nn.Module):
    def forward(self, frame): return frame * 0.8


class FakeModel(torch.nn.Module):
    def __init__(self, K=5):
        super().__init__()
        self.keypoint_detector    = _DummyKPDetector(K)
        self.dense_motion_network = _DummyDenseMotion()
        self.semantic_generator   = _DummyGenerator()
        self.fusion_module        = _DummyFusion()
        self.aux_stream           = _DummyAux()

    def forward(self, ref, drv, hidden_state=None, kp_source=None):
        if kp_source is None:
            kp_source = self.keypoint_detector(ref)
        kp_driving = self.keypoint_detector(drv)
        mo = self.dense_motion_network(ref, kp_source, kp_driving, hidden_state)
        go = self.semantic_generator(ref, mo['deformation'], mo['occlusion'])
        ax = self.aux_stream(drv)
        fu = self.fusion_module(go['semantic_frame'], ax, mo['occlusion'])
        return {
            'fused_frame':    fu['fused_frame'],
            'semantic_frame': go['semantic_frame'],
            'aux_frame':      ax,
            'warped_reference': go['warped_reference'],
            'kp_source':   kp_source,
            'kp_driving':  kp_driving,
            'deformation': mo['deformation'],
            'occlusion':   mo['occlusion'],
            'hidden_state': mo['hidden_state'],
        }


# ── Test helpers ──────────────────────────────────────────────────────────

B = 1; H = W = 64; K = 5

def _ref_img():
    return torch.rand(B, 3, H, W)

def _rand_frame():
    return np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)

def section(name):
    print(f'\n{"─"*60}\n  {name}\n{"─"*60}')

def ok(msg):  print(f'  ✓  {msg}')
def fail(msg, exc=None):
    print(f'  ✗  {msg}')
    if exc: traceback.print_exc()


# ── Tests ─────────────────────────────────────────────────────────────────

def test_dpb():
    section('1. Decoded Picture Buffer')
    from hevc_integration.dpb import (
        DecodedPictureBuffer, NEURAL_ARF_POC, RefUsage
    )

    dpb = DecodedPictureBuffer(max_size=4, inject_neural=True)
    assert len(dpb) == 1, 'Neural slot should be present at creation'
    assert dpb.has_neural_arf

    # Update neural ARF
    pixels = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    dpb.update_neural_arf(pixels)
    np.testing.assert_array_equal(dpb.neural_arf_pixels, pixels)
    ok('Neural ARF slot created and updated')

    # Insert regular frames and check they don't clobber the ARF
    for poc in range(5):
        dpb.insert(poc, np.zeros((H, W, 3), dtype=np.uint8))

    neural_entry = dpb.get(NEURAL_ARF_POC)
    assert neural_entry is not None and neural_entry.is_neural
    ok(f'DPB size={len(dpb)}, neural slot preserved after {5} inserts')

    # Build RPL — neural ARF must be L0[0]
    rpl = dpb.build_rpl(current_poc=5, max_l0=4)
    assert rpl.l0[0].is_neural, 'Neural ARF must be first in L0'
    ok(f'RPL: {rpl.summary()}')

    # Apply RPS — neural slot must survive
    dpb.apply_rps(strp_pocs=[3, 4], ltrp_pocs=[NEURAL_ARF_POC])
    assert dpb.get(NEURAL_ARF_POC) is not None
    ok('Neural ARF survives RPS application')


def test_sei_roundtrip():
    section('2. SEI Pack / Unpack Round-Trip')
    from hevc_integration.sei import NeuralARFPayload, NeuralARFSEI

    kp_src  = np.random.uniform(-1, 1, (K, 2)).astype(np.float32)
    kp_drv  = np.random.uniform(-1, 1, (K, 2)).astype(np.float32)
    jac_src = np.random.uniform(-1, 1, (K, 2, 2)).astype(np.float32)
    jac_drv = np.random.uniform(-1, 1, (K, 2, 2)).astype(np.float32)

    payload = NeuralARFPayload(kp_source=kp_src, kp_driving=kp_drv,
                               jac_source=jac_src, jac_driving=jac_drv,
                               poc=42)
    packed   = NeuralARFSEI.pack(payload)
    unpacked = NeuralARFSEI.unpack(packed)

    assert unpacked is not None
    assert unpacked.poc == 42

    # Quantisation error should be small (8-bit for kp, int8 for jac)
    kp_err  = np.abs(unpacked.kp_source - kp_src).max()
    jac_err = np.abs(unpacked.jac_source - jac_src).max()
    assert kp_err  < 0.01,  f'KP quantisation error too large: {kp_err}'
    assert jac_err < 0.02,  f'Jacobian quant error too large: {jac_err}'
    ok(f'Pack/unpack OK  kp_err={kp_err:.5f}  jac_err={jac_err:.5f}')

    # NAL unit wrapping round-trip
    sei_nal     = NeuralARFSEI.wrap_sei_nal(packed)
    raw_payload = NeuralARFSEI.unwrap_sei_nal(sei_nal)
    assert raw_payload is not None
    unpacked2 = NeuralARFSEI.unpack(raw_payload)
    assert unpacked2 is not None and unpacked2.poc == 42
    ok(f'NAL wrap/unwrap OK  size={len(sei_nal)} bytes')


def test_encoder():
    section('3. NeuralARFEncoder (simulate=True)')
    from hevc_integration.neural_arf_encoder import NeuralARFEncoder, EncoderConfig

    model  = FakeModel(K)
    ref    = _ref_img()
    cfg    = EncoderConfig(width=W, height=H, fps=25, gop_size=10,
                           arf_period=1, simulate=True)

    enc = NeuralARFEncoder(model, ref, cfg, device='cpu')

    for i in range(3):
        frame = _rand_frame()
        result = enc.encode_frame(frame)
        assert result.poc == i
        # First frame must produce an SEI
        if i == 0:
            assert result.is_intra
            assert result.sei_nal is not None, 'I-frame must carry Neural ARF SEI'
        ok(f'  Frame {i}: poc={result.poc} intra={result.is_intra}'
           f' sei={"yes" if result.sei_nal else "no"}'
           f' bits={result.bits}')


def test_decoder():
    section('4. NeuralARFDecoder (simulate=True)')
    from hevc_integration.neural_arf_encoder import NeuralARFEncoder, EncoderConfig
    from hevc_integration.neural_arf_decoder import NeuralARFDecoder, DecoderConfig

    model  = FakeModel(K)
    ref    = _ref_img()
    enc_cfg = EncoderConfig(width=W, height=H, fps=25, simulate=True, arf_period=1)
    dec_cfg = DecoderConfig(width=W, height=H, simulate=True)

    enc = NeuralARFEncoder(model, ref, enc_cfg, device='cpu')
    dec = NeuralARFDecoder(model, ref, dec_cfg)

    frame = _rand_frame()
    enc_result = enc.encode_frame(frame)
    dec_result = dec.decode_access_unit(enc_result.nal_units,
                                         sei_nal=enc_result.sei_nal)

    assert dec_result.poc == 0
    assert dec_result.pixels.shape == (H, W, 3)
    assert dec_result.neural_arf_used   # SEI was present and decoded
    ok(f'Decoded POC={dec_result.poc}  ARF_used={dec_result.neural_arf_used}  '
       f'pixels_shape={dec_result.pixels.shape}')


def test_full_pipeline():
    section('5. Full Pipeline Loopback (encode → mux → demux → decode)')
    from hevc_integration.pipeline import NeuralARFPipeline, PipelineConfig

    model  = FakeModel(K)
    ref    = _ref_img()
    cfg    = PipelineConfig(width=W, height=H, fps=25, simulate=True, device='cpu')
    pipe   = NeuralARFPipeline(model, ref, cfg)

    results = []
    for i in range(5):
        frame = _rand_frame()
        enc, dec = pipe.process_frame(frame)
        results.append((enc, dec))
        ok(f'  Frame {i}: enc_bits={enc.bits} '
           f'dec_shape={dec.pixels.shape} '
           f'arf_used={dec.neural_arf_used}')

    report = pipe.report()
    assert 'PSNR' in report
    assert 'ARF injections' in report
    print(report)


def test_mux_container():
    section('6. Mux Container Multi-Frame Round-Trip')
    from hevc_integration.neural_arf_encoder import EncoderConfig, EncodedFrame
    from hevc_integration.pipeline import mux_frame, demux_stream

    # Create 10 fake encoded frames
    frames = [
        EncodedFrame(
            poc=i,
            nal_units=bytes([0x00, 0x00, 0x00, 0x01, 0x46 + i, 0x01]),
            sei_nal=b'\x00\x00\x00\x01\x4E\x01\x05\x1C' + bytes(28) if i % 3 == 0 else None,
            is_intra=(i == 0),
            bits=(i+1) * 1000,
        )
        for i in range(10)
    ]

    # Mux all frames into a single buffer
    buf = b''.join(mux_frame(f) for f in frames)
    ok(f'Muxed buffer size: {len(buf)} bytes')

    # Demux and verify
    recovered = list(demux_stream(buf))
    assert len(recovered) == 10
    for i, (poc, hevc_bytes, sei_bytes, is_intra) in enumerate(recovered):
        assert poc == i
        assert is_intra == (i == 0)
        has_sei = sei_bytes is not None
        expected_sei = (i % 3 == 0)
        assert has_sei == expected_sei, \
            f'Frame {i}: expected sei={expected_sei}, got {has_sei}'
    ok('All 10 frames demuxed with correct POC, SEI flags, and intra flags')


def test_dpb_rpl_priority():
    section('7. RPL Priority — Neural ARF always first in L0')
    from hevc_integration.dpb import DecodedPictureBuffer, NEURAL_ARF_POC

    dpb = DecodedPictureBuffer(max_size=8, inject_neural=True)
    dpb.update_neural_arf(np.zeros((H, W, 3), dtype=np.uint8))

    # Insert 6 past frames
    for poc in range(0, 12, 2):
        dpb.insert(poc, np.zeros((H, W, 3), dtype=np.uint8))

    # Current POC = 12
    rpl = dpb.build_rpl(current_poc=12, max_l0=5, max_l1=2)

    assert len(rpl.l0) > 0
    assert rpl.l0[0].poc == NEURAL_ARF_POC, \
        f'Expected neural ARF at L0[0], got POC={rpl.l0[0].poc}'
    assert rpl.neural_arf_l0_index() == 0

    ok(f'L0={[e.poc for e in rpl.l0]}  — neural ARF correctly at index 0')
    ok(f'L1={[e.poc for e in rpl.l1]}')


# ── Runner ────────────────────────────────────────────────────────────────

TESTS = [
    test_dpb,
    test_sei_roundtrip,
    test_encoder,
    test_decoder,
    test_full_pipeline,
    test_mux_container,
    test_dpb_rpl_priority,
]

if __name__ == '__main__':
    import sys
    sys.path.insert(0, __file__.replace('hevc_integration/test_arf_pipeline.py', ''))

    print('=' * 60)
    print('  Neural ARF Pipeline — Integration Smoke Tests')
    print(f'  Frames: {B}×3×{H}×{W}   Keypoints: K={K}')
    print('=' * 60)

    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            passed += 1
        except Exception as e:
            fail(fn.__name__, e)
            failed += 1

    print('\n' + '=' * 60)
    print(f'  Results: {passed}/{len(TESTS)} passed, {failed} failed')
    print('=' * 60)
    sys.exit(0 if failed == 0 else 1)
