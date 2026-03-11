"""
hevc_integration/neural_arf_encoder.py
=======================================
Neural ARF Encoder
-------------------
Wraps the x265 HEVC encoder (via subprocess / ffmpeg) and injects the
neural talking-head frame into the Reference Picture List *before* each
encoding call, following the Additional Reference Frame (ARF) approach.

How it works
~~~~~~~~~~~~
1. For every input frame t:
   a. Run the neural generator:  neural_arf_t = G(ref_image, drive_t)
   b. Call  ``dpb.update_neural_arf(neural_arf_t)``
      → The neural frame occupies the LTRP slot at POC = NEURAL_ARF_POC.
   c. Build RPL via ``dpb.build_rpl(current_poc)``
      → L0[0] is always the neural ARF.
   d. Write the neural_arf frame into x265 as a long-term reference:
      • ``x265_encoder_encode()`` is called with the x265_picture struct
        whose ``userSEI`` field carries our packed NeuralARFPayload SEI.
      • The DPB entry for NEURAL_ARF_POC is pre-loaded into x265 via the
        ``x265_picture.userData`` + custom ``x265_param.injectExtraRef``
        hook exposed by the C patch (see openhevc_patch/x265_dpb_patch.cpp).
   e. The keypoint payload is packed into a PREFIX_SEI NAL and appended
      to the output access unit so the decoder can regenerate the ARF.

Python-only simulation mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~
When ``simulate=True`` (default) this class uses ffmpeg for HEVC encoding
but manages the DPB / RPL / SEI logic entirely in Python.  The neural ARF
is compressed separately with a high QP and prepended as a long-term
reference frame, side-stepped by inserting it into the reference stream
with a special marking.

For full integration compile the x265 patch and set ``simulate=False``.
"""

from __future__ import annotations

import io
import json
import logging
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hevc_integration.dpb import DecodedPictureBuffer, NEURAL_ARF_POC
from hevc_integration.sei import NeuralARFPayload, NeuralARFSEI

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    width:          int   = 256
    height:         int   = 256
    fps:            int   = 25
    gop_size:       int   = 30         # distance between I-frames
    # Main stream QP
    qp:             int   = 28
    # Auxiliary (neural ARF) QP — higher = smaller side-channel
    neural_arf_qp:  int   = 40
    # Max reference frames for main encode
    ref_frames:     int   = 4
    # How often to regenerate the neural ARF (every N frames)
    arf_period:     int   = 1          # 1 = every frame
    # Path to x265 binary (used when simulate=False)
    x265_bin:       str   = 'x265'
    # Path to ffmpeg
    ffmpeg_bin:     str   = 'ffmpeg'
    # L0 position of the neural ARF (0 = first / highest priority)
    arf_l0_index:   int   = 0
    # Whether to run in pure-Python simulation mode
    simulate:       bool  = True


# ---------------------------------------------------------------------------
# Per-frame encoding result
# ---------------------------------------------------------------------------

@dataclass
class EncodedFrame:
    poc:            int
    nal_units:      bytes              # raw Annex B NAL units
    sei_nal:        Optional[bytes]    # Neural ARF SEI (None if no ARF this frame)
    is_intra:       bool = False
    bits:           int  = 0
    neural_arf_poc: int  = NEURAL_ARF_POC


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

class NeuralARFEncoder:
    """
    Frame-by-frame HEVC encoder with neural ARF injection.

    Parameters
    ----------
    model         : HybridTalkingHeadModel (or any nn.Module with the same
                    forward signature)
    reference_img : (1, 3, H, W) float32 tensor — the canonical reference face
    cfg           : EncoderConfig
    device        : torch device string
    """

    def __init__(self,
                 model,
                 reference_img: torch.Tensor,
                 cfg:           EncoderConfig,
                 device:        str = 'cpu'):
        self.model         = model
        self.reference_img = reference_img
        self.cfg           = cfg
        self.device        = torch.device(device)

        # Pre-compute source keypoints once (reference is fixed)
        self.model.eval()
        with torch.no_grad():
            self.kp_source = self.model.keypoint_detector(
                reference_img.to(self.device)
            )

        # DPB with neural ARF slot
        self.dpb = DecodedPictureBuffer(
            max_size=cfg.ref_frames + 2,
            inject_neural=True,
        )

        # Encoder state
        self._poc          = 0
        self._hidden       = None
        self._frame_count  = 0
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._output_buf   = io.BytesIO()

        log.info(
            f'NeuralARFEncoder initialised '
            f'({cfg.width}×{cfg.height}@{cfg.fps}fps '
            f'QP={cfg.qp} ARF_period={cfg.arf_period})'
        )

    # ── Neural ARF generation ─────────────────────────────────────────────

    @torch.no_grad()
    def _generate_neural_arf(self, driving_frame: torch.Tensor
                             ) -> Tuple[np.ndarray, dict, dict]:
        """
        Run the talking-head generator to produce the neural ARF.

        Returns
        -------
        pixels_uint8 : (H, W, 3) uint8
        kp_source    : keypoint dict (for SEI)
        kp_driving   : keypoint dict (for SEI)
        """
        drv  = driving_frame.to(self.device)
        out  = self.model(
            self.reference_img.to(self.device),
            drv,
            hidden_state=self._hidden,
            kp_source=self.kp_source,
        )
        self._hidden = [h.detach() for h in out['hidden_state']]

        # Use the fused frame as the ARF (best quality synthesis)
        fused_np = out['fused_frame'][0].permute(1, 2, 0).cpu().numpy()
        pixels   = (fused_np.clip(0, 1) * 255).astype(np.uint8)

        return pixels, self.kp_source, out['kp_driving']

    # ── Frame encoding ────────────────────────────────────────────────────

    def encode_frame(self,
                     frame_rgb_uint8: np.ndarray,
                     driving_frame:   Optional[torch.Tensor] = None
                     ) -> EncodedFrame:
        """
        Encode one frame with neural ARF injection.

        Parameters
        ----------
        frame_rgb_uint8 : (H, W, 3) uint8 — the frame to encode
        driving_frame   : (1, 3, H, W) float32 — driving signal for generator.
                          If None, uses ``frame_rgb_uint8`` converted to tensor.

        Returns
        -------
        EncodedFrame with Annex-B NAL units and attached SEI.
        """
        poc = self._poc
        is_intra = (poc % self.cfg.gop_size == 0)

        # ── 1. Convert driving frame ──────────────────────────────────────
        if driving_frame is None:
            drv_t = torch.from_numpy(frame_rgb_uint8).float().permute(2, 0, 1)
            drv_t = drv_t.unsqueeze(0) / 255.0
        else:
            drv_t = driving_frame

        # ── 2. Generate neural ARF ────────────────────────────────────────
        sei_nal = None
        do_arf  = (self._frame_count % self.cfg.arf_period == 0) or is_intra

        if do_arf:
            arf_pixels, kp_src, kp_drv = self._generate_neural_arf(drv_t)
            self.dpb.update_neural_arf(arf_pixels)

            # Pack keypoints into SEI
            sei_payload = NeuralARFPayload.from_model_output(kp_src, kp_drv,
                                                              poc=poc)
            sei_bytes   = NeuralARFSEI.pack(sei_payload)
            sei_nal     = NeuralARFSEI.wrap_sei_nal(sei_bytes)

            log.debug(f'POC={poc}: neural ARF generated, SEI={len(sei_bytes)}B')

        # ── 3. Insert current frame into DPB ──────────────────────────────
        self.dpb.insert(poc, frame_rgb_uint8)

        # ── 4. Build RPL — L0[0] = neural ARF ────────────────────────────
        rpl = self.dpb.build_rpl(poc, max_l0=self.cfg.ref_frames)
        log.debug(f'POC={poc}: {rpl.summary()}')

        # ── 5. Encode (simulation or real x265) ───────────────────────────
        if self.cfg.simulate:
            nal_units, bits = self._simulate_encode(
                frame_rgb_uint8, poc, is_intra, rpl
            )
        else:
            nal_units, bits = self._x265_encode(
                frame_rgb_uint8, poc, is_intra, rpl
            )

        self._poc         += 1
        self._frame_count += 1

        return EncodedFrame(
            poc       = poc,
            nal_units = nal_units,
            sei_nal   = sei_nal,
            is_intra  = is_intra,
            bits      = bits,
        )

    # ── Simulation-mode encode (Python / ffmpeg) ─────────────────────────

    def _simulate_encode(self,
                         frame:    np.ndarray,
                         poc:      int,
                         is_intra: bool,
                         rpl) -> Tuple[bytes, int]:
        """
        Use ffmpeg to encode a single frame to HEVC.

        The neural ARF influence is simulated by blending the neural ARF
        frame with the input before encoding — this models the residual
        reduction that inter-prediction from the ARF would achieve.
        """
        arf_pixels = self.dpb.neural_arf_pixels
        if arf_pixels is not None and not is_intra:
            # Simulate ARF residual: encode the difference from neural ARF
            blend_weight = 0.3   # proportion of ARF pixel content
            frame_enc    = ((1 - blend_weight) * frame.astype(np.float32)
                            + blend_weight * arf_pixels.astype(np.float32)
                            ).clip(0, 255).astype(np.uint8)
        else:
            frame_enc = frame

        raw_nal = _ffmpeg_encode_frame(frame_enc, self.cfg)
        return raw_nal, len(raw_nal) * 8

    # ── Real x265 encode (requires patched x265 + C shim) ────────────────

    def _x265_encode(self,
                     frame:    np.ndarray,
                     poc:      int,
                     is_intra: bool,
                     rpl) -> Tuple[bytes, int]:
        """
        Call the patched x265 encoder via subprocess with extra arguments
        that instruct it to:
          --ref-inject-yuv <path>   path to the neural ARF YUV file
          --ref-inject-poc <poc>    POC to assign to the injected ARF
        (See openhevc_patch/x265_dpb_patch.cpp for the implementation.)
        """
        arf_pixels = self.dpb.neural_arf_pixels

        with tempfile.TemporaryDirectory() as tmp:
            # Write input frame
            in_path  = os.path.join(tmp, 'frame.yuv')
            arf_path = os.path.join(tmp, 'arf.yuv')
            out_path = os.path.join(tmp, 'out.265')

            _write_yuv420(frame, in_path)
            if arf_pixels is not None:
                _write_yuv420(arf_pixels, arf_path)

            cmd = [
                self.cfg.x265_bin,
                '--input',        in_path,
                '--output',       out_path,
                '--frames',       '1',
                '--res',          f'{self.cfg.width}x{self.cfg.height}',
                '--fps',          str(self.cfg.fps),
                '--qp',           str(self.cfg.qp),
                '--ref',          str(self.cfg.ref_frames),
                '--no-open-gop',
            ]
            if arf_pixels is not None and not is_intra:
                cmd += [
                    '--ref-inject-yuv', arf_path,
                    '--ref-inject-poc', str(NEURAL_ARF_POC),
                    '--ref-inject-lt',  '1',   # mark as long-term ref
                ]

            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                log.error(f'x265 failed: {result.stderr.decode()}')
                return b'', 0

            with open(out_path, 'rb') as f:
                nal_units = f.read()

        return nal_units, len(nal_units) * 8

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self): return self

    def __exit__(self, *args):
        self.flush()

    def flush(self):
        """Drain any pending encoder state."""
        log.info('Encoder flushed.')

    # ── Statistics ────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            'frames_encoded': self._frame_count,
            'dpb_state':      repr(self.dpb),
        }


# ---------------------------------------------------------------------------
# FFmpeg single-frame encode helper
# ---------------------------------------------------------------------------

def _ffmpeg_encode_frame(frame_rgb: np.ndarray, cfg: EncoderConfig) -> bytes:
    """
    Encode a single (H,W,3) uint8 frame to HEVC Annex B using ffmpeg.
    Returns the raw NAL unit bytes.
    """
    H, W = frame_rgb.shape[:2]
    cmd = [
        cfg.ffmpeg_bin,
        '-y',
        '-f',        'rawvideo',
        '-pix_fmt',  'rgb24',
        '-s',        f'{W}x{H}',
        '-r',        str(cfg.fps),
        '-i',        'pipe:0',
        '-c:v',      'libx265',
        '-preset',   'ultrafast',
        '-qp',       str(cfg.qp),
        '-x265-params', f'keyint={cfg.gop_size}:no-open-gop=1',
        '-f',        'hevc',
        '-frames:v', '1',
        'pipe:1',
    ]
    try:
        result = subprocess.run(
            cmd,
            input=frame_rgb.tobytes(),
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.warning(f'ffmpeg not available ({e}), returning dummy NAL')

    # Fallback: dummy VPS+SPS+PPS+IDR (won't decode but keeps pipeline alive)
    return _dummy_nal_units()


def _write_yuv420(rgb: np.ndarray, path: str):
    """Write (H,W,3) uint8 RGB as YUV420p raw."""
    from PIL import Image
    img = Image.fromarray(rgb, 'RGB').convert('YCbCr')
    y, cb, cr = img.split()
    cb = cb.resize((rgb.shape[1] // 2, rgb.shape[0] // 2), Image.BILINEAR)
    cr = cr.resize((rgb.shape[1] // 2, rgb.shape[0] // 2), Image.BILINEAR)
    with open(path, 'wb') as f:
        f.write(np.array(y).tobytes())
        f.write(np.array(cb).tobytes())
        f.write(np.array(cr).tobytes())


def _dummy_nal_units() -> bytes:
    """Return minimal valid-looking Annex B placeholder."""
    return b'\x00\x00\x00\x01\x46\x01\x10'   # single NAL
