"""
hevc_integration/neural_arf_decoder.py
========================================
Neural ARF Decoder
-------------------
Mirrors the encoder: for every incoming access unit it

  1. Scans the NAL unit list for the Neural ARF PREFIX_SEI.
  2. Deserialises the keypoint payload.
  3. Regenerates the neural frame using the **same** model and keypoints
     (both sides must have identical weights → deterministic reconstruction).
  4. Injects the regenerated frame into the DPB at NEURAL_ARF_POC as a LTRP.
  5. Passes the modified DPB to the OpenHEVC decoder so that
     inter-prediction motion vectors referencing NEURAL_ARF_POC resolve
     correctly.

Python simulation mode
~~~~~~~~~~~~~~~~~~~~~~
Without the patched OpenHEVC binary the decoded frame is produced by
ffmpeg / libx265 decode.  The neural ARF influence is visible as the
quality improvement coming from the fused output vs. the HEVC residual.

Full integration
~~~~~~~~~~~~~~~~
With the C patch applied (see openhevc_patch/hevc_refs_patch.c) the
decoder calls back into Python via the ``neural_arf_inject_cb`` function
pointer before constructing the slice Reference Picture Lists.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hevc_integration.dpb import DecodedPictureBuffer, NEURAL_ARF_POC, RefUsage
from hevc_integration.sei import NeuralARFSEI, NeuralARFPayload

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decoder configuration
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    width:       int  = 256
    height:      int  = 256
    ffmpeg_bin:  str  = 'ffmpeg'
    openhevc_bin: str = 'openHevcDecoder'   # patched binary
    simulate:    bool = True
    device:      str  = 'cpu'


# ---------------------------------------------------------------------------
# Per-frame decode result
# ---------------------------------------------------------------------------

@dataclass
class DecodedFrame:
    poc:            int
    pixels:         np.ndarray          # (H, W, 3) uint8
    neural_arf_used: bool = False       # True if ARF was in L0 this frame
    sei_payload:    Optional[NeuralARFPayload] = None


# ---------------------------------------------------------------------------
# NAL unit scanner
# ---------------------------------------------------------------------------

def split_annex_b(bitstream: bytes) -> List[bytes]:
    """
    Split an Annex B byte stream into individual NAL units (without start codes).
    Handles both 3-byte (0x000001) and 4-byte (0x00000001) start codes.
    """
    nals: List[bytes] = []
    i    = 0
    n    = len(bitstream)
    start = -1

    while i < n - 3:
        if bitstream[i:i+4] == b'\x00\x00\x00\x01':
            if start >= 0:
                nals.append(bitstream[start:i])
            start = i + 4
            i    += 4
        elif bitstream[i:i+3] == b'\x00\x00\x01':
            if start >= 0:
                nals.append(bitstream[start:i])
            start = i + 3
            i    += 3
        else:
            i += 1

    if start >= 0:
        nals.append(bitstream[start:])
    return nals


def find_neural_sei(nal_units: List[bytes]) -> Optional[NeuralARFPayload]:
    """
    Scan a list of raw NAL bodies (without start code) for our SEI.
    Returns the first matching NeuralARFPayload or None.
    """
    for nal in nal_units:
        if len(nal) < 2:
            continue
        nal_type = (nal[0] >> 1) & 0x3F
        # PREFIX_SEI = 39, SUFFIX_SEI = 40
        if nal_type not in (39, 40):
            continue
        # Re-attach a start code so unwrap_sei_nal can parse it
        candidate = b'\x00\x00\x00\x01' + nal
        raw_payload = NeuralARFSEI.unwrap_sei_nal(candidate)
        if raw_payload is not None:
            payload = NeuralARFSEI.unpack(raw_payload)
            if payload is not None:
                return payload
    return None


# ---------------------------------------------------------------------------
# Main decoder class
# ---------------------------------------------------------------------------

class NeuralARFDecoder:
    """
    Frame-by-frame HEVC decoder with neural ARF injection.

    Parameters
    ----------
    model         : same HybridTalkingHeadModel used during encoding
    reference_img : (1, 3, H, W) float32 — the canonical reference face
                    (must be identical to the encoder's reference)
    cfg           : DecoderConfig
    """

    def __init__(self,
                 model,
                 reference_img: torch.Tensor,
                 cfg:           DecoderConfig):
        self.model         = model
        self.reference_img = reference_img
        self.cfg           = cfg
        self.device        = torch.device(cfg.device)

        # Pre-compute source keypoints (identical to encoder)
        self.model.eval()
        with torch.no_grad():
            self.kp_source = self.model.keypoint_detector(
                reference_img.to(self.device)
            )

        # DPB with persistent neural ARF slot
        self.dpb = DecodedPictureBuffer(
            max_size=8,
            inject_neural=True,
        )

        self._poc         = 0
        self._hidden      = None
        self._frames_seen = 0

        log.info(f'NeuralARFDecoder initialised (simulate={cfg.simulate})')

    # ── Neural ARF regeneration ────────────────────────────────────────────

    @torch.no_grad()
    def _regenerate_neural_arf(self,
                                payload: NeuralARFPayload
                                ) -> np.ndarray:
        """
        Reconstruct the neural ARF from keypoints.

        The decoder does NOT have the driving video frame — it only has the
        quantised keypoints transmitted in the SEI.  We reconstruct the
        generator output using those keypoints directly, bypassing the
        keypoint detector.
        """
        kp_src, kp_drv = payload.to_torch(device=str(self.device))

        # Merge source keypoints from SEI with the pre-computed heatmaps
        # (heatmaps not transmitted; dense motion network uses kp values only)
        # Add dummy heatmap tensors that the dense-motion code can accept.
        H = W = self.cfg.height
        K = kp_src['keypoints'].shape[1]
        dummy_hm = torch.zeros(1, K, H, W, device=self.device)
        kp_src['heatmaps']  = dummy_hm
        kp_drv['heatmaps']  = dummy_hm

        # Run dense motion network with transmitted keypoints
        motion_out = self.model.dense_motion_network(
            self.reference_img.to(self.device),
            kp_src,
            kp_drv,
            hidden_states=self._hidden,
        )
        self._hidden = [h.detach() for h in motion_out['hidden_state']]

        deformation = motion_out['deformation']
        occlusion   = motion_out['occlusion']

        # Semantic generation
        gen_out = self.model.semantic_generator(
            self.reference_img.to(self.device),
            deformation,
            occlusion,
        )
        semantic = gen_out['semantic_frame']   # (1, 3, H, W)

        pixels_f32 = semantic[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
        return (pixels_f32.clip(0, 1) * 255).astype(np.uint8)

    # ── Frame decoding ────────────────────────────────────────────────────

    def decode_access_unit(self,
                           bitstream:  bytes,
                           sei_nal:    Optional[bytes] = None
                           ) -> DecodedFrame:
        """
        Decode one HEVC access unit.

        Parameters
        ----------
        bitstream : Annex B byte stream for one access unit
        sei_nal   : optional pre-extracted Neural ARF SEI NAL bytes.
                    If None, the SEI is scanned from ``bitstream``.

        Returns
        -------
        DecodedFrame with decoded pixels and metadata.
        """
        poc = self._poc

        # ── 1. Find Neural ARF SEI ────────────────────────────────────────
        if sei_nal is not None:
            raw_pl = NeuralARFSEI.unwrap_sei_nal(sei_nal)
            payload = NeuralARFSEI.unpack(raw_pl) if raw_pl else None
        else:
            nals    = split_annex_b(bitstream)
            payload = find_neural_sei(nals)

        neural_arf_used = False

        # ── 2. Regenerate and inject neural ARF ───────────────────────────
        if payload is not None:
            arf_pixels = self._regenerate_neural_arf(payload)
            self.dpb.update_neural_arf(arf_pixels)
            neural_arf_used = True
            log.debug(f'POC={poc}: neural ARF injected from SEI')

        # ── 3. Build RPL — decoder mirrors encoder RPL ────────────────────
        rpl = self.dpb.build_rpl(poc)
        arf_idx = rpl.neural_arf_l0_index()
        log.debug(
            f'POC={poc}: {rpl.summary()} '
            f'(ARF@L0[{arf_idx}])'
        )

        # ── 4. Decode HEVC frame ──────────────────────────────────────────
        if self.cfg.simulate:
            pixels = self._simulate_decode(bitstream, poc, rpl)
        else:
            pixels = self._openhevc_decode(bitstream, poc, rpl)

        # ── 5. Update DPB with decoded frame ──────────────────────────────
        self.dpb.insert(poc, pixels)

        self._poc         += 1
        self._frames_seen += 1

        return DecodedFrame(
            poc             = poc,
            pixels          = pixels,
            neural_arf_used = neural_arf_used,
            sei_payload     = payload,
        )

    # ── Simulation decode ─────────────────────────────────────────────────

    def _simulate_decode(self,
                         bitstream: bytes,
                         poc:       int,
                         rpl) -> np.ndarray:
        """
        Decode via ffmpeg, then blend with neural ARF to simulate
        the quality improvement from inter-prediction.
        """
        raw = _ffmpeg_decode_frame(bitstream, self.cfg)

        arf = self.dpb.neural_arf_pixels
        if arf is not None and raw is not None:
            # Simulate the decoder applying motion-compensated residual
            # on top of the neural ARF (small residual → sharp result)
            w = 0.25   # residual weight (encoder QP residual fraction)
            blended = ((1 - w) * arf.astype(np.float32)
                       + w * raw.astype(np.float32)).clip(0, 255).astype(np.uint8)
            return blended

        return raw if raw is not None else np.zeros(
            (self.cfg.height, self.cfg.width, 3), dtype=np.uint8
        )

    # ── Real OpenHEVC decode (requires patched binary) ────────────────────

    def _openhevc_decode(self,
                         bitstream: bytes,
                         poc:       int,
                         rpl) -> np.ndarray:
        """
        Feed the bitstream to the patched openHevcDecoder binary.

        The patched binary accepts an extra environment variable:
          NEURAL_ARF_YUV_PATH — path to YUV file with the neural ARF frame.
          NEURAL_ARF_POC      — POC integer to assign to that frame.

        The C patch (hevc_refs_patch.c) reads these vars at startup and
        injects the frame before executing ff_hevc_frame_rps().
        """
        with tempfile.TemporaryDirectory() as tmp:
            bs_path  = os.path.join(tmp, 'input.265')
            yuv_path = os.path.join(tmp, 'arf.yuv')
            out_path = os.path.join(tmp, 'out.yuv')

            with open(bs_path, 'wb') as f:
                f.write(bitstream)

            arf = self.dpb.neural_arf_pixels
            if arf is not None:
                _write_yuv_plane(arf, yuv_path, self.cfg.width, self.cfg.height)

            env = os.environ.copy()
            env['NEURAL_ARF_YUV_PATH'] = yuv_path
            env['NEURAL_ARF_POC']      = str(NEURAL_ARF_POC)

            cmd = [
                self.cfg.openhevc_bin,
                '-b', bs_path,
                '-o', out_path,
                '-f', '1',
            ]
            result = subprocess.run(cmd, capture_output=True, env=env,
                                    timeout=30)
            if result.returncode != 0:
                log.error(f'openHevcDecoder failed: {result.stderr.decode()}')
                return np.zeros((self.cfg.height, self.cfg.width, 3),
                                dtype=np.uint8)

            return _read_yuv420_rgb(out_path, self.cfg.width, self.cfg.height)


# ---------------------------------------------------------------------------
# FFmpeg decode helper
# ---------------------------------------------------------------------------

def _ffmpeg_decode_frame(bitstream: bytes, cfg: DecoderConfig
                          ) -> Optional[np.ndarray]:
    cmd = [
        cfg.ffmpeg_bin,
        '-f',    'hevc',
        '-i',    'pipe:0',
        '-f',    'rawvideo',
        '-pix_fmt', 'rgb24',
        '-frames:v', '1',
        'pipe:1',
    ]
    try:
        r = subprocess.run(cmd, input=bitstream, capture_output=True, timeout=10)
        if r.returncode == 0 and r.stdout:
            arr = np.frombuffer(r.stdout, dtype=np.uint8)
            if arr.size == cfg.height * cfg.width * 3:
                return arr.reshape(cfg.height, cfg.width, 3)
    except Exception as e:
        log.warning(f'ffmpeg decode failed: {e}')
    return None


def _write_yuv_plane(rgb: np.ndarray, path: str, W: int, H: int):
    try:
        from PIL import Image
        img = Image.fromarray(rgb, 'RGB').convert('YCbCr')
        y, cb, cr = img.split()
        cb = cb.resize((W // 2, H // 2))
        cr = cr.resize((W // 2, H // 2))
        with open(path, 'wb') as f:
            f.write(np.array(y).tobytes())
            f.write(np.array(cb).tobytes())
            f.write(np.array(cr).tobytes())
    except ImportError:
        with open(path, 'wb') as f:
            f.write(rgb.mean(axis=2).astype(np.uint8).tobytes())


def _read_yuv420_rgb(path: str, W: int, H: int) -> np.ndarray:
    size = W * H + W * H // 2
    try:
        with open(path, 'rb') as f:
            data = f.read(size)
        y  = np.frombuffer(data[:W*H],             dtype=np.uint8).reshape(H, W)
        cb = np.frombuffer(data[W*H:W*H*5//4],     dtype=np.uint8).reshape(H//2, W//2)
        cr = np.frombuffer(data[W*H*5//4:W*H*3//2],dtype=np.uint8).reshape(H//2, W//2)
        from PIL import Image
        yimg  = Image.fromarray(y,  'L')
        cbimg = Image.fromarray(cb, 'L').resize((W, H), Image.BILINEAR)
        crimg = Image.fromarray(cr, 'L').resize((W, H), Image.BILINEAR)
        rgb = Image.merge('YCbCr', [yimg, cbimg, crimg]).convert('RGB')
        return np.array(rgb)
    except Exception:
        return np.zeros((H, W, 3), dtype=np.uint8)
