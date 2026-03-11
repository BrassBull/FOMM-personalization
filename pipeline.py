"""
hevc_integration/pipeline.py
=============================
End-to-End Neural ARF Pipeline
--------------------------------
Orchestrates the full encode → transmit → decode cycle.

                 ┌──────────────────────────────────────────────┐
    video_in ──► │  NeuralARFEncoder                            │
                 │  ┌────────────────────────────────────────┐  │
                 │  │ for each frame:                        │  │
                 │  │  1. G(ref, drive) → arf_t              │  │
                 │  │  2. DPB.update_neural_arf(arf_t)       │  │
                 │  │  3. Build RPL  L0=[arf_t, …]           │  │
                 │  │  4. x265.encode(frame, rpl)            │  │
                 │  │  5. Pack kp → PREFIX_SEI NAL           │  │
                 │  └────────────────────────────────────────┘  │
                 └──────────────────┬───────────────────────────┘
                                    │  Annex B + SEI NAL
                          ┌─────────▼──────────┐
                          │   Bitstream mux     │
                          │  [SEI|HEVC|SEI|…]  │
                          └─────────┬──────────┘
                                    │  (network / file)
                 ┌──────────────────▼───────────────────────────┐
                 │  NeuralARFDecoder                             │
                 │  ┌────────────────────────────────────────┐  │
                 │  │ for each access unit:                  │  │
                 │  │  1. Parse SEI → keypoints              │  │
                 │  │  2. G(ref, kp) → arf_t  (same model)  │  │
                 │  │  3. DPB.update_neural_arf(arf_t)       │  │
                 │  │  4. Build RPL  L0=[arf_t, …]           │  │
                 │  │  5. openHEVC.decode(au, dpb)           │  │
                 │  └────────────────────────────────────────┘  │
                 └──────────────────────────────────────────────┘
                                    │
                              reconstructed frames

Usage
-----
    from hevc_integration.pipeline import NeuralARFPipeline, PipelineConfig

    pipeline = NeuralARFPipeline(model, reference_frame, cfg=PipelineConfig())
    results  = pipeline.process_video('input.mp4', 'output.mp4')
    print(pipeline.report())
"""

from __future__ import annotations

import io
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch

from hevc_integration.neural_arf_encoder import (
    NeuralARFEncoder, EncoderConfig, EncodedFrame
)
from hevc_integration.neural_arf_decoder import (
    NeuralARFDecoder, DecoderConfig, DecodedFrame
)
from hevc_integration.dpb import NEURAL_ARF_POC

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    width:         int   = 256
    height:        int   = 256
    fps:           int   = 25
    qp:            int   = 28
    gop_size:      int   = 30
    ref_frames:    int   = 4
    arf_period:    int   = 1
    simulate:      bool  = True
    device:        str   = 'cpu'
    ffmpeg_bin:    str   = 'ffmpeg'
    x265_bin:      str   = 'x265'
    openhevc_bin:  str   = 'openHevcDecoder'


# ---------------------------------------------------------------------------
# Mux format
# ---------------------------------------------------------------------------
#
# The multiplexed bitstream uses a simple length-prefixed container:
#
#   [frame_record]*
#   frame_record  = MAGIC(4) | poc(4) | flags(1) | hevc_len(4) | hevc_bytes
#                            | sei_len(4) | sei_bytes
#
# flags bit 0 = has_sei
# flags bit 1 = is_intra
#
_FRAME_MAGIC = b'NARF'   # Neural ARF Frame


def mux_frame(enc_frame: EncodedFrame) -> bytes:
    has_sei = enc_frame.sei_nal is not None
    flags   = (1 if has_sei else 0) | (2 if enc_frame.is_intra else 0)
    sei_b   = enc_frame.sei_nal or b''
    record  = struct.pack('>4sIBII',
                          _FRAME_MAGIC,
                          enc_frame.poc,
                          flags,
                          len(enc_frame.nal_units),
                          len(sei_b))
    return record + enc_frame.nal_units + sei_b


def demux_stream(data: bytes) -> Generator[Tuple[int, bytes, Optional[bytes], bool], None, None]:
    """
    Yield (poc, hevc_bytes, sei_bytes_or_None, is_intra) for each record.
    """
    i = 0
    n = len(data)
    while i + 17 <= n:
        magic, poc, flags, hevc_len, sei_len = struct.unpack_from('>4sIBII', data, i)
        i += 17
        if magic != _FRAME_MAGIC:
            log.error(f'Bad frame magic at offset {i-17}: {magic}')
            break
        hevc_bytes = data[i: i + hevc_len]; i += hevc_len
        sei_bytes  = data[i: i + sei_len] if sei_len else None; i += sei_len
        has_sei    = bool(flags & 1)
        is_intra   = bool(flags & 2)
        yield poc, hevc_bytes, sei_bytes, is_intra


# ---------------------------------------------------------------------------
# Pipeline statistics
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    frames_encoded:      int   = 0
    frames_decoded:      int   = 0
    arf_injections:      int   = 0
    total_bits:          int   = 0
    sei_bits:            int   = 0
    psnr_sum:            float = 0.0
    encode_time_s:       float = 0.0
    decode_time_s:       float = 0.0

    def bpp(self, W: int, H: int) -> float:
        pixels = W * H * self.frames_encoded
        return self.total_bits / max(pixels, 1)

    def avg_psnr(self) -> float:
        return self.psnr_sum / max(self.frames_decoded, 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class NeuralARFPipeline:
    """
    Combines encoder and decoder into a single object for easy testing.

    Both sides share the same reference image and model weights, which is
    the videoconferencing scenario: both parties have the same personalised
    model after the session-initialisation phase.
    """

    def __init__(self,
                 model,
                 reference_img: torch.Tensor,
                 cfg:           PipelineConfig = None):
        if cfg is None:
            cfg = PipelineConfig()
        self.cfg = cfg
        self.model = model
        self.reference_img = reference_img

        enc_cfg = EncoderConfig(
            width=cfg.width, height=cfg.height, fps=cfg.fps,
            qp=cfg.qp, gop_size=cfg.gop_size, ref_frames=cfg.ref_frames,
            arf_period=cfg.arf_period, simulate=cfg.simulate,
            ffmpeg_bin=cfg.ffmpeg_bin, x265_bin=cfg.x265_bin,
        )
        dec_cfg = DecoderConfig(
            width=cfg.width, height=cfg.height,
            ffmpeg_bin=cfg.ffmpeg_bin,
            openhevc_bin=cfg.openhevc_bin,
            simulate=cfg.simulate, device=cfg.device,
        )

        self.encoder = NeuralARFEncoder(model, reference_img, enc_cfg,
                                         device=cfg.device)
        self.decoder = NeuralARFDecoder(model, reference_img, dec_cfg)
        self.stats   = PipelineStats()

    # ── Frame-level API ───────────────────────────────────────────────────

    def process_frame(self,
                      frame_rgb:     np.ndarray,
                      driving_frame: Optional[torch.Tensor] = None
                      ) -> Tuple[EncodedFrame, DecodedFrame]:
        """
        Encode then immediately decode one frame (loopback test).

        Returns both the EncodedFrame and the DecodedFrame.
        """
        # ---- Encode ----
        t0 = time.perf_counter()
        enc = self.encoder.encode_frame(frame_rgb, driving_frame)
        self.stats.encode_time_s += time.perf_counter() - t0

        self.stats.frames_encoded += 1
        self.stats.total_bits     += enc.bits
        if enc.sei_nal:
            self.stats.sei_bits   += len(enc.sei_nal) * 8
            self.stats.arf_injections += 1

        # ---- Mux ----
        muxed = mux_frame(enc)

        # ---- Demux (simulates network transit) ----
        records = list(demux_stream(muxed))
        assert len(records) == 1
        poc, hevc_bytes, sei_bytes, is_intra = records[0]

        # ---- Decode ----
        t1 = time.perf_counter()
        dec = self.decoder.decode_access_unit(hevc_bytes, sei_nal=sei_bytes)
        self.stats.decode_time_s += time.perf_counter() - t1

        self.stats.frames_decoded += 1

        # ---- PSNR ----
        psnr = _compute_psnr(frame_rgb, dec.pixels)
        self.stats.psnr_sum += psnr

        return enc, dec

    # ── Video-level API ───────────────────────────────────────────────────

    def process_frames(self,
                       frames: List[np.ndarray],
                       driving_frames: Optional[List[torch.Tensor]] = None
                       ) -> List[Tuple[EncodedFrame, DecodedFrame]]:
        """Process a list of (H,W,3) uint8 frames."""
        results = []
        for i, frame in enumerate(frames):
            drv = driving_frames[i] if driving_frames else None
            results.append(self.process_frame(frame, drv))
            if (i + 1) % 10 == 0:
                log.info(f'Processed {i+1}/{len(frames)} frames')
        return results

    def process_video(self,
                      input_path:  str,
                      output_path: str) -> List[DecodedFrame]:
        """
        Read a video file, process all frames, write decoded output.
        Requires ffmpeg.
        """
        frames      = _read_video_ffmpeg(input_path, self.cfg)
        dec_frames  = []

        mux_buf = io.BytesIO()
        for i, frame in enumerate(frames):
            enc, dec = self.process_frame(frame)
            dec_frames.append(dec)
            mux_buf.write(mux_frame(enc))

        _write_video_ffmpeg(
            [df.pixels for df in dec_frames],
            output_path,
            self.cfg,
        )
        log.info(f'Output written to {output_path}')
        return dec_frames

    # ── Reporting ─────────────────────────────────────────────────────────

    def report(self) -> str:
        s   = self.stats
        W   = self.cfg.width
        H   = self.cfg.height
        fps = self.cfg.fps

        lines = [
            '═' * 60,
            '  Neural ARF Pipeline — Session Report',
            '═' * 60,
            f'  Frames encoded          : {s.frames_encoded}',
            f'  Frames decoded          : {s.frames_decoded}',
            f'  ARF injections          : {s.arf_injections}',
            f'  Total bits              : {s.total_bits / 1e3:.1f} kbit',
            f'  SEI overhead            : {s.sei_bits / 1e3:.2f} kbit '
            f'({100*s.sei_bits/max(s.total_bits,1):.2f}%)',
            f'  Bitrate (simulated)     : '
            f'{s.total_bits/max(s.frames_encoded,1)*fps/1e3:.1f} kbps',
            f'  Bits/pixel              : {s.bpp(W, H):.4f}',
            f'  Average PSNR            : {s.avg_psnr():.2f} dB',
            f'  Encode time             : {s.encode_time_s:.3f}s',
            f'  Decode time             : {s.decode_time_s:.3f}s',
            f'  Neural ARF POC sentinel : {NEURAL_ARF_POC} (0x{NEURAL_ARF_POC:04X})',
            '═' * 60,
        ]
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _read_video_ffmpeg(path: str, cfg: PipelineConfig) -> List[np.ndarray]:
    """Read all frames from a video file via ffmpeg."""
    cmd = [
        cfg.ffmpeg_bin, '-i', path,
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-vf', f'scale={cfg.width}:{cfg.height}',
        'pipe:1',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        raw = r.stdout
        frame_size = cfg.width * cfg.height * 3
        frames = []
        for i in range(0, len(raw) - frame_size + 1, frame_size):
            frames.append(
                np.frombuffer(raw[i:i+frame_size], dtype=np.uint8
                              ).reshape(cfg.height, cfg.width, 3).copy()
            )
        return frames
    except Exception as e:
        log.error(f'Video read failed: {e}')
        return []


def _write_video_ffmpeg(frames: List[np.ndarray], path: str,
                        cfg: PipelineConfig):
    """Write decoded frames to a video file via ffmpeg."""
    cmd = [
        cfg.ffmpeg_bin, '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{cfg.width}x{cfg.height}',
        '-r', str(cfg.fps),
        '-i', 'pipe:0',
        '-c:v', 'libx264', '-preset', 'fast',
        path,
    ]
    try:
        raw = b''.join(f.tobytes() for f in frames)
        subprocess.run(cmd, input=raw, capture_output=True, timeout=120)
    except Exception as e:
        log.error(f'Video write failed: {e}')


def _compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255.0 ** 2 / mse)


import subprocess   # noqa: E402 (needed by helpers above)
