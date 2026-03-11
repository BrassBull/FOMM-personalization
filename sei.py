"""
hevc_integration/sei.py
=======================
SEI NAL Unit — Neural ARF Side-Channel
---------------------------------------
Carries keypoint data from encoder to decoder inside HEVC Supplemental
Enhancement Information (SEI) messages, specifically payload type 5:
``user_data_unregistered`` (H.265 Annex D, Section D.2.7).

Wire format (all integers are big-endian)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Offset  Size   Field
  ──────  ────   ─────────────────────────────────────────────────────
       0    16   UUID  (see NEURAL_ARF_UUID below)
      16     1   version  = 1
      17     1   num_keypoints  K  (uint8, max 255)
      18     1   frame_poc_low  (POC & 0xFF)
      19     1   frame_poc_high (POC >> 8)
      20   K*2   source keypoints  [K × (x_q, y_q)]  uint8 each
   20+K*2  K*2   driving keypoints [K × (x_q, y_q)]  uint8 each
  20+2*K*2  K*4  source jacobians  [K × (j00,j01,j10,j11)] int8 each
  ...+K*4  K*4   driving jacobians [K × (j00,j01,j10,j11)] int8 each

Keypoint coordinates are quantised from float32 ∈ [-1, 1] to uint8 ∈ [0, 255].
Jacobian elements are quantised from float32 ∈ [-2, 2] to int8 ∈ [-128, 127].

Total payload size (excluding UUID + version): 4 + 4K + 4K + 4K + 4K = 4 + 16K bytes
For K=15: 4 + 240 = 244 bytes per frame — negligible overhead.
"""

from __future__ import annotations

import struct
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

# ── Registered UUID for Neural ARF SEI payload ───────────────────────────
# Generated once; both encoder and decoder must use this exact value.
NEURAL_ARF_UUID: bytes = uuid.UUID('a3f2c1d0-8e4b-4a7f-9c6e-1b2d3e4f5a6b').bytes
assert len(NEURAL_ARF_UUID) == 16

_SEI_VERSION: int = 1

# ── Quantisation helpers ──────────────────────────────────────────────────

def _quant_kp(x: float) -> int:
    """float ∈ [-1,1] → uint8 ∈ [0,255]"""
    return int(np.clip((x + 1.0) / 2.0 * 255.0 + 0.5, 0, 255))


def _dequant_kp(q: int) -> float:
    """uint8 ∈ [0,255] → float ∈ [-1,1]"""
    return q / 255.0 * 2.0 - 1.0


def _quant_jac(x: float) -> int:
    """float ∈ [-2,2] → int8 ∈ [-128,127]"""
    return int(np.clip(x / 2.0 * 127.0, -128, 127))


def _dequant_jac(q: int) -> float:
    """int8 ∈ [-128,127] → float ∈ [-2,2]"""
    return q / 127.0 * 2.0


# ---------------------------------------------------------------------------
# Packed keypoint payload
# ---------------------------------------------------------------------------

@dataclass
class NeuralARFPayload:
    """
    Decoded contents of a Neural ARF SEI message.

    kp_source  : (K, 2)   float32 — source keypoint coordinates in [-1,1]
    kp_driving : (K, 2)   float32 — driving keypoint coordinates in [-1,1]
    jac_source : (K, 2,2) float32 — source jacobian matrices
    jac_driving: (K, 2,2) float32 — driving jacobian matrices
    poc        : int       — picture order count of the driving frame
    """
    kp_source:   np.ndarray   # (K, 2)  float32
    kp_driving:  np.ndarray   # (K, 2)  float32
    jac_source:  np.ndarray   # (K, 2, 2) float32
    jac_driving: np.ndarray   # (K, 2, 2) float32
    poc:         int = 0

    @classmethod
    def from_model_output(cls,
                          kp_source_dict:  dict,
                          kp_driving_dict: dict,
                          poc: int = 0,
                          batch_idx: int = 0) -> "NeuralARFPayload":
        """Build from the dicts returned by KeypointDetector.forward()."""
        def _to_np(t: torch.Tensor) -> np.ndarray:
            return t[batch_idx].detach().cpu().float().numpy()

        return cls(
            kp_source   = _to_np(kp_source_dict['keypoints']),   # (K,2)
            kp_driving  = _to_np(kp_driving_dict['keypoints']),  # (K,2)
            jac_source  = _to_np(kp_source_dict['jacobians']),   # (K,2,2)
            jac_driving = _to_np(kp_driving_dict['jacobians']),  # (K,2,2)
            poc         = poc,
        )

    def to_torch(self, device: str = 'cpu') -> Tuple[dict, dict]:
        """Reconstruct kp_source and kp_driving dicts (batch size 1)."""
        def _wrap(arr):
            return torch.from_numpy(arr).unsqueeze(0).to(device)

        kp_src = {
            'keypoints': _wrap(self.kp_source),    # 1×K×2
            'jacobians': _wrap(self.jac_source),   # 1×K×2×2
        }
        kp_drv = {
            'keypoints': _wrap(self.kp_driving),
            'jacobians': _wrap(self.jac_driving),
        }
        return kp_src, kp_drv


# ---------------------------------------------------------------------------
# SEI serialisation / deserialisation
# ---------------------------------------------------------------------------

class NeuralARFSEI:
    """
    Encodes and decodes Neural ARF keypoint data into/from SEI payloads.
    """

    # ── Pack ─────────────────────────────────────────────────────────────

    @staticmethod
    def pack(payload: NeuralARFPayload) -> bytes:
        """
        Serialise a NeuralARFPayload into raw bytes suitable for embedding
        as a user_data_unregistered SEI message body.
        """
        K = payload.kp_source.shape[0]
        assert payload.kp_source.shape  == (K, 2)
        assert payload.kp_driving.shape == (K, 2)
        assert payload.jac_source.shape  == (K, 2, 2)
        assert payload.jac_driving.shape == (K, 2, 2)

        buf = bytearray()
        buf += NEURAL_ARF_UUID                        # 16 bytes
        buf += struct.pack('B', _SEI_VERSION)         #  1 byte
        buf += struct.pack('B', K)                    #  1 byte (num_kp)
        buf += struct.pack('>H', payload.poc & 0xFFFF)#  2 bytes (poc)

        # Keypoints: K × 2 uint8  (source then driving)
        for kp in payload.kp_source:
            buf += bytes([_quant_kp(kp[0]), _quant_kp(kp[1])])
        for kp in payload.kp_driving:
            buf += bytes([_quant_kp(kp[0]), _quant_kp(kp[1])])

        # Jacobians: K × 4 int8  (source then driving)
        for jac in payload.jac_source.reshape(K, 4):
            buf += struct.pack('4b', *[_quant_jac(v) for v in jac])
        for jac in payload.jac_driving.reshape(K, 4):
            buf += struct.pack('4b', *[_quant_jac(v) for v in jac])

        return bytes(buf)

    # ── Unpack ───────────────────────────────────────────────────────────

    @staticmethod
    def unpack(data: bytes) -> Optional[NeuralARFPayload]:
        """
        Deserialise bytes from a SEI message body.
        Returns None if the UUID does not match (not our payload).
        """
        if len(data) < 20:
            return None
        if data[:16] != NEURAL_ARF_UUID:
            return None

        offset = 16
        version, K = struct.unpack_from('BB', data, offset); offset += 2
        if version != _SEI_VERSION:
            raise ValueError(f'Unsupported Neural ARF SEI version {version}')
        poc, = struct.unpack_from('>H', data, offset);       offset += 2

        # Keypoints
        kp_src  = np.zeros((K, 2), dtype=np.float32)
        kp_drv  = np.zeros((K, 2), dtype=np.float32)
        for i in range(K):
            kp_src[i, 0]  = _dequant_kp(data[offset]);     offset += 1
            kp_src[i, 1]  = _dequant_kp(data[offset]);     offset += 1
        for i in range(K):
            kp_drv[i, 0]  = _dequant_kp(data[offset]);     offset += 1
            kp_drv[i, 1]  = _dequant_kp(data[offset]);     offset += 1

        # Jacobians (signed bytes)
        jac_src  = np.zeros((K, 2, 2), dtype=np.float32)
        jac_drv  = np.zeros((K, 2, 2), dtype=np.float32)
        for i in range(K):
            raw = struct.unpack_from('4b', data, offset);   offset += 4
            jac_src[i] = np.array([_dequant_jac(v) for v in raw],
                                   dtype=np.float32).reshape(2, 2)
        for i in range(K):
            raw = struct.unpack_from('4b', data, offset);   offset += 4
            jac_drv[i] = np.array([_dequant_jac(v) for v in raw],
                                   dtype=np.float32).reshape(2, 2)

        return NeuralARFPayload(
            kp_source   = kp_src,
            kp_driving  = kp_drv,
            jac_source  = jac_src,
            jac_driving = jac_drv,
            poc         = poc,
        )

    # ── HEVC Annex B NAL unit wrapping ────────────────────────────────────

    @staticmethod
    def wrap_sei_nal(payload_bytes: bytes) -> bytes:
        """
        Wrap raw SEI payload bytes in a minimal HEVC PREFIX_SEI NAL unit
        with Annex B start code.

        HEVC NAL unit structure (simplified):
          [start_code] [nal_unit_header] [sei_message]
          start_code       = 0x00 0x00 0x00 0x01  (4 bytes)
          nal_unit_header  = nal_unit_type(39)<<9 | nuh_layer_id(0)<<3 | nuh_temporal_id(1)
                           = 0x4E 0x01             (2 bytes, PREFIX_SEI)
          sei_message      = payloadType(5) payloadSize payload …
        """
        sei_type = 5                       # user_data_unregistered
        size     = len(payload_bytes)

        # Encode payload size in exp-Golomb-style run (each byte 0xFF = +255)
        size_bytes = bytearray()
        remaining  = size
        while remaining >= 255:
            size_bytes.append(0xFF)
            remaining -= 255
        size_bytes.append(remaining)

        sei_msg = bytes([sei_type]) + bytes(size_bytes) + payload_bytes

        # NAL unit header for PREFIX_SEI (nal_unit_type=39=0x27)
        # header = (39 << 9) | (0 << 3) | 1 = 0x4E01
        nal_header = bytes([0x4E, 0x01])

        # Emulation prevention (insert 0x03 before 0x00 0x00 0x00/01/02/03)
        nal_body = _emulation_prevention(nal_header + sei_msg)

        return b'\x00\x00\x00\x01' + nal_body

    @staticmethod
    def unwrap_sei_nal(nal_bytes: bytes) -> Optional[bytes]:
        """
        Extract the SEI payload bytes from an Annex B NAL unit.
        Returns None if start code / NAL type does not match.
        """
        if len(nal_bytes) < 8:
            return None

        # Strip start code
        if nal_bytes[:4] == b'\x00\x00\x00\x01':
            raw = _remove_emulation_prevention(nal_bytes[4:])
        elif nal_bytes[:3] == b'\x00\x00\x01':
            raw = _remove_emulation_prevention(nal_bytes[3:])
        else:
            return None

        # NAL header
        nal_type = (raw[0] >> 1) & 0x3F
        if nal_type != 39:   # PREFIX_SEI
            return None

        # Parse SEI messages
        offset = 2   # skip 2-byte NAL header
        while offset < len(raw):
            sei_type = raw[offset]; offset += 1
            size = 0
            while offset < len(raw):
                b = raw[offset]; offset += 1
                size += b
                if b != 0xFF:
                    break
            if sei_type == 5 and size > 0:
                return raw[offset: offset + size]
            offset += size
        return None


# ---------------------------------------------------------------------------
# Emulation prevention helpers
# ---------------------------------------------------------------------------

def _emulation_prevention(data: bytes) -> bytes:
    """Insert 0x03 before 0x000000, 0x000001, 0x000002, 0x000003."""
    out = bytearray()
    i, n = 0, len(data)
    zeros = 0
    for b in data:
        if zeros >= 2 and b <= 3:
            out.append(0x03)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def _remove_emulation_prevention(data: bytes) -> bytes:
    """Remove emulation prevention bytes (0x000003 → 0x0000)."""
    out = bytearray()
    i = 0
    while i < len(data):
        if (i + 2 < len(data)
                and data[i] == 0x00
                and data[i+1] == 0x00
                and data[i+2] == 0x03):
            out += b'\x00\x00'
            i += 3
        else:
            out.append(data[i])
            i += 1
    return bytes(out)
