"""
hevc_integration/dpb.py
=======================
Decoded Picture Buffer (DPB) with Neural ARF Support
-----------------------------------------------------
Models the HEVC Decoded Picture Buffer as defined in:
  ISO/IEC 23008-2 (H.265), Section 8.3 — Decoded picture buffering

Standard HEVC DPB rules (relevant subset)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Capacity: max_dec_pic_buffering_minus1 + 1 pictures.
* Each frame in the DPB has a POC (Picture Order Count), a usage flag
  (short-term reference / long-term reference / unused for reference),
  and a pixel buffer.
* The Reference Picture Set (RPS) embedded in each slice header selects
  which DPB frames belong to L0/L1 reference lists for the current slice.

Neural ARF extension
~~~~~~~~~~~~~~~~~~~~
A synthetically generated frame (from the talking-head network) is injected
as a **Long-Term Reference Picture (LTRP)** with a reserved POC sentinel:

    NEURAL_ARF_POC = 0x7FFF   (positive, outside normal GOP range)

Both encoder and decoder must agree on:
  1. The POC sentinel value.
  2. The frame generation procedure (same model weights, same keypoints).
  3. The position in L0 at which the neural ARF appears (index 0 by default,
     so the encoder naturally tries it first).

This module provides:
  * ``DecodedPictureBuffer`` — full DPB simulation with LTRP management.
  * ``ReferencePictureList``  — builds L0/L1 lists from DPB state.
  * ``NeuralARFSlot``         — wrapper that marks a frame as a neural ARF.
"""

from __future__ import annotations

import enum
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Reserved POC for the neural additional reference frame ─────────────────
NEURAL_ARF_POC: int = 0x7FFF   # 32 767 — safely outside any realistic GOP


# ---------------------------------------------------------------------------
# Frame usage flags (match HEVC spec Table 8-1)
# ---------------------------------------------------------------------------

class RefUsage(enum.IntFlag):
    UNUSED          = 0
    SHORT_TERM_REF  = 1   # STRP
    LONG_TERM_REF   = 2   # LTRP  ← neural ARF uses this
    OUTPUT_PENDING  = 4   # frame needs to be output but not yet used as ref


# ---------------------------------------------------------------------------
# Single entry in the DPB
# ---------------------------------------------------------------------------

@dataclass
class DPBEntry:
    poc:        int
    pixels:     np.ndarray          # (H, W, 3) uint8 or (3, H, W) float32
    usage:      RefUsage = RefUsage.SHORT_TERM_REF
    is_neural:  bool     = False    # True → injected by neural ARF mechanism
    frame_num:  int      = 0        # coded frame number (for LT bookkeeping)

    def mark_unused(self):
        self.usage = RefUsage.UNUSED

    def mark_short_term(self):
        self.usage = RefUsage.SHORT_TERM_REF

    def mark_long_term(self):
        self.usage = RefUsage.LONG_TERM_REF

    @property
    def is_reference(self) -> bool:
        return bool(self.usage & (RefUsage.SHORT_TERM_REF | RefUsage.LONG_TERM_REF))


# ---------------------------------------------------------------------------
# Neural ARF slot  — thin wrapper that forces LTRP semantics
# ---------------------------------------------------------------------------

@dataclass
class NeuralARFSlot(DPBEntry):
    """
    A DPBEntry permanently assigned POC = NEURAL_ARF_POC and LTRP usage.
    The slot is updated in-place each time the neural generator produces a
    new prediction, rather than being evicted and re-inserted.
    """
    poc:       int      = field(default=NEURAL_ARF_POC, init=False)
    usage:     RefUsage = field(default=RefUsage.LONG_TERM_REF, init=False)
    is_neural: bool     = field(default=True, init=False)

    def update(self, new_pixels: np.ndarray, frame_num: int):
        """Replace pixel content while keeping POC and LTRP status."""
        self.pixels    = new_pixels
        self.frame_num = frame_num
        self.usage     = RefUsage.LONG_TERM_REF   # always keep as LTRP


# ---------------------------------------------------------------------------
# Reference Picture List  (L0 / L1)
# ---------------------------------------------------------------------------

class ReferencePictureList:
    """
    Constructs the per-slice L0 and L1 reference lists following
    H.265 Section 8.3.4, with the neural ARF always prepended to L0.

    L0 order  (default): [neural_arf | STRP nearest-past … | LTRP …]
    L1 order  (default): [STRP nearest-future … | LTRP …]
    """

    def __init__(self, dpb: "DecodedPictureBuffer",
                 current_poc: int,
                 max_l0: int = 4,
                 max_l1: int = 2):
        self.dpb         = dpb
        self.current_poc = current_poc
        self.max_l0      = max_l0
        self.max_l1      = max_l1
        self._l0: List[DPBEntry] = []
        self._l1: List[DPBEntry] = []
        self._build()

    def _build(self):
        # ── Separate neural ARF, short-term, long-term entries ────────────
        neural  = [e for e in self.dpb if e.is_neural and e.is_reference]
        strp    = [e for e in self.dpb
                   if not e.is_neural
                   and bool(e.usage & RefUsage.SHORT_TERM_REF)]
        ltrp    = [e for e in self.dpb
                   if not e.is_neural
                   and bool(e.usage & RefUsage.LONG_TERM_REF)]

        # ── L0: descending POC distance (past frames first) ───────────────
        past_strp   = sorted([e for e in strp if e.poc < self.current_poc],
                             key=lambda e: e.poc, reverse=True)
        future_strp = sorted([e for e in strp if e.poc > self.current_poc],
                             key=lambda e: e.poc)

        # Per spec: L0 = [past STRP desc, future STRP asc, LTRP], then neural
        # We prepend the neural ARF so the encoder sees it as the *first*
        # candidate — giving it the highest implicit priority.
        base_l0 = past_strp + future_strp + ltrp
        self._l0 = (neural + base_l0)[:self.max_l0]

        # ── L1: future first, then past (for B-frames) ────────────────────
        base_l1 = future_strp + past_strp + ltrp
        self._l1 = (base_l1)[:self.max_l1]

    # ── Public accessors ──────────────────────────────────────────────────

    @property
    def l0(self) -> List[DPBEntry]:
        return list(self._l0)

    @property
    def l1(self) -> List[DPBEntry]:
        return list(self._l1)

    def neural_arf_l0_index(self) -> Optional[int]:
        """Return the index of the neural ARF in L0, or None."""
        for i, e in enumerate(self._l0):
            if e.is_neural:
                return i
        return None

    def summary(self) -> str:
        def _fmt(lst):
            return [f"POC={e.poc}{'(N)' if e.is_neural else ''}" for e in lst]
        return f"L0={_fmt(self._l0)}  L1={_fmt(self._l1)}"


# ---------------------------------------------------------------------------
# Decoded Picture Buffer
# ---------------------------------------------------------------------------

class DecodedPictureBuffer:
    """
    Full DPB implementation with:
      * Automatic STRP/LTRP management
      * One persistent Neural ARF slot (LTRP at NEURAL_ARF_POC)
      * Sliding-window eviction per H.265 Section 8.3.2

    Parameters
    ----------
    max_size        : DPB capacity in frames (max_dec_pic_buffering_minus1+1)
    inject_neural   : whether to maintain a neural ARF slot
    """

    def __init__(self, max_size: int = 8, inject_neural: bool = True):
        self.max_size      = max_size
        self._entries: OrderedDict[int, DPBEntry] = OrderedDict()
        self._neural_slot: Optional[NeuralARFSlot] = None
        self._frame_num    = 0

        if inject_neural:
            self._neural_slot = NeuralARFSlot(
                pixels=np.zeros((256, 256, 3), dtype=np.uint8)
            )
            # Reserve the neural POC slot in the buffer
            self._entries[NEURAL_ARF_POC] = self._neural_slot
            log.debug(f'Neural ARF slot created (POC={NEURAL_ARF_POC})')

    # ── Neural ARF interface ──────────────────────────────────────────────

    def update_neural_arf(self, pixels: np.ndarray):
        """
        Replace the content of the neural ARF with a newly generated frame.
        Called by both encoder and decoder each time the generator fires.

        pixels : (H, W, 3) uint8  or  (H, W, 3) float32 in [0,1]
        """
        if self._neural_slot is None:
            raise RuntimeError('DPB was created without neural ARF support.')

        if pixels.dtype != np.uint8:
            pixels = (pixels.clip(0, 1) * 255).astype(np.uint8)

        self._neural_slot.update(pixels, self._frame_num)
        log.debug(f'Neural ARF updated (frame_num={self._frame_num})')

    @property
    def neural_arf_pixels(self) -> Optional[np.ndarray]:
        if self._neural_slot is not None:
            return self._neural_slot.pixels
        return None

    @property
    def has_neural_arf(self) -> bool:
        return self._neural_slot is not None

    # ── Standard DPB operations ───────────────────────────────────────────

    def insert(self, poc: int, pixels: np.ndarray,
               usage: RefUsage = RefUsage.SHORT_TERM_REF):
        """Insert a decoded frame into the DPB (never overwrites neural slot)."""
        if poc == NEURAL_ARF_POC:
            raise ValueError(
                f'POC {NEURAL_ARF_POC} is reserved for the neural ARF.')

        if len(self._entries) >= self.max_size + (1 if self._neural_slot else 0):
            self._evict_one()

        self._entries[poc] = DPBEntry(
            poc=poc, pixels=pixels, usage=usage, frame_num=self._frame_num
        )
        self._frame_num += 1
        log.debug(f'DPB insert POC={poc}  size={len(self._entries)}')

    def mark_long_term(self, poc: int):
        entry = self._entries.get(poc)
        if entry is None:
            raise KeyError(f'POC {poc} not in DPB')
        entry.mark_long_term()

    def mark_unused(self, poc: int):
        if poc == NEURAL_ARF_POC:
            return   # neural slot is never evicted via this path
        entry = self._entries.get(poc)
        if entry:
            entry.mark_unused()

    def remove_unused(self):
        """Evict all frames flagged as UNUSED (except the neural slot)."""
        to_remove = [poc for poc, e in self._entries.items()
                     if e.usage == RefUsage.UNUSED and poc != NEURAL_ARF_POC]
        for poc in to_remove:
            del self._entries[poc]
            log.debug(f'DPB evicted POC={poc}')

    def apply_rps(self, strp_pocs: List[int], ltrp_pocs: List[int]):
        """
        Apply a Reference Picture Set from a slice header.
        Frames NOT in the RPS are marked UNUSED and will be evicted.
        Neural ARF is never touched.
        """
        keep = set(strp_pocs) | set(ltrp_pocs) | {NEURAL_ARF_POC}
        for poc, entry in self._entries.items():
            if poc in keep:
                if poc in ltrp_pocs:
                    entry.mark_long_term()
                elif poc != NEURAL_ARF_POC:
                    entry.mark_short_term()
            else:
                entry.mark_unused()
        self.remove_unused()

    def get(self, poc: int) -> Optional[DPBEntry]:
        return self._entries.get(poc)

    def build_rpl(self, current_poc: int,
                  max_l0: int = 4,
                  max_l1: int = 2) -> ReferencePictureList:
        """Build L0/L1 lists for the current frame."""
        return ReferencePictureList(self, current_poc, max_l0, max_l1)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _evict_one(self):
        """Evict the oldest unused-for-reference STRP (never the neural ARF)."""
        candidates = [
            (poc, e) for poc, e in self._entries.items()
            if poc != NEURAL_ARF_POC
            and bool(e.usage & RefUsage.SHORT_TERM_REF)
        ]
        if not candidates:
            log.warning('DPB full but no evictable STRP — forcing oldest entry')
            candidates = [(poc, e) for poc, e in self._entries.items()
                          if poc != NEURAL_ARF_POC]
        if candidates:
            oldest_poc = min(candidates, key=lambda x: x[1].frame_num)[0]
            del self._entries[oldest_poc]
            log.debug(f'DPB auto-evicted POC={oldest_poc}')

    # ── Iteration / inspection ────────────────────────────────────────────

    def __iter__(self) -> Iterator[DPBEntry]:
        return iter(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        entries = ', '.join(
            f"POC={e.poc}{'(N)' if e.is_neural else ''}[{e.usage.name}]"
            for e in self._entries.values()
        )
        return f'DPB([{entries}])'

    def dump(self) -> str:
        lines = ['DPB contents:']
        for e in self._entries.values():
            flag  = '(NEURAL)' if e.is_neural else ''
            lines.append(
                f'  POC={e.poc:6d}  usage={e.usage.name:17s}  '
                f'frame_num={e.frame_num}  {flag}'
            )
        return '\n'.join(lines)
