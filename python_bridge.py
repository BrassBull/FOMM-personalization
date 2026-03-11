"""
hevc_integration/openhevc_patch/python_bridge.py
=================================================
Python ↔ C Bridge for Patched OpenHEVC / x265
-----------------------------------------------
Provides ctypes bindings that let the Python NeuralARFDecoder set the
``g_neural_arf_callback`` function pointer in the patched libavcodec.so
and let the NeuralARFEncoder call ``DPB::injectNeuralARF`` via the thin
C shim ``libneural_arf_x265.so``.

Architecture
~~~~~~~~~~~~
                   Python process
    ┌──────────────────────────────────────────────────┐
    │  NeuralARFDecoder                                │
    │         │                                        │
    │  python_bridge.set_decoder_callback(cb)          │
    │         │ ctypes                                 │
    │         ▼                                        │
    │  libopenhevc.so ── neural_arf_interface.c        │
    │         │  g_neural_arf_callback = cb            │
    │         │                                        │
    │  ff_hevc_frame_rps()                             │
    │    └─► neural_arf_inject()                       │
    │            └─► cb(frame, userdata)  ◄── Python   │
    └──────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │  NeuralARFEncoder                                │
    │         │                                        │
    │  python_bridge.inject_encoder_arf(yuv, poc)      │
    │         │ ctypes                                 │
    │         ▼                                        │
    │  libneural_arf_x265.so                           │
    │    └─► x265_neural_arf_inject(yuv_path, poc)     │
    │            └─► DPB::injectNeuralARF(...)         │
    └──────────────────────────────────────────────────┘

Usage (once the shared libraries are built)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from hevc_integration.openhevc_patch.python_bridge import (
        OpenHEVCBridge, X265Bridge
    )

    decoder_bridge = OpenHEVCBridge('/usr/lib/libopenhevc.so')
    encoder_bridge = X265Bridge('/usr/local/lib/libneural_arf_x265.so')

    # Register Python callback — called from C on every frame
    decoder_bridge.register_neural_arf_callback(my_neural_arf_generator)

    # Inject ARF before encoding a frame
    encoder_bridge.inject_arf(arf_yuv_path, poc=0x7FFF)
"""

from __future__ import annotations

import ctypes
import logging
import os
import tempfile
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

# ── C struct mirrors ──────────────────────────────────────────────────────
NEURAL_ARF_POC = 0x7FFF


class NeuralARFFrameC(ctypes.Structure):
    """
    Mirror of the C ``NeuralARFFrame`` struct from neural_arf_interface.h.
    """
    _fields_ = [
        ('y_plane', ctypes.POINTER(ctypes.c_uint8)),
        ('u_plane', ctypes.POINTER(ctypes.c_uint8)),
        ('v_plane', ctypes.POINTER(ctypes.c_uint8)),
        ('width',   ctypes.c_int),
        ('height',  ctypes.c_int),
        ('poc',     ctypes.c_int),
        ('valid',   ctypes.c_int),
    ]


# Callback type:  int (*)(NeuralARFFrame*, void*)
_NeuralARFCallbackType = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(NeuralARFFrameC),
    ctypes.c_void_p,
)


# ---------------------------------------------------------------------------
# OpenHEVC decoder bridge
# ---------------------------------------------------------------------------

class OpenHEVCBridge:
    """
    Loads the patched libopenhevc.so and wires up the neural ARF callback.

    Parameters
    ----------
    lib_path : path to the patched shared library
               (e.g. '/usr/lib/libopenhevc.so')
    """

    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f'Patched OpenHEVC library not found: {lib_path}\n'
                f'Build it with: cmake -DENABLE_NEURAL_ARF=ON'
            )
        self._lib     = ctypes.CDLL(lib_path)
        self._cb_ref  = None   # keep Python callback alive

        # Resolve symbols
        self._set_callback = self._lib.neural_arf_set_callback
        self._set_callback.argtypes = [_NeuralARFCallbackType, ctypes.c_void_p]
        self._set_callback.restype  = None

        log.info(f'OpenHEVCBridge loaded: {lib_path}')

    def register_neural_arf_callback(
            self,
            py_callback: Callable[[np.ndarray], np.ndarray]):
        """
        Register a Python function that generates the neural ARF frame.

        py_callback signature::
            def generate(frame_out: np.ndarray) -> None:
                # fill frame_out (H×W uint8 Y,U,V planes in-place)

        The C decoder will call this synchronously inside
        ``neural_arf_inject()`` every time a slice with a reference to
        NEURAL_ARF_POC is about to be decoded.
        """

        @_NeuralARFCallbackType
        def _c_callback(frame_ptr: ctypes.POINTER(NeuralARFFrameC),
                        userdata:  ctypes.c_void_p) -> int:
            frame = frame_ptr.contents
            W, H  = frame.width, frame.height

            try:
                # Call Python generator — it returns (H,W,3) uint8 RGB
                rgb = py_callback()
                if rgb is None or rgb.shape != (H, W, 3):
                    log.warning('Neural ARF callback returned bad shape')
                    return -1

                # Convert RGB → YCbCr and copy into C struct buffers
                y, u, v = _rgb_to_yuv420(rgb)

                ctypes.memmove(frame.y_plane, y.tobytes(), W * H)
                ctypes.memmove(frame.u_plane, u.tobytes(), (W//2) * (H//2))
                ctypes.memmove(frame.v_plane, v.tobytes(), (W//2) * (H//2))

                frame.poc   = NEURAL_ARF_POC
                frame.valid = 1
                return 0

            except Exception as exc:
                log.error(f'Neural ARF callback error: {exc}')
                return -1

        self._cb_ref = _c_callback   # prevent GC
        self._set_callback(_c_callback, None)
        log.info('Neural ARF decoder callback registered')

    def unregister(self):
        """Remove the callback (decoder falls back to env-var method)."""
        null_cb = _NeuralARFCallbackType(0)
        self._set_callback(null_cb, None)
        self._cb_ref = None


# ---------------------------------------------------------------------------
# x265 encoder bridge
# ---------------------------------------------------------------------------

class X265Bridge:
    """
    Loads the patched libneural_arf_x265.so and exposes the ARF injection API.

    The shared library exposes one C function:

        int x265_neural_arf_inject(x265_encoder* enc,
                                   const char*   yuv_path,
                                   int           poc,
                                   int           is_long_term);

    Parameters
    ----------
    lib_path : path to the thin C shim library
    """

    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f'Patched x265 ARF library not found: {lib_path}'
            )
        self._lib = ctypes.CDLL(lib_path)

        self._inject_fn = self._lib.x265_neural_arf_inject
        self._inject_fn.argtypes = [
            ctypes.c_void_p,   # x265_encoder*
            ctypes.c_char_p,   # yuv_path
            ctypes.c_int,      # poc
            ctypes.c_int,      # is_long_term
        ]
        self._inject_fn.restype = ctypes.c_int

        self._enc_ptr: Optional[ctypes.c_void_p] = None
        log.info(f'X265Bridge loaded: {lib_path}')

    def bind_encoder(self, encoder_ptr: int):
        """Set the x265_encoder* pointer (obtained from x265_encoder_open())."""
        self._enc_ptr = ctypes.c_void_p(encoder_ptr)

    def inject_arf(self,
                   pixels_rgb: np.ndarray,
                   poc: int = NEURAL_ARF_POC) -> bool:
        """
        Write the neural ARF frame to a temp YUV file and call the C inject.

        Parameters
        ----------
        pixels_rgb : (H, W, 3) uint8
        poc        : should always be NEURAL_ARF_POC

        Returns
        -------
        True on success.
        """
        if self._enc_ptr is None:
            raise RuntimeError('Encoder pointer not bound. Call bind_encoder().')

        y, u, v = _rgb_to_yuv420(pixels_rgb)

        with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as tf:
            tf.write(y.tobytes())
            tf.write(u.tobytes())
            tf.write(v.tobytes())
            yuv_path = tf.name

        try:
            ret = self._inject_fn(
                self._enc_ptr,
                yuv_path.encode('utf-8'),
                ctypes.c_int(poc),
                ctypes.c_int(1),   # always long-term
            )
            if ret != 0:
                log.error(f'x265_neural_arf_inject returned {ret}')
                return False
            return True
        finally:
            os.unlink(yuv_path)


# ---------------------------------------------------------------------------
# YUV conversion helper
# ---------------------------------------------------------------------------

def _rgb_to_yuv420(rgb: np.ndarray):
    """
    Convert (H, W, 3) uint8 RGB to Y, U, V numpy arrays (YUV 4:2:0).
    """
    try:
        from PIL import Image
        img = Image.fromarray(rgb, 'RGB').convert('YCbCr')
        y_img, cb_img, cr_img = img.split()
        H, W = rgb.shape[:2]
        cb_ds = cb_img.resize((W // 2, H // 2), Image.BILINEAR)
        cr_ds = cr_img.resize((W // 2, H // 2), Image.BILINEAR)
        return (np.array(y_img,  dtype=np.uint8),
                np.array(cb_ds, dtype=np.uint8),
                np.array(cr_ds, dtype=np.uint8))
    except ImportError:
        # Fallback: approximate BT.601 conversion
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)
        H, W = rgb.shape[:2]
        Y = ( 0.299  * r + 0.587  * g + 0.114  * b).clip(0, 255).astype(np.uint8)
        U = (-0.16874* r - 0.33126* g + 0.5    * b + 128).clip(0,255).astype(np.uint8)
        V = ( 0.5    * r - 0.41869* g - 0.08131* b + 128).clip(0,255).astype(np.uint8)
        U_ds = U[::2, ::2]
        V_ds = V[::2, ::2]
        return Y, U_ds, V_ds
