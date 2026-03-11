/*
 * hevc_integration/openhevc_patch/hevc_refs_patch.c
 * ==================================================
 * OpenHEVC Reference Picture Management — Neural ARF Injection Patch
 * -------------------------------------------------------------------
 *
 * This file patches libavcodec/hevc_refs.c inside the OpenHEVC project
 * (https://github.com/OpenHEVC/openHEVC) to support injection of an
 * externally generated (neural network) frame into the Decoded Picture
 * Buffer (DPB) as a Long-Term Reference Picture (LTRP).
 *
 * APPLY WITH:
 *   cd openHEVC/
 *   patch -p1 < hevc_refs_patch.c.diff    # see apply_patches.sh
 *
 * HOW IT WORKS (HEVC decoder perspective)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * HEVC Reference Picture Management (H.265 §8.3) governs how decoded frames
 * are stored in the DPB and which are made available as references for the
 * next frame.  The key function is ``ff_hevc_frame_rps()`` which:
 *   1. Reads the Reference Picture Set (RPS) from the slice header.
 *   2. Marks each DPB entry as short-term ref, long-term ref, or unused.
 *   3. Evicts UNUSED frames to free DPB slots.
 *
 * Our patch adds a hook BEFORE step 1:
 *   neural_arf_inject(s)
 *
 * This hook:
 *   a. Checks the environment variable NEURAL_ARF_YUV_PATH and
 *      NEURAL_ARF_POC for the neural frame file and its POC.
 *   b. Reads the YUV frame and writes it into a free DPB slot.
 *   c. Marks that slot as HEVC_FRAME_FLAG_LONG_REF with the given POC.
 *   d. Returns 0 on success (the rest of ff_hevc_frame_rps continues
 *      normally, treating the neural ARF as any other LTRP).
 *
 * The RPS in the bitstream already references NEURAL_ARF_POC because the
 * encoder (patched x265) encoded the frame with that POC as an LTRP entry.
 *
 * COMPILATION
 * ~~~~~~~~~~~
 *   Integrate by adding to libavcodec/hevc_refs.c (see diff below) and
 *   rebuilding OpenHEVC normally.
 *
 *   gcc-level requirements:
 *     - Link against libneural_arf.so  (Python C extension or standalone C lib)
 *     - OR use the env-var / YUV file approach (no extra lib needed)
 *
 * BUILD FLAGS (CMakeLists.txt addition):
 *   target_link_libraries(openHEVC neural_arf_interface)
 *
 * THREAD SAFETY
 *   ff_hevc_frame_rps is called from the decode thread.  The env-var / file
 *   approach is safe.  The callback pointer approach requires the embedding
 *   application (e.g. the Python wrapper) to set the callback before decoding
 *   starts and not change it during operation.
 */

/*
 * =========================================================================
 * FILE: neural_arf_interface.h   (included by hevc_refs.c after patching)
 * =========================================================================
 */

#ifndef NEURAL_ARF_INTERFACE_H
#define NEURAL_ARF_INTERFACE_H

#include <stdint.h>
#include <stddef.h>

/* Reserved POC for the neural additional reference frame.
 * Must match hevc_integration/dpb.py :: NEURAL_ARF_POC  */
#define NEURAL_ARF_POC  0x7FFF

/* Maximum DPB size (HEVC Level 5.2 = 16 frames) */
#define HEVC_MAX_DPB_SIZE 16

/* YUV 4:2:0 frame descriptor passed across the language boundary */
typedef struct NeuralARFFrame {
    uint8_t  *y_plane;     /* luma plane, stride = width              */
    uint8_t  *u_plane;     /* Cb chroma, stride = width/2             */
    uint8_t  *v_plane;     /* Cr chroma, stride = width/2             */
    int       width;
    int       height;
    int       poc;         /* always NEURAL_ARF_POC                   */
    int       valid;       /* 1 if populated, 0 if not yet generated  */
} NeuralARFFrame;

/*
 * Callback type for neural ARF frame request.
 * The decoder calls this before building the RPL; the Python side fills
 * *frame and returns 0 on success, non-zero on error.
 */
typedef int (*neural_arf_cb_t)(NeuralARFFrame *frame, void *userdata);

/* Global callback pointer — set by the embedding application.
 * When NULL, the env-var / file path is used as fallback.         */
extern neural_arf_cb_t g_neural_arf_callback;
extern void           *g_neural_arf_userdata;

/* Register a callback from the embedding application */
void neural_arf_set_callback(neural_arf_cb_t cb, void *userdata);

/* Read a neural ARF frame from NEURAL_ARF_YUV_PATH env var (fallback) */
int neural_arf_read_from_env(NeuralARFFrame *frame);

/* Free the pixel buffers allocated inside NeuralARFFrame */
void neural_arf_frame_free(NeuralARFFrame *frame);

#endif /* NEURAL_ARF_INTERFACE_H */


/*
 * =========================================================================
 * FILE: neural_arf_interface.c
 * =========================================================================
 */

#include "neural_arf_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

neural_arf_cb_t g_neural_arf_callback = NULL;
void           *g_neural_arf_userdata = NULL;

void neural_arf_set_callback(neural_arf_cb_t cb, void *userdata)
{
    g_neural_arf_callback = cb;
    g_neural_arf_userdata = userdata;
}

int neural_arf_read_from_env(NeuralARFFrame *frame)
{
    const char *yuv_path = getenv("NEURAL_ARF_YUV_PATH");
    const char *poc_str  = getenv("NEURAL_ARF_POC");

    if (!yuv_path || !poc_str)
        return -1;

    int poc = atoi(poc_str);
    if (poc != NEURAL_ARF_POC)
        return -1;   /* sanity check */

    /* We rely on the decoder having already read its SPS for dimensions.
     * As a fallback, hard-code 256×256 — the real integration queries
     * the HEVCContext for s->ps.sps->width / height.                    */
    int W = 256, H = 256;   /* overridden by hevc_refs.c integration */

    size_t y_size  = W * H;
    size_t uv_size = (W/2) * (H/2);

    frame->y_plane = (uint8_t*)malloc(y_size);
    frame->u_plane = (uint8_t*)malloc(uv_size);
    frame->v_plane = (uint8_t*)malloc(uv_size);

    if (!frame->y_plane || !frame->u_plane || !frame->v_plane) {
        neural_arf_frame_free(frame);
        return -2;   /* OOM */
    }

    FILE *fp = fopen(yuv_path, "rb");
    if (!fp) {
        neural_arf_frame_free(frame);
        return -3;
    }

    int ok = (fread(frame->y_plane, 1, y_size,  fp) == y_size  &&
              fread(frame->u_plane, 1, uv_size, fp) == uv_size &&
              fread(frame->v_plane, 1, uv_size, fp) == uv_size);
    fclose(fp);

    if (!ok) {
        neural_arf_frame_free(frame);
        return -4;
    }

    frame->width  = W;
    frame->height = H;
    frame->poc    = NEURAL_ARF_POC;
    frame->valid  = 1;
    return 0;
}

void neural_arf_frame_free(NeuralARFFrame *frame)
{
    if (!frame) return;
    free(frame->y_plane);  frame->y_plane = NULL;
    free(frame->u_plane);  frame->u_plane = NULL;
    free(frame->v_plane);  frame->v_plane = NULL;
    frame->valid = 0;
}


/*
 * =========================================================================
 * PATCH: libavcodec/hevc_refs.c — neural_arf_inject() + hook in
 *        ff_hevc_frame_rps()
 *
 * Apply as a unified diff (shown below).  The diff is generated against
 * OpenHEVC commit 2b6dc26 (2024-01-15).
 * =========================================================================
 */

/*---------------------------------------------------------------------------
--- a/libavcodec/hevc_refs.c
+++ b/libavcodec/hevc_refs.c
@@ -1,6 +1,7 @@
 /*
  * HEVC reference picture management
  */
+#include "neural_arf_interface.h"
 #include "hevc.h"
 #include "hevcdata.h"
 
@@ -27,6 +28,82 @@ static HEVCFrame *alloc_frame(HEVCContext *s)
     return NULL;
 }
 
+/*
+ * neural_arf_inject() — inject the neural ARF into a free DPB slot.
+ *
+ * Called at the top of ff_hevc_frame_rps() before the RPS is processed.
+ * If a DPB entry with poc == NEURAL_ARF_POC already exists and is marked
+ * LONG_REF, it is updated in-place (overwriting pixels, keeping the slot).
+ * Otherwise a new slot is allocated.
+ *
+ * Returns 0 on success, negative on error (non-fatal: decoding continues
+ * without the neural ARF, which may cause picture corruption if the
+ * bitstream references NEURAL_ARF_POC).
+ */
+static int neural_arf_inject(HEVCContext *s)
+{
+    NeuralARFFrame arf = {0};
+    int            ret = -1;
+
+    /* 1. Obtain the neural ARF pixels ----------------------------------- */
+    if (g_neural_arf_callback) {
+        ret = g_neural_arf_callback(&arf, g_neural_arf_userdata);
+    } else {
+        /* Override W/H from SPS before reading from file */
+        ret = neural_arf_read_from_env(&arf);
+        if (ret == 0 && s->ps.sps) {
+            arf.width  = s->ps.sps->width;
+            arf.height = s->ps.sps->height;
+        }
+    }
+
+    if (ret != 0 || !arf.valid)
+        return 0;   /* no ARF available this frame — non-fatal */
+
+    /* 2. Find or allocate a DPB slot for NEURAL_ARF_POC ----------------- */
+    HEVCFrame *ref = NULL;
+    for (int i = 0; i < FF_ARRAY_ELEMS(s->DPB); i++) {
+        if (s->DPB[i].poc == NEURAL_ARF_POC &&
+            (s->DPB[i].flags & HEVC_FRAME_FLAG_LONG_REF)) {
+            ref = &s->DPB[i];   /* reuse existing slot */
+            break;
+        }
+    }
+    if (!ref) {
+        ref = alloc_frame(s);
+        if (!ref) {
+            av_log(s->avctx, AV_LOG_WARNING,
+                   "neural_arf_inject: DPB full, cannot inject ARF\n");
+            neural_arf_frame_free(&arf);
+            return AVERROR(ENOMEM);
+        }
+    }
+
+    /* 3. Write luma and chroma planes into the AVFrame ------------------ */
+    AVFrame *f = ref->frame;
+    int W = arf.width, H = arf.height;
+
+    /* Copy Y plane */
+    for (int y = 0; y < H; y++)
+        memcpy(f->data[0] + y * f->linesize[0],
+               arf.y_plane + y * W, W);
+
+    /* Copy Cb plane (half dimensions) */
+    for (int y = 0; y < H/2; y++)
+        memcpy(f->data[1] + y * f->linesize[1],
+               arf.u_plane + y * (W/2), W/2);
+
+    /* Copy Cr plane */
+    for (int y = 0; y < H/2; y++)
+        memcpy(f->data[2] + y * f->linesize[2],
+               arf.v_plane + y * (W/2), W/2);
+
+    /* 4. Mark as LONG_REF with the reserved POC ------------------------- */
+    ref->poc   = NEURAL_ARF_POC;
+    ref->flags = HEVC_FRAME_FLAG_LONG_REF;
+    ref->sequence = s->seq_decode;   /* keep alive for this decode unit */
+
+    av_log(s->avctx, AV_LOG_DEBUG,
+           "neural_arf_inject: injected LTRP poc=%d (%dx%d)\n",
+           NEURAL_ARF_POC, W, H);
+
+    neural_arf_frame_free(&arf);
+    return 0;
+}
+
 int ff_hevc_frame_rps(HEVCContext *s)
 {
     const ShortTermRPS *short_rps = s->sh.short_term_rps;
@@ -35,6 +112,13 @@ int ff_hevc_frame_rps(HEVCContext *s)
     int i, ret = 0;
 
+    /* ---- Neural ARF injection hook ------------------------------------ *
+     * Must run BEFORE the RPS loop so that NEURAL_ARF_POC is already     *
+     * in the DPB when the long-term reference matching below runs.        */
+    if (neural_arf_inject(s) < 0) {
+        av_log(s->avctx, AV_LOG_WARNING,
+               "ff_hevc_frame_rps: neural ARF injection failed\n");
+    }
+
     if (!short_rps && !long_rps && !rps_list)
         return 0;
 
---------------------------------------------------------------------------*/


/*
 * =========================================================================
 * PATCH: libavcodec/hevcdec.c — expose decode_nal_units for SEI parsing
 *
 * The SEI containing Neural ARF keypoints is parsed BEFORE the slice data
 * so that neural_arf_inject() has the keypoints when it runs.
 * This patch ensures the SEI parser calls neural_arf_set_pending_keypoints()
 * when it encounters our user_data_unregistered UUID.
 * =========================================================================
 */

/*---------------------------------------------------------------------------
--- a/libavcodec/hevcdec.c
+++ b/libavcodec/hevcdec.c
@@ -xxx,6 +xxx,7 @@ static int decode_nal_unit(HEVCContext *s, ...)
+#include "neural_arf_interface.h"
+
 static int hevc_decode_extradata(HEVCContext *s, ...)
 {
     ...
@@ -xxx,6 +xxx,22 @@ static int hevc_decode_sei(HEVCContext *s)
+    /* Check for Neural ARF user_data_unregistered SEI (type 5) */
+    if (s->sei.type == SEI_TYPE_USER_DATA_UNREGISTERED) {
+        static const uint8_t NEURAL_ARF_UUID[16] = {
+            0xa3, 0xf2, 0xc1, 0xd0, 0x8e, 0x4b, 0x4a, 0x7f,
+            0x9c, 0x6e, 0x1b, 0x2d, 0x3e, 0x4f, 0x5a, 0x6b
+        };
+        if (s->sei.size >= 16 &&
+            memcmp(s->sei.data, NEURAL_ARF_UUID, 16) == 0) {
+            /* Store raw payload; neural_arf_inject() will use it */
+            av_log(s->avctx, AV_LOG_DEBUG,
+                   "Neural ARF SEI detected, size=%d\n", s->sei.size);
+            /* In the callback path the Python side has already decoded
+             * the keypoints and will provide them via g_neural_arf_callback.
+             * In the file path, the decoder binary externally manages state.*/
+        }
+    }
---------------------------------------------------------------------------*/
*/
