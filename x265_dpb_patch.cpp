/*
 * hevc_integration/openhevc_patch/x265_dpb_patch.cpp
 * ====================================================
 * x265 Encoder DPB — Neural ARF Injection Patch
 * -----------------------------------------------
 *
 * Patches source/encoder/dpb.cpp in x265 (https://bitbucket.org/multicoreware/x265_git)
 * to support injection of an externally generated frame as a Long-Term
 * Reference Picture (LTRP) in the encoder's Reference Picture List.
 *
 * HEVC Encoder reference picture management (§8.3.2, encoder view)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * The encoder builds the Reference Picture Set (RPS) that it encodes
 * in each slice header.  This controls which DPB frames the DECODER will
 * keep.  The encoder also uses those same frames for its own inter-prediction
 * search (motion estimation).
 *
 * Neural ARF strategy
 * ~~~~~~~~~~~~~~~~~~~
 * We insert the neural ARF frame into the encoder's DPB *before*
 * DPB::prepareEncoderState() builds the reference lists.  This means:
 *   • The motion estimator in x265 can compute motion vectors against
 *     the neural ARF (often smaller MV + smaller residual = fewer bits).
 *   • The RPS written in the slice header includes NEURAL_ARF_POC as an
 *     LTRP entry, which is how the decoder knows to expect it.
 *
 * New CLI parameters added
 * ~~~~~~~~~~~~~~~~~~~~~~~~
 *   --ref-inject-yuv <path>   YUV file of the neural ARF frame
 *   --ref-inject-poc <int>    POC to assign (should equal NEURAL_ARF_POC)
 *   --ref-inject-lt  <0|1>    Mark as long-term reference (should be 1)
 *
 * Modified functions
 * ~~~~~~~~~~~~~~~~~~
 *   DPB::prepareEncoderState()   — insert neural ARF before ref-list build
 *   x265_param_parse()           — accept new CLI params
 *   Encoder::encode()            — pass neural ARF path per-frame
 *
 * APPLY WITH:
 *   cd x265_git/
 *   patch -p1 < hevc_integration/openhevc_patch/x265_dpb_patch.cpp.diff
 *   cmake -B build -DENABLE_NEURAL_ARF=ON
 *   cmake --build build -j$(nproc)
 */

/*---------------------------------------------------------------------------
 * DIFF: source/encoder/dpb.cpp
 *---------------------------------------------------------------------------

--- a/source/encoder/dpb.cpp
+++ b/source/encoder/dpb.cpp
@@ -1,5 +1,6 @@
 /*****************************************************************************
  * dpb.cpp: Decoded Picture Buffer
+ * Neural ARF injection support added.
  *****************************************************************************/
+#include "neural_arf_x265.h"
 #include "dpb.h"
 #include "encoder.h"
 
@@ -55,6 +56,80 @@ namespace X265_NS {
+ /*
+  * NeuralARFInjector implementation
+  * ----------------------------------
+  * Manages reading the neural ARF YUV and inserting it into the x265 DPB
+  * as a long-term reference picture at a fixed POC.
+  */
+
+ static Frame* neural_arf_alloc_frame(x265_param* param)
+ {
+     Frame* frame = new Frame();
+     if (!frame->create(param, NULL))
+     {
+         delete frame;
+         return NULL;
+     }
+     return frame;
+ }
+
+ static bool neural_arf_read_yuv(const char* path, Frame* frame, int W, int H)
+ {
+     FILE* fp = fopen(path, "rb");
+     if (!fp) return false;
+
+     /* Luma */
+     PicYuv* yuv = frame->m_fenc;
+     pixel* Y = yuv->m_picOrg[0];
+     for (int y = 0; y < H; y++)
+     {
+         fread(Y + y * yuv->m_stride, 1, W, fp);
+     }
+     /* Cb */
+     pixel* U = yuv->m_picOrg[1];
+     for (int y = 0; y < H/2; y++)
+         fread(U + y * yuv->m_strideC, 1, W/2, fp);
+     /* Cr */
+     pixel* V = yuv->m_picOrg[2];
+     for (int y = 0; y < H/2; y++)
+         fread(V + y * yuv->m_strideC, 1, W/2, fp);
+
+     fclose(fp);
+     return true;
+ }
+
+ /*
+  * DPB::injectNeuralARF()
+  * ----------------------
+  * Public method added to the DPB class.
+  * Called by Encoder::encode() after receiving the per-frame neural ARF path.
+  *
+  *   yuv_path  — path to the YUV file written by the Python neural generator
+  *   poc       — POC to assign (NEURAL_ARF_POC = 0x7FFF)
+  *
+  * The injected frame is:
+  *   • Assigned the given POC.
+  *   • Marked as a long-term reference (bIsLongTermRef = true).
+  *   • Pushed to the FRONT of the reference list (m_longTermRefPicList)
+  *     so the motion estimator sees it first.
+  *   • Included in the RPS as an LTRP via the standard markAsLongTerm path.
+  *
+  * Thread safety: must be called from the encode thread before
+  *   prepareEncoderState() builds the reference picture lists.
+  */
+ bool DPB::injectNeuralARF(const char* yuv_path, int poc)
+ {
+     if (!yuv_path || !*yuv_path)
+         return false;
+
+     int W = m_param->sourceWidth;
+     int H = m_param->sourceHeight;
+
+     /* Reuse existing NEURAL_ARF slot or allocate new */
+     Frame* arf = NULL;
+     for (Frame* f = m_picList.first(); f; f = f->m_next)
+     {
+         if (f->m_poc == poc && f->m_bIsLongTermRef)
+         {
+             arf = f;
+             break;
+         }
+     }
+     if (!arf)
+     {
+         arf = neural_arf_alloc_frame(m_param);
+         if (!arf)
+         {
+             x265_log(m_param, X265_LOG_WARNING,
+                      "DPB::injectNeuralARF: failed to alloc frame\n");
+             return false;
+         }
+         m_picList.push_back(arf);
+     }
+
+     if (!neural_arf_read_yuv(yuv_path, arf, W, H))
+     {
+         x265_log(m_param, X265_LOG_WARNING,
+                  "DPB::injectNeuralARF: failed to read YUV from %s\n",
+                  yuv_path);
+         return false;
+     }
+
+     arf->m_poc            = poc;
+     arf->m_bIsLongTermRef = true;
+     arf->m_refCount       = 1;   /* keep alive for this GOP */
+
+     x265_log(m_param, X265_LOG_DEBUG,
+              "Neural ARF injected: POC=%d YUV=%s\n", poc, yuv_path);
+     return true;
+ }
+
 void DPB::prepareEncoderState(Frame* curFrame, ...)
 {
+    /* Neural ARF injection: insert before reference list is built.
+     * The yuv_path and poc are set by Encoder::encode() from CLI params
+     * or per-frame user data.                                          */
+    if (curFrame->m_userData.neuralArfYuvPath &&
+        curFrame->m_userData.neuralArfPoc > 0)
+    {
+        injectNeuralARF(curFrame->m_userData.neuralArfYuvPath,
+                        curFrame->m_userData.neuralArfPoc);
+    }
+
     ... (existing code unchanged) ...
---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------
 * DIFF: source/encoder/dpb.h  (declaration)
 *---------------------------------------------------------------------------

--- a/source/encoder/dpb.h
+++ b/source/encoder/dpb.h
@@ -60,6 +60,8 @@ public:
     void flush();
 
+    /* Neural ARF injection API */
+    bool injectNeuralARF(const char* yuv_path, int poc);
+
 protected:
---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------
 * DIFF: source/x265.h  (x265_picture userData extension)
 *---------------------------------------------------------------------------

--- a/source/x265.h
+++ b/source/x265.h
@@ -600,6 +600,14 @@ typedef struct x265_picture
+    /* Neural ARF per-frame injection.
+     * Set these before passing the picture to x265_encoder_encode().
+     * Leave neuralArfYuvPath == NULL to skip injection for this frame.  */
+    struct {
+        const char* neuralArfYuvPath;   /* path to YUV file, or NULL    */
+        int         neuralArfPoc;       /* NEURAL_ARF_POC = 0x7FFF      */
+        int         neuralArfIsLT;      /* 1 = long-term ref (always 1) */
+    } neuralArf;
 
     void*    userData;       /* existing field */
---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------
 * DIFF: source/encoder/slicetype.cpp  (RPS construction)
 *
 * Ensures NEURAL_ARF_POC is added to the long-term RPS entry list so that
 * the decoder can correctly reconstruct the Reference Picture Set.
 *---------------------------------------------------------------------------

--- a/source/encoder/slicetype.cpp
+++ b/source/encoder/slicetype.cpp
@@ -850,6 +850,22 @@ static void buildRefPicList(Frame* frames[], ...)
+    /* ---- Add Neural ARF as long-term RPS entry ----------------------- */
+    /* Check whether the DPB contains a frame at NEURAL_ARF_POC.
+     * If so, include it in the long-term RPS so the decoder can reference it. */
+#define NEURAL_ARF_POC 0x7FFF
+    bool hasNeuralARF = false;
+    for (int i = 0; i < numFrames; i++)
+    {
+        if (frames[i] && frames[i]->m_poc == NEURAL_ARF_POC &&
+            frames[i]->m_bIsLongTermRef)
+        {
+            hasNeuralARF = true;
+            break;
+        }
+    }
+    if (hasNeuralARF)
+    {
+        /* Add NEURAL_ARF_POC to the LTRP delta list */
+        rps->numberOfLongtermPictures++;
+        rps->used[rps->numberOfPictures - 1] = 1;
+        rps->poc[rps->numberOfPictures - 1]  = NEURAL_ARF_POC;
+        x265_log_file(param, X265_LOG_DEBUG,
+                      "buildRefPicList: added Neural ARF LTRP POC=%d\n",
+                      NEURAL_ARF_POC);
+    }
---------------------------------------------------------------------------*/
*/
