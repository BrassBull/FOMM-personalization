#!/usr/bin/env bash
# hevc_integration/openhevc_patch/apply_patches.sh
# =================================================
# Build patched OpenHEVC and x265 with Neural ARF support.
#
# Prerequisites
# -------------
#   apt-get install build-essential cmake git libsdl2-dev yasm nasm
#   pip install torch torchvision   (for the Python bridge)
#
# Usage
#   chmod +x apply_patches.sh
#   ./apply_patches.sh [--openhevc-only | --x265-only | --all (default)]
#
# After successful build:
#   libs/libopenhevc.so      — patched decoder library
#   libs/libneural_arf_x265.so — encoder ARF injection shim
#   bin/openHevcDecoder      — patched decoder binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build_hevc"
LIBS_DIR="${PROJECT_ROOT}/libs"
BIN_DIR="${PROJECT_ROOT}/bin"

MODE="${1:---all}"

mkdir -p "${BUILD_DIR}" "${LIBS_DIR}" "${BIN_DIR}"

# ── Colour output ─────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }


# ══════════════════════════════════════════════════════════════════════════
# 1. Build patched OpenHEVC decoder
# ══════════════════════════════════════════════════════════════════════════
build_openhevc() {
    info "Building patched OpenHEVC decoder …"

    OPENHEVC_DIR="${BUILD_DIR}/openHEVC"

    if [ ! -d "${OPENHEVC_DIR}" ]; then
        git clone --depth 1 \
            https://github.com/OpenHEVC/openHEVC.git \
            "${OPENHEVC_DIR}"
    else
        info "OpenHEVC source already present, skipping clone."
    fi

    # ── Copy the neural ARF interface files ───────────────────────────────
    cp "${SCRIPT_DIR}/neural_arf_interface.h" \
       "${OPENHEVC_DIR}/libavcodec/neural_arf_interface.h"

    # Generate the .c file from the combined patch/doc file
    python3 - <<'PYEOF'
import re, pathlib
src = pathlib.Path("hevc_refs_patch.c").read_text()
# Extract everything between the first #include and the closing /*...*/
c_src = []
in_c = False
for line in src.splitlines():
    if line.startswith("#ifndef NEURAL_ARF_INTERFACE_H"):
        in_c = True
    if line.startswith("/*-----------"):
        break
    if in_c:
        c_src.append(line)
pathlib.Path("neural_arf_interface.c").write_text("\n".join(c_src))
print("Generated neural_arf_interface.c")
PYEOF

    cp neural_arf_interface.c \
       "${OPENHEVC_DIR}/libavcodec/neural_arf_interface.c"

    # ── Apply the hevc_refs.c patch ────────────────────────────────────────
    # The diff is embedded in hevc_refs_patch.c between the /*--- markers.
    # Extract and apply it.
    python3 - <<'PYEOF'
import re, pathlib, subprocess, sys
content = pathlib.Path("hevc_refs_patch.c").read_text()
diffs = re.findall(r'/\*-{10,}\n(.*?)\n-{10,}\*/', content, re.DOTALL)
if not diffs:
    print("No diffs found — manual patch required"); sys.exit(0)
for i, diff in enumerate(diffs):
    diff_path = f"/tmp/neural_arf_{i}.diff"
    pathlib.Path(diff_path).write_text(diff)
    result = subprocess.run(
        ["patch", "-p1", "--dry-run", "-i", diff_path],
        cwd="openHEVC", capture_output=True
    )
    if result.returncode == 0:
        subprocess.run(["patch", "-p1", "-i", diff_path], cwd="openHEVC")
        print(f"Applied diff {i}")
    else:
        print(f"Diff {i} already applied or conflicts — skipping")
PYEOF

    # ── CMakeLists.txt modification ────────────────────────────────────────
    cat >> "${OPENHEVC_DIR}/CMakeLists.txt" << 'CMAKE_PATCH'

# Neural ARF interface
target_sources(LibOpenHevcWrapper PRIVATE
    libavcodec/neural_arf_interface.c
)
target_compile_definitions(LibOpenHevcWrapper PUBLIC ENABLE_NEURAL_ARF=1)
CMAKE_PATCH

    # ── Build ──────────────────────────────────────────────────────────────
    cmake -S "${OPENHEVC_DIR}" \
          -B "${OPENHEVC_DIR}/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON 2>&1 | tail -5

    cmake --build "${OPENHEVC_DIR}/build" -j"$(nproc)" 2>&1 | tail -10

    # Copy outputs
    find "${OPENHEVC_DIR}/build" -name "*.so*"    -exec cp {} "${LIBS_DIR}/" \;
    find "${OPENHEVC_DIR}/build" -name "openHevc*" -executable \
         -exec cp {} "${BIN_DIR}/" \; 2>/dev/null || true

    info "OpenHEVC built → ${LIBS_DIR}/ and ${BIN_DIR}/"
}


# ══════════════════════════════════════════════════════════════════════════
# 2. Build patched x265 encoder + Neural ARF shim
# ══════════════════════════════════════════════════════════════════════════
build_x265() {
    info "Building patched x265 encoder + Neural ARF shim …"

    X265_DIR="${BUILD_DIR}/x265_git"

    if [ ! -d "${X265_DIR}" ]; then
        git clone --depth 1 \
            https://bitbucket.org/multicoreware/x265_git.git \
            "${X265_DIR}"
    else
        info "x265 source already present, skipping clone."
    fi

    # ── Apply x265 DPB patch ───────────────────────────────────────────────
    python3 - <<'PYEOF'
import re, pathlib, subprocess, sys
content = pathlib.Path("x265_dpb_patch.cpp").read_text()
diffs = re.findall(r'/\*-{10,}\n(.*?)\n-{10,}\*/', content, re.DOTALL)
for i, diff in enumerate(diffs):
    diff_path = f"/tmp/x265_neural_arf_{i}.diff"
    pathlib.Path(diff_path).write_text(diff)
    result = subprocess.run(
        ["patch", "-p1", "--dry-run", "-i", diff_path],
        cwd="x265_git", capture_output=True
    )
    if result.returncode == 0:
        subprocess.run(["patch", "-p1", "-i", diff_path], cwd="x265_git")
        print(f"Applied x265 diff {i}")
    else:
        print(f"x265 diff {i}: already applied or conflicts")
PYEOF

    # ── Build x265 ─────────────────────────────────────────────────────────
    cmake -S "${X265_DIR}/source" \
          -B "${X265_DIR}/build" \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_SHARED=ON \
          -DENABLE_NEURAL_ARF=ON 2>&1 | tail -5

    cmake --build "${X265_DIR}/build" -j"$(nproc)" 2>&1 | tail -10

    # ── Build Neural ARF shim (thin C wrapper around DPB::injectNeuralARF) ─
    cat > "${BUILD_DIR}/neural_arf_x265_shim.c" << 'SHIM_SRC'
/*
 * neural_arf_x265_shim.c
 * Exposes x265_neural_arf_inject() for ctypes calling from Python.
 * Compile as a shared library linked against libx265.
 */
#include <x265.h>
#include <string.h>
#include <stdio.h>

#define NEURAL_ARF_POC 0x7FFF

/* x265_encoder is an opaque struct; we access it through the public API.
 * The DPB injection is triggered via the x265_picture.neuralArf fields
 * that were added by the patch. We store the pending ARF data and inject
 * it via a pre-encode hook.                                               */

static char   g_pending_yuv_path[4096] = {0};
static int    g_pending_poc            = 0;
static int    g_pending_lt             = 0;

int x265_neural_arf_inject(void*       encoder,
                            const char* yuv_path,
                            int         poc,
                            int         is_long_term)
{
    if (!yuv_path || !encoder) return -1;
    strncpy(g_pending_yuv_path, yuv_path, sizeof(g_pending_yuv_path) - 1);
    g_pending_poc = poc;
    g_pending_lt  = is_long_term;
    fprintf(stderr, "[NeuralARF shim] queued: %s poc=%d lt=%d\n",
            yuv_path, poc, is_long_term);
    return 0;
}

/* Called by the patched x265 encoder before encoding each frame to pick
 * up the pending neural ARF data.                                        */
void x265_neural_arf_get_pending(char* path_out, int* poc_out, int* lt_out)
{
    strncpy(path_out, g_pending_yuv_path, 4095);
    *poc_out = g_pending_poc;
    *lt_out  = g_pending_lt;
    g_pending_yuv_path[0] = 0;   /* clear after read */
}
SHIM_SRC

    gcc -O2 -fPIC -shared \
        -I"${X265_DIR}/source" \
        -o "${LIBS_DIR}/libneural_arf_x265.so" \
        "${BUILD_DIR}/neural_arf_x265_shim.c" \
        -L"${X265_DIR}/build" -lx265 \
        -Wl,-rpath,"${LIBS_DIR}"

    info "libneural_arf_x265.so built → ${LIBS_DIR}/"
}


# ══════════════════════════════════════════════════════════════════════════
# 3. Generate neural_arf_interface.h standalone file
# ══════════════════════════════════════════════════════════════════════════
generate_header() {
    info "Generating standalone neural_arf_interface.h …"
    cat > "${LIBS_DIR}/neural_arf_interface.h" << 'HDR'
#ifndef NEURAL_ARF_INTERFACE_H
#define NEURAL_ARF_INTERFACE_H
#include <stdint.h>
#define NEURAL_ARF_POC  0x7FFF
typedef struct NeuralARFFrame {
    uint8_t *y_plane, *u_plane, *v_plane;
    int      width, height, poc, valid;
} NeuralARFFrame;
typedef int (*neural_arf_cb_t)(NeuralARFFrame*, void*);
extern neural_arf_cb_t g_neural_arf_callback;
extern void           *g_neural_arf_userdata;
void neural_arf_set_callback(neural_arf_cb_t cb, void *userdata);
int  neural_arf_read_from_env(NeuralARFFrame *frame);
void neural_arf_frame_free(NeuralARFFrame *frame);
#endif
HDR
    info "Header written to ${LIBS_DIR}/neural_arf_interface.h"
}


# ══════════════════════════════════════════════════════════════════════════
# 4. Entry point
# ══════════════════════════════════════════════════════════════════════════
cd "${SCRIPT_DIR}"
generate_header

case "${MODE}" in
    --openhevc-only) build_openhevc ;;
    --x265-only)     build_x265 ;;
    --all|*)         build_openhevc; build_x265 ;;
esac

info "════════════════════════════════════════"
info " Neural ARF build complete!"
info " Shared libs : ${LIBS_DIR}/"
info " Binaries    : ${BIN_DIR}/"
info ""
info " To use:"
info "   export LD_LIBRARY_PATH=${LIBS_DIR}:\$LD_LIBRARY_PATH"
info "   python train.py --mode encode_test \\"
info "       --openhevc-lib ${LIBS_DIR}/libopenhevc.so \\"
info "       --x265-lib     ${LIBS_DIR}/libneural_arf_x265.so"
info "════════════════════════════════════════"
