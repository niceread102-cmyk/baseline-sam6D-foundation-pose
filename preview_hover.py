"""
preview_hover.py — Move the arm to hover position and show a live camera feed.
Use this to centre objects in the bin before running the full pipeline.

    python preview_hover.py

Controls:
  s       — save current frame to scripts/aruco_markers/hover_preview_saved.png
  q / Esc — quit and return arm to home
"""
import time
from pathlib import Path

import numpy as np
import cv2
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ── Load config ───────────────────────────────────────────────────────────────
_CONF_DIR = str(Path(__file__).resolve().parent / "conf")
with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
    cfg = compose(config_name="config")

HOVER_POSITION = list(OmegaConf.to_container(cfg.arm.hover_position))
START_ANGLES   = list(OmegaConf.to_container(cfg.arm.start_angles))

print(f"hover_position : {HOVER_POSITION}")
print(f"start_angles   : {START_ANGLES}")

# ── Arm ───────────────────────────────────────────────────────────────────────
import hydra
ik  = hydra.utils.instantiate(cfg.ik)
arm = hydra.utils.instantiate(cfg.arm_controller, ik_solver=ik)

print("Homing…")
arm.move_to_angles(np.array(START_ANGLES), safe=False)
print("Moving to hover position…")
ok = arm.move_to_xyz(np.array(HOVER_POSITION))
if not ok:
    print(f"WARNING: IK failed for hover {HOVER_POSITION} — arm did not move.")
    print("  Update arm.hover_position in conf/config.yaml to a reachable position.")
    print("  Known reachable: [0.25,0,0.54], [0.30,0,0.50], [0.35,0,0.45]")
else:
    print(f"At hover {HOVER_POSITION}.")

# ── Camera — use the same RealSenseCamera as the binpicker ───────────────────
print("Opening camera…")
cam = hydra.utils.instantiate(cfg.camera)

# Grab one frame to confirm actual resolution
_rgb0, _, _K0 = cam.capture()
h0, w0 = _rgb0.shape[:2]
print(f"Camera ready: {w0}×{h0} px  "
      f"fx={_K0[0,0]:.1f}  fy={_K0[1,1]:.1f}  cx={_K0[0,2]:.1f}  cy={_K0[1,2]:.1f}")

# ── ArUco detector (optional overlay) ────────────────────────────────────────
import cv2.aruco as aruco
_aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
_detector   = aruco.ArucoDetector(_aruco_dict, aruco.DetectorParameters())

SAVE_PATH = Path(__file__).resolve().parent / "scripts" / "aruco_markers" / "hover_preview_saved.png"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Depth colourmap helper ────────────────────────────────────────────────────
def _depth_colormap(depth_mm: np.ndarray,
                    d_lo_mm: float = 100.0, d_hi_mm: float = 700.0) -> np.ndarray:
    """TURBO BGR colourmap clamped to [d_lo_mm, d_hi_mm].  Invalid pixels → grey."""
    d_norm = np.clip((depth_mm - d_lo_mm) / max(d_hi_mm - d_lo_mm, 1), 0, 1)
    d_bgr  = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    d_bgr[depth_mm <= 0] = (70, 70, 70)
    return d_bgr


# ── Live preview ─────────────────────────────────────────────────────────────
WIN = "Hover preview  [s=save  q=quit]   LEFT=RGB+ArUco  RIGHT=Depth (TURBO 100-700 mm)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 480)

print(f"Live feed running.  s=save frame to {SAVE_PATH}  q/Esc=quit.")
try:
    while True:
        result = cam.peek()
        if result is None:
            time.sleep(0.01)
            continue
        rgb, depth_mm, K = result

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        # ArUco overlay on RGB panel
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = _detector.detectMarkers(gray)
        if ids is not None:
            aruco.drawDetectedMarkers(bgr, corners, ids)
            label = f"Markers: {sorted(ids.flatten().tolist())}"
            cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        # Crosshair on RGB panel
        cx, cy = w // 2, h // 2
        cv2.line(bgr, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(bgr, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
        cv2.circle(bgr, (cx, cy), 30, (0, 255, 0), 1)

        # Depth panel — TURBO colourmap with validity stats overlay
        d_bgr  = _depth_colormap(depth_mm)
        valid  = int((depth_mm > 0).sum())
        pct    = 100.0 * valid / max(depth_mm.size, 1)
        center_d = float(depth_mm[cy, cx]) if depth_mm[cy, cx] > 0 else 0.0
        cv2.putText(d_bgr, f"Valid: {pct:.0f}%  center: {center_d:.0f} mm",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(d_bgr, "100 mm (purple) — 700 mm (red)",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        composite = np.concatenate([bgr, d_bgr], axis=1)
        cv2.imshow(WIN, composite)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(str(SAVE_PATH), composite)
            print(f"Saved {w*2}×{h} frame → {SAVE_PATH}")
        elif key in (ord('q'), 27):
            break
finally:
    cam.stop()
    cv2.destroyAllWindows()
    print("Returning to home…")
    arm.move_to_angles(np.array(START_ANGLES), safe=False)
    print("Done.")
