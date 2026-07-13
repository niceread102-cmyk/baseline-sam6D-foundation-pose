#!/home/joon/.virtualenvs/torch_obj_tracking/bin/python3
"""
scripts/calibrate_T_cam_base.py  —  Determine T_cam_base via 3D point correspondences.

Procedure
---------
1. (Once) Run with --generate to save printable ArUco PNG images.
   Print them at ≥ 50 mm × 50 mm per marker and tape flat in the workspace,
   spread across the area the arm will pick from.

2. For each marker, manually jog the arm EE tip to touch the marker center
   (hold the Lite6 manual button and drag, or jog via xArm Studio), then
   press ENTER.  The arm TCP position (base frame, metres) is recorded.

3. Move arm to hover → script captures RGBD → auto-detects all visible markers
   → uses depth to get their 3D positions in camera frame.

4. Kabsch algorithm fits the rigid transform → T_cam_base (4×4).

5. Prints the result and optionally updates arm.T_cam_base in conf/config.yaml.

Usage examples
--------------
  # Generate marker images to print
  python scripts/calibrate_T_cam_base.py --generate

  # Run calibration (hover auto-read from arm.hover_position in conf/config.yaml)
  # --tcp-offset is required — measure it from your end-effector's geometry
  # (flange -> physical tip that touches the markers), mm/deg.
  python scripts/calibrate_T_cam_base.py --tcp-offset 0 0 80 0 0 0

  # Fewer markers, custom hover
  python scripts/calibrate_T_cam_base.py --tcp-offset 0 0 80 0 0 0 \\
      --markers 4 --hover 0.30 0.0 0.50

  # Skip arm connection for camera-only dry-run (see what markers are visible)
  python scripts/calibrate_T_cam_base.py --dry-run
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup (must come before any local/venv imports) ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure packages from the project virtualenv are reachable regardless of which
# python3 binary was used to invoke this script (shebang is ignored when the
# interpreter is called explicitly as "python3 script.py").
_VENV_SITE = "/home/joon/.virtualenvs/torch_obj_tracking/lib/python3.12/site-packages"
if _VENV_SITE not in sys.path:
    sys.path.insert(1, _VENV_SITE)

import argparse
import time

import cv2
import numpy as np

from src.binpick.ik_solver import Lite6IKSolver, LITE6_INIT_THETA


ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
DEFAULT_CONFIG = PROJECT_ROOT / "conf" / "config.yaml"


# ── Marker generation ─────────────────────────────────────────────────────────

def generate_markers(output_dir: Path, n: int = 6, size_px: int = 600) -> None:
    """Save individual ArUco PNGs and a combined A4 PDF ready to print."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    MARKER_MM  = 70      # printed size per marker (including white quiet-zone border)
    DPI        = 200     # raster resolution embedded in the PDF
    A4_W_MM, A4_H_MM = 210.0, 297.0
    COLS       = 2
    ROWS       = (n + COLS - 1) // COLS

    def mm2px(mm: float) -> int:
        return round(mm / 25.4 * DPI)

    A4_W_PX   = mm2px(A4_W_MM)
    A4_H_PX   = mm2px(A4_H_MM)
    MARKER_PX = mm2px(MARKER_MM)

    # Layout: fixed top margin for title, even row gaps, bottom margin for note
    MARGIN_TOP_MM    = 18.0
    MARGIN_BOTTOM_MM = 14.0
    MARGIN_X_MM      = (A4_W_MM - COLS * MARKER_MM) / 2.0
    available_mm     = A4_H_MM - MARGIN_TOP_MM - MARGIN_BOTTOM_MM - ROWS * MARKER_MM
    gap_mm           = available_mm / (ROWS + 1)   # gaps above row 0, between rows, below last row

    # --- individual PNGs ---
    marker_imgs: list[np.ndarray] = []
    for i in range(n):
        img = cv2.aruco.generateImageMarker(d, i, size_px)
        # Keep a proportional quiet zone (white border) around the marker
        border = size_px // 8
        img = cv2.copyMakeBorder(img, border, border, border, border,
                                  cv2.BORDER_CONSTANT, value=255)
        path = output_dir / f"aruco_marker_{i:02d}.png"
        cv2.imwrite(str(path), img)
        marker_imgs.append(img)

    # --- A4 canvas (PIL, RGB) ---
    canvas = Image.new("RGB", (A4_W_PX, A4_H_PX), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    for idx, img_cv in enumerate(marker_imgs):
        row, col = divmod(idx, COLS)
        x_mm = MARGIN_X_MM + col * MARKER_MM
        y_mm = MARGIN_TOP_MM + (row + 1) * gap_mm + row * MARKER_MM
        x_px = mm2px(x_mm)
        y_px = mm2px(y_mm)

        # Resize to exact target pixel size with nearest-neighbour (no blur)
        img_pil = Image.fromarray(img_cv, mode="L").convert("RGB")
        img_pil = img_pil.resize((MARKER_PX, MARKER_PX), Image.NEAREST)
        canvas.paste(img_pil, (x_px, y_px))

        # Label below marker
        lx = x_px + MARKER_PX // 2
        ly = y_px + MARKER_PX + mm2px(2)
        draw.text((lx, ly), f"ID {idx}", fill=(0, 0, 0), anchor="mt")

    # Title
    draw.text((A4_W_PX // 2, mm2px(6)),
              "ArUco DICT_4X4_50  —  calibration markers",
              fill=(0, 0, 0), anchor="mt")
    # Bottom note
    draw.text((A4_W_PX // 2, A4_H_PX - mm2px(5)),
              f"Print at 100% / Actual Size (no fit-to-page).  "
              f"Each marker should measure {MARKER_MM} mm × {MARKER_MM} mm.",
              fill=(80, 80, 80), anchor="mb")

    pdf_path = output_dir / "aruco_markers_A4.pdf"
    canvas.save(str(pdf_path), "PDF", resolution=DPI)

    print(f"Saved {n} marker PNGs + A4 PDF → {pdf_path}")
    print(f"Print at 100 % / Actual Size (no fit-to-page).  "
          f"Each marker will be {MARKER_MM} mm × {MARKER_MM} mm.")


# ── Depth visualisation helper ────────────────────────────────────────────────

def _depth_colormap(depth_mm: np.ndarray,
                    d_lo_mm: float = 100.0, d_hi_mm: float = 700.0) -> np.ndarray:
    """TURBO BGR colourmap clamped to [d_lo_mm, d_hi_mm].  Invalid pixels → grey."""
    d_norm = np.clip((depth_mm - d_lo_mm) / max(d_hi_mm - d_lo_mm, 1), 0, 1)
    d_bgr  = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    d_bgr[depth_mm <= 0] = (70, 70, 70)
    return d_bgr


def log_depth_sanity(depth_mm: np.ndarray, label: str = "") -> None:
    """Print depth validity statistics to the console for later debugging."""
    total = depth_mm.size
    valid = int((depth_mm > 0).sum())
    pct   = 100.0 * valid / max(total, 1)
    tag   = f"[depth sanity{' ' + label if label else ''}]"
    if valid > 0:
        d_v = depth_mm[depth_mm > 0]
        print(f"  {tag}  {valid}/{total} valid px ({pct:.1f}%)  "
              f"range=[{d_v.min():.0f}, {d_v.max():.0f}] mm  "
              f"median={float(np.median(d_v)):.0f} mm")
        if pct < 50.0:
            print(f"  {tag}  WARNING: fewer than 50 % valid pixels — "
                  f"depth unreliable; check disparity_shift setting.")
    else:
        print(f"  {tag}  ALL ZERO ({total} pixels) — depth completely invalid!")
        print(f"  {tag}  Check: disparity_shift is 0? "
              f"camera distance ≤ 0.6 m at 640×480? USB bandwidth OK?")


# ── ArUco detection ───────────────────────────────────────────────────────────

def detect_markers_3d(
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
) -> dict[int, np.ndarray]:
    """Detect ArUco markers and return {marker_id → p_cam (metres)}."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx0, cy0 = float(K[0, 2]), float(K[1, 2])
    h, w = depth_mm.shape[:2]
    out: dict[int, np.ndarray] = {}

    for cset, mid in zip(corners, ids.flatten()):
        # Pixel coords of marker center
        u, v = cset[0].mean(axis=0)
        ui, vi = int(round(u)), int(round(v))
        if not (3 <= ui < w - 3 and 3 <= vi < h - 3):
            print(f"  Marker {mid}: center pixel ({ui},{vi}) is too close to image edge, skipping.")
            continue

        # Median depth in 7×7 patch — robust against holes / noise
        patch = depth_mm[vi - 3: vi + 4, ui - 3: ui + 4]
        valid = patch[patch > 10]    # exclude zero / invalid readings
        patch_total = patch.size
        patch_valid = len(valid)
        if patch_valid < 5:
            # Log the actual depth values to help diagnose the problem
            print(f"  Marker {mid}: pixel=({ui},{vi})  "
                  f"only {patch_valid}/{patch_total} valid depth pixels at center, SKIPPING.")
            print(f"    patch values (mm): {patch.flatten().tolist()}")
            if patch.max() <= 10:
                print(f"    → All-zero patch. Likely disparity_shift too high for this distance, "
                      f"or object in blind-spot of stereo pair.")
            continue

        D = float(np.median(valid)) / 1000.0   # mm → m
        x_cam = (u - cx0) * D / fx
        y_cam = (v - cy0) * D / fy
        out[int(mid)] = np.array([x_cam, y_cam, D])
        print(f"  Marker {mid}: pixel=({ui},{vi})  depth={D*1000:.1f}mm  "
              f"patch={patch_valid}/{patch_total} valid  "
              f"p_cam=[{x_cam:.4f}, {y_cam:.4f}, {D:.4f}]m")

    return out


def visualise_detections(rgb: np.ndarray, depth_mm: np.ndarray, K: np.ndarray,
                         save_path: Path | None = None) -> None:
    """Save an annotated detection image (no display required)."""
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(d, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        for cset, mid in zip(corners, ids.flatten()):
            u, v = cset[0].mean(axis=0)
            cv2.putText(vis, str(mid), (int(u) - 5, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "No markers detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    if save_path is not None:
        cv2.imwrite(str(save_path), vis)
        print(f"  Visualisation saved → {save_path}")


# ── Kabsch rigid-body fit ─────────────────────────────────────────────────────

def kabsch(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Fit T_cam_base (4×4) such that p_base ≈ T_cam_base @ [p_cam; 1].

    src_pts : (N, 3)  points in camera frame
    dst_pts : (N, 3)  same points in robot-base frame
    Returns  : (4, 4) homogeneous transform
    """
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    sc = src.mean(axis=0)
    dc = dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Correct for reflection
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dc - R @ sc
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def residuals_mm(T: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Per-point base-frame residual in millimetres."""
    errs = []
    for p_cam, p_base in zip(src_pts, dst_pts):
        p_pred = (T @ np.append(p_cam, 1.0))[:3]
        errs.append(np.linalg.norm(p_pred - p_base) * 1000.0)
    return np.array(errs)


def check_touch_height(
    touch_points: dict[int, np.ndarray], expected_z: float | None, tol: float = 0.02
) -> None:
    """Warn if the mean touch-point Z is far from a known reference height.

    The existing per-marker spread check only catches touch points that are
    INCONSISTENT with each other — a wrong/unset TCP offset instead produces a
    roughly CONSTANT bias (the flange-to-tip distance) across every marker, so
    the spread check passes while every point is off by the same amount. This
    catches that case by comparing against an absolute reference (e.g. 0 for a
    flat table the arm is mounted flush with).
    """
    if expected_z is None or not touch_points:
        return
    z_vals = np.array([p[2] for p in touch_points.values()])
    mean_z = float(z_vals.mean())
    dz = mean_z - expected_z
    print(f"Touch-point mean Z vs expected: {mean_z*1000:.1f} mm vs {expected_z*1000:.1f} mm "
          f"(Δ={dz*1000:+.1f} mm)")
    if abs(dz) > tol:
        print(f"  WARNING: mean touch Z is {dz*1000:+.1f} mm from the expected "
              f"{expected_z*1000:.1f} mm.")
        print(f"  This usually means --tcp-offset doesn't match the physical end-effector "
              f"(get_position() is reporting the wrong point) — check it before trusting "
              f"this calibration.")


# ── Config update ─────────────────────────────────────────────────────────────

def format_T_for_yaml(T: np.ndarray, key: str = "T_cam_ee") -> str:
    """Format a 4×4 matrix as a YAML block matching the existing config style."""
    rows = []
    for row in T.tolist():
        vals = ", ".join(f"{v:.8f}" for v in row)
        rows.append(f"    - [{vals}]")   # 4-space indent: unambiguously nested under key
    return f"  {key}:\n" + "\n".join(rows)


def update_config(config_path: Path, T: np.ndarray) -> None:
    """Replace arm.T_cam_ee in the YAML config file."""
    import re
    text = config_path.read_text()
    new_block = format_T_for_yaml(T, key="T_cam_ee")
    # Match "  T_cam_ee: null" or a full block, up to the next same-level key
    pattern = r"  T_cam_ee:.*?(?=\n  [a-zA-Z#]|\n[a-zA-Z#]|\Z)"
    if not re.search(pattern, text, flags=re.DOTALL):
        print(f"  Could not locate arm.T_cam_ee in {config_path} — add it manually.")
        return
    text_new = re.sub(pattern, new_block, text, flags=re.DOTALL)
    config_path.write_text(text_new)
    print(f"  Updated arm.T_cam_ee in {config_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate", action="store_true",
                   help="Generate ArUco marker PNG images and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Connect to camera only (no arm), show detected markers.")
    p.add_argument("--markers", type=int, default=6, metavar="N",
                   help="Number of markers to use (default: 6, min: 3).")
    p.add_argument("--hover", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   default=None,
                   help="Operational hover position in metres (default: read arm.hover_position "
                        "from conf/config.yaml).  The arm is commanded here via IK so calibration "
                        "and runtime FK cancel exactly.  MUST match arm.hover_position in config.")
    p.add_argument("--robot-ip", default="192.168.1.150",
                   help="xArm IP address (default: 192.168.1.150).")
    p.add_argument("--tcp-offset", type=float, nargs=6, default=None,
                   metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                   help="Required. TCP offset from the flange to the physical point that "
                        "touches markers (mm for X/Y/Z, degrees for ROLL/PITCH/YAW), measured "
                        "from your end-effector's geometry. get_position() reports this exact "
                        "point, so a wrong/unset value silently corrupts every touch point. "
                        "Pass 0 0 0 0 0 0 to explicitly calibrate in the flange frame.")
    p.add_argument("--expected-touch-z", type=float, default=None, metavar="Z_M",
                   help="Expected touch-point Z in robot-base metres (e.g. 0.0 if the arm is "
                        "mounted flush with the table the markers sit on). If given, warns when "
                        "the mean recorded touch Z deviates by more than --touch-z-tol.")
    p.add_argument("--touch-z-tol", type=float, default=0.02, metavar="M",
                   help="Tolerance in metres for the --expected-touch-z check (default 0.02).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"Config YAML to update (default: {DEFAULT_CONFIG}).")
    p.add_argument("--marker-dir", type=Path,
                   default=PROJECT_ROOT / "scripts" / "aruco_markers",
                   help="Directory for generated marker images.")
    return p.parse_args()


def connect_arm(robot_ip: str, tcp_offset: list[float] | None = None):
    """Connect to the xArm and set its TCP offset before any position is read.

    get_position() reports the SDK's get_tcp_pose(), i.e. whichever physical
    point the controller's currently-configured TCP offset defines — not a
    fixed flange position. If that offset doesn't match the real end-effector
    (or is left at its previous/default value), every "touch" point recorded
    downstream is silently shifted by the flange-to-tip distance. tcp_offset
    is therefore required here rather than left to whatever the controller
    already has stored: pass the measured [x, y, z, roll, pitch, yaw] (mm,
    deg) from the flange to the physical point that will touch markers/
    objects, or [0, 0, 0, 0, 0, 0] to explicitly calibrate in the flange frame.
    """
    from xarm import XArmAPI
    print(f"Connecting to xArm at {robot_ip}...")
    arm = XArmAPI(port=robot_ip, is_radian=True)
    arm.connect()
    if not arm.connected:
        raise RuntimeError(f"Failed to connect to xArm at {robot_ip}")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    print("  Connected.")

    time.sleep(0.3)  # let the real-time report stream populate tcp_offset
    stored = arm.tcp_offset
    stored_mm_deg = [round(v, 3) for v in list(stored[:3]) + list(np.degrees(stored[3:6]))]
    print(f"  Controller-stored TCP offset (before this run): {stored_mm_deg}  "
          f"[x,y,z mm, roll,pitch,yaw deg]")

    if tcp_offset is None:
        raise RuntimeError(
            "No --tcp-offset given. get_position() reports whatever point the TCP "
            "offset defines, so an unset/wrong offset silently corrupts every touch "
            "point recorded during calibration. Pass --tcp-offset X Y Z ROLL PITCH YAW "
            "(mm, deg) measured from the flange to the physical point that touches "
            "markers/objects, or --tcp-offset 0 0 0 0 0 0 to explicitly confirm "
            f"flange-referenced calibration. (Currently stored: {stored_mm_deg})"
        )

    code = arm.set_tcp_offset(list(tcp_offset), is_radian=False, wait=True)
    if isinstance(code, tuple):
        code = code[0]
    if code not in (0, None):
        raise RuntimeError(f"set_tcp_offset failed (code={code})")
    print(f"  TCP offset set to: {list(tcp_offset)}  [x,y,z mm, roll,pitch,yaw deg] "
          f"— get_position() now reports this point.")

    return arm


def move_to_operational_hover(arm, cam, hover_xyz: np.ndarray, preview_dir: Path):
    """Command arm to the OPERATIONAL hover via IK, capture the calibration image there.

    This is critical: T_cam_ee = inv(FK(hover_theta)) @ T_cam_base.  At runtime the
    binpicker ALSO moves to hover_xyz via the same IK, producing the same hover_theta.
    The FK pair cancels exactly, so any small FK model errors drop out.  Calibrating
    from a DIFFERENT hover (e.g. a manually dragged position) breaks this cancellation
    and leaves a systematic ~2-3° rotation error → ~10 mm lateral miss at pick depth.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nMoving arm to operational hover {hover_xyz.tolist()} m via IK...")

    ik = Lite6IKSolver()
    theta_seed = np.array([0.0, 0.7, 1.57, 0.0, 0.87, 0.0])
    theta_hover, ok = ik.solve(hover_xyz, theta_seed)
    if not ok:
        raise RuntimeError(
            f"IK failed for hover position {hover_xyz.tolist()}.\n"
            f"  This position is outside the Lite6 IK solver's reachable workspace.\n"
            f"  Update arm.hover_position in conf/config.yaml to a reachable point.\n"
            f"  Known reachable examples: [0.30, 0.0, 0.50], [0.35, 0.0, 0.45], [0.35, 0.0, 0.30]"
        )

    code = arm.set_servo_angle(angle=theta_hover.tolist(), is_radian=True,
                               speed=0.3, mvacc=5.0, wait=True)
    if isinstance(code, tuple):
        code = code[0]
    if code not in (0, None):
        raise RuntimeError(f"set_servo_angle failed (code={code})")
    time.sleep(0.5)  # let the arm settle

    # Read actual joint angles the arm achieved
    code, angles = arm.get_servo_angle(is_radian=True)
    if code != 0 or angles is None:
        raise RuntimeError(f"get_servo_angle failed (code={code})")
    actual_theta = np.array(angles[:6])

    code2, pos = arm.get_position(is_radian=True)
    if code2 == 0 and pos is not None:
        print(f"  EE position:    [{pos[0]/1000:.4f}, {pos[1]/1000:.4f}, {pos[2]/1000:.4f}] m")
    print(f"  Joint angles:   {np.round(actual_theta, 5).tolist()}")
    print(f"  IK-commanded:   {np.round(theta_hover, 5).tolist()}")
    print(f"  Max joint Δ:    {abs(actual_theta - theta_hover).max():.4f} rad")

    # Live preview: stream frames with ArUco detection until user confirms
    print("\n  Live preview window open — LEFT: RGB+ArUco  RIGHT: Depth (TURBO 100-700 mm)")
    print("  Press SPACE or ENTER to capture this frame, Q to abort.\n")
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    rgb = depth_mm = K = None

    WIN = "Calibration Hover Preview [SPACE=capture  Q=abort]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 480)

    while True:
        rgb, depth_mm, K = cam.capture()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # Left panel: RGB + ArUco
        vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_bgr, corners, ids)
            label = f"Detected IDs: {sorted(ids.flatten().tolist())}"
            color = (0, 200, 0)
        else:
            label = "No markers detected"
            color = (0, 0, 220)
        cv2.putText(vis_bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Right panel: depth colourmap + stats
        d_bgr = _depth_colormap(depth_mm)
        valid_px  = int((depth_mm > 0).sum())
        valid_pct = 100.0 * valid_px / max(depth_mm.size, 1)
        h_px, w_px = depth_mm.shape[:2]
        center_d = float(depth_mm[h_px // 2, w_px // 2])
        cv2.putText(d_bgr, f"Valid: {valid_pct:.0f}%  center: {center_d:.0f} mm",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(d_bgr, "100 mm (purple)  700 mm (red)",
                    (12, h_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        composite = np.concatenate([vis_bgr, d_bgr], axis=1)
        cv2.imshow(WIN, composite)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord(' '), 13):   # SPACE or ENTER
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise RuntimeError("User aborted hover preview.")

    cv2.destroyAllWindows()

    if ids is not None:
        print(f"  Captured frame — detected {len(ids)} marker(s): "
              f"IDs {sorted(ids.flatten().tolist())}")
    else:
        print("  WARNING: no markers detected in captured frame.")

    print(f"  Camera K: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    log_depth_sanity(depth_mm, "full frame")

    preview_path = preview_dir / "operational_hover_detection.png"
    cv2.imwrite(str(preview_path), vis_bgr)
    print(f"  Detection image saved → {preview_path}")

    depth_path = preview_dir / "operational_hover_depth.png"
    cv2.imwrite(str(depth_path), _depth_colormap(depth_mm))
    print(f"  Depth image saved     → {depth_path}")

    return actual_theta, rgb, depth_mm, K


def collect_touch_points(arm, n_markers: int) -> dict[int, np.ndarray]:
    """Interactively collect EE-touch base-frame positions for each marker."""
    print(f"\n{'='*60}")
    print("STEP 1: Touch each ArUco marker with the EE tip")
    print("  - Hold the Lite6 manual button (back of wrist) to drag the arm")
    print("  - OR jog using xArm Studio")
    print("  - Press ENTER once EE tip is centred on the marker")
    print("  - Type SKIP to skip a marker, DONE when ≥ 3 are recorded")
    print(f"{'='*60}\n")

    points: dict[int, np.ndarray] = {}

    for i in range(n_markers):
        while True:
            cmd = input(f"Position EE on marker #{i} then press ENTER (or SKIP / DONE): ").strip().lower()
            if cmd == "done":
                if len(points) >= 3:
                    return points
                print(f"  Need at least 3 points, have {len(points)}. Continue.")
                continue
            if cmd == "skip":
                print(f"  Skipping marker {i}.")
                break

            code, pos = arm.get_position(is_radian=True)
            if code != 0:
                print(f"  ERROR: get_position failed (code={code}). Try again.")
                continue

            # pos = [x, y, z, rx, ry, rz] in mm
            p_base = np.array(pos[:3]) / 1000.0   # mm → m
            points[i] = p_base
            print(f"  Recorded marker #{i}: p_base = [{p_base[0]:.4f}, {p_base[1]:.4f}, {p_base[2]:.4f}] m")
            break

    return points


def main() -> None:
    args = parse_args()

    # ── Generate markers and exit ─────────────────────────────────────────────
    if args.generate:
        generate_markers(args.marker_dir, n=args.markers)
        print(f"\nPrint and place {args.markers} markers in the workspace, then re-run without --generate.")
        return

    # ── Connect to camera ─────────────────────────────────────────────────────
    print("Connecting to RealSense camera...")
    from src.binpick.camera import RealSenseCamera
    import yaml as _yaml

    # Read camera params from the binpick config so width/fps/exposure stay in sync.
    # Override disparity_shift=0: at hover height the camera-to-marker distance is
    # ~0.4–0.6 m, and disparity_shift=50 clips the measurable range at ~0.60 m
    # (640×480, fx≈605, B≈50 mm), producing all-zero depth.
    _cfg_data  = _yaml.safe_load(args.config.read_text())
    _cam_yaml  = {k: v for k, v in _cfg_data.get("camera", {}).items()
                  if not k.startswith("_")}   # strip Hydra _target_ key
    _orig_ds   = _cam_yaml.get("disparity_shift", 50)
    _cam_yaml["disparity_shift"] = 0
    print(f"  Camera config from {args.config.name}: "
          f"{_cam_yaml.get('width',640)}×{_cam_yaml.get('height',480)} "
          f"fps={_cam_yaml.get('fps',30)} "
          f"depth_exposure={_cam_yaml.get('depth_exposure_us',6000)} µs "
          f"laser={_cam_yaml.get('laser_power_frac',0.5)}")
    if _orig_ds != 0:
        print(f"  disparity_shift: config={_orig_ds} → overridden to 0 "
              f"(required for hover-distance depth; binpick config also updated to 0)")
    cam = RealSenseCamera(**_cam_yaml)
    print("  Camera ready.")

    # ── Dry-run: live preview to verify marker placement ─────────────────────
    if args.dry_run:
        print("\nDRY-RUN — live camera preview with ArUco detection.")
        print("Adjust marker placement as needed.  SPACE/ENTER to exit, Q to quit.\n")
        d_dry = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        det_dry = cv2.aruco.ArucoDetector(d_dry, cv2.aruco.DetectorParameters())
        WIN_DRY = "Dry-run Preview [SPACE=exit  Q=quit]   LEFT=RGB+ArUco  RIGHT=Depth"
        cv2.namedWindow(WIN_DRY, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_DRY, 1280, 480)
        last_rgb = last_depth = last_K = None
        while True:
            last_rgb, last_depth, last_K = cam.capture()
            gray_dry = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2GRAY)
            corners_dry, ids_dry, _ = det_dry.detectMarkers(gray_dry)
            vis_dry = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
            if ids_dry is not None:
                cv2.aruco.drawDetectedMarkers(vis_dry, corners_dry, ids_dry)
                lbl = f"Detected IDs: {sorted(ids_dry.flatten().tolist())}"
                cv2.putText(vis_dry, lbl, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
            else:
                cv2.putText(vis_dry, "No markers detected", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2)
            d_bgr_dry = _depth_colormap(last_depth)
            valid_pct = 100.0 * int((last_depth > 0).sum()) / max(last_depth.size, 1)
            cv2.putText(d_bgr_dry, f"Valid depth: {valid_pct:.0f}%",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(WIN_DRY, np.concatenate([vis_dry, d_bgr_dry], axis=1))
            k = cv2.waitKey(30) & 0xFF
            if k in (ord(' '), 13):
                break
            if k == ord('q'):
                cv2.destroyAllWindows()
                cam.stop()
                return
        cv2.destroyAllWindows()
        if last_K is not None:
            print(f"Camera intrinsics: fx={last_K[0,0]:.1f} fy={last_K[1,1]:.1f} "
                  f"cx={last_K[0,2]:.1f} cy={last_K[1,2]:.1f}")
        if last_depth is not None:
            log_depth_sanity(last_depth, "dry-run frame")
        dry_path = args.marker_dir / "dry_run_detection.png"
        args.marker_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dry_path), vis_dry)
        print(f"Snapshot saved → {dry_path}")
        cam.stop()
        return

    # ── Connect to arm ────────────────────────────────────────────────────────
    arm = connect_arm(args.robot_ip, tcp_offset=args.tcp_offset)

    try:
        # ── STEP 1: Touch each marker ─────────────────────────────────────────
        touch_points = collect_touch_points(arm, args.markers)
        print(f"\nCollected {len(touch_points)} touch points: markers {sorted(touch_points.keys())}")
        check_touch_height(touch_points, args.expected_touch_z, args.touch_z_tol)

        if len(touch_points) < 3:
            print("ERROR: Need at least 3 touch points. Exiting.")
            return

        # ── STEP 2: Move to OPERATIONAL hover via IK ─────────────────────────
        # The arm is commanded to exactly the hover position the binpicker uses.
        # This ensures calibration FK and runtime FK are identical → T_cam_ee is
        # correct regardless of FK model imperfections.
        if args.hover is not None:
            hover_xyz = np.array(args.hover)
        else:
            # Auto-read from config so calibration and runtime always stay in sync
            import yaml
            cfg = yaml.safe_load(args.config.read_text())
            hover_xyz = np.array(cfg["arm"]["hover_position"], dtype=float)
            print(f"  (hover not specified — using arm.hover_position from config: "
                  f"{hover_xyz.tolist()})")

        print(f"\n{'='*60}")
        print("STEP 2: Moving arm to OPERATIONAL hover via IK")
        print(f"  Target: {hover_xyz.tolist()} m  (must match arm.hover_position in config)")
        print(f"{'='*60}")
        preview_dir = PROJECT_ROOT / "scripts" / "aruco_markers" / "hover_previews"
        hover_theta, rgb, depth_mm, K = move_to_operational_hover(arm, cam, hover_xyz, preview_dir)

        # ── STEP 3: Detect markers in the operational hover frame ─────────────
        print(f"\n{'='*60}")
        print("STEP 3: Detecting markers in operational hover frame")
        print(f"{'='*60}")

        log_depth_sanity(depth_mm, "hover frame")
        print("\nDetected markers (camera frame):")
        detected = detect_markers_3d(rgb, depth_mm, K)
        print(f"  → Detected IDs: {sorted(detected.keys())}")

        visualise_detections(rgb, depth_mm, K, save_path=preview_dir / "hover_final_detection.png")

        # ── STEP 4: Match and solve ───────────────────────────────────────────
        common = sorted(set(touch_points) & set(detected))
        print(f"\nMatched {len(common)} markers: {common}")

        if len(common) < 3:
            print("ERROR: Need ≥ 3 matched markers. Make sure markers are visible from hover.")
            return

        src_pts = np.array([detected[i] for i in common])       # camera frame
        dst_pts = np.array([touch_points[i] for i in common])   # base frame

        T_cam_base = kabsch(src_pts, dst_pts)

        # ── Derive T_cam_ee = inv(FK(hover_theta)) @ T_cam_base ──────────────
        # T_cam_base = FK(θ) @ T_cam_ee  →  T_cam_ee = inv(FK(θ)) @ T_cam_base
        # hover_theta is the OPERATIONAL hover joints (from IK in Step 2).
        # At runtime: T_cam_base_rt = FK(hover_theta_rt) @ T_cam_ee
        #           = FK(hover_theta_rt) @ inv(FK(hover_theta_calib)) @ T_cam_base_kabsch
        # If hover_theta_rt ≈ hover_theta_calib (same commanded hover), this equals
        # T_cam_base_kabsch.  Any FK model error cancels out.
        ik = Lite6IKSolver()
        T_ee_base = ik.fk_matrix(hover_theta)
        T_cam_ee  = np.linalg.inv(T_ee_base) @ T_cam_base

        # ── Results ───────────────────────────────────────────────────────────
        errs = residuals_mm(T_cam_base, src_pts, dst_pts)
        print(f"\n{'='*60}")
        print("CALIBRATION RESULT")
        print(f"{'='*60}")
        print(f"Markers used: {common}")
        print(f"Residuals:  mean={np.mean(errs):.1f} mm   max={np.max(errs):.1f} mm\n")

        # Per-marker residuals table
        print(f"  {'ID':>4}  {'depth_mm':>8}  {'p_cam (m)':>28}  "
              f"{'p_base_touch (m)':>28}  {'p_base_pred (m)':>28}  {'err_mm':>7}")
        for i, mid in enumerate(common):
            pc = src_pts[i]
            pb_touch = dst_pts[i]
            pb_pred  = (T_cam_base @ np.append(pc, 1.0))[:3]
            cam_z_to_base = T_cam_base[:3, :3] @ pc + T_cam_base[:3, 3]
            print(f"  {mid:>4}  {pc[2]*1000:>8.1f}  "
                  f"[{pc[0]:+.4f},{pc[1]:+.4f},{pc[2]:+.4f}]  "
                  f"[{pb_touch[0]:+.4f},{pb_touch[1]:+.4f},{pb_touch[2]:+.4f}]  "
                  f"[{pb_pred[0]:+.4f},{pb_pred[1]:+.4f},{pb_pred[2]:+.4f}]  "
                  f"{errs[i]:>7.1f}")
        print()
        print(f"\nT_cam_base (at hover)  =\n{np.round(T_cam_base, 6)}")
        print(f"\nT_cam_ee (fixed, store in config) =\n{np.round(T_cam_ee, 6)}")

        # ── Sanity checks ─────────────────────────────────────────────────────
        # Touch-point Z spread: all markers should lie on the same flat surface
        z_touch = dst_pts[:, 2]
        z_spread = z_touch.max() - z_touch.min()
        print(f"Touch-point Z: mean={z_touch.mean()*1000:.1f} mm  "
              f"spread={z_spread*1000:.1f} mm  "
              f"(expect < 5 mm for flat table; large spread = EE tip not on surface)")
        if z_spread > 0.010:
            print(f"  WARNING: Z spread {z_spread*1000:.1f} mm > 10 mm — "
                  f"some touch points may not have been on the marker surface.")

        # Camera-frame depth vs base-frame Z consistency
        for i, mid in enumerate(common):
            pc = src_pts[i]
            pb_touch = dst_pts[i]
            pb_pred  = (T_cam_base @ np.append(pc, 1.0))[:3]
            dz = (pb_pred[2] - pb_touch[2]) * 1000
            if abs(dz) > 10:
                print(f"  Marker {mid}: Z error {dz:+.1f} mm "
                      f"(predicted {pb_pred[2]*1000:.1f} mm vs touch {pb_touch[2]*1000:.1f} mm) "
                      f"— may indicate depth-scale or T_cam_ee Z-translation error")

        # Camera optical axis at hover
        cam_z_base = T_cam_base[:3, 2]
        dominant = int(np.argmax(np.abs(cam_z_base)))
        axes = ["base-X (forward)", "base-Y (lateral)", "base-Z (vertical)"]
        sign = "+" if cam_z_base[dominant] > 0 else "-"
        print(f"\nCamera optical axis at hover: [{cam_z_base[0]:.3f}, {cam_z_base[1]:.3f}, {cam_z_base[2]:.3f}]")
        print(f"  → Camera looking toward {sign}{axes[dominant]}")

        if np.mean(errs) > 15.0:
            print("\nWARNING: Mean residual > 15 mm — calibration may be inaccurate.")
            print("  Tips: use more markers, spread them further apart, ensure EE tip")
            print("  is precisely on the marker centre when recording touch points.")

        # Lateral spread check — a key source of rotation errors
        if len(src_pts) >= 3:
            lateral_range = src_pts[:, 0].max() - src_pts[:, 0].min()  # cam-X spread
            if lateral_range < 0.10:
                print(f"\nWARNING: Markers only span {lateral_range*1000:.0f} mm in camera-X "
                      f"(< 100 mm).")
                print("  This under-constrains the cam-Z rotation and causes systematic")
                print("  lateral errors proportional to object offset from center.")
                print("  Place markers across the FULL image width (left edge, center, right).")

        # Print YAML block for manual copy-paste
        print(f"\nYAML (copy this into conf/config.yaml → arm.T_cam_ee):\n")
        print(format_T_for_yaml(T_cam_ee))

        # ── Save ──────────────────────────────────────────────────────────────
        ans = input(f"\nUpdate arm.T_cam_ee in {args.config}? [y/N] ").strip().lower()
        if ans == "y":
            update_config(args.config, T_cam_ee)
        else:
            print("Not saved — copy the YAML block above manually.")

    finally:
        arm.disconnect()
        cam.stop()


if __name__ == "__main__":
    main()
