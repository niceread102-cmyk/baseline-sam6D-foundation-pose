#!/home/joon/.virtualenvs/cnos2/bin/python3
"""
scripts/calibrate_T_cam_base_phoxi.py — Determine T_cam_base for the fixed-mount PhoXi.

Unlike the RealSense wrist camera (scripts/calibrate_T_cam_base.py), the Photoneo
PhoXi is rigidly mounted in the workspace, not on the arm. There is no per-capture
FK recompute and no T_cam_ee to derive — the Kabsch fit *is* the final T_cam_base,
and it never changes regardless of arm pose.

Procedure
---------
1. (Once) Run with --generate to save printable ArUco PNG images.
   Print them at ≥ 50 mm × 50 mm per marker and tape flat in the workspace,
   spread across the area the arm will pick from.

2. For each marker, manually jog the arm EE tip to touch the marker center
   (hold the Lite6 manual button and drag, or jog via xArm Studio), then
   press ENTER. The arm TCP position (base frame, metres) is recorded.

3. The arm is moved to arm.start_angles (clear of the fixed camera's view),
   then the PhoXi triggers a scan → auto-detects all visible markers → uses
   depth to get their 3D positions in camera frame.

4. Kabsch algorithm fits the rigid transform → T_cam_base (4×4), directly.

5. Prints the result and optionally updates camera.T_cam_base in
   conf/camera/phoxi.yaml.

Usage examples
--------------
  # Generate marker images to print (shared with the RealSense script)
  python scripts/calibrate_T_cam_base_phoxi.py --generate

  # Run calibration
  python scripts/calibrate_T_cam_base_phoxi.py

  # Fewer markers, explicit PhoXi device / robot IP
  python scripts/calibrate_T_cam_base_phoxi.py --markers 4 --device-id 2019-11-011-LC3 \\
      --robot-ip 192.168.1.150

  # Skip arm connection for camera-only dry-run (see what markers are visible)
  python scripts/calibrate_T_cam_base_phoxi.py --dry-run
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
_VENV_SITE = "/home/joon/.virtualenvs/cnos2/lib/python3.12/site-packages"
if _VENV_SITE not in sys.path:
    sys.path.insert(1, _VENV_SITE)

import argparse
import time

import cv2
import numpy as np

# Shared ArUco marker generation / detection / Kabsch-fit logic — lives in the
# RealSense calibration script since none of it is camera-specific.
import calibrate_T_cam_base as shared

DEFAULT_CONFIG        = PROJECT_ROOT / "conf" / "config.yaml"
DEFAULT_CAMERA_CONFIG = PROJECT_ROOT / "conf" / "camera" / "phoxi.yaml"


# ── PhoXi capture + live preview ──────────────────────────────────────────────

def preview_and_capture(cam, preview_dir: Path):
    """Trigger PhoXi scans until the user accepts one; return (rgb, depth_mm, K).

    The PhoXi is a trigger-per-frame structured-light scanner, not a streaming
    video camera, so each retry re-triggers a fresh scan rather than reading a
    live feed.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    print("\n  SPACE/ENTER: accept this frame   R: retrigger scan   Q: abort\n")

    d = cv2.aruco.getPredefinedDictionary(shared.ARUCO_DICT_ID)
    detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())

    WIN = "PhoXi Capture Preview [SPACE=accept  R=retrigger  Q=abort]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 480)

    try:
        while True:
            print("  Triggering scan...")
            rgb, depth_mm, K = cam.capture()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(vis_bgr, corners, ids)
                label = f"Detected IDs: {sorted(ids.flatten().tolist())}"
                color = (0, 200, 0)
            else:
                label = "No markers detected"
                color = (0, 0, 220)
            cv2.putText(vis_bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            d_bgr = shared._depth_colormap(depth_mm)
            valid_pct = 100.0 * int((depth_mm > 0).sum()) / max(depth_mm.size, 1)
            cv2.putText(d_bgr, f"Valid depth: {valid_pct:.0f}%",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(WIN, np.concatenate([vis_bgr, d_bgr], axis=1))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord(' '), 13):   # SPACE or ENTER
                break
            if key == ord('r'):
                continue
            if key == ord('q'):
                raise RuntimeError("User aborted PhoXi capture.")
    finally:
        cv2.destroyAllWindows()

    print(f"  Camera K: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    shared.log_depth_sanity(depth_mm, "phoxi frame")

    preview_path = preview_dir / "phoxi_detection.png"
    cv2.imwrite(str(preview_path), vis_bgr)
    print(f"  Detection image saved → {preview_path}")

    depth_path = preview_dir / "phoxi_depth.png"
    cv2.imwrite(str(depth_path), d_bgr)
    print(f"  Depth image saved     → {depth_path}")

    return rgb, depth_mm, K


# ── Config update (camera.T_cam_base is top-level/flat, unlike arm.T_cam_ee) ──

def format_T_cam_base_yaml(T: np.ndarray) -> str:
    rows = []
    for row in T.tolist():
        vals = ", ".join(f"{v:.8f}" for v in row)
        rows.append(f"  - [{vals}]")
    return "T_cam_base:\n" + "\n".join(rows)


def update_camera_config(config_path: Path, T: np.ndarray) -> None:
    import re
    text = config_path.read_text()
    new_block = format_T_cam_base_yaml(T)
    pattern = r"^T_cam_base:.*?(?=\n[a-zA-Z#]|\Z)"
    if not re.search(pattern, text, flags=re.DOTALL | re.MULTILINE):
        print(f"  Could not locate T_cam_base in {config_path} — add it manually.")
        return
    text_new = re.sub(pattern, new_block, text, count=1, flags=re.DOTALL | re.MULTILINE)
    config_path.write_text(text_new)
    print(f"  Updated T_cam_base in {config_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generate", action="store_true",
                   help="Generate ArUco marker PNG images and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Connect to the PhoXi only (no arm), show detected markers.")
    p.add_argument("--markers", type=int, default=6, metavar="N",
                   help="Number of markers to use (default: 6, min: 3).")
    p.add_argument("--robot-ip", default="192.168.1.150",
                   help="xArm IP address (default: 192.168.1.150).")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"Top-level config, used to read arm.start_angles — the arm is "
                        f"parked there before capture so it doesn't occlude the fixed "
                        f"camera's view (default: {DEFAULT_CONFIG}).")
    p.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG,
                   help=f"Camera config YAML to update with T_cam_base "
                        f"(default: {DEFAULT_CAMERA_CONFIG}).")
    p.add_argument("--marker-dir", type=Path,
                   default=PROJECT_ROOT / "scripts" / "aruco_markers",
                   help="Directory for generated marker images (shared with the "
                        "RealSense script).")
    p.add_argument("--device-id", default=None,
                   help="PhoXi device id (default: auto-select first discovered PhoXi3DScan).")
    p.add_argument("--address", default=None,
                   help="PhoXi direct-connection address, if not auto-discovered.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Generate markers and exit ─────────────────────────────────────────────
    if args.generate:
        shared.generate_markers(args.marker_dir, n=args.markers)
        print(f"\nPrint and place {args.markers} markers in the workspace, then re-run without --generate.")
        return

    # ── Connect to PhoXi camera ────────────────────────────────────────────────
    print("Connecting to PhoXi camera...")
    from src.binpick.camera_phoxi import PhoXiCamera
    cam = PhoXiCamera(device_id=args.device_id, address=args.address)
    print("  Camera ready.")
    print(f"  Camera K: fx={cam.K[0,0]:.1f}  fy={cam.K[1,1]:.1f}  "
          f"cx={cam.K[0,2]:.1f}  cy={cam.K[1,2]:.1f}")

    preview_dir = args.marker_dir / "phoxi_previews"

    # ── Dry-run: verify marker placement without the arm ─────────────────────
    if args.dry_run:
        print("\nDRY-RUN — trigger PhoXi scans with live ArUco detection preview.")
        preview_and_capture(cam, preview_dir)
        cam.stop()
        return

    # ── Connect to arm ─────────────────────────────────────────────────────────
    arm = shared.connect_arm(args.robot_ip)

    try:
        # ── STEP 1: Touch each marker ──────────────────────────────────────────
        touch_points = shared.collect_touch_points(arm, args.markers)
        print(f"\nCollected {len(touch_points)} touch points: markers {sorted(touch_points.keys())}")

        if len(touch_points) < 3:
            print("ERROR: Need at least 3 touch points. Exiting.")
            return

        # ── STEP 2: Park the arm clear of the fixed camera's view ────────────
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        start_angles = np.array(cfg["arm"]["start_angles"], dtype=float)
        print(f"\n{'='*60}")
        print("STEP 2: Parking arm at arm.start_angles (clear of camera view)")
        print(f"  Target joint angles: {start_angles.tolist()}")
        print(f"{'='*60}")
        code = arm.set_servo_angle(angle=start_angles.tolist(), is_radian=True,
                                   speed=0.3, mvacc=5.0, wait=True)
        if isinstance(code, tuple):
            code = code[0]
        if code not in (0, None):
            raise RuntimeError(f"set_servo_angle failed (code={code})")
        time.sleep(0.5)  # let the arm settle

        # ── STEP 3: Capture PhoXi frame + detect markers ──────────────────────
        print(f"\n{'='*60}")
        print("STEP 3: Capturing PhoXi frame and detecting markers")
        print(f"{'='*60}")
        rgb, depth_mm, K = preview_and_capture(cam, preview_dir)

        print("\nDetected markers (camera frame):")
        detected = shared.detect_markers_3d(rgb, depth_mm, K)
        print(f"  → Detected IDs: {sorted(detected.keys())}")
        shared.visualise_detections(rgb, depth_mm, K,
                                    save_path=preview_dir / "phoxi_final_detection.png")

        # ── STEP 4: Match and solve ────────────────────────────────────────────
        common = sorted(set(touch_points) & set(detected))
        print(f"\nMatched {len(common)} markers: {common}")

        if len(common) < 3:
            print("ERROR: Need ≥ 3 matched markers. Make sure markers are visible to the PhoXi.")
            return

        src_pts = np.array([detected[i] for i in common])       # camera frame
        dst_pts = np.array([touch_points[i] for i in common])   # base frame

        T_cam_base = shared.kabsch(src_pts, dst_pts)
        errs = shared.residuals_mm(T_cam_base, src_pts, dst_pts)

        # ── Results ─────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("CALIBRATION RESULT")
        print(f"{'='*60}")
        print(f"Markers used: {common}")
        print(f"Residuals:  mean={np.mean(errs):.1f} mm   max={np.max(errs):.1f} mm\n")

        print(f"  {'ID':>4}  {'depth_mm':>8}  {'p_cam (m)':>28}  "
              f"{'p_base_touch (m)':>28}  {'p_base_pred (m)':>28}  {'err_mm':>7}")
        for i, mid in enumerate(common):
            pc = src_pts[i]
            pb_touch = dst_pts[i]
            pb_pred  = (T_cam_base @ np.append(pc, 1.0))[:3]
            print(f"  {mid:>4}  {pc[2]*1000:>8.1f}  "
                  f"[{pc[0]:+.4f},{pc[1]:+.4f},{pc[2]:+.4f}]  "
                  f"[{pb_touch[0]:+.4f},{pb_touch[1]:+.4f},{pb_touch[2]:+.4f}]  "
                  f"[{pb_pred[0]:+.4f},{pb_pred[1]:+.4f},{pb_pred[2]:+.4f}]  "
                  f"{errs[i]:>7.1f}")
        print()
        print(f"\nT_cam_base (fixed mount, store in config) =\n{np.round(T_cam_base, 6)}")

        # ── Sanity checks ───────────────────────────────────────────────────────
        z_touch = dst_pts[:, 2]
        z_spread = z_touch.max() - z_touch.min()
        print(f"Touch-point Z: mean={z_touch.mean()*1000:.1f} mm  "
              f"spread={z_spread*1000:.1f} mm  "
              f"(expect < 5 mm for flat table; large spread = EE tip not on surface)")
        if z_spread > 0.010:
            print(f"  WARNING: Z spread {z_spread*1000:.1f} mm > 10 mm — "
                  f"some touch points may not have been on the marker surface.")

        for i, mid in enumerate(common):
            pc = src_pts[i]
            pb_touch = dst_pts[i]
            pb_pred  = (T_cam_base @ np.append(pc, 1.0))[:3]
            dz = (pb_pred[2] - pb_touch[2]) * 1000
            if abs(dz) > 10:
                print(f"  Marker {mid}: Z error {dz:+.1f} mm "
                      f"(predicted {pb_pred[2]*1000:.1f} mm vs touch {pb_touch[2]*1000:.1f} mm)")

        cam_z_base = T_cam_base[:3, 2]
        dominant = int(np.argmax(np.abs(cam_z_base)))
        axes = ["base-X (forward)", "base-Y (lateral)", "base-Z (vertical)"]
        sign = "+" if cam_z_base[dominant] > 0 else "-"
        print(f"\nCamera optical axis: [{cam_z_base[0]:.3f}, {cam_z_base[1]:.3f}, {cam_z_base[2]:.3f}]")
        print(f"  → Camera looking toward {sign}{axes[dominant]}")

        if np.mean(errs) > 15.0:
            print("\nWARNING: Mean residual > 15 mm — calibration may be inaccurate.")
            print("  Tips: use more markers, spread them further apart, ensure EE tip")
            print("  is precisely on the marker centre when recording touch points.")

        if len(src_pts) >= 3:
            lateral_range = src_pts[:, 0].max() - src_pts[:, 0].min()  # cam-X spread
            if lateral_range < 0.10:
                print(f"\nWARNING: Markers only span {lateral_range*1000:.0f} mm in camera-X "
                      f"(< 100 mm).")
                print("  This under-constrains the cam-Z rotation and causes systematic")
                print("  lateral errors proportional to object offset from center.")
                print("  Place markers across the FULL image width (left edge, center, right).")

        # Print YAML block for manual copy-paste
        print(f"\nYAML (copy this into conf/camera/phoxi.yaml → T_cam_base):\n")
        print(format_T_cam_base_yaml(T_cam_base))

        # ── Save ────────────────────────────────────────────────────────────────
        ans = input(f"\nUpdate T_cam_base in {args.camera_config}? [y/N] ").strip().lower()
        if ans == "y":
            update_camera_config(args.camera_config, T_cam_base)
        else:
            print("Not saved — copy the YAML block above manually.")

    finally:
        arm.disconnect()
        cam.stop()


if __name__ == "__main__":
    main()
