"""Live preview thread for the bin-picking loop.

Renders a 2×2 composite OpenCV window:

  ┌─────────────────────┬─────────────────────┐
  │  Live RGBD feed     │  SAM6D detections   │
  │  (depth heat HUD)   │  (masks overlaid)   │
  ├─────────────────────┼─────────────────────┤
  │  FoundationPose     │  Status panel       │
  │  (axes + pts)       │  Stage / Gripper /  │
  │  [IK path arrows]   │  FPS / detections   │
  └─────────────────────┴─────────────────────┘

The main thread calls update_*() to swap panel buffers; the preview thread
composites and calls cv2.imshow() at a fixed FPS to avoid blocking inference.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Colour palette for per-detection overlays (BGR for OpenCV)
_PALETTE = [
    (0, 120, 255), (0, 200, 80), (220, 60, 0),
    (180, 0, 220), (0, 200, 200), (220, 180, 0),
    (120, 0, 160), (0, 160, 120), (200, 100, 0),
]


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── Drawing utilities (called by BinPicker, not the preview thread) ───────────

def overlay_masks(rgb: np.ndarray, results: dict | None) -> np.ndarray:
    """Draw SAM6D mask overlays and bounding-box labels on a copy of rgb."""
    vis = rgb.copy()
    if results is None:
        return vis

    masks  = results.get("masks")    # (N, H, W) bool
    boxes  = results.get("boxes")    # (N, 4)  xyxy
    scores = results.get("ism_scores")

    if masks is None or len(masks) == 0:
        return vis

    overlay = vis.astype(np.float32)
    for i, mask in enumerate(masks):
        colour = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)  # BGR
        colour_rgb = colour[[2, 1, 0]]  # → RGB
        overlay[mask] = overlay[mask] * 0.45 + colour_rgb * 0.55

    vis = overlay.clip(0, 255).astype(np.uint8)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            c = tuple(int(v) for v in _PALETTE[i % len(_PALETTE)][::-1])  # BGR
            vis_bgr = cv2.rectangle(_to_bgr(vis), (x1, y1), (x2, y2), c, 2)
            label = f"#{i}"
            if scores is not None and i < len(scores):
                label += f" {scores[i]:.2f}"
            cv2.putText(vis_bgr, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    return vis


def overlay_axes(
    rgb: np.ndarray,
    fp_results: list[dict] | None,
    K: np.ndarray,
    mesh_pts_by_model: dict[str, np.ndarray] | None = None,
    axis_len_m: float = 0.02,
    n_pts: int = 500,
) -> np.ndarray:
    """Draw FoundationPose axes (and optional point clouds) on a copy of rgb.

    Parameters
    ----------
    mesh_pts_by_model:
        Pre-sampled point clouds in object frame (metres), keyed by model_id.
        Pass None to skip point cloud drawing (axes only).
    axis_len_m:
        Length of coordinate axis arrows in metres.
    """
    vis = rgb.copy()
    if fp_results is None:
        return vis

    K = np.asarray(K, dtype=float)

    def project(pts_cam: np.ndarray) -> np.ndarray:
        """(N,3) camera-frame → (N,2) pixel coords."""
        z = pts_cam[:, 2:3].clip(1e-6, None)
        uv = (K[:2, :2] @ (pts_cam[:, :2] / z).T).T
        uv[:, 0] += K[0, 2]
        uv[:, 1] += K[1, 2]
        return uv.astype(np.int32)

    for i, result in enumerate(fp_results):
        pose = np.asarray(result["pose"])  # 4×4
        R, t = pose[:3, :3], pose[:3, 3]
        c_bgr = tuple(int(v) for v in _PALETTE[i % len(_PALETTE)])  # BGR palette

        model_id = result.get("model_id", "")
        vis_bgr  = _to_bgr(vis)

        # Point cloud overlay
        if mesh_pts_by_model and model_id in mesh_pts_by_model:
            pts_obj = mesh_pts_by_model[model_id]
            if len(pts_obj) > n_pts:
                idx = np.random.choice(len(pts_obj), n_pts, replace=False)
                pts_obj = pts_obj[idx]
            pts_cam  = (R @ pts_obj.T).T + t
            valid    = pts_cam[:, 2] > 0
            if valid.any():
                uv = project(pts_cam[valid])
                H, W = vis_bgr.shape[:2]
                uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < W) &
                        (uv[:, 1] >= 0) & (uv[:, 1] < H)]
                for u, v in uv:
                    cv2.circle(vis_bgr, (u, v), 1, c_bgr, -1)

        # Coordinate axes
        origin_cam = t
        axes_dirs  = {"X": (R[:, 0], (255, 50, 50)),   # red
                      "Y": (R[:, 1], (50, 200, 50)),    # green
                      "Z": (R[:, 2], (50, 50, 255))}    # blue

        if origin_cam[2] > 0:
            uv0 = project(origin_cam[np.newaxis])[0]
            for _label, (ax, ax_bgr) in axes_dirs.items():
                tip = origin_cam + ax * axis_len_m
                if tip[2] > 0:
                    uv1 = project(tip[np.newaxis])[0]
                    cv2.arrowedLine(vis_bgr, tuple(uv0), tuple(uv1),
                                    ax_bgr, 2, tipLength=0.25, line_type=cv2.LINE_AA)

        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    return vis


# ── LivePreview thread ─────────────────────────────────────────────────────────

class LivePreview(threading.Thread):
    """Composite 2×2 OpenCV preview window running on a dedicated daemon thread."""

    def __init__(
        self,
        panel_w: int = 640,
        panel_h: int = 480,
        fps: int = 15,
        window_name: str = "BinPicker Preview",
        ik_solver: Any = None,        # IKSolver, for IK path overlay
        T_cam_base: np.ndarray | None = None,
        enabled: bool = True,         # accepted so Hydra can pass preview.enabled without error
        capture_dir: str | None = None,  # directory to save JPEG captures (1/sec); None = disabled
    ):
        super().__init__(daemon=True, name="LivePreview")
        self._pw   = panel_w
        self._ph   = panel_h
        self._fps  = fps
        self._name = window_name
        self._ik   = ik_solver
        self._T_cb = np.asarray(T_cam_base) if T_cam_base is not None else None

        self._lock    = threading.Lock()
        self._running = True

        if capture_dir is not None:
            self._capture_dir: Path | None = Path(capture_dir)
            self._capture_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._capture_dir = None
        self._last_capture_t = 0.0

        black = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        self._p_live   = black.copy()
        self._p_sam6d  = black.copy()
        self._p_fp     = black.copy()

        # Status state
        self._stage           = "Initializing"
        self._gripper_open    = True
        self._n_detections    = 0
        self._pick_target_uv: tuple[int, int] | None = None
        self._grip_line_uv:   tuple | None = None  # (uv_jaw1, uv_center, uv_jaw2)

        # Optional callback for continuous live feed: () -> (rgb, depth, K) | None
        self._frame_source = None

    # ── public update API (called from main thread) ───────────────────────

    def set_T_cam_base(self, T: np.ndarray | None) -> None:
        """Update the camera-to-base transform used for IK path projection.

        Call this each time the arm moves to a new capture pose so that the
        overlay stays correct for eye-on-hand setups.
        """
        with self._lock:
            self._T_cb = np.asarray(T) if T is not None else None

    def set_frame_source(self, fn) -> None:
        """Register a callback that returns (rgb, depth, K) | None for live streaming."""
        self._frame_source = fn

    def update_live(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        panel = self._make_rgbd_panel(rgb, depth)
        with self._lock:
            self._p_live = panel

    def update_sam6d_panel(self, panel_rgb: np.ndarray) -> None:
        """Accept a pre-rendered SAM6D overlay (RGB) and resize if needed."""
        p = _resize(panel_rgb, self._pw, self._ph)
        with self._lock:
            self._p_sam6d = p

    def update_fp_panel(self, panel_rgb: np.ndarray) -> None:
        """Accept a pre-rendered FoundationPose overlay (RGB) and resize if needed."""
        p = _resize(panel_rgb, self._pw, self._ph)
        with self._lock:
            self._p_fp = p
            # Clear plan overlays; they'll be repopulated if a plan is found this cycle
            self._pick_target_uv = None
            self._grip_line_uv = None

    def update_status(self, stage: str, gripper_open: bool, n_detections: int) -> None:
        with self._lock:
            self._stage        = stage
            self._gripper_open = gripper_open
            self._n_detections = n_detections

    def update_pick_target(self, pick_xyz_base: np.ndarray, K: np.ndarray) -> None:
        """Project the planned jaw midpoint (base frame) into the FP hover-image panel.

        Uses the T_cam_base stored from the last hover capture, so the marker
        aligns with the static hover snapshot shown in the FP panel.
        """
        if self._T_cb is None:
            return

        T_base_to_cam = np.linalg.inv(self._T_cb)
        p_cam = T_base_to_cam[:3, :3] @ np.asarray(pick_xyz_base, dtype=float) + T_base_to_cam[:3, 3]
        if p_cam[2] <= 1e-6:
            return

        K = np.asarray(K, dtype=float)
        u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
        v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])

        sx = self._pw / 640.0
        sy = self._ph / 480.0
        u_s = int(u * sx)
        v_s = int(v * sy)

        if 0 <= u_s < self._pw and 0 <= v_s < self._ph:
            with self._lock:
                self._pick_target_uv = (u_s, v_s)
        else:
            with self._lock:
                self._pick_target_uv = None

    def update_grip_line(
        self,
        center_cam: np.ndarray,
        axis_cam: np.ndarray,
        width_m: float,
        K: np.ndarray,
    ) -> None:
        """Project the gripper jaw span into the FP panel for overlay.

        Draws from jaw1 = center - axis*(width/2) to jaw2 = center + axis*(width/2),
        with perpendicular ticks at each jaw end to show the finger plates.
        """
        K = np.asarray(K, dtype=float)
        center_cam = np.asarray(center_cam, dtype=float)
        axis = np.asarray(axis_cam, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return
        axis = axis / norm
        half = float(width_m) / 2.0

        def _proj(p: np.ndarray) -> tuple[int, int] | None:
            if p[2] <= 1e-6:
                return None
            u = int(K[0, 0] * p[0] / p[2] + K[0, 2])
            v = int(K[1, 1] * p[1] / p[2] + K[1, 2])
            return (u, v)

        uv1 = _proj(center_cam - axis * half)
        uv_c = _proj(center_cam)
        uv2 = _proj(center_cam + axis * half)

        if uv1 is None or uv_c is None or uv2 is None:
            return

        # Scale uvs to panel resolution
        H_src, W_src = 480, 640  # source camera resolution
        sx, sy = self._pw / W_src, self._ph / H_src
        def scale_uv(uv):
            return (int(uv[0] * sx), int(uv[1] * sy))

        with self._lock:
            self._grip_line_uv = (scale_uv(uv1), scale_uv(uv_c), scale_uv(uv2))

    # ── panel rendering ───────────────────────────────────────────────────

    def _make_rgbd_panel(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Blend depth colourmap (TURBO) at 30% over RGB."""
        d_valid = depth[depth > 0]
        if d_valid.size > 0:
            d_min, d_max = float(d_valid.min()), float(d_valid.max())
        else:
            d_min, d_max = 0.0, 1.0

        d_norm  = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0, 1)
        d_u8    = (d_norm * 255).astype(np.uint8)
        d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)          # BGR
        d_rgb   = cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB)

        mask         = (depth > 0)[..., np.newaxis]
        blended      = np.where(mask, rgb * 0.7 + d_rgb * 0.3, rgb).astype(np.uint8)
        return _resize(blended, self._pw, self._ph)

    def _render_status(
        self, stage: str, gripper_open: bool, n_det: int, display_fps: float
    ) -> np.ndarray:
        panel = np.zeros((self._ph, self._pw, 3), dtype=np.uint8)
        lines = [
            ("BinPicker",                    (200, 200, 200), 0.7, 2),
            ("",                             (0, 0, 0),       0.5, 1),
            (f"Stage:  {stage}",             (220, 220, 100), 0.55, 1),
            ("",                             (0, 0, 0),       0.5, 1),
            (f"Gripper: {'OPEN' if gripper_open else 'CLOSED'}",
             (80, 220, 80) if gripper_open else (220, 80, 80), 0.55, 1),
            ("",                             (0, 0, 0),       0.5, 1),
            (f"Detections: {n_det}",         (200, 200, 200), 0.55, 1),
            ("",                             (0, 0, 0),       0.5, 1),
            (f"Preview FPS: {display_fps:.1f}", (160, 160, 160), 0.45, 1),
        ]
        if self._T_cb is None:
            lines += [
                ("",                                   (0, 0, 0),    0.5, 1),
                ("Pick target: N/A",                   (120, 80, 80), 0.4, 1),
                ("(calibrate arm.T_cam_ee first)",     (100, 60, 60), 0.38, 1),
            ]

        y = 40
        for text, colour, scale, thick in lines:
            if text:
                cv2.putText(panel, text, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale,
                            colour, thick, cv2.LINE_AA)
            y += int(scale * 55 + 10)

        return panel

    # ── thread main loop ──────────────────────────────────────────────────

    def run(self) -> None:
        # On Linux/GTK, this starts an X11 event-processing thread so imshow()
        # updates even when the main thread is blocked by GPU inference.
        try:
            cv2.startWindowThread()
        except Exception:
            pass

        t_last = time.time()

        while self._running:
            t_now   = time.time()
            elapsed = t_now - t_last
            target  = 1.0 / self._fps
            if elapsed < target:
                time.sleep(target - elapsed)
            fps_actual = 1.0 / max(t_now - t_last, 1e-6)
            t_last = time.time()

            # Pull the latest camera frame and refresh the live panel
            if self._frame_source is not None:
                frame = self._frame_source()
                if frame is not None:
                    rgb, depth, _ = frame
                    panel = self._make_rgbd_panel(rgb, depth)
                    with self._lock:
                        self._p_live = panel

            with self._lock:
                p_live       = self._p_live.copy()
                p_sam6d      = self._p_sam6d.copy()
                p_fp         = self._p_fp.copy()
                stage        = self._stage
                g_open       = self._gripper_open
                n_det        = self._n_detections
                pick_target  = self._pick_target_uv
                grip_line    = self._grip_line_uv

            # Convert RGB panels to BGR for OpenCV
            p_live_bgr  = _to_bgr(p_live)
            p_sam6d_bgr = _to_bgr(p_sam6d)
            p_fp_bgr    = _to_bgr(p_fp)

            H, W = p_fp_bgr.shape[:2]

            # Draw pick-target marker on FP panel: circle + crosshair at jaw midpoint
            if pick_target is not None:
                px, py = pick_target
                cv2.circle(p_fp_bgr,    (px, py), 16, (0, 220, 255), 3, cv2.LINE_AA)
                cv2.circle(p_fp_bgr,    (px, py),  4, (0, 220, 255), -1)
                tick = 20
                cv2.line(p_fp_bgr, (px - tick, py), (px + tick, py),
                         (0, 220, 255), 2, cv2.LINE_AA)
                cv2.line(p_fp_bgr, (px, py - tick), (px, py + tick),
                         (0, 220, 255), 2, cv2.LINE_AA)

            # Draw grip line (jaw span) on FP panel
            if grip_line is not None:
                uv1, uv_c, uv2 = grip_line
                if (0 <= uv1[0] < W and 0 <= uv1[1] < H and
                        0 <= uv2[0] < W and 0 <= uv2[1] < H):
                    # Jaw span line
                    cv2.line(p_fp_bgr, uv1, uv2, (0, 255, 120), 5, cv2.LINE_AA)
                    # Center dot
                    cv2.circle(p_fp_bgr, uv_c, 8, (0, 255, 120), -1)
                    # Perpendicular jaw-plate ticks at each end
                    dx = uv2[0] - uv1[0]
                    dy = uv2[1] - uv1[1]
                    span = max((dx * dx + dy * dy) ** 0.5, 1.0)
                    pdx = int(-dy / span * 20)
                    pdy = int(dx / span * 20)
                    for uv_jaw in (uv1, uv2):
                        cv2.line(p_fp_bgr,
                                 (uv_jaw[0] - pdx, uv_jaw[1] - pdy),
                                 (uv_jaw[0] + pdx, uv_jaw[1] + pdy),
                                 (0, 200, 255), 3, cv2.LINE_AA)

            p_status = self._render_status(stage, g_open, n_det, fps_actual)

            # Panel labels (drawn in-place on BGR copies)
            for panel, label in [
                (p_live_bgr,  "Live RGBD"),
                (p_sam6d_bgr, "SAM6D"),
                (p_fp_bgr,    "FoundationPose"),
            ]:
                cv2.putText(panel, label, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # Composite 2×2 grid (OpenCV expects BGR)
            top = np.concatenate([p_live_bgr,  p_sam6d_bgr], axis=1)
            bot = np.concatenate([p_fp_bgr,    p_status],    axis=1)
            composite = np.concatenate([top, bot], axis=0)

            cv2.imshow(self._name, composite)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False
                break

            # Save a JPEG capture once per second
            if self._capture_dir is not None:
                now = time.time()
                if now - self._last_capture_t >= 1.0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    cv2.imwrite(str(self._capture_dir / f"preview_{ts}.jpg"), composite)
                    self._last_capture_t = now

        cv2.destroyWindow(self._name)

    def stop(self) -> None:
        self._running = False


# ── No-op preview (preview disabled) ─────────────────────────────────────────

class NullPreview:
    """Drop-in replacement for LivePreview when preview is disabled.

    All methods are no-ops so BinPicker needs no conditional guards.
    """

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_T_cam_base(self, T) -> None:
        pass

    def set_frame_source(self, fn) -> None:
        pass

    def update_live(self, rgb, depth) -> None:
        pass

    def update_sam6d_panel(self, panel) -> None:
        pass

    def update_fp_panel(self, panel) -> None:
        pass

    def update_status(self, stage, gripper_open, n_detections) -> None:
        pass

    def update_pick_target(self, pick_xyz_base, K) -> None:
        pass

    def update_grip_line(self, center_cam, axis_cam, width_m, K) -> None:
        pass
