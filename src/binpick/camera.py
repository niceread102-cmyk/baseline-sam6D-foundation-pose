"""RealSense D4xx RGBD camera wrapper.

Mirrors the notebook configuration from sam6d_example.ipynb.
Returns depth in millimetres (consistent with SAM6D/FoundationPose input convention).

A background streaming thread continuously grabs frames so that:
  - peek()    → non-blocking, always returns the latest frame (for live preview)
  - capture() → blocks briefly until a fresh frame arrives (for inference)
"""
from __future__ import annotations

import threading
import time

import numpy as np


class RealSenseCamera:
    """Wraps pyrealsense2 for aligned RGBD capture.

    Parameters
    ----------
    depth_scale:
        Multiplier applied to raw uint16 depth pixels to produce millimetres.
        Use 1.0 if the camera streams depth already in mm (most RealSense modes).
    depth_min_mm / depth_max_mm:
        Clamp range for the RealSense depth table (advanced mode).
        Pixels outside this range are zeroed by the sensor firmware.
        Tune to the expected camera-to-scene distance at hover height.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        depth_scale: float = 1.0,
        depth_exposure_us: int = 6000,
        laser_power_frac: float = 0.5,
        depth_gain: int = 16,
        color_exposure_us: int = 83,
        color_white_balance_k: int = 4600,
        disparity_shift: int = 0,
        depth_min_mm: int = 100,
        depth_max_mm: int = 700,
    ):
        import pyrealsense2 as rs

        self._depth_scale = float(depth_scale)

        # Hardware reset to clear stuck pipeline state
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device found — check USB connection and udev rules.")
        devices[0].hardware_reset()
        time.sleep(2.0)

        pipeline = rs.pipeline()
        config   = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8,  fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16,   fps)
        profile  = pipeline.start(config)

        device = profile.get_device()

        # Depth sensor settings
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.exposure,    depth_exposure_us)
        depth_sensor.set_option(rs.option.gain,        depth_gain)
        depth_sensor.set_option(rs.option.laser_power, int(laser_power_frac * 360))

        # Color sensor settings
        color_sensor = device.query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure,    0)
        color_sensor.set_option(rs.option.exposure,                color_exposure_us)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        color_sensor.set_option(rs.option.white_balance,           color_white_balance_k)

        # Disparity shift + depth clamp via advanced mode
        try:
            adv = rs.rs400_advanced_mode(device)
            params = adv.get_depth_table()
            params.disparityShift  = disparity_shift
            params.depthClampMin   = depth_min_mm
            params.depthClampMax   = depth_max_mm
            adv.set_depth_table(params)
        except Exception:
            pass  # advanced mode not available on all firmware versions

        # Camera intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._K = np.array(
            [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            dtype=np.float32,
        )

        self._pipeline = pipeline
        self._align    = rs.align(rs.stream.color)

        # Shared frame buffer — written by stream thread, read by capture()/peek()
        self._latest_rgb:   np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._frame_lock  = threading.Lock()
        self._new_frame   = threading.Event()
        self._streaming   = True

        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="CameraStream"
        )
        self._stream_thread.start()

        # Block until the first frame arrives (up to 10 s)
        if not self._new_frame.wait(timeout=10.0):
            raise RuntimeError("RealSense: no frame received within 10 s of startup")

    # ── background grab loop ──────────────────────────────────────────────────

    def _stream_loop(self) -> None:
        while self._streaming:
            try:
                frames  = self._pipeline.wait_for_frames(timeout_ms=2000)
                aligned = self._align.process(frames)
                rgb     = np.asanyarray(aligned.get_color_frame().get_data()).copy()
                depth   = (np.asanyarray(aligned.get_depth_frame().get_data())
                           .astype(np.float32) * self._depth_scale)
                with self._frame_lock:
                    self._latest_rgb   = rgb
                    self._latest_depth = depth
                self._new_frame.set()
            except Exception:
                pass  # timeout or pipeline stopping; loop exits when _streaming=False

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        """3×3 camera intrinsic matrix."""
        return self._K.copy()

    def peek(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Non-blocking: return the latest (rgb, depth, K), or None if not yet ready.

        Intended for live preview — always returns the most recent frame without
        waiting for a fresh one.
        """
        with self._frame_lock:
            if self._latest_rgb is None:
                return None
            return self._latest_rgb.copy(), self._latest_depth.copy(), self._K.copy()

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Block until a fresh RGBD frame is available and return it.

        Returns
        -------
        rgb:   (H, W, 3) uint8
        depth: (H, W) float32 in millimetres
        K:     (3, 3) float32 camera intrinsics
        """
        self._new_frame.clear()
        if not self._new_frame.wait(timeout=5.0):
            raise RuntimeError("RealSense: no fresh frame within 5 s")
        with self._frame_lock:
            return self._latest_rgb.copy(), self._latest_depth.copy(), self._K.copy()

    def stop(self) -> None:
        self._streaming = False
        self._new_frame.set()   # unblock any waiting capture() call
        self._pipeline.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
