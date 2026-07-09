"""Photoneo PhoXi 3D structured-light scanner wrapper.

Mirrors the capture logic from photo_neo_capture.ipynb.  Returns depth in
**millimetres** (consistent with SAM6D/FoundationPose convention, and with
RealSenseCamera's ``depth_units`` contract).

Unlike RealSenseCamera this is a single-shot trigger scanner, not a
continuously-streaming video camera: every ``capture()`` call blocks while
PhoXiControl runs a full structured-light scan (typically ~0.3-1 s). There is
no background streaming thread; ``peek()`` simply returns the last triggered
frame instead of a live feed.

Requires PhoXi Control running (``/usr/bin/PhoXiControl``) and the scanner
discovered in its device list (use ``Tools -> Add Device via Address`` if
network discovery doesn't find it).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class PhoXiCamera:
    """Wraps phoxi_api for single-shot RGBD capture from a Photoneo PhoXi scanner.

    Parameters
    ----------
    device_id:
        PhoXi device id to connect to. If ``None``, the first non-file-camera
        ``PhoXi3DScan`` device reported by PhoXiControl is used.
    address:
        Optional IP/hostname for a direct connection, forwarded to
        ``PhoXiControl.connect`` (needed if the scanner isn't discovered
        automatically).
    """

    def __init__(self, device_id: str | None = None, address: str | None = None):
        from phoxi_api import PhoXiControl, PhoXiDevice

        self._phc = PhoXiControl()

        if device_id is None:
            devices = self._phc.get_device_list(refresh=True)
            device_id = next(
                (d["id"] for d in devices if d["type"] == "PhoXi3DScan" and not d["file_camera"]),
                None,
            )
            if device_id is None:
                raise RuntimeError(
                    "No PhoXi3DScan device found — check PhoXi Control is running and the "
                    "scanner is discovered (Tools -> Add Device via Address if network "
                    "discovery doesn't find it)."
                )

        self._device = self._phc.connect(device_id, address=address)
        self._device.set_trigger_mode(PhoXiDevice.TriggerMode.SOFTWARE)
        self._device.set_frame_settings({"DepthMap": True, "Texture": True})
        self._device.start_acquisition()

        # Grab one frame up front so K / last frame are available immediately.
        self._K: np.ndarray
        self._last_rgb, self._last_depth, self._K = self._grab()

    def _grab(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame_id = self._device.trigger_frame(wait_grabbing_end=True)
        frame = self._device.get_frame(frame_id=frame_id, timeout_ms=60_000)

        K = np.array(frame.Info["CameraMatrix"], dtype=np.float32).reshape(3, 3)
        depth = np.array(frame.DepthMap, dtype=np.float32).squeeze()

        texture = np.array(frame.Texture).squeeze()
        rgb = np.stack([texture] * 3, axis=-1) if texture.ndim == 2 else texture
        t_max = float(rgb.max())
        rgb = (rgb / t_max * 255).clip(0, 255).astype(np.uint8) if t_max > 0 else rgb.astype(np.uint8)

        return rgb, depth, K

    # ── public API (mirrors RealSenseCamera) ──────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        """3×3 camera intrinsic matrix."""
        return self._K.copy()

    @property
    def depth_units(self) -> str:
        """Unit of the depth values returned by ``peek()`` and ``capture()``. Always ``"mm"``."""
        return "mm"

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trigger a fresh structured-light scan and return (rgb, depth_mm, K).

        Blocks for the duration of the scan (no cached/streamed frame).
        """
        self._last_rgb, self._last_depth, K = self._grab()
        return self._last_rgb.copy(), self._last_depth.copy(), K

    def peek(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the most recently triggered frame without starting a new scan."""
        if self._last_rgb is None:
            return None
        return self._last_rgb.copy(), self._last_depth.copy(), self._K.copy()

    def stop(self) -> None:
        self._device.disconnect(stop_acquisition=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
