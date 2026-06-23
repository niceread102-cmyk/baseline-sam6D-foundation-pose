# Camera — RealSense D4xx Reference

This document describes `RealSenseCamera` in [`src/binpick/camera.py`](../camera.py).

---

## 1. Output Contract

| Output | Shape | dtype | Unit |
|--------|-------|-------|------|
| `rgb` | `(H, W, 3)` | `uint8` | — |
| `depth` | `(H, W)` | `float32` | **millimetres** |
| `K` | `(3, 3)` | `float32` | pixels / metres |

**Depth is always in millimetres.**  This is the convention expected by both SAM-6D and FoundationPose.  Confirm programmatically with:

```python
assert camera.depth_units == "mm"
```

The raw RealSense `z16` format stores depth in millimetres as `uint16`.  The `depth_scale` parameter is multiplied into the raw value before returning; with the default `depth_scale=1.0`, no unit conversion occurs.

---

## 2. Thread Model

```
Main thread                        CameraStream thread (daemon)
     │                                      │
     │                             pipeline.wait_for_frames()
     │                                      │
     │                             align depth → color
     │                                      │
     │                             write (rgb, depth) under lock
     │                             set new_frame event
     │                                      │
     ├── peek() ──────────────────► read under lock (non-blocking)
     │   returns latest or None             │
     │                                      │
     ├── capture() ─────────────── wait for new_frame event
     │   blocks until fresh frame  read under lock
     │   raises after 5 s timeout           │
```

- `peek()` is used by `LivePreview` to display the most recent frame without blocking inference.
- `capture()` is used by `BinPicker` to get a fresh frame for detection.  It clears the event before waiting so it always gets a frame that arrived *after* the call.

---

## 3. Startup Sequence

On construction the camera performs a hardware reset to clear any stuck pipeline state:

```
devices[0].hardware_reset()
sleep(2.0)          # wait for USB re-enumeration
pipeline.start()    # configure and start streaming
wait(10.0)          # block until first frame arrives
```

The hardware reset is necessary when the previous process did not call `stop()` cleanly (e.g. crashed).  Without it the pipeline may fail to start or produce stale frames.

---

## 4. Depth Configuration

### Depth range clamping

`depth_min_mm` and `depth_max_mm` are applied by the sensor firmware via the RealSense advanced mode `DepthTable`.  Pixels outside the range are zeroed before transmission.

At hover height (`hover_position ≈ [0.3, 0.0, 0.44]`), the camera-to-bin distance is roughly 300–500 mm.  The defaults (`100–700 mm`) cover this range.  If your setup has a different hover height, adjust accordingly.

### Disparity shift

`disparity_shift` pushes the depth sensing range toward near objects.  Set it to `0` (default) unless you need to see very close objects (< 100 mm).  Increasing it compresses the usable range at the far end.

### Advanced mode availability

Advanced mode settings (`disparity_shift`, `depth_min_mm`, `depth_max_mm`) require a firmware version that supports `rs400_advanced_mode`.  If unavailable, a warning is logged and the sensor uses its firmware defaults.

---

## 5. Color Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `color_exposure_us` | `83 µs` | Manual exposure; disables auto |
| `color_white_balance_k` | `4600 K` | Manual white balance; disables auto |

Manual settings ensure consistent colour across frames.  Adjust for different lighting conditions.

---

## 6. Limitations

- **Single device**: only `devices[0]` is used.  Multiple devices are not supported.
- **No IR avoidance**: if another active IR emitter is in the scene (another RealSense, a structured-light projector), depth quality may degrade.
- **No temporal filtering**: the raw depth from the sensor is used directly.  Consider enabling RealSense temporal or spatial filters if depth noise is a problem.
- **Fixed alignment**: depth is aligned to the color frame by the SDK.  The intrinsic matrix `K` is extracted from the color stream profile and is correct for the aligned depth.
