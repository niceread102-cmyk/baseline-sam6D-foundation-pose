"""
run_binpicker.py — Edit the USER SETTINGS section below, then run:

    python run_binpicker.py
"""
import logging
from datetime import datetime
from pathlib import Path


class _FlushFileHandler(logging.FileHandler):
    """FileHandler that flushes after every emit — logs survive Ctrl+C / force-stop."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ── USER SETTINGS ─────────────────────────────────────────────────────────────

CAD_MODELS = [
    # Add one entry per object type.  path must be an absolute or repo-relative path.
    # {"path": "/home/joon/models/part_a.stl", "id": "part_a"},
    # {"path": "/home/joon/models/part_b.obj", "id": "part_b"},
]

# Camera and grip planner — None uses the defaults in conf/config.yaml
# (camera: phoxi — fixed-mount scanner with a static T_cam_base;
#  grip_planner: manual — hand-authored 6-DOF grasps, see GRASPS_JSON_PATH below).
# Override here for a quick swap without editing conf/config.yaml, e.g.
# CAMERA = "realsense" for the eye-on-hand wrist camera, or
# GRIP_PLANNER = "simple" for PCA-based centroid grips.
CAMERA       = None   # e.g. "realsense"
GRIP_PLANNER = None   # e.g. "simple"

# Path to hand-authored 6-DOF grasps (grip_planner: manual only). Each grasp
# defines a grip point + full approach orientation + hover backoff, per model
# id — see the "Manual 6DoF grasp definition" section of
# photo_neo_capture_test.ipynb for how to author/preview these before saving.
# Set to None to use conf/grip_planner/manual.yaml's default (grasps/manual_grasps.json).
GRASPS_JSON_PATH = None   # e.g. "/home/joon/Projects/cnos/grasps/manual_grasps.json"

# Arm positions in metres (robot-base frame, X forward, Y left, Z up)
# Set to None to use the value from conf/config.yaml (recommended — keeps config as source of truth).
HOVER_POSITION     = None   # e.g. [0.30, 0.0, 0.44]
PLACEMENT_POSITION = None   # e.g. [0.35, 0.20, 0.20]

# Joint angles (radians) for the safe home position.
# Set to None to use the value from conf/config.yaml.
START_ANGLES = None   # e.g. [0.0, 0.7, 1.57, 0.0, 0.87, 0.0]

# Gripper limits (metres)
MIN_GRIP_M = 0.02   # 20 mm
MAX_GRIP_M = 0.16   # 160 mm — covers 3× physical_scale grip widths (~107mm, ~155mm)

# FoundationPose scaling
#   mesh_scale:    unit conversion from CAD units → metres (0.001 for mm-unit STL files)
#   physical_scale: compensate if the CAD model is scaled differently from the real object
MESH_SCALE     = 0.001
PHYSICAL_SCALE = 3.0

# Eye-on-hand calibration — 4×4 transform from camera frame to EE (wrist) frame.
# This is FIXED (camera rigidly mounted on wrist); T_cam_base is computed dynamically
# each capture as: T_cam_base = fk_matrix(theta) @ T_cam_ee
# Set to None to use the value already saved in conf/config.yaml (recommended).
T_CAM_EE = None
# T_CAM_EE = [
#     [r00, r01, r02, tx],
#     [r10, r11, r12, ty],
#     [r20, r21, r22, tz],
#     [0.0, 0.0, 0.0, 1.0],
# ]

# SAM6D / detection thresholds (lower = more detections, higher = fewer false positives)
DET_SCORE_THRESH = 0.35
POSE_SCORE_THRESH = 0.30

# Blender path — required on ARM64; leave None on x86-64
# BLENDER_PATH = "/home/joon/blender/blender-arm64"
BLENDER_PATH = None

# Set to False to skip the OpenCV preview window (headless / SSH sessions)
PREVIEW_ENABLED = True

# ── CONFIG ASSEMBLY ───────────────────────────────────────────────────────────

_CONF_DIR = str(Path(__file__).resolve().parent / "conf")


def build_overrides() -> list[str]:
    overrides = []

    # Camera / grip planner config-group swaps
    if CAMERA is not None:
        overrides.append(f"camera={CAMERA}")
    if GRIP_PLANNER is not None:
        overrides.append(f"grip_planner={GRIP_PLANNER}")
    if GRASPS_JSON_PATH is not None:
        overrides.append(f"grip_planner.grasps_path={GRASPS_JSON_PATH}")

    # CAD models — if empty here, falls back to whatever is set in conf/config.yaml
    if CAD_MODELS:
        models_str = "[" + ",".join(
            f"{{path: {m['path']}, id: {m['id']}}}" for m in CAD_MODELS
        ) + "]"
        overrides.append(f"cad_models={models_str}")

    # Arm positions — only override if explicitly set; otherwise conf/config.yaml wins
    if HOVER_POSITION is not None:
        overrides.append(f"arm.hover_position={HOVER_POSITION}")
    if PLACEMENT_POSITION is not None:
        overrides.append(f"arm.placement_position={PLACEMENT_POSITION}")
    if START_ANGLES is not None:
        overrides.append(f"arm.start_angles={START_ANGLES}")

    # T_cam_ee (override only if explicitly set; otherwise uses conf/config.yaml value)
    if T_CAM_EE is not None:
        overrides.append(f"arm.T_cam_ee={T_CAM_EE}")

    # Gripper
    overrides.append(f"gripper.min_grip_m={MIN_GRIP_M}")
    overrides.append(f"gripper.max_grip_m={MAX_GRIP_M}")

    # FoundationPose scaling
    overrides.append(f"foundationpose.mesh_scale={MESH_SCALE}")
    overrides.append(f"foundationpose.physical_scale={PHYSICAL_SCALE}")

    # SAM6D thresholds
    overrides.append(f"sam6d.det_score_thresh={DET_SCORE_THRESH}")
    overrides.append(f"sam6d.pose_score_thresh={POSE_SCORE_THRESH}")

    # Blender path (ARM64)
    if BLENDER_PATH is not None:
        overrides.append(f"sam6d.blender_path={BLENDER_PATH}")

    # Preview
    if not PREVIEW_ENABLED:
        overrides.append("preview.enabled=false")

    return overrides


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")

    # Console handler
    _ch = logging.StreamHandler()
    _ch.setFormatter(_fmt)

    # File handler — flushes every line so logs survive Ctrl+C or kill
    _log_dir = Path(__file__).resolve().parent / "logs"
    _log_dir.mkdir(exist_ok=True)
    _log_file = _log_dir / f"binpicker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _fh = _FlushFileHandler(str(_log_file))
    _fh.setFormatter(_fmt)

    logging.basicConfig(level=logging.INFO, handlers=[_ch, _fh])
    logging.getLogger(__name__).info("Logging to: %s", _log_file)

    with initialize_config_dir(config_dir=_CONF_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=build_overrides())

    logging.getLogger(__name__).info(
        "Config:\n" + OmegaConf.to_yaml(cfg)
    )

    from src.binpick.binpicker import BinPicker
    BinPicker(cfg).run()
