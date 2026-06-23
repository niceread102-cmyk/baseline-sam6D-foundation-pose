# BinPicker — Architecture Overview

The `src/binpick/` module is a self-contained, ROS-free bin-picking pipeline built around a Hydra configuration system.  It integrates two deep-learning perception models (SAM-6D and FoundationPose) with a pure-numpy IK solver and direct xArm SDK control.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BINPICKER LOOP (per cycle)                        │
│                                                                             │
│  ┌──────────┐   RGBD (mm depth)  ┌──────────┐  masks + poses   ┌─────────┐  │
│  │  Camera  ├──────────────────► │  SAM-6D  ├───────────────►  │  Found- │  │
│  │(RealSense│                    │(segment) │                  │ationPose│  │
│  │  D4xx)   │                    └──────────┘                  │(refine) │  │
│  └────┬─────┘                                                  └────┬────┘  │
│       │ K (intrinsics)                                              │       │
│       │                                          poses in cam frame │       │
│       │                                                (metres)     │       │
│  FK(θ)│ + T_cam_ee                                                  │       │
│       ▼                                                             ▼       │
│  T_cam_base ─────────────────────────────────► ObjectSelector ◄── grips     │
│  (cam→base)                                    (NearestObject)      │       │
│                                                      │        GripPlanner   │
│                                               PickPlan              │       │
│                                               (theta_pick,          │       │
│                                                theta_place)    ◄──  │       │
│                                                      │                      │
│                                                      ▼                      │
│                                              ArmController                  │
│                                              (XArmController)               │
│                                                      │                      │
│                                                      ▼                      │
│                                              Lite 6 Robot Arm               │
└─────────────────────────────────────────────────────────────────────────────┘
```

The live preview window (`LivePreview`) runs on a separate daemon thread and displays all four stages simultaneously.

---

## Component Reference

| Component | Class | Config file | Output units |
|-----------|-------|-------------|--------------|
| Camera | `RealSenseCamera` | `conf/camera/realsense.yaml` | depth: **mm**, RGB: uint8, K: px |
| Detection | `SAM6D` (external) | `conf/sam6d/default.yaml` | masks, scores |
| Pose refinement | `FoundationPoseRefiner` (external) | `conf/foundationpose/default.yaml` | 4×4 poses, cam frame, **metres** |
| IK / FK | `Lite6IKSolver` | `conf/ik/lite6.yaml` | angles: **rad**, pos: **metres** |
| Arm control | `XArmController` | `conf/arm_controller/xarm.yaml` | — |
| Grip planning | `SimpleGripPlanner` | `conf/grip_planner/simple.yaml` | positions: **metres** |
| Object selection | `NearestObjectSelector` | `conf/object_selector/nearest.yaml` | theta: **rad** |
| Visualizer | `LivePreview` | `conf/preview/default.yaml` | display only |

Each component class has an abstract base (`IKSolver`, `ArmController`, `GripPlanner`, `ObjectSelector`) so alternative implementations can be swapped in via Hydra config without changing `BinPicker`.

---

## Configuration System (Hydra)

`conf/config.yaml` defines the composition defaults:

```yaml
defaults:
  - sam6d: default
  - foundationpose: default
  - ik: lite6
  - arm_controller: xarm
  - grip_planner: simple
  - object_selector: nearest
  - camera: realsense
  - preview: default
  - _self_
```

### Entry points

| Script | Purpose |
|--------|---------|
| `python -m src.binpick.binpicker` | Direct Hydra entry point; all overrides via CLI |
| `run_binpicker.py` | User-friendly wrapper; sets constants at top of file |

> **Important**: `run_binpicker.py` unconditionally applies its own Python constants (`MESH_SCALE`, `MAX_GRIP_M`, etc.) as Hydra CLI overrides.  Changes to `conf/config.yaml` for these parameters are **silently ignored** when running via `run_binpicker.py`.  Prefer CLI overrides (e.g. `foundationpose.physical_scale=2.0`) or edit `run_binpicker.py` directly for permanent changes.

### CLI override examples

```bash
# Change detection thresholds
python -m src.binpick.binpicker \
    'cad_models=[{path: /models/part.stl, id: part}]' \
    sam6d.det_score_thresh=0.4

# Override hover and placement positions (metres, robot-base frame)
python -m src.binpick.binpicker \
    'cad_models=[{path: /models/part.stl, id: part}]' \
    arm.hover_position='[0.35, 0.0, 0.25]' \
    arm.placement_position='[0.35, 0.20, 0.08]'

# Scale adjustment (STL in mm, real object is 2× the CAD model)
python -m src.binpick.binpicker \
    'cad_models=[{path: /models/part.stl, id: part}]' \
    foundationpose.mesh_scale=0.001 \
    foundationpose.physical_scale=2.0
```

---

## The Pick-and-Place Loop

`BinPicker.run()` executes an 8-step loop until the bin is empty:

| Step | Name | Description |
|------|------|-------------|
| 1 | Home | Move joints to `start_angles` |
| 2 | Hover | Move EE to `hover_position` above the bin |
| 3 | Capture | Grab RGBD frame; compute `T_cam_base = fk_matrix(θ) @ T_cam_ee` |
| 4a | SAM-6D | Segment the scene; exit loop if nothing found |
| 4b | FoundationPose | Refine 6-DOF object poses from masks |
| 5 | Plan | `NearestObjectSelector` picks best object + grip |
| 6 | Execute | Open → pick → close → place → open |
| 7 | Recover | On failure: recover arm, reset FP tracker, re-home |
| 8 | Repeat | From step 2 |

Recovery uses `safe=False` for the home move because the arm may be in an unexpected position after a fault.

---

## Key Coordinate Frames

See [`COORDINATE_FRAMES.md`](COORDINATE_FRAMES.md) for the full treatment.  In brief:

- **Camera frame**: origin at the camera optical centre; Z points forward (into the scene).
- **Robot-base frame**: origin at the arm's base; Z points up.
- **Object frame**: origin at the mesh centroid; axes aligned to the CAD model.
- **EE frame**: origin at the gripper midpoint; defined by the DH chain in `ik_solver.py`.

The critical transform `T_cam_base` maps camera→base and is recomputed every capture cycle.

---

## Scaling Conventions

FoundationPose requires mesh vertices in **metres**.  Two config parameters control this:

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `mesh_scale` | CAD unit → metres | `0.001` (STL in mm) |
| `physical_scale` | Compensate CAD ≠ real size | `3.0` (real is 3× the CAD) |

`effective_scale = mesh_scale × physical_scale` is computed in `BinPicker.__init__` and passed to both `FoundationPoseRefiner` (for rendering) and `SimpleGripPlanner` (for grip width computation).
