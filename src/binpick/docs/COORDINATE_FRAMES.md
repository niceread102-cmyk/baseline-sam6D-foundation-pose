# Coordinate Frames

This document defines all coordinate frames used in the bin-picking pipeline and shows how they relate to one another.

---

## 1. Frame Definitions

| Frame | Symbol | Origin | Z-axis | Used by |
|-------|--------|--------|--------|---------|
| **Robot base** | base | Arm mounting point (floor) | Pointing up | IK solver, arm controller |
| **End-effector** | EE | Gripper midpoint | Defined by DH chain | IK solver FK output |
| **Camera** | cam | Camera optical centre | Pointing into scene (forward) | RealSense, SAM-6D, FoundationPose |
| **Object** | obj | CAD mesh centroid | Defined by CAD model axes | GripPlanner, FoundationPose |

All pipeline-internal distances are in **metres**.  Depth from the camera is in **millimetres** (SAM-6D / FoundationPose convention).

---

## 2. Transform Chain

```
Camera frame ──T_cam_base──► Robot-base frame
     │                              │
     │  T_obj_cam (FoundationPose)  │
     ▼                              ▼
Object frame ──────────────────────────────► Grip centroid in base frame
  (centroid at origin)   R_obj, p_obj
```

### Key transforms

| Symbol | Direction | Source |
|--------|-----------|--------|
| `T_cam_base` | camera → base | FK(θ) @ T_cam_ee (computed each capture) |
| `T_cam_ee` | camera → EE | Static calibration (config `arm.T_cam_ee`) |
| `T_obj_cam` | object → camera | FoundationPose output |
| `T_obj_base` | object → base | `T_cam_base @ T_obj_cam` |

### Point transformation

To transform a camera-frame point $\mathbf{p}_{cam}$ to robot-base frame:

$$\mathbf{p}_{base} = T_{cam\_base} \begin{bmatrix} \mathbf{p}_{cam} \\ 1 \end{bmatrix}$$

where $T_{cam\_base}$ is a 4×4 homogeneous transform.

---

## 3. The `T_cam_base` Convention

> **Critical**: `T_cam_base` maps **camera → base** (forward direction).
>
> - $\mathbf{p}_{base} = T_{cam\_base} \cdot \mathbf{p}_{cam\_hom}$ ✓
> - `T_cam_base` is NOT base → camera.

This was the source of a past bug (see project memory `feedback_t_cam_base.md`): using `inv(T_cam_base)` instead of `T_cam_base` caused the arm to move in the mirror-image direction.

### Sanity check (logged at startup)

At hover height, `BinPicker` logs:

```
T_cam_base at hover:
  Camera origin in base: [x, y, z]m
  Cam-Z (optical axis) → base [dx, dy, dz]  dominant: +base-X (forward)
```

The optical axis (cam-Z) should point roughly forward (+base-X) and slightly downward (−base-Z) for a wrist-mounted camera facing into a bin.

---

## 4. How `T_cam_base` Is Computed

`T_cam_base` is recomputed every capture cycle because the camera moves with the wrist:

```python
T_cam_base = ik.fk_matrix(current_theta) @ T_cam_ee
```

- `fk_matrix(θ)` returns the 4×4 EE pose in base frame: $T_{EE\_base}$.
- `T_cam_ee` is the fixed camera-to-EE transform: $T_{cam\_EE}$.
- Their product gives the camera-to-base transform: $T_{cam\_base} = T_{EE\_base} \cdot T_{cam\_EE}$.

---

## 5. Calibrating `T_cam_ee`

`T_cam_ee` is the rigid-body transform from the camera optical frame to the EE frame.  It must be measured once after physical installation.

A common calibration procedure:
1. Place a calibration target (checkerboard, ArUco marker) at a known location in the scene.
2. Move the arm to several configurations and record:
   - The EE pose in base frame from FK: $T_{EE\_base}$
   - The target pose in camera frame from the camera: $T_{target\_cam}$
3. Solve for $T_{cam\_EE}$ such that the target's base-frame pose is consistent across all configurations.

Tools: `easy_handeye` (ROS), `opencv_contrib` hand-eye calibration, or any standard eye-on-hand calibration library.

The result goes into `conf/config.yaml` under `arm.T_cam_ee` as a 4×4 matrix.  A `null` value disables pose-based picking (arm will not move to picked objects).

---

## 6. Object Frame and Grip Transformation

FoundationPose outputs $T_{obj\_cam}$ — the object's pose in camera frame.  The `ObjectSelector` converts it to base frame:

```python
T_obj_base = T_cam_base @ T_obj_cam     # 4×4
R_obj = T_obj_base[:3, :3]             # object orientation in base
p_obj = T_obj_base[:3, 3]             # object position in base (metres)
```

Grip candidates (from `GripPlanner`) are in object frame.  They are rotated to base frame:

```python
approach_base = R_obj @ grip.approach_axis   # unit vector in base frame
pick_xyz = R_obj @ grip.position_m + p_obj   # centroid in base frame
```

Since `SimpleGripPlanner` sets `grip.position_m = [0, 0, 0]` (centroid at origin), the pick position simplifies to `pick_xyz = p_obj`.

---

## 7. Depth Sanity Check

The startup depth sanity check in `BinPicker` verifies the full transform chain:

```python
p_cam = [(px - cx) * D_m / fx, (py - cy) * D_m / fy, D_m]   # D in metres
p_base = T_cam_base @ [*p_cam, 1.0]
```

The logged `p_base` for the centre pixel should be close to `hover_position` in X/Y and slightly below it in Z (the bin surface is below the camera).  If the values are wildly wrong, check `T_cam_ee`.
