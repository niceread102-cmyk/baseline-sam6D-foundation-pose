# Grip Planning

This document describes `SimpleGripPlanner` and `GripFeasibilityChecker` in [`src/binpick/grip_planner.py`](../grip_planner.py) and [`src/binpick/grip_feasibility.py`](../grip_feasibility.py).

---

## 1. Overview

The grip planning pipeline has two stages:

```
CAD mesh ──► SimpleGripPlanner ──► GripCandidate[] (object frame)
                                          │
                                          │ (after FoundationPose)
                                          │
                                          ▼
                               GripFeasibilityChecker
                               ├── height check (pick_xyz[Z] > min_height)
                               └── 6-DOF IK reachability check
                                          │
                                          ▼
                                  feasible candidates ──► IK solve ──► PickPlan
```

Grip candidates are computed **once at startup** from the CAD model.  Feasibility is checked **per detection** after FoundationPose provides the object's pose in the scene.

---

## 2. `SimpleGripPlanner` — PCA-Based Grip Generation

### Scaling

Before analysis, mesh vertices are scaled from their raw CAD units to metres:

```python
vertices_m = raw_vertices * effective_scale
```

`effective_scale = mesh_scale × physical_scale`, where:

| Parameter | Purpose | Example |
|-----------|---------|---------|
| `mesh_scale` | CAD unit → metres | `0.001` (CAD in mm) |
| `physical_scale` | CAD model ≠ real object size | `3.0` (real object is 3× bigger than CAD) |

For example, if your STL file is in millimetres and the real part is exactly as modelled: `mesh_scale=0.001`, `physical_scale=1.0`.  If the CAD is a 1:3 scale mock and the STL is in mm: `mesh_scale=0.001`, `physical_scale=3.0`.

### Principal Component Analysis

The planner centres the vertex cloud and finds its principal axes via SVD:

```
centred = vertices_m - mean(vertices_m)
U, S, Vt = SVD(centred)
```

The rows of `Vt` are the three principal axes in **descending variance order**:

```
axis_0: longest extent   (most variance)
axis_1: second longest
axis_2: shortest extent  (least variance)
```

### Grip width computation

For each principal axis, the grip width is the cloud's extent along that axis:

$$w_i = \max(\text{centred} \cdot \hat{a}_i) - \min(\text{centred} \cdot \hat{a}_i)$$

Axes with $w_i \in [\texttt{min\_grip\_m},\, \texttt{max\_grip\_m}]$ become `GripCandidate`s.

### Visualisation

```
                axis_0 (widest) ─────────────────
              ╔══════════════════════════════════╗
              ║                                  ║
axis_1 ─────  ║        Object centroid (+)       ║
(medium)      ║                                  ║
              ╚══════════════════════════════════╝

  Gripper fingers close along axis_0 or axis_1 (whichever is in range).
  Grip candidates are sorted tightest-first (smallest width first).
```

### Centroid assumption

`GripCandidate.position_m` is always `[0, 0, 0]` because the mesh is centred before PCA.  The actual pick position in robot-base frame is therefore just the object's detected centroid:

$$\mathbf{p}_{pick} = R_{obj} \cdot [0,0,0] + \mathbf{p}_{obj} = \mathbf{p}_{obj}$$

**Limitation**: this assumption works well for symmetric, convex objects.  For elongated, multi-lobe, or L-shaped objects the geometric centroid may not be a valid grasp point (e.g. it might be in mid-air above a concavity).

---

## 3. `GripFeasibilityChecker` — Two-Stage Validation

Feasibility is checked after FoundationPose gives the object's pose, so all checks happen in **robot-base frame**.

### Check 1: Height floor

```python
if pick_xyz[2] < min_pick_height_m:
    return False, "pick Z below table level"
```

Default `min_pick_height_m = 0.02 m` (2 cm).  This prevents the planner from asking the arm to reach through the table for detections whose centroid is at or below Z = 0.

### Check 2: 6-DOF IK reachability

Constructs a target EE rotation from the grip's approach axis and verifies the arm can reach it via `Lite6IKSolver.solve_6dof()`.

#### Building the target EE rotation (`_build_target_R`)

Given the grip's closing axis $\hat{c}$ (in base frame), the checker builds an orthonormal EE frame:

1. **Closing direction** $\hat{c}$: the gripper fingers close along this axis.
2. **Approach direction** $\hat{a}$: how the gripper descends onto the object.
   - Start with $\hat{d} = [0, 0, -1]$ (straight down in base frame).
   - Remove the component along $\hat{c}$: $\hat{a}_{raw} = \hat{d} - (\hat{d} \cdot \hat{c})\hat{c}$.
   - If $\hat{c}$ is vertical ($\|\hat{a}_{raw}\| < \varepsilon$), fall back to $[1, 0, 0]$.
   - Normalise: $\hat{a} = \hat{a}_{raw} / \|\hat{a}_{raw}\|$.
3. **Third axis** $\hat{t} = \hat{a} \times \hat{c}$ — ensures a right-handed frame.

```
  Z (up)
  │
  │    ╔═══════╗
  │    ║       ║   ← gripper body
  │    ║   +   ║   ← pick_xyz (centroid)
  │    ║       ║
  │    ╚═══╤═══╝
  │        │ ĉ (closing axis, e.g. along object's long axis)
  └────────────── X/Y (base frame horizontal)

  â = component of −Ẑ perpendicular to ĉ
      (gripper approaches from above, tilted to align with ĉ)
```

The rotation matrix is assembled by assigning $\hat{a}$, $\hat{c}$, $\hat{t}$ to columns based on `gripper_closing_axis`:

| `gripper_closing_axis` | EE X-col | EE Y-col | EE Z-col |
|------------------------|----------|----------|----------|
| 0 (X = closing) | $\hat{c}$ | $\hat{t}$ | $\hat{a}$ |
| 1 (Y = closing) | $\hat{a}$ | $\hat{c}$ | $\hat{t}$ |
| 2 (Z = closing) | $\hat{t}$ | $\hat{a}$ | $\hat{c}$ |

For the Lite 6 the default is `gripper_closing_axis=1` (Y-axis closes).

---

## 4. Configuration

### `conf/grip_planner/simple.yaml`

`cad_models`, `effective_scale`, `min_grip_m`, and `max_grip_m` are injected by `BinPicker.__init__`.

### `conf/object_selector/nearest.yaml`

| Parameter | Default | Effect |
|-----------|---------|--------|
| `min_pick_height_m` | `0.02` | Height floor for check 1 |
| `check_grip_orientation` | `true` | Enable/disable check 2 |
| `gripper_closing_axis` | `1` | EE axis for gripper closing direction |

### `conf/config.yaml`

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gripper.min_grip_m` | `0.02` | Minimum grip width (metres) |
| `gripper.max_grip_m` | `0.0375` | Maximum grip width (metres) |

---

## 5. Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| Centroid-as-pick-point | Fails for non-convex or multi-lobe objects | Override `GripCandidate.position_m` in a custom planner |
| PCA gives symmetric grips | Both axis directions (±) could be tried; only one is used | Extend `SimpleGripPlanner` to generate candidates for both axis directions |
| Gripper orientation check is approximate | `_build_target_R` assumes downward approach; may miss valid tilted approaches | Add more approach directions to the feasibility check |
| No collision checking | Arm may collide with bin walls for feasible IK poses | Add workspace constraints to the feasibility checker |
