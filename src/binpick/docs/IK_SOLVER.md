# IK Solver — Mathematical Reference

This document describes the mathematics behind `Lite6IKSolver` in [`src/binpick/ik_solver.py`](../ik_solver.py).

---

## 1. Coordinate Frames

| Frame | Origin | Orientation |
|-------|--------|-------------|
| **Base** | Arm mounting point | Z up, X forward |
| **Joint i** | Joint i axis intersection | Z along rotation axis |
| **End-effector (EE)** | Gripper tip | DH chain (6 joints) + a fixed tool offset (§12) — without the offset this is the bare wrist flange, not the tip |

All FK outputs and IK targets are expressed in the **robot-base frame**.

---

## 2. Forward Kinematics

The FK chain is defined by six joints.  Each joint contributes a fixed offset transform (position + ZYX Euler rotation from the URDF) followed by a variable rotation about the joint's local Z-axis:

$$T_{EE} = \prod_{i=1}^{6} T_{\text{offset},i} \cdot T_{z}(\theta_i)$$

where

$$T_{\text{offset},i} = \begin{bmatrix} R(\text{rpy}_i) & \mathbf{p}_i \\ 0 & 1 \end{bmatrix}, \quad T_z(\theta) = \begin{bmatrix} R_z(\theta) & 0 \\ 0 & 1 \end{bmatrix}$$

and $R_z(\theta)$ is a rotation by $\theta$ radians about Z.

The Euler convention is **ZYX** (applied as $R_z R_y R_x$):

$$R(\text{rpy}) = R_z(yaw) \cdot R_y(pitch) \cdot R_x(roll)$$

**Implementation**: `_fk_full()` iterates the chain, recording:
- $\hat{z}_i$ — Z-axis of joint $i$ in base frame (used as Jacobian rotation axes)
- $\mathbf{p}_i$ — origin of joint $i$ in base frame (used to compute moment arms)

---

## 3. Jacobian Construction

For a 6-DOF revolute-joint chain, the geometric Jacobian relates joint velocities $\dot{\boldsymbol{\theta}}$ to EE velocity $[\dot{\mathbf{p}}_{EE}; \boldsymbol{\omega}]$:

$$\mathbf{J} = \begin{bmatrix} \hat{z}_1 \times (\mathbf{p}_{EE} - \mathbf{p}_1) & \cdots & \hat{z}_6 \times (\mathbf{p}_{EE} - \mathbf{p}_6) \\ \hat{z}_1 & \cdots & \hat{z}_6 \end{bmatrix} \in \mathbb{R}^{6 \times 6}$$

- **Linear rows** (top 3): $J_{v,i} = \hat{z}_i \times (\mathbf{p}_{EE} - \mathbf{p}_i)$ — cross product gives the moment arm.
- **Angular rows** (bottom 3): $J_{\omega,i} = \hat{z}_i$ — revolute joint angular velocity = Z-axis of that joint frame.

---

## 4. Damped Pseudoinverse

Near kinematic singularities the plain pseudoinverse $J^+ = J^\top (J J^\top)^{-1}$ diverges because $J J^\top$ becomes ill-conditioned.  The damped pseudoinverse adds a regularisation term:

$$J^\dagger = J^\top \left( J J^\top + \lambda^2 I \right)^{-1}$$

where $\lambda$ is the `damping` parameter.  Larger $\lambda$ trades accuracy for stability near singularities.

The joint update per iteration is:

$$\Delta\boldsymbol{\theta} = \text{clip}(J^\dagger \Delta\mathbf{X},\; \pm\delta_{clip})$$
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \Delta\boldsymbol{\theta}$$

where $\alpha$ (`alpha`) is a step-size scaling factor and $\delta_{clip}$ (`step_clip`) is the per-element clip applied before scaling.

---

## 5. Rotation Error

For 6-DOF IK the error vector $\Delta\mathbf{X} \in \mathbb{R}^6$ stacks position and orientation errors:

$$\Delta\mathbf{X} = \begin{bmatrix} \mathbf{p}_{target} - \mathbf{p}_{EE} \\ \boldsymbol{\omega}_{err} \end{bmatrix}$$

The angular error $\boldsymbol{\omega}_{err}$ is extracted from the rotation error matrix:

$$R_{err} = R_{target} \cdot R_{current}^\top$$

For small angular errors $R_{err} \approx I + [\boldsymbol{\omega}]_\times$ (skew-symmetric perturbation), so:

$$\boldsymbol{\omega}_{err} = \frac{1}{2} \begin{bmatrix} R_{err,32} - R_{err,23} \\ R_{err,13} - R_{err,31} \\ R_{err,21} - R_{err,12} \end{bmatrix}$$

This is a first-order approximation that works well when the angular error is small.  It becomes inaccurate for large orientation mismatches (> ~90°).

---

## 6. Convergence and Joint Limits

The solver stops when $\|\Delta\mathbf{X}\| < \varepsilon$ (the `threshold` parameter, in mixed metres + radians units).

After each joint update, angles are:
1. **Wrapped** to $[-\pi, \pi]$ via $\theta \leftarrow \text{atan2}(\sin\theta, \cos\theta)$.
2. **Clamped** to the Lite 6 joint limits `_LITE6_JOINT_MIN` / `_LITE6_JOINT_MAX`.

If the solver exhausts `max_steps` without converging it returns `(None, False)`.

---

## 7. Algorithm Pseudocode

```
function SOLVE_INNER(target_pos, target_R, theta_init):
    theta = theta_init
    for step = 1 to max_steps:
        T, pos, _, z_axes, origins = FK_FULL(theta)
        R_err = target_R @ T[:3,:3].T
        dP = target_pos - pos
        dW = skew_to_vec(R_err) * 0.5
        dX = [dP; dW]                       # 6-vector
        if norm(dX) < threshold:
            return theta, converged=True
        J = build_jacobian(z_axes, origins, pos)   # 6×6
        J_pinv = J.T @ inv(J @ J.T + damping² * I)
        d_theta = clip(J_pinv @ dX, ±step_clip)
        theta = theta + alpha * d_theta
        theta = wrap_to_pi(theta)
        theta = clamp(theta, joint_min, joint_max)
    return None, converged=False
```

---

## 8. Trajectory Interpolation

`trajectory_xyz(theta_start, theta_end, n_steps)` generates a linearly-interpolated joint trajectory and returns the corresponding EE Cartesian positions.  This is used only for visualization (IK path arrows in the preview).

The interpolation is in **joint space** (straight line in $\theta$), which does not generally produce a straight line in Cartesian space.

---

## 9. Limitations

| Limitation | Effect | Mitigation |
|-----------|--------|------------|
| No explicit singularity detection | Near-singular poses cause slow convergence or failure | Increase `damping`; avoid configurations near elbow singularities |
| Fixed `target_rpy` in `solve()` | Cannot specify per-call EE orientation | Use `solve_6dof()` with an explicit `target_R` |
| Hard joint-limit clamping | Solver can stall near limits even when a solution exists inside the workspace | Provide a better `theta_init` away from the limits |
| Small-angle rotation error approximation | Inaccurate for large orientation mismatches | Ensure `theta_init` is close to the target configuration |
| Convergence not guaranteed at workspace boundary | Returns `(None, False)` | Reachability should be pre-checked with `GripFeasibilityChecker` |
| Wrist-branch ambiguity (joints 4-6) | **Mitigated as of §11** — the solver now always keeps the joint-space-nearest wrist branch. Shoulder/elbow (joints 1-3) redundancy and genuinely unreachable poses are still unmitigated. | See §11 |
| Tool offset not auto-detected offline | `Lite6IKSolver` defaults to zero TCP offset (bare flange) unless something sets it | `XArmController` reads the real, controller-configured offset at connect time (§12); for offline/standalone use, call `set_tcp_offset()` explicitly if the FK/IK result needs to represent a real gripper tip |

---

## 10. Configuration Parameters (`conf/ik/lite6.yaml`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `damping` | `0.1` | Regularisation λ in $(JJ^\top + \lambda^2 I)^{-1}$ |
| `alpha` | `0.5` | Step-size scaling |
| `max_steps` | `200` | Iteration cap |
| `threshold` | `1e-3` | Convergence tolerance (m + rad) |
| `step_clip` | `0.1` | Per-element joint velocity clamp (rad) |
| `target_rpy` | `[π, 0, 0]` | Fixed EE orientation (ZYX Euler, rad) |
| `tcp_offset_xyz` | `[0, 0, 0]` | Flange→tool-tip translation (metres) — see §12. Normally left at the default and set at runtime from the connected arm, not configured statically here. |
| `tcp_offset_rpy` | `[0, 0, 0]` | Flange→tool-tip rotation (ZYX Euler, rad) — see §12 |

---

## 11. Wrist-Flip Branch Correction

The Lite 6's wrist (joints 4-6) is spherical — their axes intersect at a point — which means, for
any shoulder/elbow configuration $(\theta_1, \theta_2, \theta_3)$, there are exactly two joint
solutions that produce the *identical* EE pose:

$$(\theta_4, \theta_5, \theta_6) \quad \text{and} \quad (\theta_4 + \pi,\; -\theta_5,\; \theta_6 + \pi)$$

This was verified numerically (not just asserted): sampling 5000 random joint configurations
across the full joint range and comparing `fk(theta)` against `fk(flip(theta))` gives position and
orientation differences on the order of `1e-16` — an exact identity for this DH chain, not an
approximation.

**Why this matters**: the single-seed damped-pseudoinverse solve in `_solve_core` has no reason to
prefer one branch over the other — it just follows the local gradient from `theta_init`. Near the
$\theta_5 \approx 0$ crossing, a *small* orientation-target change can cross from one branch to the
other, producing a joint solution that is technically valid (same EE pose) but requires a large,
unnecessary swing in joints 4-6 relative to `theta_init` — even when the much-closer other branch
achieves the exact same pose. This was the root cause of a real hardware incident: a ~20° tilt
between a scan pose and a grasp pose's target orientation produced a >2 rad swing on two wrist
joints, tripping `XArmController`'s `max_joint_delta` safety reject.

**The fix** (`Lite6IKSolver._wrist_flip` / `_closer_branch`, wrapping `_solve_core` inside
`_solve_inner`): after any successful solve, compute the flipped alternative and keep whichever is
closer to `theta_init` (per-joint max-abs distance). Since the flip is a closed-form
re-derivation — not a re-solve — this costs nothing extra on the common path, can only reduce or
leave unchanged the joint delta from `theta_init`, and is applied unconditionally to every `solve()`
and `solve_6dof()` call. If the primary seed fails to converge at all, `_solve_inner` retries once
from the flipped seed before giving up.

The flip is always joint-limit-safe: joints 4 and 6 have $\pm 6.28$ rad range (a $+\pi$ wrap always
fits), and joint 5's magnitude is unchanged by negation, so it stays within $\pm 2.16$ rad whenever
the input did.

This mitigates the wrist-branch case specifically. It does **not** address shoulder/elbow
(joints 1-3) redundancy, nor genuinely unreachable poses — those remain open limitations (see the
table above).

---

## 12. TCP (Tool) Offset — Flange vs. Real Fingertip

`_fk_full`'s DH chain (`_LITE6_JOINTS`) has exactly six links, ending at the 6th joint's own
rotation — i.e. the bare wrist **flange**, not wherever the physical gripper's fingers actually are.
Every `fk()`/`solve()`/`solve_6dof()` call computed the flange's pose unless something appends a
final flange→tool-tip transform.

**This was the root cause of a second real hardware collision** (after the wrist-branch and
EE-frame-remap fixes above had already resolved two earlier ones). The xArm controller already had
a TCP offset configured via `scripts/calibrate_T_cam_base.py --tcp-offset` (persisted on the
controller itself, so `get_position()` correctly reports the true fingertip) — but `Lite6IKSolver`
is a separate, from-scratch Python reimplementation of the kinematics with no knowledge of that
controller-side setting. So every Cartesian target we ever commanded (`hover_xyz`, `pick_xyz`,
`placement_position`, the scan `hover_position`) was positioning the **flange** there, meaning the
**real fingertip ended up one tool-length further along the approach axis than intended** — for a
grasp with only a 60mm hover backoff and a ~80mm tool length, the "hover" pose alone already put the
real fingertip *past* the intended grasp point, before any "descent" even began.

**Confirmed, not guessed**: `XArmController`'s own long-standing startup "FK check" log line
(comparing `get_position()` against `Lite6IKSolver.fk()` at the same joint angles) had been showing
a ~80mm discrepancy in *every single run's log*, consistently oriented along the flange's own local
Z axis, across multiple different arm poses (79.3mm / 80.1mm / 81.1mm measured across three distinct
configurations) — the signature of a fixed tool length, not noise or a DH bug elsewhere in the chain.

**The fix**: `Lite6IKSolver.set_tcp_offset(xyz_m, rpy_rad)` sets a fixed final transform (stored as
`self._tcp_T`, applied to `T` at the very end of `_fk_full`, after all six joints) representing
flange→tool-tip. `XArmController.__init__` reads the arm's own configured `tcp_offset` property
right after connecting (a brief sleep first, matching `calibrate_T_cam_base.py`'s own pattern, so
the real-time report stream has populated it) and pushes it into the IK solver — so the two always
agree, regardless of what the calibrated tool length actually is, without a second config value that
could silently drift out of sync. The Jacobian math needed **no changes**: `joint_origins`/`z_axes`
are captured before the tool transform is applied (they describe the joints themselves, unaffected
by whatever is rigidly attached past them), and the standard revolute-joint Jacobian column
`ẑᵢ × (pos − pᵢ)` is valid for *any* point rigidly attached to the end-effector — tool tip included —
so simply changing what `pos` means (tool tip instead of flange) is sufficient; verified the
wrist-flip identity (§11) also still holds exactly with a nonzero offset active.

Defaults to no offset (`[0,0,0]`, i.e. bare flange) for backward compatibility and for
offline/standalone use (e.g. `scripts/verify_ik_wrist_flip.py`) where there is no connected arm to
read a real offset from — call `set_tcp_offset()` explicitly in that case if the result needs to
represent a real gripper tip.
