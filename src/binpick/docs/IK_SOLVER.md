# IK Solver — Mathematical Reference

This document describes the mathematics behind `Lite6IKSolver` in [`src/binpick/ik_solver.py`](../ik_solver.py).

---

## 1. Coordinate Frames

| Frame | Origin | Orientation |
|-------|--------|-------------|
| **Base** | Arm mounting point | Z up, X forward |
| **Joint i** | Joint i axis intersection | Z along rotation axis |
| **End-effector (EE)** | Gripper midpoint | Defined by DH chain |

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
