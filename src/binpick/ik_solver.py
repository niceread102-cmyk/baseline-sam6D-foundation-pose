"""IK solver abstractions for the bin-picking pipeline.

``Lite6IKSolver`` is a pure-numpy, damped-pseudoinverse IK solver for the
UFactory Lite 6 arm, derived from arm_lite6/custom_ik.py but decoupled from
ROS so it can be used for offline planning and simulation.

Coordinate convention:
  - All positions are in **metres**, robot-base frame.
  - All angles are in **radians**.
  - Quaternions use the ``[w, x, y, z]`` (scalar-first) convention.
  - The end-effector frame is defined by the DH chain in ``_LITE6_JOINTS``.

The abstract base class ``IKSolver`` defines the interface used by the rest of
the pipeline; swap in a different solver by implementing the three abstract
methods (``solve``, ``fk``, and optionally ``solve_6dof``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# ── Lite 6 kinematic constants ────────────────────────────────────────────────
# Each entry: [xyz_offset (m), rpy_offset (rad)] — the fixed joint-frame
# offset applied *before* the variable joint rotation about Z.
# Values measured from the Lite 6 URDF (consistent with arm_lite6/custom_ik.py).
_LITE6_JOINTS = [
    [[0,       0,       0.2435], [0,        0,        0       ]],
    [[0,       0,       0     ], [np.pi/2, -np.pi/2,  np.pi   ]],
    [[0.2002,  0,       0     ], [-np.pi,   0,        np.pi/2 ]],
    [[0.087,  -0.22761, 0     ], [np.pi/2,  0,        0       ]],
    [[0,       0,       0     ], [np.pi/2,  0,        0       ]],
    [[0,       0.0625,  0     ], [-np.pi/2, 0,        0       ]],
]
_LITE6_JOINT_MIN = np.array([-6.28, -2.61, -0.06, -6.28, -2.16, -6.28])  # rad
_LITE6_JOINT_MAX = np.array([ 6.28,  2.61,  5.23,  6.28,  2.16,  6.28])  # rad

LITE6_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LITE6_INIT_THETA  = np.array([0.0, 0.7, 1.57, 0.0, 0.87, 0.0])  # rad — safe upright pose


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _euler_R(x: float, y: float, z: float) -> np.ndarray:
    """ZYX Euler rotation matrix: R = Rz @ Ry @ Rx (angles in radians)."""
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_T(xyz, rpy) -> np.ndarray:
    """Build a 4×4 homogeneous transform from xyz (m) and ZYX Euler rpy (rad)."""
    T = np.eye(4)
    T[:3, :3] = _euler_R(*rpy)
    T[:3, 3] = xyz
    return T


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion [w, x, y, z] (scalar-first).

    Uses the numerically stable Shepperd method — selects the formula that
    avoids dividing by a near-zero denominator.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,                 (R[1, 2] + R[2, 1]) / s])
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


# ── Abstract base ─────────────────────────────────────────────────────────────

class IKSolver(ABC):
    """Interface for IK/FK solvers used by the bin-picking pipeline.

    All positions are in **metres** (robot-base frame) and all angles are in
    **radians**.  Quaternions follow ``[w, x, y, z]`` scalar-first convention.
    """

    @abstractmethod
    def solve(
        self, target_xyz: np.ndarray, theta_init: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        """Position-only IK: find joint angles that place the EE at target_xyz.

        Parameters
        ----------
        target_xyz:
            Desired end-effector position in robot-base frame (metres).
        theta_init:
            Starting joint configuration for the iterative solver (radians).

        Returns
        -------
        theta:
            Joint angles (rad) achieving the target, or ``None`` if failed.
        converged:
            True if the solver reached the convergence threshold.
        """

    @abstractmethod
    def fk(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics.

        Returns
        -------
        pos:
            End-effector position in robot-base frame (metres), shape (3,).
        quat:
            End-effector orientation as [w, x, y, z] quaternion, shape (4,).
        """

    def fk_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Forward kinematics as a 4×4 SE(3) pose matrix in robot-base frame.

        The default implementation reconstructs the matrix from ``fk()`` pos +
        quat.  Subclasses with direct access to the full transform chain should
        override this for efficiency and numerical precision.
        """
        pos, q = self.fk(theta)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),  2*(y*z - w*x)],
            [    2*(x*z - w*y),   2*(y*z + w*x),  1 - 2*(x*x + y*y)],
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    def solve_6dof(
        self,
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        theta_init: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Full 6-DOF IK: position + orientation.

        The base implementation returns ``(None, False)`` (unsupported).
        Override in subclasses that support orientation control.

        Parameters
        ----------
        target_xyz:
            Desired EE position (metres, robot-base frame).
        target_R:
            Desired EE orientation as a 3×3 rotation matrix (robot-base frame).
        theta_init:
            Starting joint configuration (radians).
        """
        return None, False

    def trajectory_xyz(
        self, theta_start: np.ndarray, theta_end: np.ndarray, n_steps: int = 10
    ) -> np.ndarray:
        """EE Cartesian positions along a linearly-interpolated joint trajectory.

        Parameters
        ----------
        theta_start / theta_end:
            Start and end joint configurations (radians).
        n_steps:
            Number of samples along the trajectory (inclusive of endpoints).

        Returns
        -------
        positions:
            Shape ``(n_steps, 3)`` array of EE positions (metres, robot-base frame).
        """
        positions = []
        for a in np.linspace(0.0, 1.0, n_steps):
            pos, _ = self.fk((1 - a) * theta_start + a * theta_end)
            positions.append(pos)
        return np.array(positions)

    @staticmethod
    def move_duration(theta_from: np.ndarray, theta_to: np.ndarray) -> float:
        """Rough estimate of move time in seconds.

        Formula: ``max(3.0, max_abs_joint_delta * 8.0)``.  The constant 8.0
        is empirically tuned for the Lite 6 at a nominal speed of ~0.3 rad/s
        with some margin for acceleration ramp-up.  This is used for display
        only — the arm controller sets its own velocity limits.
        """
        return max(3.0, float(np.max(np.abs(theta_to - theta_from))) * 8.0)


# ── Concrete: UFactory Lite 6 ─────────────────────────────────────────────────

class Lite6IKSolver(IKSolver):
    """Damped-pseudoinverse IK/FK for the UFactory Lite 6 (pure numpy, no ROS).

    Uses an iterative Jacobian-based solver with a damped pseudoinverse to
    avoid singularity blow-up.  Both position-only (``solve``) and full 6-DOF
    (``solve_6dof``) IK are supported.

    The default end-effector orientation ``target_rpy=[π, 0, 0]`` points the
    gripper straight down (suitable for top-down grasps).

    Limitations
    -----------
    - No explicit singularity detection; relies on the damping factor to limit
      step size near singularities.
    - ``solve()`` uses a fixed ``target_rpy`` orientation — orientation is not
      a per-call argument.  Use ``solve_6dof()`` when the EE orientation matters.
    - Convergence is not guaranteed near or outside the workspace boundary.
    - Joint limits are clamped hard; this can cause the solver to stall near
      limit boundaries even if a solution exists in the interior.
    """

    def __init__(
        self,
        damping: float = 0.1,
        alpha: float = 0.5,
        max_steps: int = 200,
        threshold: float = 1e-3,
        step_clip: float = 0.1,
        target_rpy: list[float] | None = None,
        tcp_offset_xyz: list[float] | None = None,
        tcp_offset_rpy: list[float] | None = None,
    ):
        """
        Parameters
        ----------
        damping:
            Regularisation factor λ in the damped pseudoinverse
            ``J† = Jᵀ (J Jᵀ + λ²I)⁻¹``.  Larger values trade accuracy for
            stability near singularities.
        alpha:
            Step-size scaling (0 < α ≤ 1).  Values below 1 add gradient-descent
            damping; the clipped step is ``Δθ = clip(J† dX, ±step_clip) * α``.
        max_steps:
            Maximum number of Jacobian iterations before declaring failure.
        threshold:
            Convergence threshold on ``‖[dP; dW]‖`` (combined position error in
            metres and angular error in radians).
        step_clip:
            Maximum joint angle change per iteration (radians), applied
            element-wise before the alpha scaling.
        target_rpy:
            Fixed EE orientation as ZYX Euler angles [r, p, y] (radians).
            Defaults to ``[π, 0, 0]`` (gripper pointing straight down).
        tcp_offset_xyz / tcp_offset_rpy:
            Fixed tool transform (metres / radians, ZYX Euler) applied AFTER
            the 6th joint — i.e. flange → actual gripper-tip. Without this,
            ``fk``/``solve``/``solve_6dof`` compute the bare wrist flange, not
            the real fingertip, so every Cartesian target is silently reached
            short (or, if the caller thinks in fingertip terms, the real
            fingertip overshoots by the tool length along the approach axis —
            this was the root cause of a real hardware collision, confirmed
            by comparing this solver's ``fk()`` against the xArm SDK's own
            ``get_position()`` at matching joint angles: a ~80mm discrepancy,
            consistent in magnitude across multiple different arm poses,
            aligned almost entirely with the flange's own local Z axis).
            Defaults to no offset (``[0,0,0]``) for backward compatibility and
            standalone/offline testing; ``XArmController`` sets this from the
            connected arm's actual configured ``tcp_offset`` at startup, since
            that (not a hardcoded config value) is ground truth for whatever
            gripper is physically attached. Use ``set_tcp_offset()`` to update
            after construction.
        """
        self._damping   = damping
        self._alpha     = alpha
        self._max_steps = max_steps
        self._threshold = threshold
        self._step_clip = step_clip
        self._target_R  = _euler_R(*(target_rpy or [np.pi, 0.0, 0.0]))
        self._tcp_T     = np.eye(4)
        self.set_tcp_offset(tcp_offset_xyz, tcp_offset_rpy)

    def set_tcp_offset(
        self, xyz_m: list[float] | None = None, rpy_rad: list[float] | None = None
    ) -> None:
        """Set (or clear, with both args None) the flange→tool-tip transform.

        Applied as a fixed final link after the 6th joint in ``_fk_full`` — see
        the constructor docstring for why this exists and how its value was
        derived. Safe to call any time after construction (e.g. once the real
        arm's configured TCP offset is known, right after connecting).
        """
        xyz = xyz_m or [0.0, 0.0, 0.0]
        rpy = rpy_rad or [0.0, 0.0, 0.0]
        self._tcp_T = _make_T(xyz, rpy)

    # ── forward kinematics ────────────────────────────────────────────────

    def _fk_full(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
        """Full forward kinematics returning all quantities needed by the IK solver.

        Chains the six joint transforms in order.  Each joint contributes:
        1. A fixed offset transform ``_make_T(jnt.xyz, jnt.rpy)`` that positions
           the joint frame relative to the previous one.
        2. A variable rotation about the joint's Z-axis by ``theta[i]``.
        Finally, the fixed flange→tool-tip transform (``self._tcp_T``, see
        ``set_tcp_offset``) is applied — ``T_ee``/``pos``/``quat`` below are
        the actual gripper tip, not the bare wrist flange.

        Returns
        -------
        T_ee:
            4×4 EE (tool-tip) pose in robot-base frame.
        pos:
            EE (tool-tip) position (metres), shape (3,).
        quat:
            EE (tool-tip) quaternion [w, x, y, z], shape (4,).
        z_axes:
            List of 6 unit vectors — the Z-axis of each joint frame in
            robot-base coordinates.  Used as the Jacobian rotation axes.
        joint_origins:
            List of 6 positions (metres) — the origin of each joint frame in
            robot-base coordinates.  Used to compute the moment arm for the
            linear Jacobian columns.  Unaffected by the tool transform — the
            Jacobian's ``z_i × (pos - joint_origins[i])`` formula is valid for
            any point rigidly attached to the end-effector, tool tip included.
        """
        z_axes, joint_origins = [], []
        T = np.eye(4)
        for i, jnt in enumerate(_LITE6_JOINTS):
            T = T @ _make_T(jnt[0], jnt[1])
            z_axes.append(T[:3, 2].copy())
            joint_origins.append(T[:3, 3].copy())
            T = T @ _make_T([0, 0, 0], [0, 0, theta[i]])
        T = T @ self._tcp_T
        return T, T[:3, 3].copy(), _R_to_quat(T[:3, :3]), z_axes, joint_origins

    def fk(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, pos, quat, _, _ = self._fk_full(theta)
        return pos, quat

    def fk_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Forward kinematics → 4×4 EE pose in robot-base frame (direct from DH chain)."""
        T_ee, _, _, _, _ = self._fk_full(theta)
        return T_ee.copy()

    # ── inverse kinematics ────────────────────────────────────────────────

    def solve(
        self, target_xyz: np.ndarray, theta_init: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        """Position-only IK using the fixed ``target_rpy`` orientation."""
        return self._solve_inner(target_xyz, self._target_R, theta_init)

    def solve_6dof(
        self,
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        theta_init: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """IK for an explicit 6-DOF target (position + orientation)."""
        return self._solve_inner(target_xyz, np.asarray(target_R, dtype=float), theta_init)

    def _solve_inner(
        self,
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        theta_init: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Solve, then correct for the Lite 6's wrist-branch ambiguity.

        The spherical wrist (joints 4-6) has two joint-space solutions for any
        given EE orientation: ``(θ4, θ5, θ6)`` and ``(θ4+π, −θ5, θ6+π)`` are
        exactly equivalent (see ``_wrist_flip``). A single-seed Jacobian solve
        can land on either branch — usually the nearby one, but near the
        θ5≈0 crossing it can jump to the far one even for a small orientation
        change, producing a huge (but technically valid) joint delta from
        ``theta_init``. Since the flip is a free, exact re-derivation (no
        extra solve), always keep whichever branch is closer to ``theta_init``.
        """
        theta, ok = self._solve_core(target_xyz, target_R, theta_init)
        if ok:
            return self._closer_branch(theta, theta_init), True

        # Primary seed failed to converge at all — retry once from the
        # analytically-equivalent flipped seed before giving up.
        theta2, ok2 = self._solve_core(target_xyz, target_R, self._wrist_flip(theta_init))
        if ok2:
            return self._closer_branch(theta2, theta_init), True
        return None, False

    @staticmethod
    def _wrist_flip(theta: np.ndarray) -> np.ndarray:
        """Analytically-equivalent alternate joint solution for the SAME EE pose.

        Joints 4-6 form a spherical wrist (their axes intersect at a point):
        for any θ1-3, ``FK(θ)`` and ``FK(θ with θ4+=π, θ5=−θ5, θ6+=π)`` are
        identical — verified numerically to float precision across the full
        joint range (random-sample FK-equivalence check; see
        ``docs/IK_SOLVER.md`` §11). Always joint-limit-safe: joints 4 and 6
        have ±6.28 rad range (a +π wrap always fits), and joint 5's magnitude
        is unchanged by negation, so it stays within ±2.16 rad whenever the
        input did.
        """
        flipped = theta.copy()
        flipped[3] = np.arctan2(np.sin(flipped[3] + np.pi), np.cos(flipped[3] + np.pi))
        flipped[4] = -flipped[4]
        flipped[5] = np.arctan2(np.sin(flipped[5] + np.pi), np.cos(flipped[5] + np.pi))
        return np.clip(flipped, _LITE6_JOINT_MIN, _LITE6_JOINT_MAX)

    @staticmethod
    def _closer_branch(theta: np.ndarray, theta_init: np.ndarray) -> np.ndarray:
        """Return whichever of ``{theta, _wrist_flip(theta)}`` is nearer ``theta_init``.

        Uses per-joint max-abs distance, matching ``XArmController``'s
        per-joint ``max_joint_delta`` safety-gate semantics.
        """
        flipped = Lite6IKSolver._wrist_flip(theta)
        d      = np.max(np.abs(theta - theta_init))
        d_flip = np.max(np.abs(flipped - theta_init))
        return flipped if d_flip < d else theta

    def _solve_core(
        self,
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        theta_init: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Damped-pseudoinverse iterative IK solver (shared by solve and solve_6dof).

        Algorithm (per iteration)
        -------------------------
        1. Compute FK to get current EE pose ``(T, pos)``.
        2. Position error: ``dP = target_pos - pos``.
        3. Rotation error: ``R_err = target_R @ T[:3,:3]ᵀ``, extract angular
           velocity ``dW = skew_to_vec(R_err) * 0.5`` (small-angle approximation
           of the axis-angle difference).
        4. Stack: ``dX = [dP; dW]`` (6-vector).  Stop if ``‖dX‖ < threshold``.
        5. Build 6×6 Jacobian:
           - Linear rows: ``J[:3, i] = ẑᵢ × (p_EE − pᵢ)``
           - Angular rows: ``J[3:, i] = ẑᵢ``
        6. Damped pseudoinverse: ``J† = Jᵀ (J Jᵀ + λ²I)⁻¹``.
        7. Joint update: ``Δθ = clip(J† dX, ±step_clip)``;
           ``θ ← θ + α Δθ``.
        8. Wrap angles to ``[-π, π]`` via ``atan2(sin, cos)``.
        9. Clamp to joint limits.
        """
        theta      = theta_init.copy()
        target_pos = np.asarray(target_xyz, dtype=float)

        for _ in range(self._max_steps):
            T, pos, _, z_axes, joint_origins = self._fk_full(theta)

            R_err = target_R @ T[:3, :3].T
            dP = target_pos - pos
            dW = np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ]) * 0.5
            dX = np.append(dP, dW)

            if np.linalg.norm(dX) < self._threshold:
                return theta, True

            J = np.zeros((6, 6))
            for i in range(6):
                r = pos - joint_origins[i]
                J[:3, i] = np.cross(z_axes[i], r)
                J[3:, i] = z_axes[i]

            J_pinv  = J.T @ np.linalg.inv(J @ J.T + self._damping ** 2 * np.eye(6))
            d_theta = np.clip(J_pinv @ dX, -self._step_clip, self._step_clip)
            theta   = theta + self._alpha * d_theta
            theta   = np.arctan2(np.sin(theta), np.cos(theta))
            theta   = np.clip(theta, _LITE6_JOINT_MIN, _LITE6_JOINT_MAX)

        return None, False
