"""IK solver classes for bin-picking.

Lite6IKSolver is a pure-numpy port of the damped-pseudoinverse IK from
arm_lite6/custom_ik.py, decoupled from ROS for offline planning.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# ── Lite 6 kinematic constants ────────────────────────────────────────────────
# [xyz_offset, rpy_offset] per joint (from arm_lite6/custom_ik.py)
_LITE6_JOINTS = [
    [[0,       0,       0.2435], [0,        0,        0       ]],
    [[0,       0,       0     ], [np.pi/2, -np.pi/2,  np.pi   ]],
    [[0.2002,  0,       0     ], [-np.pi,   0,        np.pi/2 ]],
    [[0.087,  -0.22761, 0     ], [np.pi/2,  0,        0       ]],
    [[0,       0,       0     ], [np.pi/2,  0,        0       ]],
    [[0,       0.0625,  0     ], [-np.pi/2, 0,        0       ]],
]
_LITE6_JOINT_MIN = np.array([-6.28, -2.61, -0.06, -6.28, -2.16, -6.28])
_LITE6_JOINT_MAX = np.array([ 6.28,  2.61,  5.23,  6.28,  2.16,  6.28])

LITE6_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
LITE6_INIT_THETA  = np.array([0.0, 0.7, 1.57, 0.0, 0.87, 0.0])


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _euler_R(x: float, y: float, z: float) -> np.ndarray:
    Rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _euler_R(*rpy)
    T[:3, 3] = xyz
    return T


def _R_to_quat(R: np.ndarray) -> np.ndarray:
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
    @abstractmethod
    def solve(
        self, target_xyz: np.ndarray, theta_init: np.ndarray
    ) -> tuple[np.ndarray | None, bool]:
        """Solve IK for target_xyz. Returns (theta, converged)."""

    @abstractmethod
    def fk(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics → (position_3, quaternion_wxyz_4)."""

    def fk_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Forward kinematics → 4×4 EE pose in robot-base frame.

        Default implementation reconstructs the matrix from fk() pos + quat.
        Subclasses with direct access to the full transform should override.
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
        """Solve IK for explicit position + orientation. Returns (None, False) if unsupported."""
        return None, False

    def trajectory_xyz(
        self, theta_start: np.ndarray, theta_end: np.ndarray, n_steps: int = 10
    ) -> np.ndarray:
        """EE positions along a linear joint interpolation, shape (n_steps, 3)."""
        positions = []
        for a in np.linspace(0.0, 1.0, n_steps):
            pos, _ = self.fk((1 - a) * theta_start + a * theta_end)
            positions.append(pos)
        return np.array(positions)

    @staticmethod
    def move_duration(theta_from: np.ndarray, theta_to: np.ndarray) -> float:
        """Estimated move time in seconds (mirrors custom_ik.py formula)."""
        return max(3.0, float(np.max(np.abs(theta_to - theta_from))) * 8.0)


# ── Concrete: UFactory Lite 6 ─────────────────────────────────────────────────

class Lite6IKSolver(IKSolver):
    """Damped-pseudoinverse IK for the UFactory Lite 6 (pure numpy, no ROS)."""

    def __init__(
        self,
        damping: float = 0.1,
        alpha: float = 0.5,
        max_steps: int = 200,
        threshold: float = 1e-3,
        step_clip: float = 0.1,
        jump_limit: float = 1.5,
        target_rpy: list[float] | None = None,
    ):
        self._damping   = damping
        self._alpha     = alpha
        self._max_steps = max_steps
        self._threshold = threshold
        self._step_clip = step_clip
        self._jump_limit = jump_limit
        self._target_R  = _euler_R(*(target_rpy or [np.pi, 0.0, 0.0]))

    # ── forward kinematics ────────────────────────────────────────────────

    def _fk_full(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
        """Returns (T_ee 4x4, pos 3, quat 4, z_axes, joint_origins)."""
        z_axes, joint_origins = [], []
        T = np.eye(4)
        for i, jnt in enumerate(_LITE6_JOINTS):
            T = T @ _make_T(jnt[0], jnt[1])
            z_axes.append(T[:3, 2].copy())
            joint_origins.append(T[:3, 3].copy())
            T = T @ _make_T([0, 0, 0], [0, 0, theta[i]])
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
        return self._solve_inner(target_xyz, self._target_R, theta_init)

    def solve_6dof(
        self,
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        theta_init: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """IK for an explicit 6-DOF target: position + orientation."""
        return self._solve_inner(target_xyz, np.asarray(target_R, dtype=float), theta_init)

    def _solve_inner(
        self,
        target_xyz: np.ndarray,
        target_R: np.ndarray,
        theta_init: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
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
