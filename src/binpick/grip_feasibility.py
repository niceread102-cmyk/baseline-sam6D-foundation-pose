"""Grip feasibility checker.

Filters grip candidates AFTER FoundationPose has provided the object pose,
so checks are performed in robot-base frame with the actual detected orientation.

Two checks are available (either or both can be enabled):

  1. Height check — pick_xyz[2] must be above min_pick_height_m. Catches grips
     whose centroid, once transformed to base frame, is at or below table level.

  2. Orientation check — constructs a target EE rotation from the grip's approach
     axis (approach_axis_base = R_obj @ grip.approach_axis) and runs a 6-DOF IK
     solve to verify the arm can actually reach that pose.  Uses solve_6dof() on
     the IK solver; the Lite6IKSolver supports this.

The gripper_closing_axis parameter maps the grip's approach axis to an EE frame
column.  For the Lite6 the closing direction is the EE Y-axis (axis=1).

Usage (inside NearestObjectSelector.select()):

    checker = GripFeasibilityChecker(ik_solver, min_pick_height_m=0.02)
    ok, reason = checker.check(pick_xyz, approach_axis_base, theta_init)
    if not ok:
        log.info("SKIP grip: %s", reason)
        continue
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Minimum norm-squared of the approach component perpendicular to the closing axis.
# If below this threshold the closing axis is considered vertical and the fallback
# (X_base) is used as the approach direction instead.
_VERTICAL_THRESHOLD = 1e-6


class GripFeasibilityChecker:
    """Validates a grip candidate (in robot-base frame) before committing to IK.

    Parameters
    ----------
    ik_solver:
        IKSolver instance; must implement solve_6dof() for orientation checks.
    min_pick_height_m:
        Minimum allowed Z of the pick point in robot-base frame (metres).
        Objects sitting on a table at Z≈0 should have pick_xyz[2] > 0; a small
        positive threshold prevents asking the arm to reach through the table.
    check_orientation:
        If True, construct a target EE orientation from the approach axis and
        verify reachability via 6-DOF IK.  Adds some planning latency per grip.
    gripper_closing_axis:
        Which column of the EE rotation matrix corresponds to the gripper's
        closing direction (0=X, 1=Y, 2=Z).  For the Lite6, use 1 (Y-axis).
    """

    def __init__(
        self,
        ik_solver,
        min_pick_height_m: float = 0.02,
        check_orientation: bool = True,
        gripper_closing_axis: int = 1,
    ):
        self._ik           = ik_solver
        self._min_height   = float(min_pick_height_m)
        self._check_orient = check_orientation
        self._closing_axis = int(gripper_closing_axis)

        if self._closing_axis not in (0, 1, 2):
            raise ValueError(f"gripper_closing_axis must be 0, 1, or 2; got {self._closing_axis}")

    # ── public API ────────────────────────────────────────────────────────────

    def check(
        self,
        pick_xyz: np.ndarray,
        approach_axis_base: np.ndarray,
        theta_init: np.ndarray,
        target_R: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """Return (feasible, reason_string).

        Called AFTER the pose from FoundationPose has been applied so that
        pick_xyz and approach_axis_base are in robot-base frame.

        Parameters
        ----------
        target_R:
            Explicit target EE orientation (3×3, robot-base frame), used as-is
            for the 6-DOF IK check. Pass this when the grip candidate already
            defines a full orientation (``ManualGripPlanner``). If omitted, an
            orientation is heuristically constructed from ``approach_axis_base``
            via ``_build_target_R`` (legacy PCA-based grips).
        """
        pick_xyz = np.asarray(pick_xyz, dtype=float)

        # 1. Height check — independent of orientation
        if pick_xyz[2] < self._min_height:
            return False, (
                f"pick Z={pick_xyz[2]:.3f}m < min_pick_height={self._min_height:.3f}m "
                f"(grip centroid is at or below table level)"
            )

        # 2. 6-DOF orientation check
        if self._check_orient:
            R = target_R if target_R is not None else self._build_target_R(approach_axis_base)
            _, ok = self._ik.solve_6dof(pick_xyz, R, theta_init)
            if not ok:
                ax = approach_axis_base
                return False, (
                    f"6-DOF IK could not reach pose for "
                    f"approach_base=[{ax[0]:.3f},{ax[1]:.3f},{ax[2]:.3f}] "
                    f"at xyz=[{pick_xyz[0]:.3f},{pick_xyz[1]:.3f},{pick_xyz[2]:.3f}]"
                )

        return True, "ok"

    # ── internal helpers ──────────────────────────────────────────────────────

    def _build_target_R(self, approach_axis_base: np.ndarray) -> np.ndarray:
        """Construct a 3×3 EE rotation from the grip's approach (closing) axis.

        Convention
        ----------
        - The gripper's closing direction  = column self._closing_axis of R.
        - The "approach from above" direction = another column, constructed as the
          component of −Z_base perpendicular to the closing axis.
        - The third column = cross product to keep the matrix right-handed.

        If the closing axis is exactly vertical (parallel to Z_base), falls back
        to using X_base as the approach direction.
        """
        closing = approach_axis_base / np.linalg.norm(approach_axis_base)

        # Approach direction: downward in base frame, made perpendicular to closing
        down = np.array([0.0, 0.0, -1.0])
        approach_raw = down - np.dot(down, closing) * closing
        if np.linalg.norm(approach_raw) < _VERTICAL_THRESHOLD:
            # closing is vertical — fall back to X_base as approach
            approach_raw = np.array([1.0, 0.0, 0.0])
            approach_raw = approach_raw - np.dot(approach_raw, closing) * closing
        approach = approach_raw / np.linalg.norm(approach_raw)

        # Third axis: approach × closing (gives a right-handed tripod)
        third = np.cross(approach, closing)

        # {approach, closing, third} are mutually orthogonal unit vectors with:
        #   approach × closing = third
        #   closing  × third   = approach
        #   third    × approach = closing
        # Assign to matrix columns so that det(R)=1 (right-handed):
        if self._closing_axis == 0:       # X=closing → [c, t, a]  (t×a = c = X) ✓
            return np.column_stack([closing, third, approach])
        elif self._closing_axis == 1:     # Y=closing → [a, c, t]  (a×c = t = Z) ✓
            return np.column_stack([approach, closing, third])
        else:                             # Z=closing → [t, a, c]  (t×a = c = Z) ✓
            return np.column_stack([third, approach, closing])
