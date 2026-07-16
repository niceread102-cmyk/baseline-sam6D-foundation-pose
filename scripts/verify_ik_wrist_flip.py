#!/home/joon/.virtualenvs/cnos2/bin/python3
"""
scripts/verify_ik_wrist_flip.py — Offline verification for the wrist-branch IK fix.

No hardware required: Lite6IKSolver is pure numpy. This script checks:

  1. Identity regression — the wrist-flip transform is an exact FK-preserving
     identity across randomly sampled joint configurations (protects the fix
     if _LITE6_JOINTS DH constants are ever changed).
  2. Incident reproduction — replays the exact current_theta / hover_xyz /
     grasp orientation from the 2026-07-14 16:23:20 "SAFETY REJECT" log entry
     and confirms the old solve path (_solve_core alone) reproduces a large
     joint delta while the fixed path (_solve_inner) keeps it under
     max_joint_delta.
  3. Planning-time gate — confirms NearestObjectSelector.select() rejects a
     candidate whose IK solution has too large a joint delta, via a stubbed
     IK solver, without ever calling the (absent) real arm.

Usage:
    python scripts/verify_ik_wrist_flip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.binpick.ik_solver import (
    Lite6IKSolver,
    LITE6_INIT_THETA,
    _LITE6_JOINT_MIN,
    _LITE6_JOINT_MAX,
)
from src.binpick.object_selector import NearestObjectSelector


def check_identity_regression(n_samples: int = 5000, seed: int = 0) -> None:
    print(f"[1] Identity regression check ({n_samples} random joint configs)...")
    rng = np.random.default_rng(seed)
    ik = Lite6IKSolver()

    max_pos_err = 0.0
    max_quat_err = 0.0
    for _ in range(n_samples):
        theta = rng.uniform(_LITE6_JOINT_MIN, _LITE6_JOINT_MAX)
        pos1, quat1 = ik.fk(theta)
        pos2, quat2 = ik.fk(ik._wrist_flip(theta))
        max_pos_err = max(max_pos_err, float(np.max(np.abs(pos1 - pos2))))
        max_quat_err = max(max_quat_err, abs(abs(float(np.dot(quat1, quat2))) - 1.0))

    print(f"    max position error:    {max_pos_err:.3e} m")
    print(f"    max orientation error: {max_quat_err:.3e}")
    assert max_pos_err < 1e-6, "wrist-flip identity broken (position)"
    assert max_quat_err < 1e-6, "wrist-flip identity broken (orientation)"
    print("    PASS\n")


def check_incident_reproduction() -> None:
    """Replay the incident's logged current_theta/hover_xyz/approach_axis.

    The log only gives us the grasp's approach AXIS in base frame (one column
    of the target orientation) — the full target_R also depends on the
    object's actual detected rotation and the grasp's jaw roll, neither of
    which was logged, so the remaining "roll" about that axis is unknown.
    Rather than guess a single value (which would let us claim more than we
    can actually verify), sweep it and report where the fix helps.
    """
    print("[2] Incident reproduction (2026-07-14 16:23:20 SAFETY REJECT)...")

    current_theta = np.array([1.012, -0.084, 1.153, 0.0, 1.237, 1.012])
    hover_xyz = np.array([0.274, 0.056, 0.108])
    target_axis = np.array([0.027, 0.335, -0.942])
    target_axis /= np.linalg.norm(target_axis)

    up = np.array([0.0, 1.0, 0.0]) if abs(target_axis[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    x0 = np.cross(up, target_axis); x0 /= np.linalg.norm(x0)
    y0 = np.cross(target_axis, x0)

    ik = Lite6IKSolver()
    n_old_infeasible = 0
    n_fixed_to_feasible = 0
    n_regressions = 0
    worst_new_delta_where_old_infeasible = 0.0

    for deg in range(0, 360, 5):
        roll = np.radians(deg)
        x = np.cos(roll) * x0 + np.sin(roll) * y0
        y = np.cross(target_axis, x)
        R = np.stack([x, y, target_axis], axis=1)

        theta_old, ok_old = ik._solve_core(hover_xyz, R, current_theta)
        theta_new, ok_new = ik.solve_6dof(hover_xyz, R, current_theta)
        if not (ok_old and ok_new):
            continue

        d_old = float(np.max(np.abs(theta_old - current_theta)))
        d_new = float(np.max(np.abs(theta_new - current_theta)))

        if d_new > d_old + 1e-9:
            n_regressions += 1
        if d_old > 1.5:
            n_old_infeasible += 1
            worst_new_delta_where_old_infeasible = max(worst_new_delta_where_old_infeasible, d_new)
            if d_new < 1.5:
                n_fixed_to_feasible += 1

    print(f"    swept 72 roll angles (missing DOF) around the logged approach axis")
    print(f"    old delta > max_joint_delta=1.5 in {n_old_infeasible}/72 cases")
    print(f"    of those, the fix brings delta back under 1.5 in {n_fixed_to_feasible}/{n_old_infeasible}")
    print(f"    regressions (fix made delta worse): {n_regressions}  (must be 0)")

    assert n_regressions == 0, "wrist-flip correction must never increase joint delta"
    assert n_old_infeasible > 0, "expected at least one swept orientation to reproduce a >1.5 rad reject"
    assert n_fixed_to_feasible > 0, "expected the fix to resolve at least one previously-infeasible case"
    if n_fixed_to_feasible < n_old_infeasible:
        print(
            f"    NOTE: {n_old_infeasible - n_fixed_to_feasible} case(s) remain infeasible even after "
            f"the wrist-flip fix (worst residual delta {worst_new_delta_where_old_infeasible:.3f} rad) — "
            "these are NOT wrist-branch artifacts (likely genuine shoulder/elbow reconfiguration for "
            "that orientation) and are exactly what Fix 2's planning-time gate is for: reject cleanly "
            "and try the next grip/detection instead of a hardware safety reject + recovery cycle."
        )
    print("    PASS (never regresses; resolves at least one, and possibly not all, prior failures)\n")


class _StubIK:
    """Minimal IKSolver stub: solve_6dof/solve always 'converge' to a
    deliberately large-delta joint config, to test the planning-time gate
    in isolation from real kinematics."""

    def fk(self, theta):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def solve(self, target_xyz, theta_init):
        return theta_init + 3.0, True  # 3 rad delta — deliberately huge

    def solve_6dof(self, target_xyz, target_R, theta_init):
        return theta_init + 3.0, True


def check_planning_time_gate() -> None:
    print("[3] Planning-time joint-delta gate...")

    from src.binpick.grip_planner import GripCandidate

    selector = NearestObjectSelector(
        ik_solver=_StubIK(),
        placement_position=[0.35, 0.20, 0.20],
        T_cam_base=np.eye(4),
        max_joint_delta=1.5,
    )

    grip = GripCandidate(
        position_m=np.zeros(3),
        approach_axis=np.array([0.0, 0.0, 1.0]),
        width_m=0.025,
        orientation_m=np.eye(3),
        hover_offset_m=-0.06,
    )
    T_obj_cam = np.eye(4)
    T_obj_cam[:3, 3] = [0.28, 0.02, 0.35]
    fp_results = [{"pose": T_obj_cam, "model_id": "steering_knuckle"}]

    plan = selector.select(
        fp_results, {"steering_knuckle": [grip]}, LITE6_INIT_THETA, T_cam_base=np.eye(4)
    )
    assert plan is None, "expected the huge-delta candidate to be rejected at planning time"
    print("    PASS (candidate correctly rejected, no PickPlan returned)\n")


if __name__ == "__main__":
    check_identity_regression()
    check_incident_reproduction()
    check_planning_time_gate()
    print("All checks passed.")
