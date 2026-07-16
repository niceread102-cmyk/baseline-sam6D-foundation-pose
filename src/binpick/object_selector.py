"""Object selection and pick planning.

NearestObjectSelector picks the closest feasible object to the current
end-effector position. It requires T_cam_base (camera-to-robot-base 4×4
transform) to convert FoundationPose camera-frame poses into base frame.

If T_cam_base is None, poses are used as-is (camera frame ≈ base frame),
which is only valid for unit-testing without a physical arm.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .grip_feasibility import GripFeasibilityChecker
from .grip_planner import GripCandidate
from .ik_solver import IKSolver

log = logging.getLogger(__name__)


def _grasp_to_ee_frame(R_grasp: np.ndarray, closing_axis: int) -> np.ndarray:
    """Remap a grasp's authoring-frame rotation into the robot EE's own frame.

    ``ManualGripPlanner`` grasps (see ``grip_planner._grasp_rotation`` and the
    "Manual 6DoF grasp definition" section of ``photo_neo_capture_test.ipynb``)
    are authored in a frame where column 0 is the jaw/closing axis and column
    2 is the approach axis. That is generic to however the grasp was drawn —
    it is NOT the Lite 6's own physical EE-frame convention, where column 2 is
    always the approach axis (confirmed by the validated top-down default,
    ``Lite6IKSolver``'s ``target_rpy=[pi,0,0]``, whose Z column points straight
    down) but the physical closing axis is ``gripper_closing_axis`` (Y for the
    Lite 6 — see ``conf/object_selector/nearest.yaml``), not X.

    Feeding a grasp's rotation straight to ``solve_6dof`` without this remap
    points the robot's OWN X axis (not its physical closing axis) along the
    intended jaw direction, while its actual closing axis (Y) ends up wherever
    the grasp's incidental third column points — i.e. the gripper executes the
    correct approach direction but rolled ~90° around it, so it can strike the
    object from an unintended side while still descending along a
    plausible-looking line. This was confirmed both by rederiving the
    remap analytically and by checking it numerically (proper rotation,
    approach axis preserved, jaw axis relocated to the physical closing
    axis) across randomized grasps.

    ``closing_axis=0`` needs no remap (the grasp's own convention already
    matches). ``closing_axis=1`` swaps jaw onto column 1 and recomputes
    column 0 as the cross product to keep a right-handed frame.
    ``closing_axis=2`` is invalid for this remap: a parallel gripper's closing
    axis can't coincide with its approach axis (column 2).
    """
    if closing_axis == 0:
        return R_grasp
    jaw = R_grasp[:, 0]
    approach = R_grasp[:, 2]
    if closing_axis == 1:
        third = np.cross(jaw, approach)
        return np.stack([third, jaw, approach], axis=1)
    raise ValueError(
        f"gripper_closing_axis={closing_axis} is incompatible with manual 6-DOF "
        "grasps: the closing axis can't coincide with the approach axis "
        "(column 2). Use 0 or 1."
    )


@dataclass
class PickPlan:
    """Fully resolved pick-and-place plan returned by ``ObjectSelector.select()``.

    Attributes
    ----------
    fp_result:
        FoundationPose result dict for the chosen object.  Contains at least
        ``"pose"`` (4×4 transform, camera frame, metres) and ``"model_id"``.
    grip:
        Selected ``GripCandidate`` (position and approach axis in object frame).
    theta_pick:
        IK solution (radians, shape (6,)) for the pick position in robot-base frame.
    theta_place:
        IK solution (radians, shape (6,)) for the placement position in robot-base frame.
    theta_hover:
        IK solution (radians, shape (6,)) for the grip's pre-grasp/retreat hover
        pose (same orientation as ``theta_pick``, backed off along the approach
        axis), or ``None`` if the grip has no orientation / hover offset
        (``SimpleGripPlanner`` grips, or a ``ManualGripPlanner`` grip with
        ``hover_dist_m == 0``).
    """

    fp_result:   dict
    grip:        GripCandidate
    theta_pick:  np.ndarray
    theta_place: np.ndarray
    theta_hover: np.ndarray | None = None


class ObjectSelector(ABC):
    @abstractmethod
    def select(
        self,
        fp_results: list[dict],
        grips: dict[str, list[GripCandidate]],
        current_theta: np.ndarray,
        T_cam_base: np.ndarray | None = None,
    ) -> PickPlan | None:
        """Choose one object and return a fully resolved PickPlan, or None."""


class NearestObjectSelector(ObjectSelector):
    """Selects the object whose grip position is nearest to the current EE.

    Iterates all detected objects × all grip candidates, solves IK for both
    pick and placement, and returns the closest feasible option.

    Parameters
    ----------
    ik_solver:
        Used for IK planning (not the ROS execution node).
    placement_position:
        Target XYZ in robot-base frame (metres) where objects are deposited.
    T_cam_base:
        4×4 homogeneous transform from camera frame to robot-base frame,
        i.e.  p_base = T_cam_base @ p_cam_hom.
        If None, object poses are assumed to already be in base frame.
    max_joint_delta:
        Reject a plan at planning time if any waypoint (hover/pick/place)
        requires a per-joint change exceeding this many radians from the
        joint config the arm will actually be commanded from at that point —
        mirrors ``XArmController.move_to_angles``'s hardware safety gate, so
        a bad candidate is skipped (and the next grip/detection tried)
        *before* the gripper opens and a move is attempted, rather than only
        being caught late by the arm controller (which then has to recover
        and re-home). ``0`` disables the check (same convention as
        ``XArmController``'s ``max_joint_delta``).
    """

    def __init__(
        self,
        ik_solver: IKSolver,
        placement_position: list[float],
        T_cam_base: np.ndarray | None = None,
        min_pick_height_m: float = 0.02,
        check_grip_orientation: bool = True,
        gripper_closing_axis: int = 1,
        max_joint_delta: float = 0.0,
    ):
        self._ik        = ik_solver
        self._place_xyz = np.asarray(placement_position, dtype=float)
        self._T_cb      = np.asarray(T_cam_base) if T_cam_base is not None else None
        self._max_joint_delta = float(max_joint_delta)
        self._closing_axis    = int(gripper_closing_axis)

        if self._T_cb is None:
            log.warning(
                "NearestObjectSelector: T_cam_base is None — object poses used "
                "in camera frame. Set arm.T_cam_ee in config for physical operation."
            )

        self._feasibility = GripFeasibilityChecker(
            ik_solver=ik_solver,
            min_pick_height_m=min_pick_height_m,
            check_orientation=check_grip_orientation,
            gripper_closing_axis=gripper_closing_axis,
        )

    def _to_base(self, T_obj_cam: np.ndarray, T_cb: np.ndarray | None) -> np.ndarray:
        """Convert object pose from camera frame to robot-base frame."""
        if T_cb is None:
            return T_obj_cam
        # T_cam_base maps cam→base: p_base = T_cam_base @ p_cam_hom
        return T_cb @ T_obj_cam

    def _delta_ok(
        self, theta_new: np.ndarray, theta_from: np.ndarray, label: str, i_grip: int
    ) -> bool:
        """Planning-time joint-delta feasibility gate (see max_joint_delta docstring)."""
        if self._max_joint_delta <= 0:
            return True
        delta = np.abs(theta_new - theta_from)
        if np.any(delta > self._max_joint_delta):
            log.info(
                "  SKIP grip #%d: %s joint delta %s rad exceeds max_joint_delta=%.2f "
                "(rejected at planning time, before opening gripper / committing to the move)",
                i_grip, label, np.round(delta, 3).tolist(), self._max_joint_delta,
            )
            return False
        return True

    def select(
        self,
        fp_results: list[dict],
        grips: dict[str, list[GripCandidate]],
        current_theta: np.ndarray,
        T_cam_base: np.ndarray | None = None,
    ) -> PickPlan | None:
        """Select the closest feasible object and return a resolved ``PickPlan``.

        Parameters
        ----------
        fp_results:
            List of FoundationPose result dicts.  Each dict must contain:
            - ``"pose"``: 4×4 transform in **camera frame**, translation in **metres**.
            - ``"model_id"``: string key into ``grips``.
        grips:
            Pre-computed grip candidates per model (``model_id → list[GripCandidate]``),
            with positions and axes in **object frame**.
        current_theta:
            Current arm joint angles (radians), used to compute the EE position for
            nearest-object ranking and as the IK seed for pick planning.
        T_cam_base:
            4×4 transform mapping camera-frame points to robot-base frame
            (``p_base = T_cam_base @ p_cam_hom``).  Per-call value takes precedence
            over the instance-level default.  Pass ``None`` to use the default.

        Returns
        -------
        ``PickPlan`` for the closest feasible object/grip pair, or ``None`` if no
        feasible plan was found.
        """
        # Per-call T_cam_base takes precedence; fall back to instance-level (fixed-camera mode)
        T_cb = np.asarray(T_cam_base) if T_cam_base is not None else self._T_cb

        current_ee, _ = self._ik.fk(current_theta)
        best: PickPlan | None = None
        best_dist = float("inf")

        for result in fp_results:
            model_id   = result["model_id"]
            candidates = grips.get(model_id, [])
            if not candidates:
                log.debug(f"No grip candidates for model '{model_id}', skipping.")
                continue

            T_obj_cam  = np.asarray(result["pose"])  # 4×4, translation in metres

            # Negative Z in camera frame means the pose is physically behind the camera —
            # a failed FoundationPose detection; skip it.
            z_cam = T_obj_cam[2, 3]
            if z_cam < 0:
                log.warning(
                    "Object '%s' has cam-frame Z=%.3fm (< 0) — bad detection, skipping.",
                    model_id, z_cam,
                )
                continue

            T_obj_base = self._to_base(T_obj_cam, T_cb)
            R_obj      = T_obj_base[:3, :3]
            p_obj      = T_obj_base[:3, 3]

            log.info(
                "Object '%s'  p_obj_cam=[%.3f, %.3f, %.3f]m  "
                "→  p_obj_base=[%.3f, %.3f, %.3f]m",
                model_id,
                T_obj_cam[0, 3], T_obj_cam[1, 3], T_obj_cam[2, 3],
                p_obj[0], p_obj[1], p_obj[2],
            )

            for i_grip, grip in enumerate(candidates):
                # Approach axis and pick position in robot-base frame
                approach_base = R_obj @ grip.approach_axis
                pick_xyz      = R_obj @ grip.position_m + p_obj

                # Full 6-DOF target orientation + hover waypoint, only available
                # for grips with an explicit orientation (ManualGripPlanner).
                # The grasp's own authoring frame (jaw=X, approach=Z) is remapped
                # to the robot's physical EE frame (approach=Z always; jaw on
                # whichever column gripper_closing_axis names) — see
                # _grasp_to_ee_frame's docstring for why this remap is required.
                R_pick_base = (
                    _grasp_to_ee_frame(R_obj @ grip.orientation_m, self._closing_axis)
                    if grip.orientation_m is not None else None
                )
                hover_xyz = (
                    pick_xyz + R_pick_base[:, 2] * grip.hover_offset_m
                    if R_pick_base is not None and grip.hover_offset_m != 0.0
                    else None
                )

                log.info(
                    "  Grip #%d: width=%.1fmm"
                    "  pick_xyz=[%.3f, %.3f, %.3f]m"
                    "  approach_base=[%.3f, %.3f, %.3f]"
                    "  hover=%s",
                    i_grip, grip.width_m * 1000,
                    pick_xyz[0], pick_xyz[1], pick_xyz[2],
                    approach_base[0], approach_base[1], approach_base[2],
                    "none" if hover_xyz is None else
                    f"[{hover_xyz[0]:.3f}, {hover_xyz[1]:.3f}, {hover_xyz[2]:.3f}]m",
                )

                # Feasibility: height + optional 6-DOF orientation check
                ok_feas, reason = self._feasibility.check(
                    pick_xyz, approach_base, current_theta, target_R=R_pick_base
                )
                if not ok_feas:
                    log.info("  SKIP grip #%d: %s", i_grip, reason)
                    continue

                theta_hover = None
                if hover_xyz is not None:
                    theta_hover, ok_hover = self._ik.solve_6dof(hover_xyz, R_pick_base, current_theta)
                    if not ok_hover:
                        log.info("  SKIP grip #%d: hover-pose IK failed", i_grip)
                        continue
                    if not self._delta_ok(theta_hover, current_theta, "hover", i_grip):
                        continue

                seed_theta = theta_hover if theta_hover is not None else current_theta
                if R_pick_base is not None:
                    theta_pick, ok_pick = self._ik.solve_6dof(pick_xyz, R_pick_base, seed_theta)
                else:
                    theta_pick, ok_pick = self._ik.solve(pick_xyz, seed_theta)
                if not ok_pick:
                    log.info("  SKIP grip #%d: pick-pose IK failed", i_grip)
                    continue
                if not self._delta_ok(theta_pick, seed_theta, "pick", i_grip):
                    continue

                theta_place, ok_place = self._ik.solve(self._place_xyz, theta_pick)
                if not ok_place:
                    log.info("  SKIP grip #%d: place IK failed", i_grip)
                    continue
                # BinPicker._execute_pick retreats to theta_hover (if defined) before
                # _execute_place moves to theta_place, so that's the arm's actual
                # position before this move — not theta_pick's IK seed.
                place_from = theta_hover if theta_hover is not None else theta_pick
                if not self._delta_ok(theta_place, place_from, "place", i_grip):
                    continue

                dist = float(np.linalg.norm(pick_xyz - current_ee))
                if dist < best_dist:
                    best_dist = dist
                    best = PickPlan(
                        fp_result=result,
                        grip=grip,
                        theta_pick=theta_pick,
                        theta_place=theta_place,
                        theta_hover=theta_hover,
                    )
                    log.info("  → new best grip #%d  dist=%.3fm", i_grip, dist)

        if best is None:
            log.warning("No feasible pick plan found for any detected object.")
        return best
