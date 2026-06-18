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


@dataclass
class PickPlan:
    fp_result:   dict           # FoundationPose result dict for chosen object
    grip:        GripCandidate
    theta_pick:  np.ndarray    # IK solution for pick position
    theta_place: np.ndarray    # IK solution for placement position


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
    """

    def __init__(
        self,
        ik_solver: IKSolver,
        placement_position: list[float],
        T_cam_base: np.ndarray | None = None,
        min_pick_height_m: float = 0.02,
        check_grip_orientation: bool = True,
        gripper_closing_axis: int = 1,
        grasp_standoff_m: float = 0.0,
        min_grip_centroid_dist_m: float = 0.0,
    ):
        self._ik                 = ik_solver
        self._place_xyz          = np.asarray(placement_position, dtype=float)
        self._T_cb               = np.asarray(T_cam_base) if T_cam_base is not None else None
        self._standoff           = float(grasp_standoff_m)
        self._min_centroid_dist  = float(min_grip_centroid_dist_m)

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

    def select(
        self,
        fp_results: list[dict],
        grips: dict[str, list[GripCandidate]],
        current_theta: np.ndarray,
        T_cam_base: np.ndarray | None = None,
    ) -> PickPlan | None:
        # Per-call T_cam_base takes precedence; fall back to instance-level (fixed-camera mode)
        T_cb = np.asarray(T_cam_base) if T_cam_base is not None else self._T_cb

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

                log.info(
                    "  Grip #%d: width=%.1fmm"
                    "  pick_xyz=[%.3f, %.3f, %.3f]m"
                    "  approach_base=[%.3f, %.3f, %.3f]",
                    i_grip, grip.width_m * 1000,
                    pick_xyz[0], pick_xyz[1], pick_xyz[2],
                    approach_base[0], approach_base[1], approach_base[2],
                )

                # Skip grips too close to the object centroid (hub region)
                if self._min_centroid_dist > 0.0:
                    centroid_dist = float(np.linalg.norm(grip.position_m))
                    if centroid_dist < self._min_centroid_dist:
                        log.info(
                            "  SKIP grip #%d: centroid dist %.1fmm < min %.1fmm",
                            i_grip, centroid_dist * 1000, self._min_centroid_dist * 1000,
                        )
                        continue

                # Feasibility: height + optional 6-DOF orientation check
                ok_feas, reason = self._feasibility.check(pick_xyz, approach_base, current_theta)
                if not ok_feas:
                    log.info("  SKIP grip #%d: %s", i_grip, reason)
                    continue

                # 6-DOF pick IK: position + grip orientation.
                # Apply standoff so the EE flange is offset from the jaw contact
                # point by grasp_standoff_m along the approach direction.
                # approach_dir = first column of target_R (EE X-axis in base frame).
                target_R    = self._feasibility.build_pick_orientation(approach_base)
                approach_dir = target_R[:, 0]
                pick_xyz_ee  = pick_xyz - self._standoff * approach_dir
                theta_pick, ok_pick = self._ik.solve_6dof(pick_xyz_ee, target_R, current_theta)
                if not ok_pick:
                    log.info("  SKIP grip #%d: 6-DOF pick IK failed", i_grip)
                    continue
                log.info(
                    "  Grip #%d pick_xyz=[%.3f,%.3f,%.3f] standoff=%.3fm "
                    "→ EE target=[%.3f,%.3f,%.3f]",
                    i_grip,
                    pick_xyz[0], pick_xyz[1], pick_xyz[2],
                    self._standoff,
                    pick_xyz_ee[0], pick_xyz_ee[1], pick_xyz_ee[2],
                )

                theta_place, ok_place = self._ik.solve(self._place_xyz, theta_pick)
                if not ok_place:
                    log.info("  SKIP grip #%d: place IK failed", i_grip)
                    continue

                # Score by joint-space distance from hover rather than cartesian distance.
                # Cartesian distance penalises grips on the object arm (farther from EE)
                # while systematically favouring hub-area grips that close along the robot's
                # forward (base-X) axis — those require joint 4/6 rotations of >80° to align
                # the closing direction, which swings the eye-on-hand camera sideways into a
                # wall and makes the place move exceed max_joint_delta.
                # Joint-space distance from hover naturally penalises those large wrist
                # rotations and prefers grips that close in the ±Y direction.
                dist = float(np.linalg.norm(theta_pick - current_theta))
                if dist < best_dist:
                    best_dist = dist
                    best = PickPlan(
                        fp_result=result,
                        grip=grip,
                        theta_pick=theta_pick,
                        theta_place=theta_place,
                    )
                    log.info(
                        "  → new best grip #%d  joint_dist=%.3frad  pick_xyz=[%.3f,%.3f,%.3f]",
                        i_grip, dist,
                        pick_xyz[0], pick_xyz[1], pick_xyz[2],
                    )

        if best is None:
            log.warning("No feasible pick plan found for any detected object.")
        return best
