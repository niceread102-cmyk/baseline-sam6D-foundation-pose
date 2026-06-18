"""BinPicker: orchestrates the full pick-and-place loop.

Usage
-----
    # With ROS 2 arm stack already running:
    python -m src.binpick.binpicker \
        'cad_models=[{path: /path/to/model.stl, id: widget}]'

    # Override any parameter at runtime via Hydra CLI:
    python -m src.binpick.binpicker \
        sam6d.det_score_thresh=0.4 \
        foundationpose.physical_scale=2.0

The 8-step loop
---------------
1. Home    — move to start_angles
2. Hover   — move to hover_position (above pick region)
3. Capture — RGBD frame + update live preview
4. Detect  — SAM6D → FoundationPose; exit loop when no detections
5. Plan    — compute grips, select object → PickPlan
6. Execute — open → pick → close → place → open
7. Recover — on failure, return to home
8. Repeat  — from step 2 until empty scene
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .arm_controller import ArmController
from .camera import RealSenseCamera
from .grip_planner import GripPlanner
from .ik_solver import IKSolver
from .object_selector import ObjectSelector
from .visualizer import LivePreview, NullPreview, overlay_axes, overlay_masks

log = logging.getLogger(__name__)


class BinPicker:
    def __init__(self, cfg: DictConfig) -> None:
        cad_paths  = [m.path  for m in cfg.cad_models]
        model_ids  = [m.id    for m in cfg.cad_models]

        log.info("Loading SAM6D...")
        self.sam6d = hydra.utils.instantiate(
            cfg.sam6d, cad_path=cad_paths, model_ids=model_ids
        )

        log.info("Loading FoundationPose...")
        self.fp = hydra.utils.instantiate(
            cfg.foundationpose, cad_path=cad_paths, model_ids=model_ids
        )

        log.info("Setting up IK solver...")
        self.ik: IKSolver = hydra.utils.instantiate(cfg.ik)

        log.info("Setting up arm controller...")
        self.arm: ArmController = hydra.utils.instantiate(
            cfg.arm_controller, ik_solver=self.ik
        )

        log.info("Computing grip candidates...")
        effective_scale = (
            float(cfg.foundationpose.mesh_scale)
            * float(cfg.foundationpose.physical_scale)
        )
        self.grip_planner: GripPlanner = hydra.utils.instantiate(
            cfg.grip_planner,
            cad_models=cfg.cad_models,
            effective_scale=effective_scale,
            min_grip_m=float(cfg.gripper.min_grip_m),
            max_grip_m=float(cfg.gripper.max_grip_m),
        )

        log.info("Setting up object selector...")
        # T_cam_ee is the fixed camera-to-EE transform (camera rigidly mounted on wrist).
        # T_cam_base is computed dynamically each capture: T_cam_base = fk_matrix(θ) @ T_cam_ee
        self._T_cam_ee: np.ndarray | None = (
            np.array(OmegaConf.to_container(cfg.arm.T_cam_ee, resolve=True))
            if cfg.arm.T_cam_ee is not None else None
        )
        if self._T_cam_ee is None:
            log.warning(
                "arm.T_cam_ee is null — T_cam_base will not be computed. "
                "Object poses will remain in camera frame (picks will fail). "
                "Run scripts/calibrate_T_cam_base.py to generate T_cam_ee."
            )

        self.selector: ObjectSelector = hydra.utils.instantiate(
            cfg.object_selector,
            ik_solver=self.ik,
            placement_position=list(cfg.arm.placement_position),
            # T_cam_base is NOT passed here; it is injected per-call in run()
        )

        log.info("Setting up camera...")
        self.camera: RealSenseCamera = hydra.utils.instantiate(cfg.camera)

        # Pre-sample mesh point clouds for FP overlay (centred + scaled, same frame as FP output)
        self._mesh_pts: dict[str, np.ndarray] = {}
        for i, m in enumerate(cfg.cad_models):
            pts = np.asarray(self.fp._meshes[i].sample(512), dtype=float)
            self._mesh_pts[m.id] = pts

        preview_cfg = OmegaConf.select(cfg, "preview", default=None)
        preview_enabled = preview_cfg is not None and preview_cfg.get("enabled", True)
        if preview_enabled:
            log.info("Starting live preview...")
            self.preview = hydra.utils.instantiate(
                preview_cfg,
                ik_solver=self.ik,
                # T_cam_base starts null; updated dynamically each capture via set_T_cam_base()
            )
            self.preview.set_frame_source(self.camera.peek)
            self.preview.start()
        else:
            log.info("Preview disabled.")
            self.preview = NullPreview()

        self.cfg = cfg

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        cfg         = self.cfg
        start_theta = np.array(cfg.arm.start_angles)
        hover_xyz   = np.array(cfg.arm.hover_position)

        # Pre-compute grips for all models
        grips = {m.id: self.grip_planner.compute_grips(m.id) for m in cfg.cad_models}
        for mid, gc in grips.items():
            log.info(f"  {mid}: {len(gc)} grip candidate(s)")
            for i, g in enumerate(gc):
                log.info(
                    "    Grip #%d (object frame): width=%.1fmm"
                    "  approach_axis=[%.3f, %.3f, %.3f]  centroid=[%.3f, %.3f, %.3f]m",
                    i, g.width_m * 1000,
                    g.approach_axis[0], g.approach_axis[1], g.approach_axis[2],
                    g.position_m[0], g.position_m[1], g.position_m[2],
                )
                log.info(
                    "    (base-frame approach direction depends on detected pose — "
                    "see selector logs after FoundationPose runs)"
                )

        # ── Step 1: Home ──────────────────────────────────────────────────
        self.preview.update_status("Homing", gripper_open=True, n_detections=0)
        self.arm.set_gripper(open=True)
        self.arm.move_to_angles(start_theta)

        # ── Startup: move to hover and log T_cam_base sanity check ───────
        self.arm.move_to_xyz(hover_xyz)
        _rgb0, _depth0, _K0 = self.camera.capture()
        _hover_theta0 = self.arm.current_theta
        _T_cb0 = (
            self.ik.fk_matrix(_hover_theta0) @ self._T_cam_ee
            if self._T_cam_ee is not None else None
        )
        if _T_cb0 is not None:
            _R = _T_cb0[:3, :3]
            _t = _T_cb0[:3, 3]
            _cam_z = _R[:, 2]
            _dom = int(np.argmax(np.abs(_cam_z)))
            _axes = ["base-X (forward)", "base-Y (lateral)", "base-Z (vertical)"]
            _sign = "+" if _cam_z[_dom] > 0 else "-"
            log.info(
                "T_cam_base at hover:\n"
                "  Camera origin in base: [%.3f, %.3f, %.3f]m\n"
                "  Cam-Z (optical axis) → base [%.3f, %.3f, %.3f]  dominant: %s%s",
                _t[0], _t[1], _t[2],
                _cam_z[0], _cam_z[1], _cam_z[2], _sign, _axes[_dom],
            )
            _h0, _w0 = _depth0.shape[:2]
            for _px, _py, _label in [(_w0 // 2, _h0 // 2, "center"),
                                       (_w0 // 2, _h0 // 4, "top-quarter")]:
                _D = float(_depth0[_py, _px])   # raw depth in mm (depth_scale=1.0)
                if _D > 0:
                    _D_m = _D / 1000.0           # convert mm → metres for geometry
                    _fx, _fy = float(_K0[0, 0]), float(_K0[1, 1])
                    _cx0, _cy0 = float(_K0[0, 2]), float(_K0[1, 2])
                    _p_cam = np.array([(_px - _cx0) * _D_m / _fx,
                                       (_py - _cy0) * _D_m / _fy, _D_m])
                    _p_base = (_T_cb0 @ np.append(_p_cam, 1.0))[:3]
                    log.info(
                        "Depth sanity [%s D=%.0fmm]: cam=[%.3f,%.3f,%.3f]m "
                        "→ base=[%.3f,%.3f,%.3f]m  (hover=[%.3f,%.3f,%.3f]m)",
                        _label, _D,
                        _p_cam[0], _p_cam[1], _p_cam[2],
                        _p_base[0], _p_base[1], _p_base[2],
                        hover_xyz[0], hover_xyz[1], hover_xyz[2],
                    )

        try:
            while True:
                # ── Step 2: Hover ─────────────────────────────────────────
                self.preview.update_status("Hovering", gripper_open=True, n_detections=0)
                self.arm.move_to_xyz(hover_xyz)

                # ── Step 3: Capture ───────────────────────────────────────
                self.preview.update_status("Capturing", gripper_open=True, n_detections=0)
                rgb, depth, K = self.camera.capture()
                # Recompute T_cam_base from current arm pose × fixed T_cam_ee
                hover_theta = self.arm.current_theta
                T_cam_base = (
                    self.ik.fk_matrix(hover_theta) @ self._T_cam_ee
                    if self._T_cam_ee is not None else None
                )
                self.preview.set_T_cam_base(T_cam_base)
                self.preview.update_live(rgb, depth)

                # Camera orientation gate: if the optical axis is not pointing
                # sufficiently downward the camera is facing a wall or ceiling,
                # and any FP detections will be nonsensical.
                if T_cam_base is not None:
                    _cam_z = T_cam_base[:3, 2]   # optical axis in base frame
                    if _cam_z[2] > -0.70:
                        log.error(
                            "Camera not looking downward: cam-Z=[%.2f,%.2f,%.2f] "
                            "(Z=%.2f > -0.70). Arm may be in wrong pose or "
                            "T_cam_ee needs re-calibration. Re-homing.",
                            *_cam_z
                        )
                        self.arm.move_to_angles(start_theta, safe=False)
                        continue

                # ── Step 4a: SAM6D ────────────────────────────────────────
                self.preview.update_status("SAM6D detecting", gripper_open=True, n_detections=0)
                sam6d_results = self.sam6d(rgb, depth, K)
                n_sam6d = len(sam6d_results.get("poses", []))
                log.info(f"SAM6D: {n_sam6d} detection(s)")

                sam6d_vis = overlay_masks(rgb, sam6d_results)
                self.preview.update_sam6d_panel(sam6d_vis)

                if n_sam6d == 0:
                    log.info("No objects detected — bin is empty.")
                    break

                # ── Step 4b: FoundationPose ───────────────────────────────
                self.preview.update_status(
                    "FoundationPose refining", gripper_open=True, n_detections=n_sam6d
                )
                fp_results = self.fp.refine_from_sam6d(rgb, depth, K, sam6d_results)
                log.info(f"FoundationPose: {len(fp_results)} refined pose(s)")

                fp_vis = overlay_axes(rgb, fp_results, K, self._mesh_pts)
                self.preview.update_fp_panel(fp_vis)
                self.preview.update_status(
                    "Detected", gripper_open=True, n_detections=len(fp_results)
                )

                if not fp_results:
                    log.warning("FoundationPose returned no refined poses — re-scanning.")
                    self.fp.reset()
                    continue

                # ── Step 5: Plan ──────────────────────────────────────────
                self.preview.update_status(
                    "Planning", gripper_open=True, n_detections=len(fp_results)
                )
                plan = self.selector.select(
                    fp_results, grips, hover_theta, T_cam_base=T_cam_base
                )

                if plan is None:
                    log.warning("No feasible plan — re-scanning.")
                    self.fp.reset()
                    continue

                # Pick-target marker: project jaw midpoint (base frame) into hover image
                T_obj_cam = np.asarray(plan.fp_result["pose"])
                if T_cam_base is not None:
                    T_obj_base_plan = T_cam_base @ T_obj_cam
                else:
                    T_obj_base_plan = T_obj_cam
                pick_xyz_plan = (
                    T_obj_base_plan[:3, :3] @ plan.grip.position_m
                    + T_obj_base_plan[:3, 3]
                )
                self.preview.update_pick_target(pick_xyz_plan, K)

                # Grip line: project jaw span into camera frame using FP pose (obj→cam)
                R_obj_cam = T_obj_cam[:3, :3]
                t_obj_cam = T_obj_cam[:3, 3]
                grip_center_cam = R_obj_cam @ plan.grip.position_m + t_obj_cam
                grip_axis_cam   = R_obj_cam @ plan.grip.approach_axis
                self.preview.update_grip_line(
                    grip_center_cam, grip_axis_cam, plan.grip.width_m, K
                )

                # ── Step 6: Execute ───────────────────────────────────────
                try:
                    n = len(fp_results)

                    self.preview.update_status("Opening gripper", gripper_open=True, n_detections=n)
                    self.arm.set_gripper(open=True)

                    self.preview.update_status("Moving to pick", gripper_open=True, n_detections=n)
                    if not self.arm.move_to_angles(plan.theta_pick):
                        raise RuntimeError("Pick move failed — arm rejected angles or safety check triggered")

                    self.preview.update_status("Gripping", gripper_open=False, n_detections=n)
                    self.arm.set_gripper(open=False)
                    time.sleep(0.5)

                    self.preview.update_status("Moving to place", gripper_open=False, n_detections=n)
                    if not self.arm.move_to_angles(plan.theta_place):
                        raise RuntimeError("Place move failed — arm rejected angles or safety check triggered")

                    self.preview.update_status("Releasing", gripper_open=True, n_detections=n)
                    self.arm.set_gripper(open=True)
                    time.sleep(0.3)

                    log.info("Pick-and-place succeeded.")
                    self.fp.reset()

                except Exception as exc:
                    # ── Step 7: Recover ───────────────────────────────────
                    log.error(f"Execution error: {exc}")
                    self.preview.update_status("Recovering", gripper_open=True, n_detections=0)
                    self.arm.recover()
                    self.arm.set_gripper(open=True)
                    self.arm.move_to_angles(start_theta, safe=False)
                    self.fp.reset()

                # ── Step 8: Repeat ────────────────────────────────────────
                # Loop continues from step 2

        except KeyboardInterrupt:
            log.info("Interrupted by user.")

        finally:
            self.preview.update_status("Finished", gripper_open=True, n_detections=0)
            log.info("Returning to home position.")
            self.arm.set_gripper(open=True)
            self.arm.move_to_angles(start_theta)
            time.sleep(2.0)
            self.preview.stop()
            self.camera.stop()


# ── Entry point ───────────────────────────────────────────────────────────────

@hydra.main(
    config_path=str(Path(__file__).resolve().parent.parent.parent / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    log.info("BinPicker starting.\n" + OmegaConf.to_yaml(cfg))
    BinPicker(cfg).run()


if __name__ == "__main__":
    main()
