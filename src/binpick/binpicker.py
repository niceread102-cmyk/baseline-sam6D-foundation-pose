"""BinPicker: orchestrates the full pick-and-place loop.

Usage
-----
    python -m src.binpick.binpicker \\
        'cad_models=[{path: /path/to/model.stl, id: widget}]'

    # Override any parameter at runtime via Hydra CLI:
    python -m src.binpick.binpicker \\
        sam6d.det_score_thresh=0.4 \\
        foundationpose.physical_scale=2.0

See ``run_binpicker.py`` in the repo root for a user-friendly wrapper with
pre-set overrides.  Note that ``run_binpicker.py`` unconditionally applies its
own constants (``MESH_SCALE``, ``MAX_GRIP_M``, etc.) which will silently
override ``conf/config.yaml``; prefer CLI overrides for permanent changes.

The pick-and-place loop (``BinPicker.run()``)
---------------------------------------------
1. **Home**    — move joints to ``start_angles``
2. **Hover**   — move EE to ``hover_position`` above the pick region
3. **Capture** — grab RGBD frame; compute ``T_cam_base`` from FK + ``T_cam_ee``
4. **Detect**  — SAM6D segments the scene; exit loop when nothing is found
5. **Refine**  — FoundationPose refines 6-DOF poses from SAM6D masks
6. **Plan**    — NearestObjectSelector picks the best object + grip
7. **Execute** — open gripper → approach hover (if defined) → pick → close →
                 retreat to hover (if defined) → place → open
8. **Recover** — on failure, return to home with ``safe=False``
9. **Repeat**  — from step 2 until the bin is empty

Coordinate and unit conventions
--------------------------------
- ``T_cam_base``: camera-frame → robot-base frame transform, i.e.
  ``p_base = T_cam_base @ p_cam_hom``. Computed one of two ways depending on
  the camera's mounting (see ``_compute_T_cam_base``):
    - **Fixed-mount** (e.g. PhoXi): ``camera.T_cam_base`` is a static
      extrinsic from config — the scanner doesn't move with the arm.
    - **Eye-on-hand** (e.g. RealSense): ``T_cam_ee`` is a fixed camera→EE
      transform (calibrated once), and ``T_cam_base = fk_matrix(θ) @ T_cam_ee``
      is recomputed every capture.
- FoundationPose pose matrices are in **camera frame**, translation in **metres**.
- All arm positions are in **metres** (robot-base frame); joint angles in **radians**.
- Depth from the camera is in **millimetres** (SAM6D / FoundationPose convention).
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
from .camera_phoxi import PhoXiCamera
from .grip_planner import GripPlanner
from .ik_solver import IKSolver
from .object_selector import ObjectSelector, PickPlan
from .visualizer import LivePreview, NullPreview, overlay_axes, overlay_grasps, overlay_masks

log = logging.getLogger(__name__)

# ── Timing constants ──────────────────────────────────────────────────────────
_GRIP_SETTLE_S    = 0.5   # extra wait after closing gripper before moving to place
_RELEASE_SETTLE_S = 0.3   # extra wait after opening gripper (release) before next hover
_HOME_SETTLE_S    = 2.0   # wait at home position in the finally-block for the arm to settle

# Depth sanity check: sample pixels at these image rows (as fractions of height)
_DEPTH_SAMPLE_ROWS = [0.5, 0.25]   # centre row and top-quarter row


class BinPicker:
    """Orchestrates the full bin-picking pipeline.

    Parameters
    ----------
    cfg:
        Hydra DictConfig loaded from ``conf/config.yaml`` (with any CLI overrides).
        Instantiates all sub-components (camera, IK solver, arm controller, etc.)
        via ``hydra.utils.instantiate``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        cad_paths  = [m.path  for m in cfg.cad_models]
        model_ids  = [m.id    for m in cfg.cad_models]

        log.info("Loading SAM6D...")
        self.sam6d = hydra.utils.instantiate(
            cfg.sam6d, cad_path=cad_paths, model_ids=model_ids,
            physical_scale=float(cfg.foundationpose.physical_scale),
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
        # effective_scale converts raw mesh vertices to metres for grip planning.
        # mesh_scale handles CAD unit conversion (e.g. 0.001 if CAD is in mm).
        # physical_scale compensates when the real object differs in size from the
        # CAD model (e.g. 3.0 if the CAD is a 1:3 scale mock).
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
        # T_cam_ee: fixed camera-to-EE transform (eye-on-hand, e.g. RealSense on
        # the wrist). T_cam_base is recomputed every capture from this + FK.
        self._T_cam_ee: np.ndarray | None = (
            np.array(OmegaConf.to_container(cfg.arm.T_cam_ee, resolve=True))
            if cfg.arm.T_cam_ee is not None else None
        )
        # camera.T_cam_base: static extrinsic for a fixed-mount camera (e.g.
        # PhoXi, rigidly mounted in the workspace) — takes precedence over
        # T_cam_ee since it needs no FK recompute. See _compute_T_cam_base.
        camera_T_cam_base = OmegaConf.select(cfg, "camera.T_cam_base", default=None)
        self._T_cam_base_fixed: np.ndarray | None = (
            np.array(OmegaConf.to_container(camera_T_cam_base, resolve=True))
            if camera_T_cam_base is not None else None
        )
        if self._T_cam_base_fixed is None and self._T_cam_ee is None:
            log.warning(
                "Neither camera.T_cam_base (fixed-mount) nor arm.T_cam_ee "
                "(eye-on-hand) is set — T_cam_base will not be computed. "
                "Object poses will remain in camera frame (picks will fail). "
                "Run scripts/calibrate_T_cam_base_phoxi.py or "
                "scripts/calibrate_T_cam_base.py."
            )

        # Shared safety threshold: same value the arm controller uses to hard-reject
        # a move (conf/arm_controller/xarm.yaml), reused here so bad plans are
        # rejected at planning time instead of only being caught after the gripper
        # opens and a move is attempted. conf/arm_controller/ros.yaml has no such
        # field, hence OmegaConf.select with a disabled (0.0) default.
        max_joint_delta = float(OmegaConf.select(cfg, "arm_controller.max_joint_delta", default=0.0))
        self.selector: ObjectSelector = hydra.utils.instantiate(
            cfg.object_selector,
            ik_solver=self.ik,
            placement_position=list(cfg.arm.placement_position),
            max_joint_delta=max_joint_delta,
            # T_cam_base is NOT passed here; it is injected per-call in run()
        )

        log.info("Setting up camera...")
        self.camera: RealSenseCamera | PhoXiCamera = hydra.utils.instantiate(cfg.camera)

        # Pre-sample mesh point clouds for FP overlay (centred + scaled, object frame)
        self._mesh_pts: dict[str, np.ndarray] = {}
        for i, m in enumerate(cfg.cad_models):
            pts = np.asarray(self.fp._meshes[i].sample(512), dtype=float)
            self._mesh_pts[m.id] = pts

        preview_cfg     = OmegaConf.select(cfg, "preview", default=None)
        preview_enabled = preview_cfg is not None and preview_cfg.get("enabled", True)
        if preview_enabled:
            log.info("Starting live preview...")
            self.preview = hydra.utils.instantiate(
                preview_cfg,
                ik_solver=self.ik,
                # T_cam_base starts null; updated dynamically each capture
            )
            self.preview.set_frame_source(self.camera.peek)
            self.preview.start()
        else:
            log.info("Preview disabled.")
            self.preview = NullPreview()

        self.cfg = cfg

    # ── camera extrinsics ─────────────────────────────────────────────────

    def _compute_T_cam_base(self, theta: np.ndarray) -> np.ndarray | None:
        """Camera→base transform: static for a fixed-mount camera (PhoXi), or
        recomputed via FK for an eye-on-hand camera (RealSense). Returns None
        if neither ``camera.T_cam_base`` nor ``arm.T_cam_ee`` is configured.
        """
        if self._T_cam_base_fixed is not None:
            return self._T_cam_base_fixed
        if self._T_cam_ee is not None:
            return self.ik.fk_matrix(theta) @ self._T_cam_ee
        return None

    # ── execution helpers ─────────────────────────────────────────────────

    def _execute_pick(self, plan: PickPlan, n_detections: int) -> None:
        """Open gripper, approach via the pre-grasp hover pose (if any), descend
        to the pick position, close the gripper, then retreat back through hover.

        Raises ``RuntimeError`` if any arm move fails so the caller can trigger
        the recovery path.
        """
        self.preview.update_status("Opening gripper", gripper_open=True, n_detections=n_detections)
        self.arm.set_gripper(open=True)

        if plan.theta_hover is not None:
            self.preview.update_status("Approaching", gripper_open=True, n_detections=n_detections)
            if not self.arm.move_to_angles(plan.theta_hover):
                raise RuntimeError("Hover approach move failed — arm rejected angles or safety check triggered")

        self.preview.update_status("Moving to pick", gripper_open=True, n_detections=n_detections)
        if not self.arm.move_to_angles(plan.theta_pick):
            raise RuntimeError("Pick move failed — arm rejected angles or safety check triggered")

        self.preview.update_status("Gripping", gripper_open=False, n_detections=n_detections)
        self.arm.set_gripper(open=False)
        time.sleep(_GRIP_SETTLE_S)

        if plan.theta_hover is not None:
            self.preview.update_status("Retreating", gripper_open=False, n_detections=n_detections)
            if not self.arm.move_to_angles(plan.theta_hover):
                raise RuntimeError("Retreat move failed — arm rejected angles or safety check triggered")

    def _execute_place(self, plan: PickPlan, n_detections: int) -> None:
        """Move to placement position and release the gripper.

        Raises ``RuntimeError`` if the arm move fails.
        """
        self.preview.update_status("Moving to place", gripper_open=False, n_detections=n_detections)
        if not self.arm.move_to_angles(plan.theta_place):
            raise RuntimeError("Place move failed — arm rejected angles or safety check triggered")

        self.preview.update_status("Releasing", gripper_open=True, n_detections=n_detections)
        self.arm.set_gripper(open=True)
        time.sleep(_RELEASE_SETTLE_S)

    def _do_recovery(self, reason: str, start_theta: np.ndarray) -> None:
        """Recover after an execution failure: log, re-home, and reset FP tracker.

        Uses ``safe=False`` for the home move because the arm may be in an
        unexpected position after a fault.
        """
        log.error("Execution error: %s", reason)
        self.preview.update_status("Recovering", gripper_open=True, n_detections=0)
        self.arm.recover()
        self.arm.set_gripper(open=True)
        self.arm.move_to_angles(start_theta, safe=False)
        self.fp.reset()

    # ── startup diagnostics ───────────────────────────────────────────────

    def _log_startup_diagnostics(
        self,
        T_cam_base: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        hover_xyz: np.ndarray,
    ) -> None:
        """Log T_cam_base sanity checks at hover height (optical axis + depth samples)."""
        R = T_cam_base[:3, :3]
        t = T_cam_base[:3, 3]
        cam_z = R[:, 2]
        dom   = int(np.argmax(np.abs(cam_z)))
        axes  = ["base-X (forward)", "base-Y (lateral)", "base-Z (vertical)"]
        sign  = "+" if cam_z[dom] > 0 else "-"
        log.info(
            "T_cam_base at hover:\n"
            "  Camera origin in base: [%.3f, %.3f, %.3f]m\n"
            "  Cam-Z (optical axis) → base [%.3f, %.3f, %.3f]  dominant: %s%s",
            t[0], t[1], t[2],
            cam_z[0], cam_z[1], cam_z[2], sign, axes[dom],
        )
        H, W = depth.shape[:2]
        for row_frac, label in zip(_DEPTH_SAMPLE_ROWS, ["center", "top-quarter"]):
            px, py = W // 2, int(H * row_frac)
            D = float(depth[py, px])
            if D > 0:
                fx, fy = float(K[0, 0]), float(K[1, 1])
                cx, cy = float(K[0, 2]), float(K[1, 2])
                # depth D is in mm; convert to metres for the transform
                p_cam  = np.array([(px - cx) * D / fx, (py - cy) * D / fy, D]) * 1e-3
                p_base = (T_cam_base @ np.append(p_cam, 1.0))[:3]
                log.info(
                    "Depth sanity [%s D=%.0fmm]: cam=[%.3f,%.3f,%.3f]m "
                    "→ base=[%.3f,%.3f,%.3f]m  (hover=[%.3f,%.3f,%.3f]m)",
                    label, D,
                    p_cam[0], p_cam[1], p_cam[2],
                    p_base[0], p_base[1], p_base[2],
                    hover_xyz[0], hover_xyz[1], hover_xyz[2],
                )

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the bin-picking loop until the bin is empty or the user interrupts."""
        cfg         = self.cfg
        start_theta = np.array(cfg.arm.start_angles)
        hover_xyz   = np.array(cfg.arm.hover_position)

        # Pre-compute grip candidates for all models (object frame, metres)
        grips = {m.id: self.grip_planner.compute_grips(m.id) for m in cfg.cad_models}
        for mid, gc in grips.items():
            log.info("  %s: %d grip candidate(s)", mid, len(gc))
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
        rgb0, depth0, K0 = self.camera.capture()
        hover_theta0 = self.arm.current_theta
        T_cb0 = self._compute_T_cam_base(hover_theta0)
        if T_cb0 is not None:
            self._log_startup_diagnostics(T_cb0, depth0, K0, hover_xyz)

        try:
            while True:
                # ── Step 2: Hover ─────────────────────────────────────────
                self.preview.update_status("Hovering", gripper_open=True, n_detections=0)
                self.arm.move_to_xyz(hover_xyz)

                # ── Step 3: Capture ───────────────────────────────────────
                self.preview.update_status("Capturing", gripper_open=True, n_detections=0)
                rgb, depth, K = self.camera.capture()
                # depth is in mm (camera.depth_units == "mm")
                # Fixed-mount cameras get a static T_cam_base; eye-on-hand cameras
                # get it recomputed from the current arm pose (see _compute_T_cam_base)
                hover_theta = self.arm.current_theta
                T_cam_base = self._compute_T_cam_base(hover_theta)
                self.preview.set_T_cam_base(T_cam_base)
                self.preview.update_live(rgb, depth)

                # ── Step 4a: SAM6D ────────────────────────────────────────
                self.preview.update_status("SAM6D detecting", gripper_open=True, n_detections=0)
                sam6d_results = self.sam6d(rgb, depth, K)
                n_sam6d = len(sam6d_results.get("poses", []))
                log.info("SAM6D: %d detection(s)", n_sam6d)

                sam6d_vis = overlay_masks(rgb, sam6d_results)
                self.preview.update_sam6d_panel(sam6d_vis)

                if n_sam6d == 0:
                    log.info("No objects detected — bin is empty.")
                    break

                # ── Step 4b: FoundationPose ───────────────────────────────
                # fp_results: list of dicts, each with:
                #   "pose"     — 4×4 transform in CAMERA frame, translation in metres
                #   "model_id" — model identifier string
                self.preview.update_status(
                    "FoundationPose refining", gripper_open=True, n_detections=n_sam6d
                )
                fp_results = self.fp.refine_from_sam6d(rgb, depth, K, sam6d_results)
                log.info("FoundationPose: %d refined pose(s)", len(fp_results))

                fp_vis = overlay_axes(rgb, fp_results, K, self._mesh_pts)
                fp_vis = overlay_grasps(fp_vis, fp_results, grips, K)
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

                self.preview.update_ik_path(hover_theta, plan.theta_pick, K)

                # ── Step 6: Execute ───────────────────────────────────────
                try:
                    n = len(fp_results)
                    self._execute_pick(plan, n)
                    self._execute_place(plan, n)
                    log.info("Pick-and-place succeeded.")
                    self.fp.reset()

                except Exception as exc:
                    # ── Step 7: Recover ───────────────────────────────────
                    self._do_recovery(str(exc), start_theta)

                # ── Step 8: Repeat ────────────────────────────────────────

        except KeyboardInterrupt:
            log.info("Interrupted by user.")

        finally:
            self.preview.update_status("Finished", gripper_open=True, n_detections=0)
            log.info("Returning to home position.")
            self.arm.set_gripper(open=True)
            self.arm.move_to_angles(start_theta)
            time.sleep(_HOME_SETTLE_S)
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
