"""FoundationPose wrapper: RGB-D pose refinement and tracking on top of SAM-6D detections."""
import os
import sys
import logging
import importlib
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_FP_ROOT = Path(__file__).resolve().parent.parent / "FoundationPose-main"
_FP_WEIGHTS = _FP_ROOT / "weights"

REFINER_RUN = "2023-10-28-18-33-37"
SCORER_RUN = "2024-01-11-20-02-45"
FP_GDRIVE_FOLDER_ID = "1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"


def _ensure_fp_paths():
    for p in [str(_FP_ROOT)]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Safe Python fallback for mycpp.cluster_poses
#
# mycpp.cluster_poses() segfaults on aarch64 (Eigen NEON alignment fault when
# pybind11 converts the numpy (N,4,4) array to vectorMatrix4f).  We replace it
# with an equivalent pure-Python implementation and monkey-patch the module so
# estimater.py transparently calls the Python version without any upstream edits.
# ---------------------------------------------------------------------------

_mycpp_patched = False


def _cluster_poses_python(angle_diff, dist_diff, poses_in, symmetry_tfs):
    """Pure-Python equivalent of mycpp.cluster_poses."""
    poses_in = np.asarray(poses_in, dtype=np.float32)
    symmetry_tfs = np.asarray(symmetry_tfs, dtype=np.float32)
    radian_thres = float(angle_diff) / 180.0 * np.pi
    print(f"num original candidates = {len(poses_in)}")
    out = [poses_in[0]]
    for i in range(1, len(poses_in)):
        cur = poses_in[i]
        is_new = True
        for cluster in out:
            if np.linalg.norm(cluster[:3, 3] - cur[:3, 3]) >= dist_diff:
                continue
            for tf in symmetry_tfs:
                R_cur = (cur @ tf)[:3, :3]
                R_clu = cluster[:3, :3]
                cos_a = np.clip((np.trace(R_cur @ R_clu.T) - 1.0) / 2.0, -1.0, 1.0)
                if np.arccos(cos_a) < radian_thres:
                    is_new = False
                    break
            if not is_new:
                break
        if is_new:
            out.append(cur)
    print(f"num of pose after clustering: {len(out)}")
    return out


def _patch_mycpp_cluster_poses():
    """Replace mycpp.cluster_poses with the Python fallback (idempotent)."""
    global _mycpp_patched
    if _mycpp_patched:
        return
    mycpp_mod = sys.modules.get("mycpp")
    if mycpp_mod is not None:
        mycpp_mod.cluster_poses = _cluster_poses_python
        _mycpp_patched = True


def _broadcast_thresh(val, M, default=None):
    if val is None:
        return [default] * M
    if isinstance(val, (int, float)):
        return [val] * M
    assert len(val) == M, f"Expected length {M}, got {len(val)}"
    return list(val)


class FoundationPoseRefiner:
    """
    Wraps the FoundationPose estimator (estimater.py) to accept SAM-6D output dicts
    and run pose registration (first frame) and tracking (subsequent frames).

    Supports an arbitrary number M of CAD model types simultaneously. All FoundationPose
    neural network weights (RefineNet, ScoreNet, glctx) are loaded once and shared.

    Requires CUDA; CPU-only mode is not supported by FoundationPose.
    The mycpp C++ extension must be compiled before use (see FoundationPose-main/build_all.sh).
    Model weights are auto-downloaded from Google Drive if missing.
    """

    def __init__(
        self,
        cad_path: Union[str, Path, List[Union[str, Path]]],
        model_ids: Optional[List[str]] = None,
        device: str = "cuda",
        weights_dir: Optional[Union[str, Path]] = None,
        mesh_scale: float = 1.0 / 1000.0,
        physical_scale: float = 1.0,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        max_detections: Optional[int] = None,
        min_pose_score: Union[float, List[float]] = 0.0,
        max_detections_per_model: Optional[Union[int, List[Optional[int]]]] = None,
        debug: int = 0,
        debug_dir: Optional[Union[str, Path]] = None,

        # Memory management
        half_precision: bool = False,
        lazy_load: bool = False,
    ):
        """
        Args:
            cad_path: Path to one CAD mesh file, or a list of M paths for M model types.
            model_ids: Optional names for each model (defaults to file stems). Length must
                match cad_path if provided.
            device: Must be "cuda" — FoundationPose is CUDA-only.
            weights_dir: Where to find (or download) model weights. Defaults to
                FoundationPose-main/weights/, which is the path hardcoded in
                PoseRefinePredictor / ScorePredictor and cannot be changed.
            mesh_scale: Multiplier applied to mesh vertices to convert them to
                metres for FoundationPose (default 1/1000 = mm → m). Set to 1.0
                if your mesh is already in metres.
            physical_scale: Additional scale factor applied on top of mesh_scale to
                compensate for a size mismatch between the CAD model and the physical
                object (default 1.0 = no adjustment). The effective vertex scale is
                mesh_scale * physical_scale. This affects pose estimation quality:
                mesh_diameter, crop windows, and translation step sizes inside the
                FoundationPose neural network all depend on the physical model size.
            est_refine_iter: Refinement iterations for register() (default 5).
            track_refine_iter: Refinement iterations for track_one() (default 2).
            max_detections: Global cap on detections to process per refine_from_sam6d()
                call (applied after per-model filtering). None = no global cap.
            min_pose_score: Skip SAM-6D detections below this pose_score threshold.
                Scalar (same for all models) or list of length M (per-model).
            max_detections_per_model: Cap per model type. Scalar, list of length M,
                or None (no cap per model).
            debug: Debug verbosity 0–3 (passed to internal FoundationPose estimator).
            debug_dir: Directory for debug visualizations (if debug >= 2).
        """
        if device == "cpu":
            raise NotImplementedError(
                "FoundationPose requires CUDA. CPU-only mode is not supported."
            )

        # Normalize cad_path to list
        if isinstance(cad_path, (str, Path)):
            cad_paths = [Path(cad_path)]
        else:
            cad_paths = [Path(p) for p in cad_path]
        M = len(cad_paths)

        self._cad_paths = cad_paths
        self.cad_path = cad_paths[0]  # backward compat
        self._model_ids = model_ids if model_ids is not None else [p.stem for p in cad_paths]
        assert len(self._model_ids) == M, (
            f"model_ids length {len(self._model_ids)} must match cad_path count {M}"
        )

        self.device = device
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.max_detections = max_detections
        self._min_pose_score = _broadcast_thresh(min_pose_score, M)
        self._max_det_per_model = _broadcast_thresh(max_detections_per_model, M, default=None)
        # backward compat scalar alias
        self.min_pose_score = self._min_pose_score[0]
        self.debug = debug
        self.debug_dir = str(debug_dir) if debug_dir else "/tmp/foundationpose_debug"
        if self.debug >= 2:
            os.makedirs(self.debug_dir, exist_ok=True)

        _ensure_fp_paths()
        self._check_mycpp()

        _weights_dir = Path(weights_dir) if weights_dir else _FP_WEIGHTS
        self._maybe_download_weights(_weights_dir)

        import trimesh
        effective_scale = mesh_scale * physical_scale
        self._meshes: List = []
        self._mesh_centroids: List[np.ndarray] = []
        for cp in cad_paths:
            logging.info(f"Loading mesh from {cp} (mesh_scale={mesh_scale}, physical_scale={physical_scale}) ...")
            mesh = trimesh.load(str(cp), force="mesh")
            mesh.vertices = mesh.vertices * effective_scale
            # Center the mesh so the origin sits at the geometric centroid.
            # Without centering the STL origin can be 1+ m from the actual geometry;
            # FoundationPose must then extrapolate T_obj_cam[:3,3] far outside the
            # observed depth points, so tiny rotation errors produce huge translation
            # errors (e.g. 2° error × 1.5 m offset → 52 mm positional error in
            # pick_xyz that can flip its sign).  After centering, T_obj_cam[:3,3]
            # equals the centroid directly — no amplification.
            centroid = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
            mesh.vertices = np.asarray(mesh.vertices, dtype=float) - centroid
            self._mesh_centroids.append(centroid.copy())
            mesh.process(validate=True)
            if len(mesh.vertex_normals) == 0:
                mesh = mesh.smoothed()
            self._meshes.append(mesh)
            logging.info(
                f"Mesh loaded: {len(mesh.vertices)} vertices, "
                f"diameter≈{mesh.bounding_box.extents.max():.4f} m  "
                f"(mesh_scale={mesh_scale}, physical_scale={physical_scale}, "
                f"centroid_offset=[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]m)."
            )
        # backward compat alias
        self._mesh = self._meshes[0]
        self._mesh_scale = mesh_scale
        self._physical_scale = physical_scale
        self._effective_scale = effective_scale

        self._half_precision = half_precision
        self._lazy_load = lazy_load
        self._refiner = None
        self._scorer = None
        if not lazy_load:
            self._load_models()

        # Import internal FoundationPose class under an alias to avoid name clash
        _est_mod = importlib.import_module("estimater")
        self._FPEst = _est_mod.FoundationPose

        self._glctx = None
        # Each entry: {"est": <FPEstimator>, "model_idx": int, "model_id": str}
        self._est_list: List[Dict] = []

    @property
    def effective_scale(self) -> float:
        """mesh_scale * physical_scale — the actual multiplier applied to mesh vertices."""
        return self._effective_scale

    # ------------------------------------------------------------------
    # Checks and downloads
    # ------------------------------------------------------------------

    def _check_mycpp(self):
        build_dir = _FP_ROOT / "mycpp" / "build"
        so_files = list(build_dir.glob("mycpp*.so")) if build_dir.is_dir() else []
        if not so_files:
            raise RuntimeError(
                "mycpp C++ extension not found. Build it before using FoundationPoseRefiner:\n"
                f"  cd {_FP_ROOT}/mycpp && mkdir -p build && cd build && "
                "cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j4\n"
                f"Or run: bash {_FP_ROOT}/build_all_conda.sh"
            )

    def _maybe_download_weights(self, weights_dir: Path):
        refiner_ckpt = weights_dir / REFINER_RUN / "model_best.pth"
        scorer_ckpt = weights_dir / SCORER_RUN / "model_best.pth"
        if refiner_ckpt.exists() and scorer_ckpt.exists():
            return

        logging.info("FoundationPose weights not found; attempting download via gdown ...")
        weights_dir.mkdir(parents=True, exist_ok=True)
        os.system(
            f'gdown --folder "https://drive.google.com/drive/folders/{FP_GDRIVE_FOLDER_ID}"'
            f' -O "{weights_dir}"'
        )

        if not (refiner_ckpt.exists() and scorer_ckpt.exists()):
            raise RuntimeError(
                f"FoundationPose weights not found at {weights_dir}.\n"
                "Download manually from:\n"
                "  https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i\n"
                f"Place the '{REFINER_RUN}' and '{SCORER_RUN}' folders under:\n"
                f"  {weights_dir}\n"
                "See FoundationPose-main/readme.md for details."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load PoseRefinePredictor and ScorePredictor onto GPU (idempotent)."""
        if self._refiner is None:
            logging.info("Loading PoseRefinePredictor ...")
            from learning.training.predict_pose_refine import PoseRefinePredictor
            self._refiner = PoseRefinePredictor()
            if self._half_precision:
                self._refiner.model.half()
        if self._scorer is None:
            logging.info("Loading ScorePredictor ...")
            from learning.training.predict_score import ScorePredictor
            self._scorer = ScorePredictor()
            if self._half_precision:
                self._scorer.model.half()
        logging.info("FoundationPose models loaded.")

    def _make_estimator(self, model_idx: int = 0):
        """Create a fresh internal FoundationPose estimator for the given model."""
        self._load_models()
        _patch_mycpp_cluster_poses()
        if self._glctx is None:
            import nvdiffrast.torch as dr
            self._glctx = dr.RasterizeCudaContext()
        mesh = self._meshes[model_idx]
        return self._FPEst(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=self._scorer,
            refiner=self._refiner,
            glctx=self._glctx,
            debug=self.debug,
            debug_dir=self.debug_dir,
        )

    @staticmethod
    def _depth_to_metres(depth: np.ndarray, depth_scale: float) -> np.ndarray:
        """Convert raw depth pixels to metres. depth_scale multiplies pixels to get mm."""
        return depth.astype(np.float32) * depth_scale / 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refine_from_sam6d(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        sam6d_results: Dict[str, Any],
        depth_scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Register FoundationPose on each SAM-6D detection and return refined poses.

        Resets internal tracking state; call track() on subsequent frames.

        Consumes "model_indices" and "model_ids" from sam6d_results when present
        (produced by MultiModelSAM6D). Falls back to model index 0 for single-model
        SAM6D output.

        Args:
            rgb: (H, W, 3) uint8 BGR or RGB image (same convention as SAM-6D input).
            depth: (H, W) raw depth pixels. Multiply by depth_scale to get mm.
            K: (3, 3) camera intrinsic matrix.
            sam6d_results: Output dict from SAM6D.forward() or MultiModelSAM6D.forward().
            depth_scale: Scale to convert raw depth to mm (e.g. 1000.0 for uint16 mm depth).

        Returns:
            List of dicts, one per processed detection:
                pose          — (4, 4) float32 object-in-camera transform, translation in metres.
                fp_score      — FoundationPose scorer confidence (higher is better).
                sam6d_score   — SAM-6D pose_score for this detection.
                mask          — (H, W) bool mask used for registration.
                box           — (4,) int64 xyxy bounding box.
                detection_idx — index into the sam6d_results arrays.
                model_idx     — index into self._meshes / self._cad_paths.
                model_id      — string identifier for this model.
        """
        if len(sam6d_results.get("poses", [])) == 0:
            self._est_list = []
            return []

        depth_m = self._depth_to_metres(depth, depth_scale)
        pose_scores = sam6d_results["pose_scores"]
        n_det = len(pose_scores)
        M = len(self._meshes)

        # Consume model assignments (defaults to model 0 for single-model SAM6D output)
        model_indices = sam6d_results.get(
            "model_indices", np.zeros(n_det, dtype=np.int64)
        )
        model_ids_out = sam6d_results.get(
            "model_ids", [self._model_ids[0]] * n_det
        )

        # Sort descending by pose_score; apply per-model min_pose_score and
        # max_detections_per_model, then global max_detections cap.
        order = np.argsort(pose_scores)[::-1]
        per_model_count = [0] * M
        selected = []
        for i in order:
            midx = int(model_indices[i])
            if pose_scores[i] < self._min_pose_score[midx]:
                continue
            cap = self._max_det_per_model[midx]
            if cap is not None and per_model_count[midx] >= cap:
                continue
            selected.append(i)
            per_model_count[midx] += 1

        if self.max_detections is not None:
            selected = selected[: self.max_detections]

        self._est_list = []
        results = []
        for i in selected:
            model_idx = int(model_indices[i])
            mask = sam6d_results["masks"][i].astype(np.uint8)
            est = self._make_estimator(model_idx)
            pose = est.register(
                K=K,
                rgb=rgb,
                depth=depth_m,
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )
            fp_score = float(est.scores[0].item())
            mid = model_ids_out[i]
            self._est_list.append({"est": est, "model_idx": model_idx, "model_id": mid})
            results.append(
                {
                    "pose": pose.astype(np.float32),
                    "fp_score": fp_score,
                    "sam6d_score": float(pose_scores[i]),
                    "mask": mask.astype(bool),
                    "box": sam6d_results["boxes"][i],
                    "detection_idx": int(i),
                    "model_idx": model_idx,
                    "model_id": mid,
                }
            )

        return results

    def track(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Track all active detections on a new frame using the stored estimators.

        Must be called after at least one refine_from_sam6d() call.

        Args:
            rgb: (H, W, 3) uint8 image.
            depth: (H, W) raw depth pixels.
            K: (3, 3) camera intrinsic matrix.
            depth_scale: Scale to convert raw depth to mm.

        Returns:
            List of dicts (one per active estimator):
                pose      — (4, 4) float32 object-in-camera transform in metres.
                model_idx — index into self._meshes / self._cad_paths.
                model_id  — string identifier for this model.
        """
        if not self._est_list:
            raise RuntimeError(
                "No active estimators. Call refine_from_sam6d() before track()."
            )

        depth_m = self._depth_to_metres(depth, depth_scale)
        results = []
        for entry in self._est_list:
            pose = entry["est"].track_one(
                rgb=rgb,
                depth=depth_m,
                K=K,
                iteration=self.track_refine_iter,
            )
            results.append(
                {
                    "pose": pose.astype(np.float32),
                    "model_idx": entry["model_idx"],
                    "model_id": entry["model_id"],
                }
            )

        return results

    def reset(self):
        """Clear all active estimators. Call before processing a new scene or object."""
        self._est_list = []

    def __call__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        sam6d_results: Dict[str, Any],
        depth_scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        return self.refine_from_sam6d(rgb, depth, K, sam6d_results, depth_scale)
