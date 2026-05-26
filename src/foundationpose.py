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


class FoundationPoseRefiner:
    """
    Wraps the FoundationPose estimator (estimater.py) to accept SAM-6D output dicts
    and run pose registration (first frame) and tracking (subsequent frames).

    Requires CUDA; CPU-only mode is not supported by FoundationPose.
    The mycpp C++ extension must be compiled before use (see FoundationPose-main/build_all.sh).
    Model weights are auto-downloaded from Google Drive if missing.
    """

    def __init__(
        self,
        cad_path: Union[str, Path],
        device: str = "cuda",
        weights_dir: Optional[Union[str, Path]] = None,
        mesh_scale: float = 1.0 / 1000.0,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        max_detections: Optional[int] = None,
        min_pose_score: float = 0.0,
        debug: int = 0,
        debug_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            cad_path: Path to the CAD mesh file (.obj, .stl, .ply, …).
            device: Must be "cuda" — FoundationPose is CUDA-only.
            weights_dir: Where to find (or download) model weights. Defaults to
                FoundationPose-main/weights/, which is the path hardcoded in
                PoseRefinePredictor / ScorePredictor and cannot be changed.
            mesh_scale: Multiplier applied to mesh vertices to convert them to
                metres for FoundationPose (default 1/1000 = mm → m). Set to 1.0
                if your mesh is already in metres.
            est_refine_iter: Refinement iterations for register() (default 5).
            track_refine_iter: Refinement iterations for track_one() (default 2).
            max_detections: Max SAM-6D detections to process per call (None = all).
            min_pose_score: Skip SAM-6D detections below this pose_score threshold.
            debug: Debug verbosity 0–3 (passed to internal FoundationPose estimator).
            debug_dir: Directory for debug visualizations (if debug >= 2).
        """
        if device == "cpu":
            raise NotImplementedError(
                "FoundationPose requires CUDA. CPU-only mode is not supported."
            )

        self.cad_path = Path(cad_path)
        self.device = device
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.max_detections = max_detections
        self.min_pose_score = min_pose_score
        self.debug = debug
        self.debug_dir = str(debug_dir) if debug_dir else "/tmp/foundationpose_debug"
        if self.debug >= 2:
            os.makedirs(self.debug_dir, exist_ok=True)

        _ensure_fp_paths()
        self._check_mycpp()

        _weights_dir = Path(weights_dir) if weights_dir else _FP_WEIGHTS
        self._maybe_download_weights(_weights_dir)

        import trimesh
        logging.info(f"Loading mesh from {self.cad_path} (mesh_scale={mesh_scale}) ...")
        mesh = trimesh.load(str(self.cad_path), force="mesh")
        mesh.vertices = mesh.vertices * mesh_scale  # convert to metres for FoundationPose
        mesh.process(validate=True)
        if len(mesh.vertex_normals) == 0:
            mesh = mesh.smoothed()
        self._mesh = mesh
        self._mesh_scale = mesh_scale
        logging.info(f"Mesh loaded: {len(mesh.vertices)} vertices, diameter≈{mesh.bounding_box.extents.max():.4f} m.")

        logging.info("Loading PoseRefinePredictor ...")
        from learning.training.predict_pose_refine import PoseRefinePredictor
        self._refiner = PoseRefinePredictor()
        logging.info("Loading ScorePredictor ...")
        from learning.training.predict_score import ScorePredictor
        self._scorer = ScorePredictor()
        logging.info("FoundationPose models loaded.")

        # Import internal FoundationPose class under an alias to avoid name clash
        _est_mod = importlib.import_module("estimater")
        self._FPEst = _est_mod.FoundationPose

        self._glctx = None          # lazy init on first register() call
        self._est_list: List = []   # one internal estimator per tracked detection

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

    def _make_estimator(self):
        """Create a fresh internal FoundationPose estimator, lazily init glctx."""
        if self._glctx is None:
            import nvdiffrast.torch as dr
            self._glctx = dr.RasterizeCudaContext()
        return self._FPEst(
            model_pts=self._mesh.vertices,
            model_normals=self._mesh.vertex_normals,
            mesh=self._mesh,
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

        Args:
            rgb: (H, W, 3) uint8 BGR or RGB image (same convention as SAM-6D input).
            depth: (H, W) raw depth pixels. Multiply by depth_scale to get mm.
            K: (3, 3) camera intrinsic matrix.
            sam6d_results: Output dict from SAM6D.forward().
            depth_scale: Scale to convert raw depth to mm (e.g. 1000.0 for uint16 mm depth).

        Returns:
            List of dicts, one per processed detection:
                pose        — (4, 4) float32 object-in-camera transform, translation in metres.
                fp_score    — FoundationPose scorer confidence (higher is better).
                sam6d_score — SAM-6D pose_score for this detection.
                mask        — (H, W) bool mask used for registration.
                box         — (4,) int64 xyxy bounding box.
                detection_idx — index into the sam6d_results arrays.
        """
        if len(sam6d_results.get("poses", [])) == 0:
            self._est_list = []
            return []

        depth_m = self._depth_to_metres(depth, depth_scale)

        pose_scores = sam6d_results["pose_scores"]
        order = np.argsort(pose_scores)[::-1]  # descending by score
        selected = [i for i in order if pose_scores[i] >= self.min_pose_score]
        if self.max_detections is not None:
            selected = selected[: self.max_detections]

        self._est_list = []
        results = []
        for i in selected:
            mask = sam6d_results["masks"][i].astype(np.uint8)
            est = self._make_estimator()
            pose = est.register(
                K=K,
                rgb=rgb,
                depth=depth_m,
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )
            fp_score = float(est.scores[0].item())
            self._est_list.append(est)
            results.append(
                {
                    "pose": pose.astype(np.float32),
                    "fp_score": fp_score,
                    "sam6d_score": float(pose_scores[i]),
                    "mask": mask.astype(bool),
                    "box": sam6d_results["boxes"][i],
                    "detection_idx": int(i),
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
                pose     — (4, 4) float32 object-in-camera transform in metres.
        """
        if not self._est_list:
            raise RuntimeError(
                "No active estimators. Call refine_from_sam6d() before track()."
            )

        depth_m = self._depth_to_metres(depth, depth_scale)
        results = []
        for est in self._est_list:
            pose = est.track_one(
                rgb=rgb,
                depth=depth_m,
                K=K,
                iteration=self.track_refine_iter,
            )
            results.append({"pose": pose.astype(np.float32)})

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
