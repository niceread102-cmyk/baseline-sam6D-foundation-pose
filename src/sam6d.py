"""SAM-6D wrapper: two-stage 6DoF pose estimation (ISM + PEM) for one or more CAD models.

A single SAM6D instance handles M ≥ 1 CAD model types simultaneously. Pass a list of paths
to cad_path and all shared neural network weights (SAM, DINOv2, PEM) are loaded exactly once.
"""
import os
import sys
import logging
import numpy as np
import torch
import torchvision.transforms as T
import cv2
import trimesh
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAM6D_ROOT = _REPO_ROOT / "SAM-6D-main" / "SAM-6D"
_ISM_ROOT = _SAM6D_ROOT / "Instance_Segmentation_Model"
_PEM_ROOT = _SAM6D_ROOT / "Pose_Estimation_Model"

_RGB_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SAM_WEIGHTS = {
    "vit_h": ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
    "vit_l": ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    "vit_b": ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
}

_N_TEMPLATES = 42  # views in cam_poses_level0.npy (level-0 icosphere)


def _ensure_sam6d_paths():
    for p in [
        str(_ISM_ROOT),
        str(_PEM_ROOT / "model"),
        str(_PEM_ROOT / "model" / "pointnet2"),
        str(_PEM_ROOT / "utils"),
    ]:
        if p not in sys.path:
            sys.path.append(p)


def _broadcast_thresh(val, M: int, default=None) -> list:
    """Broadcast a scalar or list threshold to a length-M list."""
    if val is None:
        return [default] * M
    if isinstance(val, (int, float)):
        return [float(val)] * M
    val = list(val)
    if len(val) != M:
        raise ValueError(f"Expected {M} threshold values, got {len(val)}")
    return val


class SAM6D:
    def __init__(
        self,
        cad_path: Union[str, Path, List[Union[str, Path]]],
        model_ids: Optional[List[str]] = None,
        device: str = "cuda",

        # Segmentor
        sam_checkpoint: Optional[Union[str, Path]] = None,
        sam_model_type: str = "vit_h",
        stability_score_thresh: float = 0.97,
        points_per_side: int = 32,            # grid density for SAM proposals (32→~1024 prompts)

        # DINOv2 checkpoint dir (parent dir containing dinov2_vitl14_pretrain.pth)
        dinov2_checkpoint_dir: Optional[Union[str, Path]] = None,

        # PEM checkpoint
        pem_checkpoint: Optional[Union[str, Path]] = None,

        # Template rendering
        template_dir: Optional[Union[str, Path]] = None,      # root dir; sub-dirs derived per model when M>1
        template_dirs: Optional[List[Union[str, Path]]] = None,  # explicit per-model override; takes precedence
        reset_templates: Union[bool, List[bool]] = False,
        mesh_color: Optional[Union[tuple, List[Optional[tuple]]]] = None,
        flat_shading: Union[bool, List[bool]] = False,
        roughness: Optional[Union[float, List[Optional[float]]]] = None,
        metallic: Optional[Union[float, List[Optional[float]]]] = None,
        light_energy: Union[float, List[float]] = 1000,

        # Matching / filtering (scalar = same for all models, list = per-model length M)
        det_score_thresh: Union[float, List[float]] = 0.2,
        confidence_thresh: Union[float, List[float]] = 0.2,
        visible_thred: float = 0.5,
        n_template_view: int = 42,
        min_component_area_ratio: float = 0.005,  # drop mask blobs < ratio * largest blob (0=off)
        mask_overlap_thresh: float = 0.9,          # drop lower-scoring mask if overlap > thresh (0=off)
        pose_score_thresh: Union[float, List[float]] = 0.0,
        max_detections_per_model: Optional[Union[int, List[Optional[int]]]] = None,

        # PEM dataset config
        pem_img_size: int = 224,
        pem_n_sample_template: int = 5000,
        pem_n_sample_observed: int = 2048,
        pem_n_sample_model: int = 1024,
        pem_rgb_mask_flag: bool = True,

        # Memory management
        half_precision: bool = False,  # fp16 for SAM + DINOv2 only (~1.8 GB saved); PEM stays fp32 (CUDA SVD not supported in fp16)
        auto_offload: bool = False,
        empty_cache_between_stages: bool = False,
    ):
        _ensure_sam6d_paths()

        self.device = torch.device(device)

        # --- Normalize cad_path to a list ---
        if isinstance(cad_path, (str, Path)):
            cad_paths = [Path(cad_path)]
        else:
            cad_paths = [Path(p) for p in cad_path]
        M = len(cad_paths)
        if M == 0:
            raise ValueError("cad_path must contain at least one model.")
        self._cad_paths = cad_paths
        self.cad_path   = cad_paths[0]  # backward compat
        self._model_ids = list(model_ids) if model_ids else [p.stem for p in cad_paths]
        if len(self._model_ids) != M:
            raise ValueError(f"model_ids length {len(self._model_ids)} != cad_path count {M}")
        self.M = M

        # --- Derive per-model template directories ---
        if template_dirs is not None:
            tmpl_dirs = [Path(d) for d in template_dirs]
            if len(tmpl_dirs) != M:
                raise ValueError(f"template_dirs length {len(tmpl_dirs)} != {M}")
        elif template_dir is not None:
            root = Path(template_dir)
            tmpl_dirs = [root] if M == 1 else [root / mid for mid in self._model_ids]
        else:
            if M == 1:
                tmpl_dirs = [Path("./sam6d_templates") / cad_paths[0].stem]
            else:
                tmpl_dirs = [Path(f"./sam6d_templates_{mid}") for mid in self._model_ids]
        for d in tmpl_dirs:
            d.mkdir(parents=True, exist_ok=True)
        self.template_dir = tmpl_dirs[0]   # backward compat
        self._tmpl_dir    = tmpl_dirs[0] / "templates"

        # --- Broadcast per-model rendering params ---
        def _bcast(val, M, default=None):
            if val is None:
                return [default] * M
            if not isinstance(val, list):  # tuples (e.g. RGB color) are treated as a single value
                return [val] * M
            val = list(val)
            if len(val) != M:
                raise ValueError(f"Expected length {M}, got {len(val)}")
            return val

        reset_templates_list = _bcast(reset_templates, M, False)
        mesh_color_list      = _bcast(mesh_color, M, None)
        flat_shading_list    = _bcast(flat_shading, M, False)
        roughness_list       = _bcast(roughness, M, None)
        metallic_list        = _bcast(metallic, M, None)
        light_energy_list    = _bcast(light_energy, M, 1000)

        # --- Broadcast per-model thresholds ---
        self._det_score_thresh_list  = _broadcast_thresh(det_score_thresh, M)
        self._confidence_thresh_list = _broadcast_thresh(confidence_thresh, M)
        self._pose_score_thresh_list = _broadcast_thresh(pose_score_thresh, M)
        self._max_det_per_model      = _bcast(max_detections_per_model, M, None)
        # Scalar backward-compat aliases (index-0 values)
        self.det_score_thresh  = self._det_score_thresh_list[0]
        self.confidence_thresh = self._confidence_thresh_list[0]
        self.pose_score_thresh = self._pose_score_thresh_list[0]

        self.visible_thred            = visible_thred
        self.n_template_view          = n_template_view
        self.min_component_area_ratio = min_component_area_ratio
        self.mask_overlap_thresh      = mask_overlap_thresh
        self.pem_img_size             = pem_img_size
        self.pem_n_sample_template    = pem_n_sample_template
        self.pem_n_sample_observed    = pem_n_sample_observed
        self.pem_n_sample_model       = pem_n_sample_model
        self.pem_rgb_mask_flag        = pem_rgb_mask_flag
        self._half_precision          = half_precision
        self._auto_offload            = auto_offload
        self._empty_cache             = empty_cache_between_stages

        _ism_ckpt = _ISM_ROOT / "checkpoints"
        _pem_ckpt = _PEM_ROOT / "checkpoints"

        # --- Download weights ---
        self._maybe_download_sam(sam_checkpoint, sam_model_type, _ism_ckpt)
        self._maybe_download_dinov2(dinov2_checkpoint_dir, _ism_ckpt)
        self._maybe_download_pem(pem_checkpoint, _pem_ckpt)
        self._maybe_download_mae(_REPO_ROOT / "checkpoints")

        _sam_path   = Path(sam_checkpoint) if sam_checkpoint else _ism_ckpt / "segment-anything" / SAM_WEIGHTS[sam_model_type][0]
        _dinov2_dir = Path(dinov2_checkpoint_dir) if dinov2_checkpoint_dir else _ism_ckpt / "dinov2"
        _pem_path   = Path(pem_checkpoint) if pem_checkpoint else _pem_ckpt / "sam-6d-pem-base.pth"

        # --- Import SAM-6D ISM modules ---
        from segment_anything import sam_model_registry
        from model.sam import CustomSamAutomaticMaskGenerator
        from model.dinov2 import CustomDINOv2
        from model.utils import Detections
        from model.loss import PairwiseSimilarity, MaskedPatch_MatrixSimilarity
        from utils.bbox_utils import CropResizePad, compute_iou
        from utils.trimesh_utils import depth_image_to_pointcloud_translate_torch

        self._Detections = Detections
        self._MaskedPatch_MatrixSimilarity = MaskedPatch_MatrixSimilarity
        self._compute_iou = compute_iou
        self._depth_image_to_pointcloud_translate_torch = depth_image_to_pointcloud_translate_torch

        # --- Load SAM ---
        logging.info(f"Loading SAM ({sam_model_type}) ...")
        sam = sam_model_registry[sam_model_type](checkpoint=str(_sam_path))
        sam.to(device=self.device)
        if half_precision:
            sam.half()
        self.segmentor = CustomSamAutomaticMaskGenerator(
            sam=sam,
            stability_score_thresh=stability_score_thresh,
            segmentor_width_size=640,
            points_per_side=points_per_side,
        )
        logging.info("SAM loaded.")

        # --- Load DINOv2 (ISM's custom version) ---
        logging.info("Loading DINOv2 (ISM) ...")
        self.descriptor = CustomDINOv2(
            model_name="dinov2_vitl14",
            token_name="x_norm_clstoken",
            image_size=224,
            chunk_size=16,
            descriptor_width_size=640,
            checkpoint_dir=str(_dinov2_dir),
        )
        self.descriptor.model = self.descriptor.model.to(self.device)
        self.descriptor.model.device = self.device
        if half_precision:
            self.descriptor.model.half()
        logging.info("DINOv2 loaded.")

        self._similarity  = PairwiseSimilarity(metric="cosine", chunk_size=16)
        self._crop_resize = CropResizePad(224)

        # --- Load PEM ---
        logging.info("Loading PEM ...")
        import gorilla
        import importlib
        cfg = gorilla.Config.fromfile(str(_PEM_ROOT / "config" / "base.yaml"))
        MODEL = importlib.import_module("pose_estimation_model")
        self.pem = MODEL.Net(cfg.model)
        self.pem = self.pem.to(self.device)
        self.pem.eval()
        gorilla.solver.load_checkpoint(model=self.pem, filename=str(_pem_path))
        logging.info("PEM loaded.")
        # PEM stays in fp32: CUDA batched SVD (used in the pose solver) is not
        # implemented for fp16, so half_precision does not apply here.

        # --- Per-model template prep loop ---
        self.ref_data    = {}
        per_model_ism    = []
        self._pem_model_pts_list = []
        self._pem_templates      = []

        for idx in range(M):
            tmpl_dir_root = tmpl_dirs[idx]
            tmpl_dir_i    = tmpl_dir_root / "templates"
            logging.info(f"Onboarding model {idx + 1}/{M}: {cad_paths[idx].stem} ...")

            self._maybe_render_templates_for(
                cad_paths[idx], tmpl_dir_root, tmpl_dir_i,
                reset_templates_list[idx],
                mesh_color_list[idx],
                flat_shading_list[idx],
                roughness_list[idx],
                metallic_list[idx],
                light_energy_list[idx],
            )

            ism_data = self._compute_ism_ref_data(tmpl_dir_i, cad_paths[idx])
            per_model_ism.append({
                "descriptors":      ism_data["descriptors"],
                "appe_descriptors": ism_data["appe_descriptors"],
                "pointcloud":       ism_data["pointcloud"],
            })
            self._pem_model_pts_list.append(ism_data["pem_model_pts"])

            logging.info(f"Extracting PEM template features for {cad_paths[idx].stem} ...")
            pts, feat = self._compute_pem_template_feats(tmpl_dir_i)
            self._pem_templates.append((pts, feat))

        # Stack ISM data across M models; poses are model-agnostic (icosphere camera viewpoints)
        self.ref_data["descriptors"]      = torch.cat([d["descriptors"]      for d in per_model_ism], dim=0)  # [M, N_t, D]
        self.ref_data["appe_descriptors"] = torch.cat([d["appe_descriptors"] for d in per_model_ism], dim=0)  # [M, N_t, P, D]
        self.ref_data["pointcloud"]       = torch.cat([d["pointcloud"]       for d in per_model_ism], dim=0)  # [M, N, 3]
        self.ref_data["poses"]            = ism_data["poses"]   # [N_level0, 4, 4], shared across all models

        # Backward-compat aliases
        self._pem_model_pts = self._pem_model_pts_list[0]
        self.all_tem_pts    = self._pem_templates[0][0]
        self.all_tem_feat   = self._pem_templates[0][1]

        logging.info(
            f"SAM6D init complete: {M} model(s), "
            f"descriptors={self.ref_data['descriptors'].shape}, "
            f"pointcloud={self.ref_data['pointcloud'].shape}"
        )

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _maybe_download_sam(self, sam_checkpoint, sam_model_type, save_dir):
        if sam_checkpoint and Path(sam_checkpoint).exists():
            return
        target = Path(save_dir) / "segment-anything" / SAM_WEIGHTS[sam_model_type][0]
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading SAM {sam_model_type} ...")
        ret = os.system(f"wget -q -O {target} {SAM_WEIGHTS[sam_model_type][1]} --no-check-certificate")
        if ret != 0:
            raise RuntimeError(
                f"Failed to download SAM {sam_model_type} weights (exit {ret}).\n"
                f"Download manually: {SAM_WEIGHTS[sam_model_type][1]}\n"
                f"Place at: {target}"
            )

    def _maybe_download_dinov2(self, dinov2_checkpoint_dir, save_dir):
        if dinov2_checkpoint_dir:
            target = Path(dinov2_checkpoint_dir) / "dinov2_vitl14_pretrain.pth"
        else:
            target = Path(save_dir) / "dinov2" / "dinov2_vitl14_pretrain.pth"
        if target.exists():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading DINOv2 weights ...")
        url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
        ret = os.system(f"wget -q -O {target} {url} --no-check-certificate")
        if ret != 0:
            raise RuntimeError(
                f"Failed to download DINOv2 weights (exit {ret}).\n"
                f"Download manually: {url}\n"
                f"Place at: {target}"
            )

    def _maybe_download_pem(self, pem_checkpoint, save_dir):
        if pem_checkpoint and Path(pem_checkpoint).exists():
            return
        target = Path(save_dir) / "sam-6d-pem-base.pth"
        if target.exists():
            return
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        logging.info("Downloading PEM checkpoint ...")
        ret = os.system(
            f"gdown --no-cookies --no-check-certificate "
            f"-O '{target}' 1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7"
        )
        if ret != 0 or not target.exists():
            raise RuntimeError(
                f"Failed to download SAM-6D PEM checkpoint (exit {ret}).\n"
                "Download manually with gdown ID: 1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7\n"
                f"Place at: {target}"
            )

    def _maybe_download_mae(self, save_dir):
        target = Path(save_dir) / "mae_pretrain_vit_base.pth"
        if target.exists():
            return
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        logging.info("Downloading MAE ViT-Base backbone ...")
        ret = os.system(f"wget -q -O '{target}' '{url}' --no-check-certificate")
        if ret != 0 or not target.exists():
            raise RuntimeError(
                f"Failed to download MAE ViT-Base backbone (exit {ret}).\n"
                f"Download manually: {url}\n"
                f"Place at: {target}"
            )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def offload_to_cpu(self):
        """Move all GPU-resident models and tensors to CPU to free VRAM.

        Call this after SAM6D.forward() completes and before running FoundationPose
        (or any other VRAM-intensive stage) to reclaim the ~5 GB held by SAM, DINOv2,
        and PEM. Use reload_to_gpu() to bring everything back before the next forward().
        """
        self.segmentor.predictor.model.cpu()
        self.descriptor.model.cpu()
        self.pem.cpu()
        for k, v in self.ref_data.items():
            if isinstance(v, torch.Tensor):
                self.ref_data[k] = v.cpu()
        self._pem_templates = [(p.cpu(), f.cpu()) for p, f in self._pem_templates]
        # Keep backward-compat aliases in sync
        self.all_tem_pts  = self._pem_templates[0][0]
        self.all_tem_feat = self._pem_templates[0][1]
        torch.cuda.empty_cache()

    def reload_to_gpu(self):
        """Reload CPU-offloaded models back to the configured CUDA device.

        Reverses offload_to_cpu() so that forward() can run again.
        """
        self.segmentor.predictor.model.to(self.device)
        self.descriptor.model.to(self.device)
        self.pem.to(self.device)
        for k, v in self.ref_data.items():
            if isinstance(v, torch.Tensor):
                self.ref_data[k] = v.to(self.device)
        self._pem_templates = [(p.to(self.device), f.to(self.device)) for p, f in self._pem_templates]
        self.all_tem_pts  = self._pem_templates[0][0]
        self.all_tem_feat = self._pem_templates[0][1]

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _maybe_render_templates_for(
        self,
        cad_path: Path,
        template_dir: Path,
        tmpl_dir: Path,
        reset_templates: bool,
        mesh_color: Optional[tuple],
        flat_shading: bool,
        roughness: Optional[float],
        metallic: Optional[float],
        light_energy: float,
    ):
        """Render templates for a specific CAD model into a specific directory."""
        n_rgb  = len(list(tmpl_dir.glob("rgb_*.png")))
        n_xyz  = len(list(tmpl_dir.glob("xyz_*.npy")))
        n_mask = len(list(tmpl_dir.glob("mask_*.png")))
        n_expected = _N_TEMPLATES
        if not reset_templates and n_rgb == n_expected and n_xyz == n_expected and n_mask == n_expected:
            logging.info(f"Templates already at {tmpl_dir}, skipping render.")
            return

        render_script = str(_SAM6D_ROOT / "Render" / "render_custom_templates.py")
        extra = ""
        if mesh_color is not None:
            r, g, b = mesh_color
            extra += f" --mesh_color {r} {g} {b}"
        if flat_shading:
            extra += " --flat_shading"
        if roughness is not None:
            extra += f" --roughness {roughness}"
        if metallic is not None:
            extra += f" --metallic {metallic}"
        if light_energy != 1000:
            extra += f" --light_energy {light_energy}"
        cmd = (
            f"env -u MPLBACKEND blenderproc run {render_script} "
            f"--output_dir {template_dir} --cad_path {cad_path}{extra}"
        )
        logging.info(f"Rendering templates: {cmd}")
        ret = os.system(cmd)
        if ret != 0 or len(list(tmpl_dir.glob("rgb_*.png"))) == 0:
            raise RuntimeError(
                f"BlenderProc rendering failed (exit {ret}). "
                f"Run manually: blenderproc run {render_script} "
                f"--output_dir {template_dir} --cad_path {cad_path}{extra}"
            )

    # ------------------------------------------------------------------
    # ISM onboarding
    # ------------------------------------------------------------------

    def _compute_ism_ref_data(self, tmpl_dir: Path, cad_path: Path) -> Dict[str, Any]:
        """Compute ISM reference descriptors and pointcloud for one CAD model.

        Returns a dict with keys: descriptors [1,N_t,D], appe_descriptors [1,N_t,P,D],
        poses [N_level0,4,4], pointcloud [1,2048,3], pem_model_pts np.ndarray[n,3].
        Does not write to self.ref_data — caller decides where to store results.
        """
        from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2

        template_paths = sorted(tmpl_dir.glob("rgb_*.png"))
        desc_cache = tmpl_dir / "ism_descriptors.pth"
        appe_cache = tmpl_dir / "ism_appe_descriptors.pth"

        boxes, templates, masks_list = [], [], []
        for path in template_paths:
            img = Image.open(path)
            mask_img = Image.open(tmpl_dir / path.name.replace("rgb_", "mask_"))
            boxes.append(mask_img.getbbox())
            image = torch.from_numpy(np.array(img.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask_img.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks_list.append(mask.unsqueeze(-1))

        templates_t = torch.stack(templates).permute(0, 3, 1, 2)   # [N, 3, H, W]
        masks_t = torch.stack(masks_list).permute(0, 3, 1, 2)      # [N, 1, H, W]
        boxes_t = torch.tensor(np.array(boxes))

        templates_proc = self._crop_resize(images=templates_t, boxes=boxes_t).to(self.device)
        masks_proc = self._crop_resize(images=masks_t, boxes=boxes_t).to(self.device)

        if desc_cache.exists():
            logging.info(f"Loading cached ISM descriptors from {desc_cache}")
            descriptors = torch.load(desc_cache, map_location=self.device)
        else:
            logging.info(f"Computing ISM semantic descriptors for {cad_path.stem} ...")
            with torch.autocast("cuda", enabled=self._half_precision):
                descriptors = self.descriptor.compute_features(templates_proc, token_name="x_norm_clstoken")
            torch.save(descriptors, desc_cache)

        if appe_cache.exists():
            logging.info(f"Loading cached ISM appearance descriptors from {appe_cache}")
            appe_descriptors = torch.load(appe_cache, map_location=self.device)
        else:
            logging.info(f"Computing ISM appearance descriptors for {cad_path.stem} ...")
            with torch.autocast("cuda", enabled=self._half_precision):
                appe_descriptors = self.descriptor.compute_masked_patch_feature(
                    templates_proc, masks_proc[:, 0, :, :]
                )
            torch.save(appe_descriptors, appe_cache)

        # Template camera poses (level-2 set → index down to level-0 subset)
        # These are camera viewpoints on the icosphere — identical for all CAD models.
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
        index_templates = load_index_level_in_level2(0, "all")
        ref_poses = poses[index_templates, :, :]   # [N_level0, 4, 4]

        # ISM point cloud: sampled from CAD mesh, centred to match renderer
        mesh = trimesh.load_mesh(str(cad_path))
        ism_pts = mesh.sample(2048).astype(np.float32) / 1000.0
        mesh_centroid = np.array(mesh.bounding_box.centroid, dtype=np.float32) / 1000.0
        ism_pts = ism_pts - mesh_centroid
        pointcloud = torch.tensor(ism_pts).unsqueeze(0).to(self.device)  # [1, 2048, 3]

        # PEM model points cached here to avoid per-frame mesh loading in forward()
        pem_model_pts = mesh.sample(self.pem_n_sample_model).astype(np.float32) / 1000.0

        logging.info(
            f"ISM ref data for {cad_path.stem}: "
            f"descriptors={descriptors.unsqueeze(0).shape}, "
            f"appe={appe_descriptors.unsqueeze(0).shape}"
        )
        return {
            "descriptors":      descriptors.unsqueeze(0),       # [1, N_t, D]
            "appe_descriptors": appe_descriptors.unsqueeze(0),  # [1, N_t, P, D]
            "poses":            ref_poses,                       # [N_level0, 4, 4]
            "pointcloud":       pointcloud,                      # [1, 2048, 3]
            "pem_model_pts":    pem_model_pts,                   # np.ndarray [n, 3]
        }

    # ------------------------------------------------------------------
    # PEM onboarding
    # ------------------------------------------------------------------

    def _get_pem_template_from(self, tmpl_dir: Path, idx: int):
        """Load one rendered view for PEM from an explicit template directory."""
        rgb_np = np.array(Image.open(tmpl_dir / f"rgb_{idx}.png").convert("RGB"))
        xyz = np.load(str(tmpl_dir / f"xyz_{idx}.npy")).astype(np.float32) / 1000.0
        mask = (np.array(Image.open(tmpl_dir / f"mask_{idx}.png").convert("L")) == 255)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            # blank template — return zeros
            rgb_t = torch.zeros(3, self.pem_img_size, self.pem_img_size)
            choose = np.zeros(self.pem_n_sample_template, dtype=np.int64)
            pts = np.zeros((self.pem_n_sample_template, 3), dtype=np.float32)
            return rgb_t, choose, pts

        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1]) + 1
        cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1]) + 1

        mask_crop = mask[rmin:rmax, cmin:cmax]
        # Match PEM's original: imageio (RGB) → [::-1] (BGR) → crop
        rgb_crop = rgb_np[rmin:rmax, cmin:cmax, :][:, :, ::-1].copy()
        if self.pem_rgb_mask_flag:
            rgb_crop = rgb_crop * (mask_crop[:, :, None] > 0).astype(np.uint8)
        rgb_crop = cv2.resize(rgb_crop, (self.pem_img_size, self.pem_img_size), interpolation=cv2.INTER_LINEAR)
        rgb_t = _RGB_TRANSFORM(np.array(rgb_crop))

        choose = mask_crop.astype(np.float32).flatten().nonzero()[0]
        n = self.pem_n_sample_template
        choose_idx = np.random.choice(len(choose), n, replace=(len(choose) <= n))
        choose = choose[choose_idx]

        xyz_pts = xyz[rmin:rmax, cmin:cmax, :].reshape(-1, 3)[choose, :]

        # Map flat indices back to resized-image pixel indices
        crop_h, crop_w = rmax - rmin, cmax - cmin
        row_idx = choose // crop_w
        col_idx = choose % crop_w
        rgb_choose = (
            np.floor(row_idx * (self.pem_img_size / crop_h)) * self.pem_img_size
            + np.floor(col_idx * (self.pem_img_size / crop_w))
        ).astype(np.int64)

        return rgb_t, rgb_choose, xyz_pts

    def _get_pem_template(self, idx: int):
        """Load one rendered view for PEM from self._tmpl_dir."""
        return self._get_pem_template_from(self._tmpl_dir, idx)

    def _compute_pem_template_feats(self, tmpl_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract PEM template features from a template directory. Returns (pts, feat)."""
        all_tem, all_tem_pts_list, all_tem_choose = [], [], []
        total_n = _N_TEMPLATES
        for v in range(self.n_template_view):
            i = int(total_n / self.n_template_view * v)
            tem, tem_choose, tem_pts = self._get_pem_template_from(tmpl_dir, i)
            all_tem.append(tem.float().unsqueeze(0).to(self.device))
            all_tem_choose.append(torch.LongTensor(tem_choose).unsqueeze(0).to(self.device))
            all_tem_pts_list.append(torch.FloatTensor(tem_pts).unsqueeze(0).to(self.device))

        with torch.no_grad():
            pts, feat = self.pem.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts_list, all_tem_choose
            )
        return pts, feat

    # ------------------------------------------------------------------
    # Mask cleaning
    # ------------------------------------------------------------------

    def _clean_mask_components(self, detections):
        """Remove disconnected blobs smaller than min_component_area_ratio * largest blob.

        Also recomputes bounding boxes from the cleaned masks so stray specs don't
        inflate the detection box used in geometric scoring.
        """
        cleaned_masks, cleaned_boxes = [], []
        for i, mask in enumerate(detections.masks):
            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            if n_labels <= 1:
                cleaned_masks.append(mask)
                cleaned_boxes.append(None)
                continue

            areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
            max_area = areas.max()
            keep_labels = np.where(areas >= self.min_component_area_ratio * max_area)[0] + 1

            cleaned = np.isin(labels, keep_labels).astype(np.uint8)
            ys, xs = np.nonzero(cleaned)
            if len(xs) == 0:
                cleaned_masks.append(mask)
                cleaned_boxes.append(None)
                continue

            box = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.long)
            cleaned_masks.append(torch.from_numpy(cleaned).to(dtype=mask.dtype, device=mask.device))
            cleaned_boxes.append(box.to(mask.device))

        detections.masks = torch.stack(cleaned_masks)
        detections.boxes = torch.stack([
            b if b is not None else detections.boxes[i]
            for i, b in enumerate(cleaned_boxes)
        ])
        return detections

    def _filter_overlapping_masks(self, masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Greedy overlap suppression — returns indices of detections to keep.

        For each pair of masks, overlap = intersection / min(area_A, area_B).
        If two masks overlap above mask_overlap_thresh, the lower-scoring one is dropped.
        Detections are processed in descending score order so the best one always wins.
        Works across model types: a lower-scoring detection from any model is suppressed.
        """
        n = len(masks)
        if n == 0:
            return np.array([], dtype=int)

        order = np.argsort(-scores)   # highest score first
        areas = masks.reshape(n, -1).sum(axis=1).astype(np.float32)
        suppressed = np.zeros(n, dtype=bool)

        for i_rank in range(n):
            i = order[i_rank]
            if suppressed[i]:
                continue
            for j_rank in range(i_rank + 1, n):
                j = order[j_rank]
                if suppressed[j]:
                    continue
                intersection = (masks[i] & masks[j]).sum()
                min_area = min(areas[i], areas[j])
                if min_area > 0 and intersection / min_area > self.mask_overlap_thresh:
                    suppressed[j] = True

        return np.where(~suppressed)[0]

    # ------------------------------------------------------------------
    # ISM scoring (ported from Instance_Segmentation_Model)
    # ------------------------------------------------------------------

    def _compute_semantic_score(self, proposal_descriptors):
        # scores: [N_proposals, M, N_templates]
        scores = self._similarity(proposal_descriptors, self.ref_data["descriptors"])
        top5 = torch.topk(scores, k=min(5, scores.shape[-1]), dim=-1)[0]
        score_per_obj = top5.mean(dim=-1)                              # [N_proposals, M]

        score_per_proposal, assigned_obj = torch.max(score_per_obj, dim=-1)  # [N_proposals]

        # Per-model confidence threshold (when M=1 assigned_obj is always 0, giving scalar behaviour)
        thresh_t = torch.tensor(
            [self._confidence_thresh_list[m] for m in assigned_obj.tolist()],
            dtype=torch.float32, device=score_per_proposal.device,
        )
        idx_selected = torch.where(score_per_proposal > thresh_t)[0]

        pred_idx_objects = assigned_obj[idx_selected]
        semantic_score   = score_per_proposal[idx_selected]

        filtered_scores = scores[idx_selected]                    # [N_keep, M, N_templates]
        _, best_tmpl_per_obj = torch.max(filtered_scores, dim=-1) # [N_keep, M]
        pred_obj_rep = pred_idx_objects[:, None].repeat(1, best_tmpl_per_obj.shape[1])
        best_template = torch.gather(best_tmpl_per_obj, dim=1, index=pred_obj_rep)[:, 0]

        return idx_selected, pred_idx_objects, semantic_score, best_template

    def _compute_appearance_score(self, best_pose, pred_objects_idx, query_appe):
        con_idx = torch.stack([pred_objects_idx, best_pose], dim=0)  # [2, N]
        ref_appe = self.ref_data["appe_descriptors"][con_idx[0], con_idx[1]]  # [N, P, D]
        aux = self._MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux.compute_straight(query_appe, ref_appe)
        return appe_scores, ref_appe

    def _project_template_to_image(self, best_pose, pred_obj_idx, depth_t, cam_K_t, depth_scale_t, proposal_masks):
        pose_R = self.ref_data["poses"][best_pose, :3, :3]         # [N, 3, 3]
        select_pc = self.ref_data["pointcloud"][pred_obj_idx]       # [N, N_pts, 3]
        N_q, N_pts, _ = select_pc.shape
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)  # [N, N_pts, 3]

        # Translation: mean 3D position of the proposal's depth pixels
        # descriptor_model.forward() adds a channel dim in-place ([N,1,H,W]); squeeze it out
        if proposal_masks.dim() == 4:
            proposal_masks = proposal_masks.squeeze(1)
        N_masks, H, W = proposal_masks.shape
        masked_depth = proposal_masks.float() * depth_t.unsqueeze(0).expand(N_masks, -1, -1)
        translate = self._depth_image_to_pointcloud_translate_torch(masked_depth, depth_scale_t, cam_K_t)
        translate = translate.to(torch.float32)                     # [N, 3]
        posed_pc = posed_pc + translate.unsqueeze(1).expand(-1, N_pts, -1)

        K_rep = cam_K_t.unsqueeze(0).expand(N_q, -1, -1).to(torch.float32)
        image_homo = torch.bmm(K_rep, posed_pc.permute(0, 2, 1)).permute(0, 2, 1)
        image_uv = (image_homo / image_homo[:, :, -1:])[:, :, :2].to(torch.int)
        image_uv[:, :, 0].clamp_(min=0, max=W - 1)
        image_uv[:, :, 1].clamp_(min=0, max=H - 1)
        return image_uv

    def _compute_geometric_score(self, image_uv, detections, query_appe, ref_appe):
        aux = self._MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux.compute_visible_ratio(query_appe, ref_appe, self.visible_thred)
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.cat([y1x1, y2x2], dim=-1)
        iou = self._compute_iou(xyxy, detections.boxes)
        # compute_iou returns 0.0 (float) when boxes don't intersect; normalise to tensor
        if not isinstance(iou, torch.Tensor):
            iou = torch.full((xyxy.shape[0],), float(iou), dtype=torch.float32, device=xyxy.device)
        return iou, visible_ratio

    # ------------------------------------------------------------------
    # Main forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Run SAM-6D on a single RGBD frame.

        Args:
            rgb:         RGB image as PIL Image or uint8 numpy array [H, W, 3].
            depth:       Depth map [H, W], values in mm (scaled by depth_scale to reach mm).
            K:           Camera intrinsic matrix [3, 3].
            depth_scale: Multiply depth pixel values to get mm. Default 1.0 (depth is already mm).

        Returns a dict with:
            poses              list of {"R": [3,3], "t": [3], "score": float}
            masks              [N, H, W] bool   ISM proposal masks
            boxes              [N, 4]           xyxy boxes
            ism_scores         [N]              final ISM confidence
            semantic_scores    [N]              DINOv2 semantic similarity
            appe_scores        [N]              appearance similarity
            geometric_scores   [N]              geometric IoU score
            visible_ratios     [N]              visible ratio from appearance matching
            best_template_ids  [N]              index of best-matching template per proposal
            pred_R             [N, 3, 3]        raw PEM rotation outputs
            pred_t             [N, 3]           raw PEM translation outputs (mm)
            pose_scores        [N]              raw PEM confidence scores
            model_ids          List[str]        which CAD model each detection belongs to
            model_indices      [N] int64        model index (always 0 for single-model SAM6D)
        """
        if isinstance(rgb, Image.Image):
            rgb_np = np.array(rgb.convert("RGB"))
        else:
            rgb_np = np.asarray(rgb)
        H, W = rgb_np.shape[:2]

        depth_t = torch.from_numpy(depth.astype(np.float32)).to(self.device)
        cam_K_t = torch.from_numpy(K.astype(np.float32)).to(self.device)
        depth_scale_t = torch.tensor(float(depth_scale), dtype=torch.float32, device=self.device)
        # Depth in metres for PEM point cloud extraction
        depth_m = depth.astype(np.float32) * depth_scale / 1000.0

        # ---- ISM: proposal generation ----
        with torch.autocast("cuda", enabled=self._half_precision):
            proposals = self.segmentor.generate_masks(rgb_np)
        if self._empty_cache:
            torch.cuda.empty_cache()
        detections = self._Detections(proposals)

        if len(detections.masks) == 0:
            return self._empty_result(H, W)

        if self.min_component_area_ratio > 0:
            detections = self._clean_mask_components(detections)

        with torch.autocast("cuda", enabled=self._half_precision):
            query_cls, query_appe = self.descriptor.forward(rgb_np, detections)
        if self._empty_cache:
            torch.cuda.empty_cache()

        idx_selected, pred_idx_objects, semantic_score, best_template = (
            self._compute_semantic_score(query_cls)
        )

        if len(idx_selected) == 0:
            return self._empty_result(H, W)

        detections.filter(idx_selected)
        query_appe = query_appe[idx_selected]

        appe_scores, ref_appe = self._compute_appearance_score(
            best_template, pred_idx_objects, query_appe
        )
        image_uv = self._project_template_to_image(
            best_template, pred_idx_objects, depth_t, cam_K_t, depth_scale_t, detections.masks
        )
        geometric_score, visible_ratio = self._compute_geometric_score(
            image_uv, detections, query_appe, ref_appe
        )
        ism_scores = (semantic_score + appe_scores + geometric_score * visible_ratio) / (2.0 + visible_ratio)

        # Keep all ISM intermediates before confidence filtering (for debugging)
        # masks may be [N,1,H,W] after descriptor forward's in-place unsqueeze
        _masks = detections.masks.squeeze(1) if detections.masks.dim() == 4 else detections.masks
        all_masks_np  = _masks.cpu().numpy().astype(bool)
        all_boxes_np  = detections.boxes.cpu().numpy()
        all_ism_np    = ism_scores.cpu().numpy()
        all_sem_np    = semantic_score.cpu().numpy()
        all_appe_np   = appe_scores.cpu().numpy()
        all_geo_np    = geometric_score.cpu().numpy()
        all_vis_np    = visible_ratio.cpu().numpy()
        all_tmpl_np   = best_template.cpu().numpy()
        pred_obj_np   = pred_idx_objects.cpu().numpy()  # model assignment per filtered proposal

        # Per-model det_score_thresh (degrades to scalar comparison when M=1)
        det_thresh_arr = np.array([self._det_score_thresh_list[m] for m in pred_obj_np])
        keep = all_ism_np > det_thresh_arr
        if not keep.any():
            return self._empty_result(H, W)
        keep_idx = np.where(keep)[0]

        # ---- PEM: pose estimation with per-instance model data ----
        from data_utils import get_point_cloud_from_depth, get_bbox, get_resize_rgb_choose

        whole_pts = get_point_cloud_from_depth(depth_m, K)  # [H, W, 3] metres

        all_rgb_t, all_cloud_t, all_rgb_choose_t, all_scores_raw = [], [], [], []
        all_model_pts_t, all_tem_pts_t, all_tem_feat_t = [], [], []
        valid_keep      = []
        valid_model_ids = []

        for ki in keep_idx:
            model_idx   = int(pred_obj_np[ki])
            model_pts_i = self._pem_model_pts_list[model_idx]  # cached at init, no disk I/O
            radius      = float(np.max(np.linalg.norm(model_pts_i, axis=1)))

            mask_np = all_masks_np[ki]
            mask_with_depth = np.logical_and(mask_np, depth_m > 0)
            if np.sum(mask_with_depth) < 32:
                continue

            bbox = get_bbox(mask_with_depth)
            y1, y2, x1, x2 = bbox
            mask_crop = mask_with_depth[y1:y2, x1:x2]
            choose    = mask_crop.astype(np.float32).flatten().nonzero()[0]

            cloud  = whole_pts[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            center = cloud.mean(axis=0)
            dist   = np.linalg.norm(cloud - center[None, :], axis=1)
            flag   = dist < radius * 1.2
            if flag.sum() < 4:
                continue
            choose = choose[flag]
            cloud  = cloud[flag]

            n_obs   = self.pem_n_sample_observed
            idx_obs = np.random.choice(len(choose), n_obs, replace=(len(choose) <= n_obs))
            choose  = choose[idx_obs]
            cloud   = cloud[idx_obs]

            # RGB crop: match PEM convention (imageio RGB → flip → BGR)
            rgb_crop = rgb_np[y1:y2, x1:x2, :][:, :, ::-1].copy()
            if self.pem_rgb_mask_flag:
                rgb_crop = rgb_crop * (mask_crop[:, :, None] > 0).astype(np.uint8)
            rgb_crop  = cv2.resize(rgb_crop, (self.pem_img_size, self.pem_img_size), interpolation=cv2.INTER_LINEAR)
            rgb_t     = _RGB_TRANSFORM(np.array(rgb_crop))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.pem_img_size)

            tem_pts_i, tem_feat_i = self._pem_templates[model_idx]  # (Tensor[1,n,3], Tensor[1,n,C])

            all_rgb_t.append(torch.FloatTensor(rgb_t))
            all_cloud_t.append(torch.FloatTensor(cloud))
            all_rgb_choose_t.append(torch.LongTensor(rgb_choose))
            all_scores_raw.append(float(all_ism_np[ki]))
            all_model_pts_t.append(torch.FloatTensor(model_pts_i))
            all_tem_pts_t.append(tem_pts_i.squeeze(0))   # [npoint, 3]
            all_tem_feat_t.append(tem_feat_i.squeeze(0)) # [npoint, C]
            valid_keep.append(ki)
            valid_model_ids.append(model_idx)

        if not valid_keep:
            return self._empty_result(H, W)

        ninstance = len(valid_keep)
        # Single batched PEM forward across all instances from all model types.
        # PEM's per-element radius normalisation handles different model sizes within one batch.
        input_data = {
            "pts":        torch.stack(all_cloud_t).to(self.device),
            "rgb":        torch.stack(all_rgb_t).to(self.device),
            "rgb_choose": torch.stack(all_rgb_choose_t).to(self.device),
            "score":      torch.FloatTensor(all_scores_raw).to(self.device),
            "model":      torch.stack(all_model_pts_t).to(self.device),      # [N, n_model, 3] — per-element
            "K":          cam_K_t.unsqueeze(0).expand(ninstance, -1, -1).contiguous(),
            "dense_po":   torch.stack(all_tem_pts_t).to(self.device),        # [N, npoint, 3] — per-element
            "dense_fo":   torch.stack(all_tem_feat_t).to(self.device),       # [N, npoint, C] — per-element
        }

        with torch.no_grad():
            out = self.pem(input_data)
        if self._empty_cache:
            torch.cuda.empty_cache()

        det_scores    = input_data["score"]
        pose_scores_t = out.get("pred_pose_score", det_scores) * det_scores
        pose_scores   = pose_scores_t.detach().cpu().numpy()
        pred_R        = out["pred_R"].detach().cpu().numpy()
        pred_t        = out["pred_t"].detach().cpu().numpy() * 1000.0  # m → mm

        poses = [
            {"R": pred_R[i], "t": pred_t[i], "score": float(pose_scores[i])}
            for i in range(ninstance)
        ]

        vk                  = np.array(valid_keep)
        valid_model_ids_arr = np.array(valid_model_ids, dtype=np.int64)
        final_masks         = all_masks_np[vk]

        # Cross-model + within-model NMS
        if self.mask_overlap_thresh > 0 and len(final_masks) > 1:
            nms_keep            = self._filter_overlapping_masks(final_masks, pose_scores)
            vk                  = vk[nms_keep]
            final_masks         = final_masks[nms_keep]
            pose_scores         = pose_scores[nms_keep]
            pred_R              = pred_R[nms_keep]
            pred_t              = pred_t[nms_keep]
            poses               = [poses[i] for i in nms_keep]
            valid_model_ids_arr = valid_model_ids_arr[nms_keep]

        # Per-model pose_score_thresh
        if any(t > 0 for t in self._pose_score_thresh_list):
            thresh_per_det      = np.array([self._pose_score_thresh_list[m] for m in valid_model_ids_arr])
            ps_keep             = np.where(pose_scores >= thresh_per_det)[0]
            vk                  = vk[ps_keep]
            final_masks         = final_masks[ps_keep]
            pose_scores         = pose_scores[ps_keep]
            pred_R              = pred_R[ps_keep]
            pred_t              = pred_t[ps_keep]
            poses               = [poses[i] for i in ps_keep]
            valid_model_ids_arr = valid_model_ids_arr[ps_keep]

        # Per-model max_detections_per_model cap
        if any(m is not None for m in self._max_det_per_model):
            keep_mask = np.ones(len(valid_model_ids_arr), dtype=bool)
            for midx in range(self.M):
                cap = self._max_det_per_model[midx]
                if cap is None:
                    continue
                positions = np.where(valid_model_ids_arr == midx)[0]
                if len(positions) > cap:
                    drop = positions[np.argsort(pose_scores[positions])[:-cap]]
                    keep_mask[drop] = False
            cap_keep            = np.where(keep_mask)[0]
            vk                  = vk[cap_keep]
            final_masks         = final_masks[cap_keep]
            pose_scores         = pose_scores[cap_keep]
            pred_R              = pred_R[cap_keep]
            pred_t              = pred_t[cap_keep]
            poses               = [poses[i] for i in cap_keep]
            valid_model_ids_arr = valid_model_ids_arr[cap_keep]

        if self._auto_offload:
            self.offload_to_cpu()

        return {
            "poses":             poses,
            "masks":             final_masks,
            "boxes":             all_boxes_np[vk],
            "ism_scores":        all_ism_np[vk],
            "semantic_scores":   all_sem_np[vk],
            "appe_scores":       all_appe_np[vk],
            "geometric_scores":  all_geo_np[vk],
            "visible_ratios":    all_vis_np[vk],
            "best_template_ids": all_tmpl_np[vk],
            "pred_R":            pred_R,
            "pred_t":            pred_t,
            "pose_scores":       pose_scores,
            "model_ids":         [self._model_ids[m] for m in valid_model_ids_arr],
            "model_indices":     valid_model_ids_arr,
        }

    def _empty_result(self, H: int, W: int) -> Dict[str, Any]:
        if self._auto_offload:
            self.offload_to_cpu()
        return {
            "poses":             [],
            "masks":             np.zeros((0, H, W), dtype=bool),
            "boxes":             np.zeros((0, 4), dtype=np.int64),
            "ism_scores":        np.zeros(0),
            "semantic_scores":   np.zeros(0),
            "appe_scores":       np.zeros(0),
            "geometric_scores":  np.zeros(0),
            "visible_ratios":    np.zeros(0),
            "best_template_ids": np.zeros(0, dtype=np.int64),
            "pred_R":            np.zeros((0, 3, 3)),
            "pred_t":            np.zeros((0, 3)),
            "pose_scores":       np.zeros(0),
            "model_ids":         [],
            "model_indices":     np.zeros(0, dtype=np.int64),
        }

    def __call__(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
    ) -> Dict[str, Any]:
        return self.forward(rgb, depth, K, depth_scale)
