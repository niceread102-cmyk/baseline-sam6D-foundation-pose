"""SAM-6D wrapper: two-stage 6DoF pose estimation (ISM + PEM) for a single object."""
import os
import sys
import logging
import numpy as np
import torch
import torchvision.transforms as T
import cv2
import trimesh
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

_SAM6D_ROOT = Path(__file__).resolve().parent.parent / "SAM-6D-main" / "SAM-6D"
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


class SAM6D:
    def __init__(
        self,
        cad_path: Union[str, Path],
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
        template_dir: Optional[Union[str, Path]] = None,
        reset_templates: bool = False,
        mesh_color: Optional[tuple] = None,   # RGB tuple (0-255) applied to colorless meshes
        flat_shading: bool = False,           # disable normal interpolation (useful for STL/CAD)
        roughness: Optional[float] = None,    # 0.0 = mirror/glossy, 1.0 = fully matte
        metallic: Optional[float] = None,     # 0.0 = plastic/dielectric, 1.0 = metallic
        light_energy: float = 1000,           # point light brightness in watts

        # Matching / filtering
        det_score_thresh: float = 0.2,
        confidence_thresh: float = 0.2,
        visible_thred: float = 0.5,
        n_template_view: int = 42,
        min_component_area_ratio: float = 0.005,  # drop mask blobs < ratio * largest blob (0=off)
        mask_overlap_thresh: float = 0.9,          # drop lower-scoring mask if overlap > thresh (0=off)
        pose_score_thresh: float = 0.0,            # drop detections with pose score below this (0=off)

        # PEM dataset config
        pem_img_size: int = 224,
        pem_n_sample_template: int = 5000,
        pem_n_sample_observed: int = 2048,
        pem_n_sample_model: int = 1024,
        pem_rgb_mask_flag: bool = True,
    ):
        _ensure_sam6d_paths()

        self.device = torch.device(device)
        self.cad_path = Path(cad_path)
        self.det_score_thresh = det_score_thresh
        self.confidence_thresh = confidence_thresh
        self.visible_thred = visible_thred
        self.n_template_view = n_template_view
        self.min_component_area_ratio = min_component_area_ratio
        self.mask_overlap_thresh = mask_overlap_thresh
        self.pose_score_thresh = pose_score_thresh
        self.pem_img_size = pem_img_size
        self.pem_n_sample_template = pem_n_sample_template
        self.pem_n_sample_observed = pem_n_sample_observed
        self.pem_n_sample_model = pem_n_sample_model
        self.pem_rgb_mask_flag = pem_rgb_mask_flag

        self.template_dir = Path(template_dir) if template_dir else Path("./sam6d_templates") / self.cad_path.stem
        self.template_dir.mkdir(parents=True, exist_ok=True)
        # Render script writes into a "templates/" subdirectory (SAM-6D convention)
        self._tmpl_dir = self.template_dir / "templates"

        _ism_ckpt = _ISM_ROOT / "checkpoints"
        _pem_ckpt = _PEM_ROOT / "checkpoints"

        # --- Download weights ---
        self._maybe_download_sam(sam_checkpoint, sam_model_type, _ism_ckpt)
        self._maybe_download_dinov2(dinov2_checkpoint_dir, _ism_ckpt)
        self._maybe_download_pem(pem_checkpoint, _pem_ckpt)

        _sam_path = Path(sam_checkpoint) if sam_checkpoint else _ism_ckpt / "segment-anything" / SAM_WEIGHTS[sam_model_type][0]
        _dinov2_dir = Path(dinov2_checkpoint_dir) if dinov2_checkpoint_dir else _ism_ckpt / "dinov2"
        _pem_path = Path(pem_checkpoint) if pem_checkpoint else _pem_ckpt / "sam-6d-pem-base.pth"

        # --- Render templates ---
        self._maybe_render_templates(reset_templates, mesh_color, flat_shading, roughness, metallic, light_energy)

        # --- Import SAM-6D ISM modules ---
        from segment_anything import sam_model_registry
        from model.sam import CustomSamAutomaticMaskGenerator
        from model.dinov2 import CustomDINOv2
        from model.utils import Detections
        from model.loss import PairwiseSimilarity, MaskedPatch_MatrixSimilarity
        from utils.bbox_utils import CropResizePad, compute_iou
        from utils.trimesh_utils import depth_image_to_pointcloud_translate_torch
        from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2

        self._Detections = Detections
        self._MaskedPatch_MatrixSimilarity = MaskedPatch_MatrixSimilarity
        self._compute_iou = compute_iou
        self._depth_image_to_pointcloud_translate_torch = depth_image_to_pointcloud_translate_torch

        # --- Load SAM ---
        logging.info(f"Loading SAM ({sam_model_type}) ...")
        sam = sam_model_registry[sam_model_type](checkpoint=str(_sam_path))
        sam.to(device=self.device)
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
        logging.info("DINOv2 loaded.")

        self._similarity = PairwiseSimilarity(metric="cosine", chunk_size=16)
        self._crop_resize = CropResizePad(224)

        # --- Pre-compute ISM reference data ---
        self.ref_data = {}
        self._precompute_ism_templates(
            get_obj_poses_from_template_level, load_index_level_in_level2
        )

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

        # --- Pre-compute PEM template features ---
        logging.info("Extracting PEM template features ...")
        self._extract_pem_templates()
        logging.info("SAM6D init complete.")

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

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------

    def _maybe_render_templates(self, reset_templates: bool, mesh_color: Optional[tuple],
                                flat_shading: bool, roughness: Optional[float], metallic: Optional[float],
                                light_energy: float):
        # The render script writes into {template_dir}/templates/
        n_rgb  = len(list(self._tmpl_dir.glob("rgb_*.png")))
        n_xyz  = len(list(self._tmpl_dir.glob("xyz_*.npy")))
        n_mask = len(list(self._tmpl_dir.glob("mask_*.png")))
        n_expected = _N_TEMPLATES
        if not reset_templates and n_rgb == n_expected and n_xyz == n_expected and n_mask == n_expected:
            logging.info(f"Templates already at {self._tmpl_dir}, skipping render.")
            return

        render_script = str(_SAM6D_ROOT / "Render" / "render_custom_templates.py")
        # Pass template_dir (not _tmpl_dir) — the script appends "templates/" itself
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
            f"--output_dir {self.template_dir} --cad_path {self.cad_path}{extra}"
        )
        logging.info(f"Rendering templates: {cmd}")
        ret = os.system(cmd)
        if ret != 0 or len(list(self._tmpl_dir.glob("rgb_*.png"))) == 0:
            raise RuntimeError(
                f"BlenderProc rendering failed (exit {ret}). "
                f"Run manually: blenderproc run {render_script} "
                f"--output_dir {self.template_dir} --cad_path {self.cad_path}{extra}"
            )

    # ------------------------------------------------------------------
    # ISM onboarding
    # ------------------------------------------------------------------

    def _precompute_ism_templates(self, get_obj_poses_from_template_level, load_index_level_in_level2):
        """Load rendered templates and compute ISM reference descriptors."""
        template_paths = sorted(self._tmpl_dir.glob("rgb_*.png"))
        n_templates = len(template_paths)

        desc_cache = self._tmpl_dir / "ism_descriptors.pth"
        appe_cache = self._tmpl_dir / "ism_appe_descriptors.pth"

        boxes, templates, masks_list = [], [], []
        for path in template_paths:
            img = Image.open(path)
            mask_img = Image.open(self._tmpl_dir / path.name.replace("rgb_", "mask_"))
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
            logging.info("Computing ISM semantic descriptors ...")
            descriptors = self.descriptor.compute_features(templates_proc, token_name="x_norm_clstoken")
            torch.save(descriptors, desc_cache)
        # Shape: [1, N_templates, D]  (1 object)
        self.ref_data["descriptors"] = descriptors.unsqueeze(0)

        if appe_cache.exists():
            logging.info(f"Loading cached ISM appearance descriptors from {appe_cache}")
            appe_descriptors = torch.load(appe_cache, map_location=self.device)
        else:
            logging.info("Computing ISM appearance descriptors ...")
            appe_descriptors = self.descriptor.compute_masked_patch_feature(
                templates_proc, masks_proc[:, 0, :, :]
            )
            torch.save(appe_descriptors, appe_cache)
        # Shape: [1, N_templates, N_patches, D]  (1 object)
        self.ref_data["appe_descriptors"] = appe_descriptors.unsqueeze(0)

        # Template camera poses (level-2 set → index down to level-0 subset)
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
        index_templates = load_index_level_in_level2(0, "all")
        self.ref_data["poses"] = poses[index_templates, :, :]   # [N_level0, 4, 4]

        # Object point cloud sampled from CAD mesh, centered to match renderer
        mesh = trimesh.load_mesh(str(self.cad_path))
        model_pts = mesh.sample(2048).astype(np.float32) / 1000.0
        mesh_centroid = np.array(mesh.bounding_box.centroid, dtype=np.float32) / 1000.0
        model_pts = model_pts - mesh_centroid
        self.ref_data["pointcloud"] = torch.tensor(model_pts).unsqueeze(0).to(self.device)  # [1, N, 3]

        logging.info(
            f"ISM ref data: descriptors={self.ref_data['descriptors'].shape}, "
            f"appe={self.ref_data['appe_descriptors'].shape}, "
            f"poses={self.ref_data['poses'].shape}"
        )

    # ------------------------------------------------------------------
    # PEM onboarding
    # ------------------------------------------------------------------

    def _get_pem_template(self, idx: int):
        """Load one rendered view for PEM (rgb tensor, pixel indices, xyz points)."""
        rgb_np = np.array(Image.open(self._tmpl_dir / f"rgb_{idx}.png").convert("RGB"))
        xyz = np.load(str(self._tmpl_dir / f"xyz_{idx}.npy")).astype(np.float32) / 1000.0
        mask = (np.array(Image.open(self._tmpl_dir / f"mask_{idx}.png").convert("L")) == 255)

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

    def _extract_pem_templates(self):
        all_tem, all_tem_pts, all_tem_choose = [], [], []
        total_n = _N_TEMPLATES
        for v in range(self.n_template_view):
            i = int(total_n / self.n_template_view * v)
            tem, tem_choose, tem_pts = self._get_pem_template(i)
            all_tem.append(tem.float().unsqueeze(0).to(self.device))
            all_tem_choose.append(torch.LongTensor(tem_choose).unsqueeze(0).to(self.device))
            all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).to(self.device))

        with torch.no_grad():
            self.all_tem_pts, self.all_tem_feat = self.pem.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose
            )
        logging.info(
            f"PEM template features: pts={self.all_tem_pts.shape}, feat={self.all_tem_feat.shape}"
        )

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
        # scores: [N_proposals, N_objects, N_templates]
        scores = self._similarity(proposal_descriptors, self.ref_data["descriptors"])
        top5 = torch.topk(scores, k=min(5, scores.shape[-1]), dim=-1)[0]
        score_per_obj = top5.mean(dim=-1)                         # [N_proposals, N_objects]

        score_per_proposal, assigned_obj = torch.max(score_per_obj, dim=-1)  # [N_proposals]
        idx_selected = torch.arange(len(score_per_proposal), device=score_per_proposal.device)[
            score_per_proposal > self.confidence_thresh
        ]
        pred_idx_objects = assigned_obj[idx_selected]
        semantic_score = score_per_proposal[idx_selected]

        filtered_scores = scores[idx_selected]                    # [N_keep, N_objects, N_templates]
        _, best_tmpl_per_obj = torch.max(filtered_scores, dim=-1) # [N_keep, N_objects]
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
        proposals = self.segmentor.generate_masks(rgb_np)
        detections = self._Detections(proposals)

        if len(detections.masks) == 0:
            return self._empty_result(H, W)

        if self.min_component_area_ratio > 0:
            detections = self._clean_mask_components(detections)

        query_cls, query_appe = self.descriptor.forward(rgb_np, detections)

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
        all_masks_np = _masks.cpu().numpy().astype(bool)
        all_boxes_np = detections.boxes.cpu().numpy()
        all_ism_np = ism_scores.cpu().numpy()
        all_sem_np = semantic_score.cpu().numpy()
        all_appe_np = appe_scores.cpu().numpy()
        all_geo_np = geometric_score.cpu().numpy()
        all_vis_np = visible_ratio.cpu().numpy()
        all_tmpl_np = best_template.cpu().numpy()

        keep = ism_scores > self.det_score_thresh
        if not keep.any():
            return self._empty_result(H, W)

        keep_idx = torch.where(keep)[0].cpu().numpy()

        # ---- PEM: pose estimation ----
        from data_utils import get_point_cloud_from_depth, get_bbox, get_resize_rgb_choose

        whole_pts = get_point_cloud_from_depth(depth_m, K)        # [H, W, 3] metres
        mesh = trimesh.load_mesh(str(self.cad_path))
        model_pts = mesh.sample(self.pem_n_sample_model).astype(np.float32) / 1000.0
        radius = float(np.max(np.linalg.norm(model_pts, axis=1)))

        all_rgb_t, all_cloud_t, all_rgb_choose_t, all_scores_raw = [], [], [], []
        valid_keep = []  # subset of keep_idx that survived PEM preparation

        for ki in keep_idx:
            mask_np = all_masks_np[ki]
            mask_with_depth = np.logical_and(mask_np, depth_m > 0)
            if np.sum(mask_with_depth) < 32:
                continue

            bbox = get_bbox(mask_with_depth)  # [rmin, rmax, cmin, cmax]
            y1, y2, x1, x2 = bbox
            mask_crop = mask_with_depth[y1:y2, x1:x2]
            choose = mask_crop.astype(np.float32).flatten().nonzero()[0]

            cloud = whole_pts[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
            center = cloud.mean(axis=0)
            dist = np.linalg.norm(cloud - center[None, :], axis=1)
            flag = dist < radius * 1.2
            if flag.sum() < 4:
                continue
            choose = choose[flag]
            cloud = cloud[flag]

            n_obs = self.pem_n_sample_observed
            idx_obs = np.random.choice(len(choose), n_obs, replace=(len(choose) <= n_obs))
            choose = choose[idx_obs]
            cloud = cloud[idx_obs]

            # RGB crop: match PEM convention (imageio RGB → flip → BGR)
            rgb_crop = rgb_np[y1:y2, x1:x2, :][:, :, ::-1].copy()
            if self.pem_rgb_mask_flag:
                rgb_crop = rgb_crop * (mask_crop[:, :, None] > 0).astype(np.uint8)
            rgb_crop = cv2.resize(rgb_crop, (self.pem_img_size, self.pem_img_size), interpolation=cv2.INTER_LINEAR)
            rgb_t = _RGB_TRANSFORM(np.array(rgb_crop))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.pem_img_size)

            all_rgb_t.append(torch.FloatTensor(rgb_t))
            all_cloud_t.append(torch.FloatTensor(cloud))
            all_rgb_choose_t.append(torch.LongTensor(rgb_choose))
            all_scores_raw.append(float(all_ism_np[ki]))
            valid_keep.append(ki)

        if not valid_keep:
            return self._empty_result(H, W)

        ninstance = len(valid_keep)
        input_data = {
            "pts":       torch.stack(all_cloud_t).to(self.device),
            "rgb":       torch.stack(all_rgb_t).to(self.device),
            "rgb_choose": torch.stack(all_rgb_choose_t).to(self.device),
            "score":     torch.FloatTensor(all_scores_raw).to(self.device),
            "model":     torch.FloatTensor(model_pts).unsqueeze(0).expand(ninstance, -1, -1).contiguous().to(self.device),
            "K":         torch.FloatTensor(K.astype(np.float32)).unsqueeze(0).expand(ninstance, -1, -1).contiguous().to(self.device),
            "dense_po":  self.all_tem_pts.expand(ninstance, -1, -1).contiguous(),
            "dense_fo":  self.all_tem_feat.expand(ninstance, -1, -1).contiguous(),
        }

        with torch.no_grad():
            out = self.pem(input_data)

        det_scores = input_data["score"]
        pose_scores_t = out.get("pred_pose_score", det_scores) * det_scores
        pose_scores = pose_scores_t.detach().cpu().numpy()
        pred_R = out["pred_R"].detach().cpu().numpy()
        pred_t = out["pred_t"].detach().cpu().numpy() * 1000.0  # m → mm

        poses = [
            {"R": pred_R[i], "t": pred_t[i], "score": float(pose_scores[i])}
            for i in range(ninstance)
        ]

        vk = np.array(valid_keep)
        final_masks = all_masks_np[vk]

        if self.mask_overlap_thresh > 0 and len(final_masks) > 1:
            keep_idx = self._filter_overlapping_masks(final_masks, pose_scores)
            vk          = vk[keep_idx]
            final_masks = final_masks[keep_idx]
            pose_scores = pose_scores[keep_idx]
            pred_R      = pred_R[keep_idx]
            pred_t      = pred_t[keep_idx]
            poses       = [poses[i] for i in keep_idx]

        if self.pose_score_thresh > 0:
            keep_idx = np.where(pose_scores >= self.pose_score_thresh)[0]
            vk          = vk[keep_idx]
            final_masks = final_masks[keep_idx]
            pose_scores = pose_scores[keep_idx]
            pred_R      = pred_R[keep_idx]
            pred_t      = pred_t[keep_idx]
            poses       = [poses[i] for i in keep_idx]

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
        }

    def _empty_result(self, H: int, W: int) -> Dict[str, Any]:
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
        }

    def __call__(
        self,
        rgb: Union[np.ndarray, Image.Image],
        depth: np.ndarray,
        K: np.ndarray,
        depth_scale: float = 1.0,
    ) -> Dict[str, Any]:
        return self.forward(rgb, depth, K, depth_scale)
