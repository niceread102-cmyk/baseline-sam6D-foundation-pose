import os
import logging
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import trimesh
import pyrender

from segment_anything import sam_model_registry
from src.model.utils import BatchedData, Detections
from src.model.loss import PairwiseSimilarity
from src.model.sam import CustomSamAutomaticMaskGenerator
from src.utils.bbox_utils import CropResizePad
from src.utils.trimesh_utils import load_mesh as load_trimesh_mesh
from src.poses.utils import get_obj_poses_from_template_level


# EGL is required for headless pyrender rendering
os.environ["PYOPENGL_PLATFORM"] = "egl"

DESCRIPTOR_SIZES = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# Standard BOP camera intrinsics at 640x480, a reasonable default for most objects
DEFAULT_INTRINSIC = np.array([[604.7352,    0.,      430.31723],
                              [  0.,      604.959,   245.13461],
                              [  0.,        0.,        1.     ]])
DEFAULT_IMG_SIZE = (480, 640)  # (H, W)


class CNOS:
    def __init__(
        self,
        cad_paths: List[Union[str, Path]],
        device: str = "cuda",

        # Segmentor
        segmentor: str = "sam",
        sam_checkpoint: Optional[Union[str, Path]] = None,
        sam_model_type: str = "vit_h",
        fastsam_checkpoint: Optional[Union[str, Path]] = None,
        fastsam_iou: float = 0.9,
        fastsam_conf: float = 0.05,
        fastsam_max_det: int = 200,
        sam_min_mask_region_area: int = 0,      # SAM post-proc: remove islands < N px (0=off)

        # DINOv2
        dinov2_model_name: str = "dinov2_vitl14",
        dinov2_hub_dir: Optional[Union[str, Path]] = None,

        # Rendering / onboarding
        rendering_type: str = "pyrender",   # "pyrender" or "blenderproc"
        template_dir: Optional[Union[str, Path]] = None,
        level_templates: int = 0,           # 0 → 42 poses, 1 → 162, 2 → 642
        reset_descriptors: bool = False,
        mesh_color: Optional[tuple] = None, # RGB tuple (0-255) applied to colorless meshes

        # Image processing
        segmentor_width: int = 640,
        image_size: int = 224,
        chunk_size: int = 16,

        # Matching
        aggregation: str = "avg_5",         # "mean" | "median" | "max" | "avg_5"
        confidence_threshold: float = 0.15,
        max_instances: int = 100,

        # Post-processing
        nms_threshold: float = 0.25,
        min_box_size: float = 0.05,
        min_mask_size: float = 3e-4,
        min_component_area_ratio: float = 0.005,  # drop blobs < ratio * largest blob (0=off)
        max_bbox_area: float = 1.0,               # drop detections whose bbox covers > fraction of image (1=off)
    ):
        self.device = torch.device(device)
        self.cad_paths = [Path(p) for p in cad_paths]
        self.rendering_type = rendering_type
        self.template_dir = Path(template_dir) if template_dir else Path("./cnos_templates")
        self.mesh_color = mesh_color
        self.level_templates = level_templates
        self.chunk_size = chunk_size
        self.aggregation = aggregation
        self.confidence_threshold = confidence_threshold
        self.max_instances = max_instances
        self.nms_threshold = nms_threshold
        # SimpleNamespace so Detections.remove_very_small_detections can consume it unchanged
        self.filter_config = SimpleNamespace(min_box_size=min_box_size, min_mask_size=min_mask_size)
        self.min_component_area_ratio = min_component_area_ratio
        self.max_bbox_area = max_bbox_area

        self.rgb_normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.proposal_processor = CropResizePad(image_size)
        self.similarity = PairwiseSimilarity()

        # --- Load DINOv2 ---
        logging.info(f"Loading DINOv2 ({dinov2_model_name}) ...")
        if dinov2_hub_dir is not None:
            torch.hub.set_dir(str(dinov2_hub_dir))
        self.dinov2 = torch.hub.load("facebookresearch/dinov2", dinov2_model_name)
        self.dinov2.eval().to(self.device)

        logging.info("DINOv2 loaded.")

        # --- Load segmentor ---
        logging.info(f"Loading segmentor ({segmentor}) ...")
        if segmentor == "sam":
            sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint))
            sam.to(device=self.device)
            self.segmentor = CustomSamAutomaticMaskGenerator(
                sam=sam,
                segmentor_width_size=segmentor_width,
                min_mask_region_area=sam_min_mask_region_area,
            )
        elif segmentor == "fastsam":
            from src.model.fast_sam import FastSAM
            fastsam_config = SimpleNamespace(
                iou_threshold=fastsam_iou,
                conf_threshold=fastsam_conf,
                max_det=fastsam_max_det,
            )
            self.segmentor = FastSAM(
                checkpoint_path=str(fastsam_checkpoint),
                config=fastsam_config,  # type: ignore[arg-type]
                segmentor_width_size=segmentor_width,
                device=device,
            )
            self.segmentor.model.setup_model(device=self.device, verbose=False)
        else:
            raise ValueError(f"Unknown segmentor '{segmentor}'. Choose 'sam' or 'fastsam'.")
        logging.info("Segmentor loaded.")

        # --- Onboarding: render templates and compute reference descriptors ---
        self.template_dir.mkdir(parents=True, exist_ok=True)
        descriptor_cache = self.template_dir / "descriptors.pth"

        n_templates = len(np.array(get_obj_poses_from_template_level(self.level_templates, "all")))
        cached_ok = False
        if descriptor_cache.exists() and not reset_descriptors:
            cached = torch.load(descriptor_cache, map_location=self.device)
            if cached.shape[1] == n_templates:
                logging.info(f"Loading cached descriptors from {descriptor_cache}")
                self.ref_descriptors = cached
                cached_ok = True
            else:
                logging.info(
                    f"Descriptor cache has {cached.shape[1]} templates but "
                    f"{n_templates} expected (level_templates={self.level_templates}) — regenerating."
                )
        if not cached_ok:
            all_descriptors = []
            for i, cad_path in enumerate(tqdm(self.cad_paths, desc="Onboarding")):
                obj_dir = self.template_dir / f"obj_{i:06d}"
                obj_dir.mkdir(parents=True, exist_ok=True)
                self._render_templates(cad_path, obj_dir)
                all_descriptors.append(self._compute_descriptors(obj_dir))
            # [N_objects x N_templates x descriptor_dim]
            self.ref_descriptors = torch.stack(all_descriptors)
            torch.save(self.ref_descriptors, descriptor_cache)
            logging.info(f"Descriptors saved to {descriptor_cache}")

        logging.info(f"Reference descriptors: {self.ref_descriptors.shape}")

    # ------------------------------------------------------------------
    # Onboarding helpers
    # ------------------------------------------------------------------

    def _render_templates(self, cad_path: Path, output_dir: Path):
        """Render RGBA template images from icosphere viewpoints. Skips if already done."""
        obj_poses = np.array(get_obj_poses_from_template_level(self.level_templates, "all"))
        existing = list(output_dir.glob("*.png"))
        if len(existing) == len(obj_poses):
            return
        # Level changed or render incomplete — wipe stale PNGs before re-rendering
        for p in existing:
            p.unlink()

        obj_poses = obj_poses.copy()
        obj_poses[:, :3, 3] *= 0.4  # scale to desired camera-to-object distance

        # load_trimesh_mesh re-centers the mesh at its bounding box center, so any
        # CAD file with an arbitrary origin is handled correctly without needing an
        # explicit re_center transform (which would have mm/meter unit issues).
        mesh = load_trimesh_mesh(str(cad_path))
        if np.linalg.norm(mesh.extents * 2) > 100:  # diameter > 100 → assume millimeters
            mesh.apply_scale(0.001)

        if self.mesh_color is not None:
            rgba = np.array(list(self.mesh_color) + [255], dtype=np.uint8)
            mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))

        if self.rendering_type == "blenderproc":
            self._render_blenderproc(cad_path, output_dir, obj_poses)
            return

        # Predefined pose translations are in mm (1000× unit sphere); convert to meters
        # to match the mesh which is now in meters after apply_scale above.
        obj_poses[:, :3, 3] /= 1000.0
        # smooth=False prevents normal interpolation across flat faces (important for STL)
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

        # Negate Y and Z to convert from OpenCV to OpenGL camera convention
        cam_pose = np.eye(4)
        cam_pose[1, 1] = -1
        cam_pose[2, 2] = -1

        fx, fy = DEFAULT_INTRINSIC[0, 0], DEFAULT_INTRINSIC[1, 1]
        cx, cy = DEFAULT_INTRINSIC[0, 2], DEFAULT_INTRINSIC[1, 2]
        H, W = DEFAULT_IMG_SIZE

        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
            ambient_light=np.array([0.01, 0.01, 0.01, 1.0]),
        )
        scene.add(
            pyrender.SpotLight(
                color=np.ones(3),
                intensity=0.6,
                innerConeAngle=np.pi / 16.0,
                outerConeAngle=np.pi / 6.0,
            ),
            pose=cam_pose,
        )
        scene.add(
            pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=100000),
            pose=cam_pose,
        )
        renderer = pyrender.OffscreenRenderer(W, H)
        cad_node = scene.add(pyrender_mesh, pose=np.eye(4), name="cad")

        for idx, pose in enumerate(obj_poses):
            scene.set_pose(cad_node, pose)
            rgb, _ = renderer.render(scene, pyrender.constants.RenderFlags.RGBA)
            Image.fromarray(np.uint8(rgb)).save(output_dir / f"{idx:06d}.png")

        renderer.delete()

    def _render_blenderproc(self, cad_path: Path, output_dir: Path, obj_poses: np.ndarray):
        """Render templates via BlenderProc subprocess. Requires blenderproc on PATH."""
        poses_path = output_dir / "template_poses.npy"
        np.save(str(poses_path), obj_poses)
        color_arg = ""
        if self.mesh_color is not None:
            r, g, b = self.mesh_color
            color_arg = f"--mesh-color {r} {g} {b}"
        # Unset MPLBACKEND: Jupyter sets it to matplotlib_inline which is invalid inside Blender's Python.
        ret = os.system(f"env -u MPLBACKEND blenderproc run src/poses/blenderproc.py {cad_path} {poses_path} {output_dir} {color_arg}")
        if ret != 0 or not any(output_dir.glob("*.png")):
            raise RuntimeError(
                f"BlenderProc rendering failed (exit code {ret}). "
                f"Run manually to see the error:\n"
                f"  blenderproc run src/poses/blenderproc.py {cad_path} {poses_path} {output_dir}"
            )

    def _compute_descriptors(self, template_dir: Path) -> torch.Tensor:
        """Load rendered PNGs, crop to object bbox, and embed with DINOv2. Returns [N x D]."""
        template_paths = sorted(template_dir.glob("*.png"))
        normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        templates, boxes = [], []
        for path in template_paths:
            img = Image.open(path)
            boxes.append(img.getbbox())           # tight bbox from the alpha channel
            templates.append(T.ToTensor()(img.convert("RGB")))

        templates = torch.stack(templates)              # [N x 3 x H x W]
        boxes = torch.tensor(np.array(boxes))
        # Crop and resize BEFORE normalizing — matches BOPTemplate's processing order
        templates = self.proposal_processor(templates, boxes).to(self.device)
        templates = normalize(templates)
        return self._dinov2_forward_by_chunk(templates) # [N x D]

    # ------------------------------------------------------------------
    # Mask cleaning
    # ------------------------------------------------------------------

    def _clean_mask_components(self, detections: Detections) -> Detections:
        """Drop connected components whose area < min_component_area_ratio * largest component.

        Recomputes bounding boxes from the cleaned masks so stray specs no longer
        inflate the detection box.
        """
        cleaned_masks, cleaned_boxes = [], []
        for mask in detections.masks:
            mask_np = mask.cpu().numpy().astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            if n_labels <= 1:
                # No foreground at all — keep as-is
                cleaned_masks.append(mask)
                cleaned_boxes.append(None)
                continue

            areas = stats[1:, cv2.CC_STAT_AREA]  # skip background (label 0)
            max_area = areas.max()
            keep_labels = np.where(areas >= self.min_component_area_ratio * max_area)[0] + 1

            cleaned = np.isin(labels, keep_labels).astype(np.uint8)
            ys, xs = np.nonzero(cleaned)
            if len(xs) == 0:
                cleaned_masks.append(mask)
                cleaned_boxes.append(None)
                continue
            box = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()], dtype=torch.long)
            cleaned_masks.append(torch.from_numpy(cleaned).bool().to(mask.device))
            cleaned_boxes.append(box.to(mask.device))

        detections.masks = torch.stack(cleaned_masks)
        detections.boxes = torch.stack([
            b if b is not None else detections.boxes[i]
            for i, b in enumerate(cleaned_boxes)
        ])
        return detections

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _dinov2_forward_by_chunk(self, images: torch.Tensor) -> torch.Tensor:
        """Run DINOv2 in chunks of chunk_size to avoid OOM. Returns [N x D]."""
        features = BatchedData(batch_size=None)
        for start in range(0, len(images), self.chunk_size):
            features.cat(self.dinov2(images[start : start + self.chunk_size]))
        return features.data

    @torch.no_grad()
    def _extract_query_descriptors(self, image_np: np.ndarray, detections: Detections) -> torch.Tensor:
        """Mask background, crop to bbox, and embed each proposal. Returns [N_proposals x D]."""
        N = len(detections.masks)
        rgb = self.rgb_normalize(image_np).to(self.device).float()  # [3 x H x W]
        rgbs = rgb.unsqueeze(0).repeat(N, 1, 1, 1)                  # [N x 3 x H x W]
        masked_rgbs = rgbs * detections.masks.unsqueeze(1)           # zero out background
        cropped = self.proposal_processor(masked_rgbs, detections.boxes)
        return self._dinov2_forward_by_chunk(cropped)

    def _aggregate_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Reduce [N x M x T] to [N x M] by aggregating the T template similarities."""
        if self.aggregation == "mean":
            return scores.mean(dim=-1)
        elif self.aggregation == "median":
            return scores.median(dim=-1)[0]
        elif self.aggregation == "max":
            return scores.max(dim=-1)[0]
        elif self.aggregation == "avg_5":
            top5 = torch.topk(scores, k=min(5, scores.shape[-1]), dim=-1)[0]
            return top5.mean(dim=-1)
        raise ValueError(f"Unknown aggregation '{self.aggregation}'. Choose: mean, median, max, avg_5.")

    # ------------------------------------------------------------------
    # Main forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Run the full CNOS pipeline on a single image.

        Args:
            image: RGB image as PIL Image or uint8 numpy array [H x W x 3].

        Returns a dict with:
            masks               Tensor [N, H, W]    binary proposal masks
            boxes               Tensor [N, 4]       xyxy bounding boxes
            object_ids          Tensor [N]          0-indexed object class IDs
            scores              Tensor [N]          per-detection confidence
            scores_per_object   Tensor [N, M]       confidence against each of M objects
            scores_per_template Tensor [N, M, T]    raw cosine similarity per template
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # --- Proposal stage ---
        proposals = self.segmentor.generate_masks(image_np)
        detections = Detections(proposals)
        if self.min_component_area_ratio > 0:
            detections = self._clean_mask_components(detections)
        detections.remove_very_small_detections(self.filter_config)

        if len(detections.masks) == 0:
            N_objects, N_templates = self.ref_descriptors.shape[:2]
            H, W = image_np.shape[:2]
            return {
                "masks":               torch.zeros((0, H, W)),
                "boxes":               torch.zeros((0, 4)),
                "object_ids":          torch.zeros(0, dtype=torch.long),
                "scores":              torch.zeros(0),
                "scores_per_object":   torch.zeros((0, N_objects)),
                "scores_per_template": torch.zeros((0, N_objects, N_templates)),
            }

        # --- Descriptor stage ---
        query_descriptors = self._extract_query_descriptors(image_np, detections)

        # --- Matching stage ---
        # scores_per_template: [N_proposals x N_objects x N_templates]
        scores_per_template = self.similarity(query_descriptors, self.ref_descriptors)
        # scores_per_object: [N_proposals x N_objects]
        scores_per_object = self._aggregate_scores(scores_per_template)

        # --- Object assignment ---
        scores, object_ids = torch.max(scores_per_object, dim=-1)
        masks = detections.masks
        boxes = detections.boxes

        # Bundle everything into one Detections so all filters apply uniformly
        detections = Detections({"masks": masks, "boxes": boxes})
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", object_ids)
        detections.add_attribute("scores_per_object", scores_per_object)
        detections.add_attribute("scores_per_template", scores_per_template)

        # --- Filtering ---
        detections.filter(detections.scores > self.confidence_threshold)

        if self.max_bbox_area < 1.0:
            H, W = image_np.shape[:2]
            image_area = H * W
            boxes = detections.boxes.float()
            bbox_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            detections.filter(bbox_areas / image_area <= self.max_bbox_area)

        if len(detections.scores) > self.max_instances:
            _, idx = torch.topk(detections.scores, k=self.max_instances)
            detections.filter(idx)

        if len(detections.scores) > 0:
            detections.apply_nms_per_object_id(self.nms_threshold)

        return {
            "masks":               detections.masks,
            "boxes":               detections.boxes,
            "object_ids":          detections.object_ids,
            "scores":              detections.scores,
            "scores_per_object":   detections.scores_per_object,
            "scores_per_template": detections.scores_per_template,
        }

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Any]:
        return self.forward(image)
