"""Grip planning from CAD models.

SimpleGripPlanner uses PCA on mesh vertices to find principal axes, then
checks which axes produce a grip width within [min_grip_m, max_grip_m].
All dimensions are in metres, consistent with FoundationPose's effective_scale.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GripCandidate:
    """A single viable grip configuration in the object's local frame.

    Attributes
    ----------
    position_m:
        Gripper midpoint in object frame (metres).  ``SimpleGripPlanner`` always
        sets this to ``[0, 0, 0]`` because the mesh is centred before PCA, so
        the centroid is the origin.  The base-frame pick position is therefore
        just the object's detected position: ``pick_xyz = R_obj @ [0,0,0] + p_obj``.

        **Limitation**: centroid-as-pick-point is only reliable for roughly
        symmetric, convex objects.  For elongated, multi-lobe, or heavily
        asymmetric shapes a centroid grip may be mechanically infeasible even
        though IK succeeds.
    approach_axis:
        Unit vector in object frame along which the gripper fingers close.
        Transformed to base frame before planning: ``approach_base = R_obj @ approach_axis``.
    width_m:
        Required gripper opening width in metres.
    """

    position_m:    np.ndarray
    approach_axis: np.ndarray
    width_m:       float


class GripPlanner(ABC):
    @abstractmethod
    def compute_grips(self, model_id: str) -> list[GripCandidate]:
        """Return grip candidates for the given model ID."""


class SimpleGripPlanner(GripPlanner):
    """PCA-based grip planner.

    For each CAD model:
    1. Applies effective_scale (mesh_scale * physical_scale from FoundationPose) to
       convert mesh vertices to metres.
    2. Finds the 3 principal axes via SVD on the centred vertex cloud.
    3. For each axis, the grip width is the point-cloud extent projected onto that axis.
    4. Axes with width in [min_grip_m, max_grip_m] become GripCandidates.
    """

    def __init__(
        self,
        cad_models: Any,          # list of DictConfig or objects with .path and .id
        effective_scale: float,   # mesh_scale * physical_scale from FoundationPose cfg
        min_grip_m: float,
        max_grip_m: float,
    ):
        self._scale   = float(effective_scale)
        self._min     = float(min_grip_m)
        self._max     = float(max_grip_m)
        self._grips: dict[str, list[GripCandidate]] = {}

        for model in cad_models:
            self._grips[model.id] = self._analyse(model.path)

    def _analyse(self, mesh_path: str | Path) -> list[GripCandidate]:
        """Load mesh, find principal axes via SVD, and return valid grip candidates.

        Steps
        -----
        1. Load mesh with trimesh; raise ``ValueError`` if the mesh is empty.
        2. Scale vertices by ``effective_scale`` → metres.
        3. Centre the cloud (subtract centroid) so PCA axes pass through the origin.
        4. Compute SVD: rows of ``Vt`` are principal axes in descending variance.
        5. For each axis, grip width = ``max − min`` projection along that axis.
        6. Axes with width in ``[min_grip_m, max_grip_m]`` become ``GripCandidate``s.
        7. Return sorted by ascending width (tightest grip first).
        """
        import trimesh

        mesh = trimesh.load_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            raise ValueError(
                f"Mesh at '{mesh_path}' has no vertices — check the file path and format."
            )
        verts    = np.asarray(mesh.vertices, dtype=float) * self._scale  # → metres
        centroid = verts.mean(axis=0)
        centred  = verts - centroid

        # PCA via SVD (rows of Vt are principal axes, descending variance)
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)

        candidates: list[GripCandidate] = []
        for axis in Vt[:3]:
            axis = axis / np.linalg.norm(axis)
            proj  = centred @ axis
            width = float(proj.max() - proj.min())
            if self._min <= width <= self._max:
                candidates.append(
                    GripCandidate(
                        # Mesh is centered so the centroid is at the origin.
                        # pick_xyz = R_obj @ [0,0,0] + p_obj_base = p_obj_base,
                        # which is exactly the centroid in robot-base frame.
                        position_m=np.zeros(3),
                        approach_axis=axis,
                        width_m=width,
                    )
                )

        # Tightest grip first (requires least opening)
        return sorted(candidates, key=lambda c: c.width_m)

    def compute_grips(self, model_id: str) -> list[GripCandidate]:
        return self._grips.get(model_id, [])
