"""Grip planning from CAD models.

SimpleGripPlanner uses PCA on mesh vertices to find principal axes, then
checks which axes produce a grip width within [min_grip_m, max_grip_m].
All dimensions are in metres, consistent with FoundationPose's effective_scale.

ManualGripPlanner loads hand-authored 6-DOF grasps (position + full
orientation + hover backoff) from a JSON file, in the format produced by the
"Manual 6DoF grasp definition" section of ``photo_neo_capture_test.ipynb``
(see ``grasps/manual_grasps.json``). Use this when a grip needs a specific
grasp point and approach direction rather than a PCA-derived centroid grip.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


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
        ``ManualGripPlanner`` sets this to the grasp's hand-authored ``t_m``,
        which need not be the centroid.

        **Limitation**: centroid-as-pick-point (``SimpleGripPlanner``) is only
        reliable for roughly symmetric, convex objects.  For elongated,
        multi-lobe, or heavily asymmetric shapes a centroid grip may be
        mechanically infeasible even though IK succeeds — use
        ``ManualGripPlanner`` for those.
    approach_axis:
        Unit vector in object frame along which the gripper fingers close
        (for ``ManualGripPlanner`` grips, this is ``orientation_m[:, 2]``).
        Transformed to base frame before planning: ``approach_base = R_obj @ approach_axis``.
    width_m:
        Required gripper opening width in metres.
    orientation_m:
        Full 3×3 rotation matrix in object frame, or ``None``.  Set by
        ``ManualGripPlanner`` from the grasp's approach direction + jaw roll;
        left ``None`` by ``SimpleGripPlanner``, in which case the EE
        orientation is instead heuristically derived from ``approach_axis``
        alone (see ``GripFeasibilityChecker._build_target_R``) and only
        position-only IK is used for the pick move.
    hover_offset_m:
        Distance to back off along ``orientation_m[:, 2]`` (the approach axis)
        for a pre-grasp/retreat hover pose, or ``0.0`` for no separate hover
        waypoint (the default, and always the case when ``orientation_m`` is
        ``None``).
    jaw_length_m:
        Finger length, for preview/visualization only (drawing the jaw
        segments in ``visualizer.overlay_grasps`` and the notebook's mesh
        preview) — not a physical constraint used anywhere in planning.
    """

    position_m:     np.ndarray
    approach_axis:  np.ndarray
    width_m:        float
    orientation_m:  np.ndarray | None = None
    hover_offset_m: float = 0.0
    jaw_length_m:   float = 0.01


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


# ── Manual 6-DOF grasp geometry ────────────────────────────────────────────────
# Ported from the "Manual 6DoF grasp definition" section of
# photo_neo_capture_test.ipynb — keep in sync with that notebook.

def _approach_vector(g: dict) -> np.ndarray:
    """Unit approach-axis direction (local +Z) from spherical angles relative to the object frame."""
    az = np.radians(g["approach_azimuth_deg"])
    el = np.radians(g["approach_elevation_deg"])
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _grasp_rotation(g: dict) -> np.ndarray:
    """Rotation matrix from an approach direction + roll about it.

    local +Z = approach axis, local +X = jaw-opening axis (see
    ``grasps/manual_grasps.json`` for the authoring convention).
    """
    z = _approach_vector(g)
    # reference "up" for building the perpendicular basis; swap to avoid degeneracy
    # when the approach axis is itself near-parallel to the primary reference
    up = np.array([0.0, 1.0, 0.0]) if abs(z[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    x0 = np.cross(up, z); x0 /= np.linalg.norm(x0)
    y0 = np.cross(z, x0)
    roll = np.radians(g["jaw_roll_deg"])
    x = np.cos(roll) * x0 + np.sin(roll) * y0
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


class ManualGripPlanner(GripPlanner):
    """Loads hand-authored 6-DOF grasps from a JSON file.

    Each grasp specifies a grip position, a full orientation (approach
    direction + jaw roll), and an optional hover backoff — richer than
    ``SimpleGripPlanner``'s PCA axes, which only give a centroid + a single
    approach axis with no hover waypoint.

    Grasps are defined in the SAME frame FoundationPose estimates poses in:
    origin at the CAD mesh's centroid, vertices scaled by ``effective_scale``
    (see ``src/foundationpose.py``) — so ``t_m`` positions need no extra
    scaling here.

    JSON schema (``model_id`` -> list of grasp dicts), matching the ``GRASPS``
    dict authored in ``photo_neo_capture_test.ipynb``::

        {
          "<model_id>": [
            {
              "t_m": [x, y, z],
              "approach_azimuth_deg": ..., "approach_elevation_deg": ...,
              "jaw_roll_deg": ..., "jaw_width_m": ..., "hover_dist_m": ...
            }
          ]
        }
    """

    def __init__(
        self,
        cad_models: Any,          # list of DictConfig or objects with .path and .id
        effective_scale: float,   # unused: t_m is already metres in FoundationPose's
                                  # centred/scaled frame; accepted only so BinPicker can
                                  # instantiate every grip planner with the same kwargs
        min_grip_m: float,
        max_grip_m: float,
        grasps_path: str | Path,
    ):
        self._min = float(min_grip_m)
        self._max = float(max_grip_m)
        self._grips: dict[str, list[GripCandidate]] = {}

        with open(grasps_path) as f:
            all_grasps: dict[str, list[dict]] = json.load(f)

        model_ids = {m.id for m in cad_models}
        for model_id, grasp_list in all_grasps.items():
            if model_id not in model_ids:
                continue

            candidates: list[GripCandidate] = []
            for g in grasp_list:
                width = float(g["jaw_width_m"])
                if not (self._min <= width <= self._max):
                    log.warning(
                        "ManualGripPlanner: skipping grasp for '%s' — width=%.1fmm "
                        "outside gripper range [%.1f, %.1f]mm",
                        model_id, width * 1000, self._min * 1000, self._max * 1000,
                    )
                    continue

                R = _grasp_rotation(g)
                candidates.append(
                    GripCandidate(
                        position_m=np.array(g["t_m"], dtype=float),
                        approach_axis=R[:, 2],
                        width_m=width,
                        orientation_m=R,
                        hover_offset_m=float(g.get("hover_dist_m", 0.0)),
                        jaw_length_m=float(g.get("jaw_length_m", 0.01)),
                    )
                )
            self._grips[model_id] = candidates

    def compute_grips(self, model_id: str) -> list[GripCandidate]:
        return self._grips.get(model_id, [])
