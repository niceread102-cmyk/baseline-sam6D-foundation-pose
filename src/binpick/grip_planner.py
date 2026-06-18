"""Grip planning from CAD models.

SimpleGripPlanner uses PCA to find principal axes, then does a slice-based
scan to locate grippable regions:

  For each axis (closing direction), the perpendicular plane is divided into an
  n_scan_bins × n_scan_bins grid.  Each cell computes the local extent of mesh
  vertices in the closing direction.  Cells where that local extent falls within
  [min_grip_m, max_grip_m] are "grippable".

  Connected grippable cells (4-connectivity) form a grip region.  Each region
  becomes a GripCandidate whose position is the weighted centroid of the region,
  biased toward the object centroid for stability.

This correctly handles asymmetric objects (e.g. a steering knuckle) where the
overall bounding box exceeds the gripper range but specific features (arms,
bosses) do not.  The global-extent check used previously would miss all such
features and always target the object centroid.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GripCandidate:
    position_m: np.ndarray    # gripper jaw midpoint in object frame (metres, relative to centroid)
    approach_axis: np.ndarray  # unit vector — gripper fingers close along this axis
    width_m: float             # required opening width (metres)


class GripPlanner(ABC):
    @abstractmethod
    def compute_grips(self, model_id: str) -> list[GripCandidate]:
        """Return grip candidates for the given model ID."""


class SimpleGripPlanner(GripPlanner):
    """Slice-based grip planner.

    Parameters
    ----------
    n_scan_bins:
        Resolution of the perpendicular-plane grid used to find grippable
        cross-sections.  Higher = finer detail but O(n²) cells; 20 is a good
        default for objects in the 50–300 mm range.
    min_pts_per_bin:
        Minimum vertex count in a grid cell before its local width is trusted.
        Raise if the mesh is very dense and you see phantom grips on thin shells.
    """

    def __init__(
        self,
        cad_models: Any,
        effective_scale: float,
        min_grip_m: float,
        max_grip_m: float,
        n_scan_bins: int = 20,
        min_pts_per_bin: int = 3,
    ):
        self._scale   = float(effective_scale)
        self._min     = float(min_grip_m)
        self._max     = float(max_grip_m)
        self._n_bins  = int(n_scan_bins)
        self._min_pts = int(min_pts_per_bin)
        self._grips: dict[str, list[GripCandidate]] = {}

        for model in cad_models:
            self._grips[model.id] = self._analyse(model.path)

    # ── analysis ──────────────────────────────────────────────────────────────

    def _analyse(self, mesh_path: str | Path) -> list[GripCandidate]:
        import trimesh

        mesh     = trimesh.load_mesh(str(mesh_path))
        verts    = np.asarray(mesh.vertices, dtype=float) * self._scale
        centroid = verts.mean(axis=0)
        centred  = verts - centroid

        _, _, Vt = np.linalg.svd(centred, full_matrices=False)

        candidates: list[GripCandidate] = []

        for i in range(3):
            close_axis = Vt[i] / np.linalg.norm(Vt[i])
            # The other two PCA axes span the perpendicular plane
            perp = [Vt[j] / np.linalg.norm(Vt[j]) for j in range(3) if j != i]

            u_c = centred @ close_axis  # projection onto closing axis  (N,)
            u_1 = centred @ perp[0]     # projection onto perp axis 1   (N,)
            u_2 = centred @ perp[1]     # projection onto perp axis 2   (N,)

            n  = self._n_bins
            e1 = np.linspace(u_1.min(), u_1.max(), n + 1)
            e2 = np.linspace(u_2.min(), u_2.max(), n + 1)

            grid_valid = np.zeros((n, n), dtype=bool)
            grid_pos   = np.zeros((n, n, 3))   # 3D grip centre per cell
            grid_width = np.zeros((n, n))       # local width per cell

            for k1 in range(n):
                mask1 = (u_1 >= e1[k1]) & (u_1 < e1[k1 + 1])
                if not mask1.any():
                    continue
                for k2 in range(n):
                    mask = mask1 & (u_2 >= e2[k2]) & (u_2 < e2[k2 + 1])
                    if mask.sum() < self._min_pts:
                        continue
                    w = float(u_c[mask].max() - u_c[mask].min())
                    if self._min <= w <= self._max:
                        grid_valid[k1, k2] = True
                        # Grip centre in closing direction: midpoint of jaw span.
                        # Grip centre in perp directions: centroid of vertices in cell.
                        close_mid = float(u_c[mask].max() + u_c[mask].min()) / 2.0
                        grid_pos[k1, k2] = (
                            close_mid            * close_axis
                            + u_1[mask].mean() * perp[0]
                            + u_2[mask].mean() * perp[1]
                        )
                        grid_width[k1, k2] = w

            if not grid_valid.any():
                continue

            # Each connected group of grippable cells → one GripCandidate
            labeled, n_comps = _label_connected(grid_valid)

            for cid in range(1, n_comps + 1):
                comp      = labeled == cid
                positions = grid_pos[comp]    # (K, 3)
                widths    = grid_width[comp]  # (K,)

                # Bias grip position toward object centroid (0, 0, 0) for balance.
                # sigma = mean distance of this region from centroid; positions
                # at the mean distance get weight exp(-1) ≈ 0.37.
                dists   = np.linalg.norm(positions, axis=1)
                sigma   = max(float(dists.mean()), 1e-3)
                weights = np.exp(-dists / sigma)

                grip_pos   = np.average(positions, axis=0, weights=weights)
                grip_width = float(np.average(widths, weights=weights))

                candidates.append(GripCandidate(
                    position_m=grip_pos,
                    approach_axis=close_axis.copy(),
                    width_m=grip_width,
                ))

        # Closest to centroid first — selector iterates in this order
        candidates.sort(key=lambda c: np.linalg.norm(c.position_m))
        return candidates

    def compute_grips(self, model_id: str) -> list[GripCandidate]:
        return self._grips.get(model_id, [])


# ── helpers ───────────────────────────────────────────────────────────────────

def _label_connected(grid: np.ndarray) -> tuple[np.ndarray, int]:
    """BFS connected-component labelling (4-connectivity) on a 2D bool grid."""
    labels     = np.zeros_like(grid, dtype=int)
    cid        = 0
    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] and labels[r, c] == 0:
                cid += 1
                queue       = [(r, c)]
                labels[r, c] = cid
                while queue:
                    cr, cc = queue.pop()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr, nc] and labels[nr, nc] == 0:
                                labels[nr, nc] = cid
                                queue.append((nr, nc))
    return labels, cid
