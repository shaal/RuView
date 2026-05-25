"""RFGS geometry evaluation (ADR-125 P3, AC5).

Scores a reconstructed ``ComplexGaussianField`` against a reference scan
(camera/LiDAR point cloud) with two standard 3D-reconstruction metrics:

  * **Chamfer distance** -- symmetric mean nearest-neighbour distance between
    the field's (opacity-weighted) Gaussian centres and the reference points.
    Lower is better; units are metres.
  * **Occupancy IoU** -- intersection-over-union of voxelized occupancy. The
    field occupies a voxel if any Gaussian centre with opacity above a
    threshold falls in it. Higher is better; in [0, 1].

Both run on CPU and need only torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import ComplexGaussianField


def chamfer_distance(a: Tensor, b: Tensor) -> float:
    """Symmetric Chamfer distance between point sets ``a[N,3]``, ``b[M,3]`` (metres)."""
    if a.numel() == 0 or b.numel() == 0:
        return float("inf")
    d = torch.cdist(a, b)            # [N, M]
    a_to_b = d.min(dim=1).values.mean()
    b_to_a = d.min(dim=0).values.mean()
    return float((a_to_b + b_to_a) * 0.5)


def _voxel_keys(points: Tensor, origin: Tensor, voxel: float) -> set[tuple[int, int, int]]:
    idx = torch.floor((points - origin) / voxel).to(torch.int64)
    return {tuple(int(v) for v in row) for row in idx}


def occupancy_iou(
    field_points: Tensor,
    reference_points: Tensor,
    *,
    voxel: float = 0.1,
) -> float:
    """Voxel-occupancy IoU between two point sets at resolution ``voxel`` (m)."""
    if field_points.numel() == 0 or reference_points.numel() == 0:
        return 0.0
    origin = torch.minimum(field_points.min(0).values, reference_points.min(0).values)
    fset = _voxel_keys(field_points, origin, voxel)
    rset = _voxel_keys(reference_points, origin, voxel)
    inter = len(fset & rset)
    union = len(fset | rset)
    return inter / union if union else 0.0


@dataclass
class GeometryReport:
    chamfer_m: float
    occupancy_iou: float
    n_field_points: int
    n_reference_points: int

    def as_dict(self) -> dict:
        return {
            "chamfer_m": self.chamfer_m,
            "occupancy_iou": self.occupancy_iou,
            "n_field_points": self.n_field_points,
            "n_reference_points": self.n_reference_points,
        }


def evaluate_geometry(
    field: ComplexGaussianField,
    reference_points: Tensor,
    *,
    min_opacity: float = 0.05,
    voxel: float = 0.1,
) -> GeometryReport:
    """Compare a field's occupied Gaussian centres to a reference point cloud."""
    with torch.no_grad():
        keep = field.opacities() > min_opacity
        pts = field.mu[keep].detach().cpu()
    ref = reference_points.detach().cpu()
    return GeometryReport(
        chamfer_m=chamfer_distance(pts, ref),
        occupancy_iou=occupancy_iou(pts, ref, voxel=voxel),
        n_field_points=int(pts.shape[0]),
        n_reference_points=int(ref.shape[0]),
    )
