"""RFGS field export (ADR-125, Field Serving context).

Two outputs:
  * ``splats-v2`` JSON  -- backward-compatible extension of ``/api/splats``:
    ``GaussianSplatV2`` adds ``rotation`` (quaternion) + ``radiance`` (complex
    coeffs, interleaved re,im) to the v1 ``{center,color,opacity,scale}``.
    v1 viewers ignore the new fields; the WebGPU viewer uses them.
  * ``.rfgs.npz``       -- full-fidelity baked field (mu, scale, quat, opacity,
    radiance) for the edge query path and the upgraded viewer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import ComplexGaussianField, directional_basis


def _amplitude_rgb(field: ComplexGaussianField) -> torch.Tensor:
    """Map mean radiance magnitude -> blue->green->red RGB for v1 viewers."""
    rad = field.radiance_complex().abs().mean(-1)           # [N]
    v = (rad / rad.max().clamp_min(1e-9)).clamp(0, 1)       # [N] in [0,1]
    r = v
    g = 1.0 - (2 * v - 1).abs()
    b = 1.0 - v
    return torch.stack([r, g, b], dim=-1)                   # [N,3]


def to_splats_v2(field: ComplexGaussianField) -> list[dict]:
    """Serialize to the ``GaussianSplatV2`` list consumed by ``/api/splats``."""
    with torch.no_grad():
        mu = field.mu.cpu()
        scale = field.scales().cpu()
        quat = (field.quat / field.quat.norm(dim=-1, keepdim=True)).cpu()
        opacity = field.opacities().cpu()
        color = _amplitude_rgb(field).cpu()
        rad = field.radiance.detach().cpu().reshape(field.num_gaussians, -1)  # re,im
    out = []
    for i in range(field.num_gaussians):
        out.append(
            {
                "center": mu[i].tolist(),
                "color": color[i].tolist(),
                "opacity": float(opacity[i]),
                "scale": scale[i].tolist(),
                "rotation": quat[i].tolist(),     # NEW (v2)
                "radiance": rad[i].tolist(),      # NEW (v2): [re0,im0,re1,im1,...]
            }
        )
    return out


def write_splats_v2_json(field: ComplexGaussianField, path: str | Path) -> Path:
    p = Path(path)
    payload = {
        "schema": "rfgs-v2",
        "count": field.num_gaussians,
        "n_radiance_coeffs": field.n_coeffs,
        "splats": to_splats_v2(field),
    }
    p.write_text(json.dumps(payload))
    return p


def write_baked_npz(field: ComplexGaussianField, path: str | Path) -> Path:
    """Pruned/quantizable baked field for edge query (ADR-125 R1)."""
    import numpy as np

    p = Path(path)
    with torch.no_grad():
        np.savez_compressed(
            p,
            mu=field.mu.detach().cpu().numpy().astype("float16"),
            scale=field.scales().detach().cpu().numpy().astype("float16"),
            quat=field.quat.detach().cpu().numpy().astype("float16"),
            opacity=field.opacities().detach().cpu().numpy().astype("float16"),
            radiance=field.radiance.detach().cpu().numpy().astype("float16"),
            n_radiance_coeffs=field.n_coeffs,
        )
    return p
