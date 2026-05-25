"""RFGS complex Gaussian field (ADR-125, Radiance Field context).

``ComplexGaussianField`` is the reconstruction artifact: a set of anisotropic
3D Gaussians (mean, scale, rotation, opacity) carrying a **complex** radiance
encoded in a small directional basis -- radio's analogue of the optical RGB
spherical-harmonic Gaussian. Invariants (unit quaternions, positive scale,
opacity in [0,1]) are enforced by parameterization, not by runtime checks.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Unit-normalize quaternions ``[N,4]`` (w,x,y,z) and return rotmats ``[N,3,3]``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def directional_basis(dirs: Tensor, n_coeffs: int) -> Tensor:
    """A small real directional basis evaluated at unit directions ``[...,3]``.

    A compact stand-in for GSRF's Fourier-Legendre / SH basis: constant + the
    three direction components + low-order products, truncated to ``n_coeffs``.
    Returns ``[..., n_coeffs]``. The complex radiance is ``sum_b coeff_b * basis_b``.
    """
    x, y, z = dirs.unbind(-1)
    feats = [
        torch.ones_like(x), x, y, z,
        x * y, y * z, x * z, x * x - y * y, 3 * z * z - 1,
    ]
    b = torch.stack(feats[:n_coeffs], dim=-1)
    if b.shape[-1] < n_coeffs:  # pad if more coeffs than features
        pad = n_coeffs - b.shape[-1]
        b = torch.cat([b, torch.zeros(*b.shape[:-1], pad, device=b.device)], dim=-1)
    return b


class ComplexGaussianField(nn.Module):
    """Aggregate root: the learnable radio radiance field."""

    def __init__(self, n: int, n_radiance_coeffs: int = 9, device: str = "cpu"):
        super().__init__()
        self.n_coeffs = n_radiance_coeffs
        self.mu = nn.Parameter(torch.zeros(n, 3, device=device))
        self.log_scale = nn.Parameter(torch.full((n, 3), -1.0, device=device))
        self.quat = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
        )
        self.opacity_logit = nn.Parameter(torch.zeros(n, device=device))
        # Complex radiance coeffs stored as real [N, B, 2] (re, im).
        rad = torch.zeros(n, n_radiance_coeffs, 2, device=device)
        rad[:, 0, 0] = 1.0  # unit DC amplitude
        self.radiance = nn.Parameter(rad)

    @property
    def num_gaussians(self) -> int:
        return self.mu.shape[0]

    def scales(self) -> Tensor:
        return torch.exp(self.log_scale)

    def opacities(self) -> Tensor:
        return torch.sigmoid(self.opacity_logit)

    def radiance_complex(self) -> Tensor:
        return torch.view_as_complex(self.radiance.contiguous())  # [N, B] complex

    def covariance(self) -> Tensor:
        """World covariance ``R S Sᵀ Rᵀ`` per Gaussian, ``[N,3,3]``."""
        rot = quat_to_rotmat(self.quat)
        s = self.scales()
        m = rot * s.unsqueeze(1)  # R @ diag(s)
        return m @ m.transpose(-1, -2)

    # -- initialization -------------------------------------------------------

    @classmethod
    def init_random(
        cls,
        n: int,
        bounds_min: tuple[float, float, float],
        bounds_max: tuple[float, float, float],
        *,
        n_radiance_coeffs: int = 9,
        device: str = "cpu",
        seed: Optional[int] = 0,
    ) -> "ComplexGaussianField":
        if seed is not None:
            torch.manual_seed(seed)
        field = cls(n, n_radiance_coeffs, device)
        lo = torch.tensor(bounds_min, device=device)
        hi = torch.tensor(bounds_max, device=device)
        with torch.no_grad():
            field.mu.copy_(lo + (hi - lo) * torch.rand(n, 3, device=device))
        return field

    @classmethod
    def init_from_pointcloud(
        cls,
        points: Tensor,  # [P, 3]
        *,
        n_radiance_coeffs: int = 9,
        device: str = "cpu",
    ) -> "ComplexGaussianField":
        """Hybrid initialization: one Gaussian per point (ADR-125 F3)."""
        points = points.to(device)
        n = points.shape[0]
        field = cls(n, n_radiance_coeffs, device)
        with torch.no_grad():
            field.mu.copy_(points)
        return field

    # -- adaptive density control (3DGS) --------------------------------------

    @torch.no_grad()
    def prune(self, min_opacity: float = 0.005) -> int:
        """Drop near-transparent Gaussians; returns the number removed."""
        keep = self.opacities() > min_opacity
        if keep.all():
            return 0
        removed = int((~keep).sum().item())
        self._reindex(keep.nonzero(as_tuple=True)[0])
        return removed

    @torch.no_grad()
    def densify_clone(self, grad_norm: Tensor, threshold: float) -> int:
        """Clone high-gradient Gaussians (under-reconstruction); returns count added.

        ``grad_norm`` may be shorter than the current count (an earlier densify
        step in the same cycle appended children); we only act on the prefix it
        covers, which keeps clone/split/prune order-independent.
        """
        n = min(grad_norm.shape[0], self.num_gaussians)
        mask = grad_norm[:n] > threshold
        if not mask.any():
            return 0
        idx = mask.nonzero(as_tuple=True)[0]
        self._append(idx)
        return int(idx.numel())

    @torch.no_grad()
    def densify_split(self, grad_norm: Tensor, threshold: float,
                      large_scale: float = 0.1) -> int:
        """Split large, high-gradient Gaussians into two smaller children.

        The 3DGS over-reconstruction remedy (vs. ``densify_clone`` for under-
        reconstruction): children inherit the parent's attributes at reduced
        scale, offset along the dominant axis. Returns the number added.
        """
        n = min(grad_norm.shape[0], self.num_gaussians)
        big = self.scales()[:n].max(dim=-1).values > large_scale
        mask = (grad_norm[:n] > threshold) & big
        if not mask.any():
            return 0
        idx = mask.nonzero(as_tuple=True)[0]
        offset = torch.randn_like(self.mu[idx]) * self.scales()[idx] * 0.5
        with torch.no_grad():
            self.log_scale[idx] -= 0.6931  # ln(2): halve the parent scale
        self._append(idx)
        # Nudge the freshly appended children off the parent center.
        n_added = idx.numel()
        self.mu.data[-n_added:] += offset
        return int(n_added)

    @torch.no_grad()
    def _reindex(self, idx: Tensor) -> None:
        self.mu = nn.Parameter(self.mu[idx])
        self.log_scale = nn.Parameter(self.log_scale[idx])
        self.quat = nn.Parameter(self.quat[idx])
        self.opacity_logit = nn.Parameter(self.opacity_logit[idx])
        self.radiance = nn.Parameter(self.radiance[idx])

    @torch.no_grad()
    def _append(self, idx: Tensor) -> None:
        jitter = torch.randn_like(self.mu[idx]) * self.scales()[idx]
        self.mu = nn.Parameter(torch.cat([self.mu, self.mu[idx] + jitter]))
        self.log_scale = nn.Parameter(torch.cat([self.log_scale, self.log_scale[idx]]))
        self.quat = nn.Parameter(torch.cat([self.quat, self.quat[idx]]))
        self.opacity_logit = nn.Parameter(
            torch.cat([self.opacity_logit, self.opacity_logit[idx]])
        )
        self.radiance = nn.Parameter(torch.cat([self.radiance, self.radiance[idx]]))
