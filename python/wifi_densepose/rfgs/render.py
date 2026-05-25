"""RFGS differentiable CSI forward model (ADR-125, Field Optimization).

Synthesizes the complex channel response ``Ĥ[antenna, subcarrier]`` an RX would
measure, given a TX pose and the Gaussian field. This is the *reference*
PyTorch backend -- pure autograd, runs on CPU (no CUDA required, ADR-125 N2).
The accurate GSRF ``complex-gaussian-tracer-csi`` CUDA kernel plugs in behind
the same ``CsiForwardModel`` interface (P3, opt-in).

Physical model (reference): each Gaussian is a re-radiating scatterer. A path
TX -> Gaussian g -> RX contributes a complex phasor whose

  * magnitude  = opacity_g * gaussian_extent_response_g * |radiance_g(dir)| / (d_tx * d_rx)
  * phase      = -2π f (d_tx + d_rx) / c        (per subcarrier frequency f)

summed coherently over all Gaussians. This captures multipath superposition and
per-subcarrier phase -- the structure RF-GS needs -- without claiming to be the
full volumetric wavefront tracer (that is the CUDA backend's job).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .geometry import SPEED_OF_LIGHT
from .model import ComplexGaussianField, directional_basis


class CsiForwardModel(ABC):
    """Backend-agnostic contract (DDD anti-corruption layer)."""

    @abstractmethod
    def render(
        self,
        field: ComplexGaussianField,
        tx_position: Tensor,   # [3]
        rx_position: Tensor,   # [3]
        freqs_hz: Tensor,      # [S]
        n_antennas: int,
    ) -> Tensor:               # complex [n_antennas, S]
        ...


class ReferenceTorchTracer(CsiForwardModel):
    """Pure-PyTorch additive-multipath forward model (the no-CUDA reference)."""

    def __init__(self, antenna_spacing_m: float = 0.03):
        self.antenna_spacing_m = antenna_spacing_m

    def render(
        self,
        field: ComplexGaussianField,
        tx_position: Tensor,
        rx_position: Tensor,
        freqs_hz: Tensor,
        n_antennas: int,
    ) -> Tensor:
        mu = field.mu                              # [N,3]
        device = mu.device
        tx = tx_position.to(device)
        rx = rx_position.to(device)

        d_tx = (mu - tx).norm(dim=-1).clamp_min(1e-3)        # [N]
        to_rx = mu - rx
        d_rx = to_rx.norm(dim=-1).clamp_min(1e-3)            # [N]
        dirs = to_rx / d_rx.unsqueeze(-1)                    # [N,3] g->rx direction

        # Directional complex radiance: coeffs · basis(dir).
        basis = directional_basis(dirs, field.n_coeffs)      # [N,B]
        rad = (field.radiance_complex() * basis).sum(-1)     # [N] complex

        opacity = field.opacities()                          # [N]
        # Anisotropic extent response: larger Gaussians radiate more broadly.
        extent = field.scales().prod(dim=-1).clamp_min(1e-6) # [N]
        amp = opacity * extent * rad.abs() / (d_tx * d_rx)   # [N] real magnitude
        base_phase = torch.angle(rad)                        # [N]

        path = (d_tx + d_rx)                                 # [N] meters
        # Per-antenna phase offset along a simple linear array (x-axis).
        ant_idx = torch.arange(n_antennas, device=device).float()
        ant_off = ant_idx * self.antenna_spacing_m           # [A]

        # Phase per (antenna, subcarrier, gaussian).
        f = freqs_hz.to(device)                              # [S]
        k = 2.0 * torch.pi * f / SPEED_OF_LIGHT              # [S] wavenumber
        # total path delay phase: -k * path  ; antenna term: -k * ant_off * dir_x
        dir_x = dirs[:, 0]                                   # [N]
        phase = (
            base_phase.view(1, 1, -1)
            - k.view(1, -1, 1) * path.view(1, 1, -1)
            - k.view(1, -1, 1) * (ant_off.view(-1, 1, 1) * dir_x.view(1, 1, -1))
        )                                                    # [A,S,N]
        contrib = amp.view(1, 1, -1) * torch.exp(1j * phase) # [A,S,N] complex
        return contrib.sum(-1)                               # [A,S] complex


class GsrfCudaTracer(CsiForwardModel):
    """Opt-in accelerated backend (ADR-125 P3) wrapping GSRF's CUDA tracer.

    GSRF (BSD-3-Clause) ships a ``complex_gaussian_tracer_csi`` CUDA extension
    that performs the true wavefront-propagation tracing the pure-PyTorch
    ``ReferenceTorchTracer`` only approximates -- and, unlike the reference, it
    makes Gaussian *position* gradients tractable (the hard inverse problem).

    It is not vendored: building it needs CUDA 12.1 + the GSRF kernel sources.
    This adapter lazily imports it so the package installs and runs without
    CUDA; constructing it without the kernel raises a clear, actionable error.
    """

    def __init__(self) -> None:
        try:
            import complex_gaussian_tracer_csi as _kernel  # type: ignore
        except ImportError as e:  # pragma: no cover - requires CUDA build
            raise ImportError(
                "GsrfCudaTracer requires the GSRF CUDA extension "
                "'complex_gaussian_tracer_csi'. Build it from "
                "https://github.com/nesl/GSRF (BSD-3-Clause, CUDA 12.1), then "
                "re-run with --backend gsrf-cuda. Falls back to the pure-PyTorch "
                "ReferenceTorchTracer (--backend reference) otherwise."
            ) from e
        self._kernel = _kernel

    def render(self, field, tx_position, rx_position, freqs_hz, n_antennas):  # pragma: no cover
        return self._kernel.trace_csi(
            field.mu, field.covariance(), field.opacities(),
            field.radiance_complex(), tx_position, rx_position, freqs_hz, n_antennas,
        )


def make_forward_model(backend: str = "reference") -> CsiForwardModel:
    """Factory selecting a forward-model backend (DDD anti-corruption layer)."""
    if backend == "reference":
        return ReferenceTorchTracer()
    if backend == "gsrf-cuda":
        return GsrfCudaTracer()
    raise ValueError(f"unknown forward-model backend: {backend!r}")


def reconstruction_loss(
    h_pred: Tensor,     # complex [A,S]
    h_meas: Tensor,     # complex [A,S]
    *,
    use_phase: bool = True,
    lambda_amp: float = 0.2,
) -> dict[str, Tensor]:
    """Global-phase-invariant complex reconstruction loss.

    The absolute phase of CSI is physically unobservable (carrier-phase offset,
    CFO) and at 2.4 GHz the per-path phase is heavily aliased -- so a naive
    angle difference is not learnable. We instead use the **normalized complex
    cross-correlation magnitude**, which is invariant to a global phase rotation
    and global scale of either signal while remaining sensitive to the *relative*
    amplitude+phase structure across subcarriers/antennas (the part that carries
    geometry and multipath):

        ncc = |<H_pred, H_meas>| / (||H_pred|| ||H_meas||)   in [0, 1]
        phase-term = 1 - ncc

    For amplitude-only sources (``use_phase=False``, DDD invariant I1) we fall
    back to a magnitude-shape term only. A small magnitude term is always added
    for optimization stability.
    """
    p = h_pred.reshape(-1)
    m = h_meas.reshape(-1)

    ap = p.abs() / p.abs().norm().clamp_min(1e-9)
    am = m.abs() / m.abs().norm().clamp_min(1e-9)
    amp_loss = torch.mean((ap - am) ** 2)

    if use_phase:
        inner = torch.sum(torch.conj(p) * m).abs()
        ncc = inner / (p.norm() * m.norm()).clamp_min(1e-9)
        phase_loss = 1.0 - ncc
        total = phase_loss + lambda_amp * amp_loss
    else:
        phase_loss = torch.zeros((), device=h_pred.device)
        total = amp_loss
    return {"total": total, "amplitude": amp_loss, "phase": phase_loss}
