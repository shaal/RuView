"""RFGS -- camera-free room reconstruction via RF Gaussian Splatting (ADR-125).

Optimizes complex-valued 3D Gaussians whose implied radio radiance field
reproduces the CSI measured by the ESP32 mesh, producing an explicit, editable
3D room model from WiFi signals alone. Foundation: GSRF (BSD-3-Clause).

Requires the ``rfgs`` extra: ``pip install "wifi-densepose[rfgs]"``.
"""

from __future__ import annotations

from .geometry import NodePose, Pose, RoomConfig, subcarrier_frequencies

__all__ = [
    "NodePose",
    "Pose",
    "RoomConfig",
    "subcarrier_frequencies",
    "CsiMeasurement",
    "CsiMeasurementDataset",
    "decode_adr018",
    "ComplexGaussianField",
    "ReferenceTorchTracer",
    "reconstruction_loss",
]


def __getattr__(name: str):
    # Lazy re-exports so importing the package does not require torch.
    if name in ("CsiMeasurement", "CsiMeasurementDataset", "decode_adr018"):
        from . import dataset as _d
        return getattr(_d, name)
    if name == "ComplexGaussianField":
        from .model import ComplexGaussianField
        return ComplexGaussianField
    if name in ("ReferenceTorchTracer", "reconstruction_loss"):
        from . import render as _r
        return getattr(_r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
