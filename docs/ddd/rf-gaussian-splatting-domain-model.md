# RF Gaussian Splatting (RFGS) Domain Model

RFGS is the camera-free room-reconstruction subsystem of RuView. It optimizes a set of complex-valued 3D Gaussians whose implied **radio radiance field** reproduces the CSI measured by the ESP32 mesh, then renders an explicit, editable 3D model of the room — geometry and multipath — from radio signals alone.

This document specifies RFGS with [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html): bounded contexts that own their data and rules, aggregates that enforce invariants, value objects that carry meaning, and domain events that connect contexts. See [ADR-125](../adr/ADR-125-rf-gaussian-splatting-room-reconstruction.md) for the decision record and [GSRF](https://github.com/nesl/GSRF) for the foundation.

**Bounded Contexts:**

| # | Context | Responsibility | Code |
|---|---------|----------------|------|
| 1 | [CSI Acquisition](#1-csi-acquisition-context) | Decode phase-preserving complex CSI from ADR-018 frames / live WS; attach TX-RX geometry | `rfgs/dataset.py`, `rfgs/geometry.py` |
| 2 | [Radiance Field](#2-radiance-field-context) | The complex Gaussian field aggregate: parameters + invariants + adaptive density control | `rfgs/model.py` |
| 3 | [Field Optimization](#3-field-optimization-context) | Forward-render CSI, compute reconstruction loss, optimize, densify/prune | `rfgs/render.py`, `rfgs/train.py` |
| 4 | [Field Serving](#4-field-serving-context) | Export to `/api/splats` v2 + `.rfgs.npz`; edge query; viewer | `rfgs/export.py`, `pointcloud/src/stream.rs` |

All Python paths are relative to `python/wifi_densepose/`.

---

## Domain-Driven Design Specification

### Ubiquitous Language

| Term | Definition |
|------|------------|
| **Radio Radiance Field** | The complex (amplitude + phase) wavefield over the room, represented explicitly as a set of complex-valued 3D Gaussians. The reconstruction target. |
| **CSI Measurement** | A single observation: complex channel response `H[antenna, subcarrier] ∈ ℂ`, tagged with TX pose, RX pose+orientation, and per-subcarrier frequency. The atomic training datum. |
| **Complex Gaussian** | One field primitive: mean `μ∈ℝ³`, covariance `Σ = R·S·Sᵀ·Rᵀ` (rotation quaternion `q`, scale `s∈ℝ³`), opacity/transmittance `ρ`, and **complex radiance** `ψ` encoded in a Fourier–Legendre (GSRF) or spherical-harmonic basis. Radio's analogue of the optical RGB Gaussian. |
| **Node Pose** | A sensing node's known position + orientation in room coordinates; the source of TX/RX geometry. |
| **Room Config** | Immutable scene description: node poses, room bounds, channel→center-frequency map, subcarrier plan. |
| **TX/RX Pair** | A (transmitter, receiver) configuration derived from the node mesh; GSRF assumes one TX + many RX, so each pair maps to a GSRF "view". |
| **Hybrid Initialization** | Seeding Gaussian means from the existing RuView fused point cloud (`/api/cloud`) before pure-RF optimization. |
| **Forward Model** | The differentiable renderer that synthesizes predicted CSI `Ĥ` for a query (TX,RX) pose from the field. Pure-PyTorch reference, with an optional GSRF CUDA tracer backend. |
| **Reconstruction Loss** | Discrepancy between predicted and measured CSI: amplitude term + circular **phase** term + optional spatial-spectrum term. |
| **Adaptive Density Control** | 3DGS densify/clone/split/prune driven by positional gradient and opacity, concentrating Gaussians where CSI residual is high. |
| **Baked Field** | A pruned + quantized, inference-only export of the field for edge query and the WebGPU viewer (`.rfgs.npz`). |
| **Re-fit** | Self-supervised re-optimization triggered when the room changes, gated by the CSI room fingerprint (`identify_location`). |
| **Splats v2** | The backward-compatible extension of `/api/splats`: `GaussianSplatV2` adds `rotation` (quaternion) and `radiance`/`sh` to the v1 `{center,color,opacity,scale}`. |

---

## Bounded Contexts

### 1. CSI Acquisition Context

**Responsibility:** Turn raw RuView CSI into phase-preserving `CsiMeasurement` value objects with attached TX/RX geometry. Owns the hard constraint that **training requires phase** — so it reads ADR-018 binary I/Q frames (which preserve phase + antennas), not the amplitude-only `.csi.jsonl` recorder.

**Aggregates**

```python
# Aggregate Root: a dataset of CSI measurements bound to one RoomConfig.
class CsiMeasurementDataset(torch.utils.data.Dataset):
    room: RoomConfig                 # immutable scene geometry
    measurements: list[CsiMeasurement]
    # Invariant: every measurement's node_id resolves to a NodePose in room.
    # Invariant: complex H is reconstructed from I/Q; amplitude-only sources
    #            yield phase=None and are excluded from phase-loss training.
```

**Value Objects**

```python
@dataclass(frozen=True)
class CsiMeasurement:
    h: Tensor          # complex64 [n_antennas, n_subcarriers]
    tx_pose: Pose      # ℝ³ position (+ orientation)
    rx_pose: Pose
    freqs_hz: Tensor   # [n_subcarriers] center frequencies
    timestamp_s: float
    node_id: int

@dataclass(frozen=True)
class Pose:            # immutable; position + unit-quaternion orientation
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
```

**Anti-Corruption Layer:** `dataset.py` adapters translate three external formats (ADR-018 binary, `.csi.jsonl`, live WS `SensingMessage`) into `CsiMeasurement`. Upstream phase sanitization (LO/CFO offset) is applied via `ruvsense/phase_align` semantics before measurements leave this context.

**Domain Events:** `MeasurementBatchReady`, `PhaseUnavailable(source)` (amplitude-only degradation), `RoomChanged(fingerprint)` (from `identify_location`, triggers Re-fit).

---

### 2. Radiance Field Context

**Responsibility:** Own the `ComplexGaussianField` aggregate and its invariants. This is the reconstruction artifact.

**Aggregates**

```python
# Aggregate Root: the learnable radio radiance field.
class ComplexGaussianField(nn.Module):
    mu: Parameter        # [N, 3] means
    scale: Parameter     # [N, 3] log-scale (exp → positive)
    quat: Parameter      # [N, 4] rotation (normalized on read)
    opacity: Parameter   # [N] logit (sigmoid → [0,1])
    radiance: Parameter  # [N, B] complex Fourier–Legendre / SH coeffs
    # Invariant: quaternions are unit-norm at read time.
    # Invariant: scale > 0 (log-parameterized); opacity ∈ [0,1] (sigmoid).
    # Invariant: N changes only via Adaptive Density Control (densify/prune).
```

**Value Objects:** `Quaternion` (unit), `LogScale`, `OpacityLogit`, `RadianceCoeffs` (complex). **Invariants** enforced via parameterization (no raw float can violate them).

**Domain Events:** `FieldDensified(added, removed)`, `FieldConverged(final_loss)`, `FieldExported(path)`.

---

### 3. Field Optimization Context

**Responsibility:** Render predicted CSI from the field, score it against measurements, optimize, and run adaptive density control. Pure-PyTorch reference forward model with a CUDA-tracer seam.

**Aggregates**

```python
# Aggregate Root: one optimization run over a dataset + field.
class RfgsTrainer:
    field: ComplexGaussianField
    dataset: CsiMeasurementDataset
    forward: CsiForwardModel        # render.py — differentiable Ĥ(field, tx, rx)
    # Invariant: loss is computed on complex H (amp + circular phase),
    #            never on amplitude alone unless phase is unavailable.
    # Invariant: densify/prune only between optimizer steps (field N is stable
    #            within a step).
```

**Value Objects:** `ReconstructionLoss { amplitude, phase, spectrum, reg }`, `DensifyPolicy { grad_threshold, opacity_threshold, every }`.

**Domain Events:** `StepCompleted(step, loss)`, `HeldOutEvaluated(metric)`, `CheckpointWritten(path)`.

**Anti-Corruption Layer:** the `CsiForwardModel` interface isolates the optimizer from the *backend* — `ReferenceTorchTracer` (default, no CUDA) vs. `GsrfCudaTracer` (opt-in, ADR-125 P3). Swapping backends must not change the optimizer's contract.

---

### 4. Field Serving Context

**Responsibility:** Make the optimized field consumable. Export to backward-compatible `splats-v2` JSON and to a baked `.rfgs.npz`; serve from `/api/splats`; render in the viewer.

**Aggregates / Surfaces**

```rust
// Extends the existing pointcloud::GaussianSplat (v1) — additive, v1-safe.
#[derive(Serialize)]
pub struct GaussianSplatV2 {
    pub center: [f32; 3],
    pub color: [f32; 3],     // amplitude-mapped RGB for v1 viewers
    pub opacity: f32,
    pub scale: [f32; 3],
    pub rotation: [f32; 4],  // NEW — quaternion (anisotropic)
    pub radiance: Vec<f32>,  // NEW — complex SH/FL coeffs (interleaved re,im)
}
```

**Invariants:** `/api/splats` without `?schema=rfgs-v2` returns v1 shape unchanged (backward compatibility, ADR-125 AC4). Baked fields are read-only at the edge — **no on-device training** (ADR-125 N3).

**Domain Events:** `SplatsServed(schema, count)`, `BakedFieldLoaded(path)`.

---

## Cross-Context Invariants (system-wide)

- **I1 — Phase integrity:** complex CSI used for training must originate from an I/Q source; amplitude-only data is structurally barred from phase-loss training (it carries `phase=None`).
- **I2 — Camera-free at inference:** the optimization target is RF only; camera depth may *initialize* means (hybrid) but is never required to render or query the field.
- **I3 — Edge is inference-only:** the baked field is immutable at the edge; ESP32 nodes relay CSI and never train.
- **I4 — License purity:** only BSD-3/Apache-2.0 code enters the tree (ADR-125 N1/AC8).
- **I5 — Privacy-by-default:** raw CSI and trained fields stay on-prem unless the operator opts in (ADR-031 alignment).

## Context Map

```
[CSI Acquisition] --MeasurementBatchReady--> [Field Optimization] --uses--> [Radiance Field]
        ^                                            |
        | RoomChanged (re-fit)                       | FieldConverged / FieldExported
        |                                            v
[csi_pipeline.identify_location]            [Field Serving] --SplatsServed--> [Viewer / Edge query]
                                                     ^
                              /api/cloud (hybrid init) ─┘
```

## Related
- [ADR-125](../adr/ADR-125-rf-gaussian-splatting-room-reconstruction.md) — RFGS decision record (SPARC)
- [Sensing Server Domain Model](sensing-server-domain-model.md) — owns `/api/splats`, Visualization context
- [RuvSense Domain Model](ruvsense-domain-model.md) — multistatic geometry, phase alignment, field model reuse
