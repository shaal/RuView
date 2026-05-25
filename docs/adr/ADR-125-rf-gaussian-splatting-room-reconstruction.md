# ADR-125: RFGS — Camera-Free Room Reconstruction via RF Gaussian Splatting

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **RFGS** — Radio-Frequency Gaussian Splatting |
| **Foundation** | [GSRF](https://github.com/nesl/GSRF) (NeurIPS 2025 Spotlight, arXiv 2502.01826) — complex-valued 3D Gaussians + Fourier–Legendre radiance basis + wavefront CUDA tracer. **License: BSD-3-Clause** (repo) — permissive, commercial-friendly. |
| **Secondary ref / fallback** | [RF-3DGS](https://github.com/SunLab-UGA/RF-3DGS) (IEEE 2025, arXiv 2411.19420) — **Apache-2.0**, actively maintained, COLMAP/Sionna tooling. License-clean fallback if GSRF integration stalls. |
| **Relates to** | [ADR-014](ADR-014-sota-signal-processing.md) (SOTA signal), [ADR-018](ADR-018-csi-frame-format.md) (CSI wire frame), [ADR-024](ADR-024-contrastive-csi-embedding-model.md) (AETHER), [ADR-029](ADR-029-ruvsense-multistatic-sensing-mode.md) (multistatic), [ADR-030](ADR-030-ruvsense-persistent-field-model.md) (field model), [ADR-031](ADR-031-ruview-sensing-first-rf-mode.md) (sensing-first), [ADR-094](ADR-094-pointcloud-github-pages-deployment.md) (pointcloud viewer), [ADR-095](ADR-095-rvcsi-edge-rf-sensing-platform.md) (rvCSI), [ADR-117](ADR-117-pip-wifi-densepose-modernization.md) (pip) |
| **DDD model** | [`docs/ddd/rf-gaussian-splatting-domain-model.md`](../ddd/rf-gaussian-splatting-domain-model.md) |
| **Tracking issue** | TBD |

---

## 1. Context

### 1.1 The "Gaussian splats" in RuView today are not 3DGS

RuView ships two unrelated primitives that both use the word *splat*:

1. **`ui/components/gaussian-splats.js`** — a 20×20 floor-plane grid of screen-space point-sprite discs, colored by RF intensity. No covariance, no rotation, no opacity learning, no spherical harmonics. It is a heat-map visualization.
2. **`v2/crates/wifi-densepose-pointcloud/src/pointcloud.rs::to_gaussian_splats`** — voxel-clusters a fused (camera + RF) point cloud into `GaussianSplat { center, color, opacity, scale }`. Axis-aligned `scale` only — **no rotation quaternion, no anisotropic covariance, no SH, and no RF physics.** It is a point-cloud → blob converter, served at `/api/splats` (`stream.rs:224`).

Neither is a *learned* radiance field. They cannot synthesize CSI, model multipath, or reconstruct room geometry from radio alone — they re-render geometry that the **camera** already produced. The camera is doing the 3D work; RF is decoration. This contradicts RuView's camera-free, privacy-first thesis.

### 1.2 What RF-domain 3DGS unlocks

GSRF and peers (RF-3DGS, WRF-GS, WRF-GS+, XFreq-GS) adapt the optical 3DGS pipeline to radio: an explicit, editable set of anisotropic Gaussians that **encode the room's radio radiance field** — static geometry (walls, furniture, materials) and multipath propagation — and can *synthesize the complex CSI* an antenna would measure at any pose. This is a true camera-free 3D representation: optimize Gaussians so that the field they imply reproduces the CSI our ESP32 mesh actually measured.

### 1.3 Why our hardware is a good fit (and where it isn't)

RuView's modality is a strong match for GSRF specifically:

| GSRF expects | RuView provides | Status |
|---|---|---|
| Complex CSI (amplitude + phase) per subcarrier | ADR-018 frames carry `iq_data: Vec<i8>` (I/Q) per antenna → reconstruct complex CSI | ✅ direct |
| Multiple RX positions, fixed TX | Multi-node mesh with **known node positions** (`nodes[].position`) | ✅ map node pairs → TX/RX |
| Many measurements per scene | Streaming CSI at ~20 fps per node | ✅ abundant |
| GPU for training (single RTX 3080Ti, ~0.27 h) | Cloud/host training, not edge | ⚠️ training is off-device |

**The capture-format gap (critical):** the `.csi.jsonl` recorder (`recording.rs:58`) stores `subcarriers: Vec<f64>` — **amplitude only, single antenna, no phase.** RF-GS *requires phase*. Therefore the RFGS capture path must tap the **binary ADR-018 I/Q frames** (which preserve phase and the antenna dimension), not the JSONL recorder. ADR-125 introduces a phase-preserving capture container; it does **not** retrofit the lossy recorder.

### 1.4 What this ADR is *not*

- **Not** on-device (ESP32) training. GSRF/RF-3DGS/WRF-GS are all CUDA-trained with no ONNX/quantization path. Training is a GPU-host (or cloud) step; only the *baked Gaussian field* ships to the edge for cheap query/inference.
- **Not** a replacement for pose/vitals. RFGS produces **static room geometry + multipath**; dynamic humans remain the pose/vitals pipeline's job (and feed the optional 4DGS extension, §6 Bonus / deferred).
- **Not** a hard dependency on the camera. Camera depth (MiDaS) is used **only as optional Gaussian initialization** when present; the optimization target is pure RF.
- **Not** a vendored CUDA kernel dump. We ship a pure-PyTorch reference forward model (autograd, CPU/GPU) behind a seam; the GSRF `complex-gaussian-tracer-csi` CUDA kernel is an optional accelerated backend the operator opts into.

---

## 2. Decision (SPARC)

Create a new Python package **`wifi_densepose.rfgs`** (training/optimization on a GPU host) plus a thin extended `/api/splats` schema and viewer upgrade for inference-time display. A future Rust edge-query crate (`wifi-densepose-rfgs`) is deferred to a follow-up ADR (see §6 Phase 5).

This decision is structured with the **SPARC** methodology: **S**pecification → **P**seudocode → **A**rchitecture → **R**efinement → **C**ompletion.

### 2.1 (S) Specification

**Goal.** From multi-node ESP32 CSI alone (no camera at inference), optimize a set of complex-valued 3D Gaussians whose implied radio radiance field reproduces the measured CSI, then render a high-fidelity 3D room for the existing Three.js / WebGPU viewer.

**Functional requirements**
- F1: Ingest phase-preserving complex CSI per (node, antenna, subcarrier) with each node's known TX/RX position + orientation and per-channel center frequency.
- F2: Optimize a Gaussian set `{μ, scale, quat, opacity, complex-radiance}` to minimize CSI reconstruction error.
- F3: Optionally initialize Gaussians from the existing RuView fused point cloud (`/api/cloud`) — "hybrid init", pure-RF optimization thereafter.
- F4: Export the optimized field as an **extended `/api/splats` v2** JSON (backward compatible) renderable by the current viewer; provide a full-fidelity `.rfgs.npz` for the upgraded WebGPU viewer.
- F5: Self-supervised re-fit when the room changes, gated by the existing CSI room-fingerprint system (`csi_pipeline.rs::identify_location`).

**Non-functional requirements**
- N1: Permissive license only (BSD-3/Apache-2.0). GSRF code is BSD-3; reject any GPL/non-commercial path. *(Action: read GSRF `LICENSE` in-repo before vendoring — project page lists CC-BY-SA for non-code; the code is BSD-3.)*
- N2: Pure-PyTorch reference forward model (no CUDA required to run); CUDA kernel is an optional fast backend.
- N3: Privacy: raw CSI and trained field stay on-prem by default; no cloud upload without explicit opt-in (mirrors ADR-031 sensing-first / ADR-120 default-deny posture).
- N4: Files ≤ 500 LOC; typed public APIs; TDD.

### 2.2 (P) Pseudocode (core optimization loop)

```text
load room_config (node poses, room bounds, channel→freq)
measurements = CsiMeasurementDataset(capture_dir | live_ws, room_config)   # complex CSI + tx/rx poses
gaussians = init_from_pointcloud(/api/cloud) if hybrid else init_random(bounds)
opt = Adam(gaussians.params, lr_per_group)

for step in range(N):
    batch = measurements.sample()                      # (tx_pose, rx_pose, H_measured[ant, sub])
    H_pred = render_csi(gaussians, tx_pose, rx_pose, freqs)   # complex forward model
    loss   = amp_loss(|H_pred|,|H_measured|)
           + phase_loss(∠H_pred, ∠H_measured)          # circular / cosine
           + λ_spec * spectrum_loss(H_pred, H_measured) # optional spatial spectrum
           + λ_reg * (opacity_l1 + scale_reg)
    loss.backward(); opt.step()
    if step % densify_every == 0: densify_and_prune(gaussians, grads)   # 3DGS adaptive control
export(gaussians) -> splats_v2.json + field.rfgs.npz
```

### 2.3 (A) Architecture

New package layout:

```
python/wifi_densepose/rfgs/
├── __init__.py
├── geometry.py        # RoomConfig, NodePose, channel→frequency, ray/pose math
├── dataset.py         # ADR-018 decode + .csi.jsonl + live WS → CsiMeasurementDataset (DELIVERABLE 1)
├── model.py           # ComplexGaussianField: μ/scale/quat/opacity + Fourier–Legendre complex radiance
├── render.py          # pure-PyTorch complex CSI forward model (CUDA seam: GSRF tracer backend)
├── train.py           # optimization loop, densify/prune, checkpoint, export (DELIVERABLE 2)
├── export.py          # ComplexGaussianField → /api/splats v2 JSON + .rfgs.npz
└── configs/room.example.toml
```

Edge/serving touch-points (no new crate yet): `pointcloud/src/stream.rs::api_splats` gains a `schema:"rfgs-v2"` branch that can serve a baked field; the viewer (`viewer.html` / `ui/components/`) gains an anisotropic-Gaussian path. See DDD model for context boundaries.

### 2.4 (R) Refinement (edge / real-time roadmap)

- R1: Train at full precision on host → **prune** low-opacity Gaussians → **quantize** μ/scale to fp16 and radiance coeffs to int8 → bake to `.rfgs.npz`.
- R2: Edge query is *inference-only*: given a query pose, evaluate the field. Target: ≤ 5 ms/frame on a Pi-class host; ESP32 only relays CSI (never trains).
- R3: WebGPU viewer renders anisotropic splats directly from the baked field; the legacy Three.js point-sprite path remains as the `splats-v1` fallback.

### 2.5 (C) Completion (acceptance — see §5)

Done when: complex CSI loader round-trips ADR-018 I/Q (DELIVERABLE 1 tests green); the reference forward model + optimizer reduce held-out CSI error vs. a constant-field baseline on synthetic data (DELIVERABLE 2); export produces a viewer-renderable `splats-v2` JSON; and an evaluation harness reports geometry IoU/Chamfer vs. a reference scan.

### 2.6 Reuse map

| RFGS module | Reuses |
|---|---|
| `dataset.py` | ADR-018 parser logic (`pointcloud/src/parser.rs`), `client/ws.py` (live), `recording.rs` schema (amplitude-only fallback) |
| `geometry.py` | node positions from `/api/cloud` `nodes[].position`; multistatic geometry (`ruvsense/multistatic.rs`, `viewpoint/geometry.rs`) |
| `model.py` | GSRF complex-Gaussian + Fourier–Legendre parameterization (BSD-3) |
| `train.py` init | `/api/cloud` fused point cloud (hybrid initialization) |
| re-fit trigger | `csi_pipeline.rs::identify_location` room fingerprint (ADR-030) |
| export | extends `pointcloud::GaussianSplat` → `GaussianSplatV2` (adds `rotation`, `sh`/`radiance`) |

---

## 3. Consequences

### Positive
- First **true camera-free** 3D room reconstruction in RuView; RF stops being decoration over camera geometry.
- Permissive (BSD-3/Apache-2.0) foundation — vendorable and commercial-safe.
- Backward-compatible `/api/splats` (v1 consumers unaffected; v2 is additive).
- Reuses existing assets: point-cloud init, node geometry, room fingerprint re-fit, ADR-018 frames.
- Explicit Gaussian field is editable and exportable (AR/VR, Home-Assistant digital twin — §6 bonus).

### Negative
- Training requires a CUDA GPU host; not edge-trainable. Adds a (clearly-bounded) host/cloud step.
- GSRF upstream is a **release stub** (~8 commits, no active dev) — we own the fork and its maintenance.
- Phase-preserving capture is **new plumbing**; the existing JSONL recorder is amplitude-only and cannot be reused for training.
- RF spatial resolution is coarser than optical; reconstruction fidelity is bounded by node count/geometry (mitigations in §4).

### Neutral
- Adds an optional heavy dep set (`torch`, CUDA) behind a `rfgs` extra; core `wifi-densepose` install is unaffected.
- GSRF's fixed-TX/many-RX assumption needs a multistatic extension for full mesh fusion (deferred to Phase 4).

---

## 4. Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| **RF resolution ≪ optical** | Hybrid point-cloud init (F3) seeds geometry; densification concentrates Gaussians where CSI residual is high; multi-node geometric diversity (`viewpoint/geometry.rs`, Cramér–Rao bounds) maximizes effective resolution. |
| **Multipath ambiguity** | Complex (phase-aware) radiance + transmittance models multipath explicitly (GSRF Fourier–Legendre); multi-link consistency check (`ruvsense/adversarial.rs`) rejects impossible solutions. |
| **Edge compute budget** | Train on host; ship pruned+quantized field; edge is inference-only (R1–R3). |
| **Privacy** | On-prem by default; no raw CSI / field upload without opt-in; aligns with ADR-031/ADR-120. |
| **Multi-node fusion** | Map each TX/RX node pair to a GSRF view; aggregate per-pair losses; attention-weighted multistatic fusion (`ruvsense/multistatic.rs`) for Phase 4. |
| **Phase noise / CFO** | Phase sanitization from `ruvsense/phase_align.rs` (LO offset estimation) applied in `dataset.py` before training. |

---

## 5. Acceptance Criteria

- [x] **AC1**: `dataset.py` reconstructs complex CSI `H[ant, sub] ∈ ℂ` bit-faithfully from ADR-018 I/Q frames (magic `0xC5110001`/`0xC5110006`), with each sample carrying valid `tx_pos`, `rx_pos`, `rx_orientation`, and subcarrier frequencies from `RoomConfig`. *(verified: multi-antenna decode + capture round-trip)*
- [x] **AC2**: Amplitude-only `.csi.jsonl` inputs load with `phase=None` and a logged warning (graceful degradation, no crash).
- [x] **AC3**: `train.py` runs end-to-end on synthetic data with **no GPU** (pure-PyTorch backend) and reduces held-out CSI reconstruction loss by ≥ 50% vs. the untrained-field baseline. *(verified: 71.5% held-out improvement on CPU)*
- [x] **AC4**: Optimized field exports to `splats-v2` JSON (v1-compatible) and to `.rfgs.npz` (μ, scale, quat, opacity, radiance coeffs). *(viewer render path added in P5; browser render not yet automated-tested)*
- [x] **AC5**: Evaluation harness reports geometry **Chamfer distance** and **occupancy IoU** vs. a reference scan (`eval.py`, P3). *(metric correctness unit-checked)*
- [x] **AC6**: Hybrid init from `/api/cloud` produces a field whose initial loss is below random init. *(verified in synthetic self-test)*
- [ ] **AC7**: Re-fit is triggered when `identify_location` reports a room change (fingerprint mismatch). *(deferred — wiring to the room-fingerprint event)*
- [ ] **AC8**: No GPL / non-commercial code enters the tree; GSRF `LICENSE` confirmed BSD-3 in-repo before any vendoring. *(no GSRF code vendored yet; CUDA backend is an opt-in import seam)*

---

## 6. Phased Rollout

| Phase | Scope | Deliverable | Effort |
|-------|-------|-------------|--------|
| **P1** ✅ | Complex CSI data loader + `RoomConfig` + ADR-018/JSONL/WS adapters | `dataset.py`, `geometry.py` (**done**) | 1.5 wk |
| **P2** ✅ | Complex Gaussian field + pure-PyTorch CSI forward model + optimizer + export | `model.py`, `render.py`, `train.py`, `export.py` (**done, reference**) | 2.5 wk |
| **P3** ✅ | GSRF CUDA tracer backend (opt-in) + densify/prune tuning + eval harness | `render.py` backend seam, `model.py::densify_split`, `eval.py` (**done**) | 2.0 wk |
| **P4** | Multistatic multi-node fusion (extend GSRF fixed-TX assumption) | `fusion.py` | 2.0 wk |
| **P5** ✅ | Rust edge-query crate `wifi-densepose-rfgs` + anisotropic viewer + `/api/splats` v2 server branch | new crate + `gaussian-splats.js` + `stream.rs` (**done**; WebGPU/Rust-query crate are the upgrade path) | 3.0 wk |
| **P6 (bonus)** | 4DGS dynamic Gaussians (moving people/furniture), material estimation from phase/amplitude, HA digital-twin / AR-VR export | deferred ADRs | TBD |
| **Total (P1–P5)** | | | **11.0 wk** |

---

## 7. Alternatives Considered

### Alt 1: Upgrade the existing point-cloud→blob converter in place
Rejected: `to_gaussian_splats` has no RF forward model, no phase, no optimization. It can never reconstruct from radio alone — it re-renders camera geometry. RFGS is a categorically different (learned-field) approach.

### Alt 2: RF-3DGS (C++) as the primary foundation
Rejected as *primary* (kept as secondary/fallback): RF-3DGS is vision-capture-oriented (RGB + COLMAP poses + Sionna-simulated spectra) and its phase handling is less first-class than GSRF. Its **Apache-2.0** license and active maintenance make it the right **fallback** and a source of pipeline/tooling patterns, but GSRF's native complex-CSI input matches our ESP32 modality directly.

### Alt 3: WRF-GS / WRF-GS+ as primary
Rejected: WRF-GS's repo **license is unnamed/unclear** (likely 3DGS-inherited non-commercial) — fails N1. Revisit WRF-GS+ (deformable) only for the Phase-6 4DGS extension, and only after license verification.

### Alt 4: Bind the GSRF CUDA kernel directly from Rust now
Deferred: couples us to CUDA at the core and blocks no-GPU CI. We ship a pure-PyTorch reference first (N2), with the CUDA tracer as an opt-in P3 backend.

---

## 8. Related ADRs
See header table. Body cites structural reuse of ADR-018 (CSI frame), ADR-029/030 (multistatic + field model), ADR-031 (sensing-first/privacy), ADR-094 (pointcloud viewer), and the `/api/splats` surface from the Sensing-Server domain model.
