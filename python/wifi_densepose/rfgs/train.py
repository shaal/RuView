"""RFGS optimization loop (ADR-125 Deliverable 2, Field Optimization context).

Optimizes a ``ComplexGaussianField`` so the CSI it renders reproduces the CSI
measured by the ESP32 mesh, then exports a viewer-renderable field. Runs on CPU
with the pure-PyTorch reference backend (ADR-125 AC3) -- no GPU needed.

Usage:
    # Train from a capture directory against a room config:
    python -m wifi_densepose.rfgs.train \\
        --room-config configs/room.example.toml \\
        --capture-dir ./captures --steps 2000 --out ./out

    # Self-test on synthetic data (no hardware, no GPU) -- proves AC3:
    python -m wifi_densepose.rfgs.train --synthetic --steps 800 --out /tmp/rfgs
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from .dataset import CsiMeasurement, CsiMeasurementDataset
from .export import write_baked_npz, write_splats_v2_json
from .geometry import NodePose, Pose, RoomConfig, subcarrier_frequencies
from .model import ComplexGaussianField
from .render import ReferenceTorchTracer, reconstruction_loss

log = logging.getLogger("rfgs.train")


# ─── Synthetic scene (for AC3 self-test) ─────────────────────────────────────


def _synthetic_room(n_rx: int = 16) -> RoomConfig:
    """1 fixed TX + ``n_rx`` RX nodes on a ring (many views constrain the field)."""
    import math
    nodes = {0: NodePose(0, Pose((0.0, 1.5, 0.0)), channel=6)}  # TX at center
    for i in range(n_rx):
        a = 2 * math.pi * i / n_rx
        nodes[i + 1] = NodePose(
            i + 1, Pose((2.4 * math.cos(a), 1.0 + 0.3 * (i % 3), 2.4 * math.sin(a))),
            channel=6)
    return RoomConfig(nodes=nodes, bounds_min=(-3, 0, -3), bounds_max=(3, 3, 3),
                      tx_node_id=0)


def _synthetic_dataset(room: RoomConfig, n_sub: int = 52, device: str = "cpu",
                       seed: int = 0) -> tuple[CsiMeasurementDataset, torch.Tensor]:
    """Render a hidden ground-truth field, snapshot it as measurements.

    Returns the dataset and the ground-truth Gaussian means -- the latter
    stands in for the RuView point cloud used to seed positions (hybrid init,
    ADR-125 F3/AC6). Recovering Gaussian *positions* from absolute CSI is a
    hard non-convex inverse problem (oscillatory carrier phase) reserved for
    the GSRF CUDA tracer + good init; the CPU reference self-test demonstrates
    the differentiable pipeline by recovering the complex *radiance* given
    point-cloud-seeded geometry.
    """
    torch.manual_seed(seed)
    gt = ComplexGaussianField.init_random(25, room.bounds_min, room.bounds_max,
                                          device=device, seed=seed)
    # Give the GT distinct, structured radiance so there is real signal to fit.
    with torch.no_grad():
        gt.radiance.copy_(torch.randn_like(gt.radiance))
        gt.opacity_logit.copy_(torch.randn_like(gt.opacity_logit))
    tracer = ReferenceTorchTracer()
    tx = torch.tensor(room.tx_pose.position)
    meas: list[CsiMeasurement] = []
    for nid in room.rx_node_ids():
        node = room.nodes[nid]
        freqs = torch.tensor(subcarrier_frequencies(node.channel, n_sub))
        with torch.no_grad():
            h = tracer.render(gt, tx, torch.tensor(node.pose.position), freqs, 1)
            h = h + 0.02 * torch.randn_like(h)  # measurement noise
        meas.append(CsiMeasurement(
            h=h, tx_position=room.tx_pose.position, rx_position=node.pose.position,
            rx_orientation=node.pose.orientation, freqs_hz=freqs,
            timestamp_s=0.0, node_id=nid, has_phase=True))
    return CsiMeasurementDataset(meas, room), gt.mu.detach().clone()


# ─── Hybrid init from /api/cloud ─────────────────────────────────────────────


def _points_from_cloud(url: str, device: str) -> Optional[torch.Tensor]:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as r:  # nosec - local loopback
            import json
            data = json.loads(r.read())
        pts = [[p["x"], p["y"], p["z"]] for p in data.get("cloud", [])]
        if not pts:
            return None
        return torch.tensor(pts, dtype=torch.float32, device=device)
    except Exception as e:  # pragma: no cover - network optional
        log.warning("hybrid init unavailable (%s); falling back to random", e)
        return None


# ─── Trainer ─────────────────────────────────────────────────────────────────


def train(
    dataset: CsiMeasurementDataset,
    *,
    steps: int = 2000,
    n_gaussians: int = 200,
    lr: float = 1e-2,
    device: str = "cpu",
    init_points: Optional[torch.Tensor] = None,
    densify_every: int = 300,
    holdout: float = 0.15,
    log_every: int = 100,
) -> tuple[ComplexGaussianField, dict]:
    room = dataset.room
    if init_points is not None:
        field = ComplexGaussianField.init_from_pointcloud(init_points, device=device)
        log.info("hybrid init: %d Gaussians from point cloud", field.num_gaussians)
    else:
        field = ComplexGaussianField.init_random(
            n_gaussians, room.bounds_min, room.bounds_max, device=device)
        log.info("random init: %d Gaussians", field.num_gaussians)

    tracer = ReferenceTorchTracer()
    train_idx, hold_idx = dataset.split(holdout)
    opt = torch.optim.Adam(field.parameters(), lr=lr)

    def batch_loss(indices: list[int], grad: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            total = torch.zeros((), device=device)
            for i in indices:
                m = dataset[i]
                h = tracer.render(field, torch.tensor(m.tx_position),
                                  torch.tensor(m.rx_position), m.freqs_hz,
                                  m.h.shape[0])
                total = total + reconstruction_loss(
                    h, m.h, use_phase=m.has_phase)["total"]
            return total / max(1, len(indices))

    base_train = float(batch_loss(train_idx, grad=False).detach())
    base_hold = float(batch_loss(hold_idx, grad=False).detach())
    history = {"baseline_train": base_train, "baseline_holdout": base_hold,
               "steps": []}
    log.info("baseline (untrained field): train=%.5f  held-out=%.5f",
             base_train, base_hold)

    for step in range(steps):
        opt.zero_grad()
        loss = batch_loss(train_idx, grad=True)   # full-batch GD (stable convergence)
        loss.backward()
        grad_norm = (field.mu.grad.norm(dim=-1).detach()
                     if field.mu.grad is not None
                     else torch.zeros(field.num_gaussians, device=device))
        opt.step()

        if densify_every and step > 0 and step % densify_every == 0:
            removed = field.prune(min_opacity=0.005)
            added = field.densify_clone(grad_norm, threshold=grad_norm.mean().item() * 2)
            if removed or added:
                opt = torch.optim.Adam(field.parameters(), lr=lr)  # params changed
                log.info("step %d densify: +%d/-%d -> %d Gaussians",
                         step, added, removed, field.num_gaussians)

        if step % log_every == 0:
            tl = float(loss.detach())
            history["steps"].append({"step": step, "train": tl})
            log.info("step %4d  train_loss=%.5f", step, tl)

    final_train = float(batch_loss(train_idx, grad=False).detach())
    final_hold = float(batch_loss(hold_idx, grad=False).detach())
    history["final_train"] = final_train
    history["final_holdout"] = final_hold
    history["improvement"] = 1.0 - (final_train / base_train if base_train > 0 else 1.0)
    history["holdout_improvement"] = (
        1.0 - (final_hold / base_hold if base_hold > 0 else 1.0))
    log.info("final: train=%.5f (%.1f%% better)  held-out=%.5f (%.1f%% better)",
             final_train, 100 * history["improvement"],
             final_hold, 100 * history["holdout_improvement"])
    return field, history


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RFGS: optimize an RF Gaussian field from CSI")
    ap.add_argument("--room-config", type=str, help="RoomConfig TOML")
    ap.add_argument("--capture-dir", type=str, help="dir of *.adr018 / *.csi.jsonl")
    ap.add_argument("--synthetic", action="store_true", help="self-test on synthetic data")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--gaussians", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--init", choices=["random", "pointcloud"], default="random")
    ap.add_argument("--cloud-url", type=str, default="http://127.0.0.1:9880/api/cloud")
    ap.add_argument("--out", type=str, default="./rfgs_out")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()

    synthetic_points = None
    if args.synthetic:
        room = _synthetic_room()
        dataset, synthetic_points = _synthetic_dataset(room, device=args.device)
        # Seed positions from the (synthetic) point cloud with jitter -> hybrid
        # init (F3/AC6). The optimizer then recovers the complex radiance.
        synthetic_points = synthetic_points + 0.05 * torch.randn_like(synthetic_points)
        log.info("synthetic dataset: %d measurements", len(dataset))
    else:
        if not (args.room_config and args.capture_dir):
            ap.error("--room-config and --capture-dir are required (or use --synthetic)")
        room = RoomConfig.load(args.room_config)
        dataset = CsiMeasurementDataset.from_capture_dir(
            args.capture_dir, room, device=args.device)
        log.info("loaded %d measurements from %s", len(dataset), args.capture_dir)

    init_points = synthetic_points
    if args.init == "pointcloud" and not args.synthetic:
        init_points = _points_from_cloud(args.cloud_url, args.device)

    field, history = train(
        dataset, steps=args.steps, n_gaussians=args.gaussians, lr=args.lr,
        device=args.device, init_points=init_points)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_splats_v2_json(field, out / "splats_v2.json")
    write_baked_npz(field, out / "field.rfgs.npz")
    (out / "history.json").write_text(__import__("json").dumps(history, indent=2))
    log.info("exported -> %s  (%.1fs, %d Gaussians)",
             out, time.time() - t0, field.num_gaussians)

    # AC3 gate: held-out CSI reconstruction loss (RX poses the optimizer never
    # trained on) must improve >= 50% vs. the untrained-field baseline -- the
    # field generalizes, not just memorizes.
    if args.synthetic and history["holdout_improvement"] < 0.5:
        log.error("AC3 FAIL: held-out improvement %.1f%% < 50%%",
                  100 * history["holdout_improvement"])
        return 1
    if args.synthetic:
        log.info("AC3 PASS: held-out improvement %.1f%% >= 50%%",
                 100 * history["holdout_improvement"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
