"""RFGS CSI -> RF-GS data loader (ADR-125 Deliverable 1, CSI Acquisition).

Turns RuView CSI into phase-preserving ``CsiMeasurement`` value objects that
the GSRF-style forward model can train on. The hard constraint (DDD invariant
I1) is that RF-GS *needs phase*: this loader reconstructs complex CSI from the
ADR-018 binary I/Q frames. The ``.csi.jsonl`` recorder is amplitude-only and
therefore supported only as a degraded (``phase=None``) fallback.

Sources supported:
  * ADR-018 binary frames        -> full complex H[ant, sub]  (preferred)
  * ``.csi.jsonl`` recordings     -> amplitude-only, phase=None (warned)
  * live WS stream (optional)     -> see ``stream_live`` (async)

Torch is imported lazily so this module loads without the ``rfgs`` extra; the
``CsiMeasurementDataset`` (a ``torch.utils.data.Dataset``) requires it.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .geometry import (
    CSI_HEADER_SIZE,
    CSI_MAGIC_V1,
    CSI_MAGIC_V6,
    RoomConfig,
)

log = logging.getLogger(__name__)

try:  # lazy: dataset needs torch, plain decoding does not.
    import torch
    from torch import Tensor
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False
    Tensor = object  # type: ignore[assignment,misc]


# ─── Decoded frame (numpy-free, torch-free) ──────────────────────────────────


@dataclass
class DecodedCsiFrame:
    """A single ADR-018 frame decoded to complex CSI.

    ``h`` is a list-of-lists ``[antenna][subcarrier]`` of (re, im) tuples so
    that decoding has no hard torch/numpy dependency. ``CsiMeasurementDataset``
    converts these to complex tensors.
    """

    node_id: int
    n_antennas: int
    n_subcarriers: int
    channel: int
    rssi: int
    noise_floor: int
    timestamp_us: int
    h: list[list[tuple[float, float]]]  # [ant][sub] complex (re, im)


def decode_adr018(data: bytes) -> Optional[DecodedCsiFrame]:
    """Decode one ADR-018 binary CSI frame into complex CSI per antenna.

    Mirrors ``pointcloud/src/parser.rs`` but, crucially, reconstructs the
    **full per-antenna complex** channel response (the Rust hot-path only
    computes amplitude/phase for antenna 0). I/Q layout after the 20-byte
    header is interleaved ``[I0,Q0,I1,Q1,...]`` per subcarrier, repeated per
    antenna. Returns ``None`` on bad magic / truncation (a hot path).
    """
    if len(data) < CSI_HEADER_SIZE:
        return None
    (magic,) = struct.unpack_from("<I", data, 0)
    if magic not in (CSI_MAGIC_V1, CSI_MAGIC_V6):
        return None

    node_id = data[4]
    n_antennas = max(1, data[5])
    (n_subcarriers,) = struct.unpack_from("<H", data, 6)
    channel = data[8]
    rssi = struct.unpack_from("<b", data, 9)[0]
    noise_floor = struct.unpack_from("<b", data, 10)[0]
    (timestamp_us,) = struct.unpack_from("<I", data, 16)

    iq_len = n_subcarriers * 2 * n_antennas
    if len(data) < CSI_HEADER_SIZE + iq_len:
        return None
    iq = struct.unpack_from(f"<{iq_len}b", data, CSI_HEADER_SIZE)

    h: list[list[tuple[float, float]]] = []
    for a in range(n_antennas):
        row: list[tuple[float, float]] = []
        base = a * n_subcarriers * 2
        for s in range(n_subcarriers):
            i_val = float(iq[base + s * 2])
            q_val = float(iq[base + s * 2 + 1])
            row.append((i_val, q_val))
        h.append(row)

    return DecodedCsiFrame(
        node_id=node_id,
        n_antennas=n_antennas,
        n_subcarriers=n_subcarriers,
        channel=channel,
        rssi=rssi,
        noise_floor=noise_floor,
        timestamp_us=timestamp_us,
        h=h,
    )


def iter_adr018_file(path: str | Path) -> Iterable[DecodedCsiFrame]:
    """Iterate length-prefixed ADR-018 frames from a capture dump.

    Capture file format (RFGS convention): a stream of records, each a
    little-endian ``u32`` length followed by that many ADR-018 frame bytes.
    This is what ``ruview-pointcloud csi-test --record`` (P3) writes; for now
    it lets us persist raw I/Q that the lossy ``.csi.jsonl`` recorder drops.
    """
    raw = Path(path).read_bytes()
    off = 0
    while off + 4 <= len(raw):
        (n,) = struct.unpack_from("<I", raw, off)
        off += 4
        if off + n > len(raw):
            break
        frame = decode_adr018(raw[off : off + n])
        off += n
        if frame is not None:
            yield frame


# ─── CsiMeasurement value object ─────────────────────────────────────────────


@dataclass(frozen=True)
class CsiMeasurement:
    """A training datum: complex CSI + TX/RX geometry + frequency axis.

    ``h`` is a complex tensor ``[n_antennas, n_subcarriers]``. ``has_phase`` is
    False for amplitude-only sources (``.csi.jsonl``); such samples are
    excluded from the phase loss (DDD invariant I1).
    """

    h: "Tensor"  # complex64 [n_ant, n_sub]
    tx_position: tuple[float, float, float]
    rx_position: tuple[float, float, float]
    rx_orientation: tuple[float, float, float, float]
    freqs_hz: "Tensor"  # [n_sub]
    timestamp_s: float
    node_id: int
    has_phase: bool = True


def sanitize_phase(h: "Tensor") -> "Tensor":
    """Remove a per-antenna linear phase ramp across subcarriers (CFO/LO).

    A first-order detrend of the unwrapped phase per antenna — the cheap
    analogue of ``ruvsense/phase_align``. Keeps multipath structure while
    removing the bulk offset/slope that would otherwise dominate the loss.
    """
    amp = h.abs()
    phase = torch.angle(h)
    n_sub = phase.shape[-1]
    k = torch.arange(n_sub, dtype=phase.dtype, device=phase.device)
    k = k - k.mean()
    denom = (k * k).sum().clamp_min(1e-9)
    # Per-antenna slope & mean via least-squares on the (wrapped) phase.
    slope = (phase * k).sum(dim=-1, keepdim=True) / denom
    mean = phase.mean(dim=-1, keepdim=True)
    detrended = phase - (slope * k + mean)
    return torch.polar(amp, detrended)


# ─── Dataset ─────────────────────────────────────────────────────────────────


class CsiMeasurementDataset:  # Dataset-compatible (len/getitem); no inherit, keeps import torch-free
    """Aggregate root: CSI measurements bound to one ``RoomConfig``.

    Build from a directory of ADR-018 captures (``*.adr018``) and/or
    ``*.csi.jsonl`` recordings. Each decoded frame becomes one
    ``CsiMeasurement`` whose TX pose is the room's fixed transmitter and whose
    RX pose is the measuring node.
    """

    def __init__(
        self,
        measurements: Sequence[CsiMeasurement],
        room: RoomConfig,
    ) -> None:
        if not _TORCH:
            raise ImportError(
                "CsiMeasurementDataset requires torch. Install with "
                '`pip install "wifi-densepose[rfgs]"`.'
            )
        self.room = room
        self.measurements = list(measurements)

    def __len__(self) -> int:
        return len(self.measurements)

    def __getitem__(self, idx: int) -> CsiMeasurement:
        return self.measurements[idx]

    # -- builders -------------------------------------------------------------

    @classmethod
    def from_frames(
        cls,
        frames: Iterable[DecodedCsiFrame],
        room: RoomConfig,
        *,
        device: str = "cpu",
        do_sanitize_phase: bool = True,
    ) -> "CsiMeasurementDataset":
        tx_pos = room.tx_pose.position
        out: list[CsiMeasurement] = []
        for fr in frames:
            if fr.node_id == room.tx_node_id:
                continue  # the TX does not measure itself
            node = room.nodes.get(fr.node_id)
            if node is None:
                log.warning("frame from unknown node_id=%s; skipping", fr.node_id)
                continue
            h = torch.tensor(fr.h, dtype=torch.float32, device=device)  # [ant,sub,2]
            h = torch.view_as_complex(h.contiguous())  # [ant, sub] complex64
            if do_sanitize_phase:
                h = sanitize_phase(h)
            freqs = torch.tensor(
                room.freqs_for_node(fr.node_id, fr.n_subcarriers),
                dtype=torch.float32,
                device=device,
            )
            out.append(
                CsiMeasurement(
                    h=h,
                    tx_position=tx_pos,
                    rx_position=node.pose.position,
                    rx_orientation=node.pose.orientation,
                    freqs_hz=freqs,
                    timestamp_s=fr.timestamp_us / 1e6,
                    node_id=fr.node_id,
                    has_phase=True,
                )
            )
        if not out:
            raise ValueError("no usable RX frames decoded for this RoomConfig")
        return cls(out, room)

    @classmethod
    def from_capture_dir(
        cls, path: str | Path, room: RoomConfig, *, device: str = "cpu"
    ) -> "CsiMeasurementDataset":
        d = Path(path)
        frames: list[DecodedCsiFrame] = []
        for f in sorted(d.glob("*.adr018")):
            frames.extend(iter_adr018_file(f))
        if frames:
            return cls.from_frames(frames, room, device=device)
        # Fallback: amplitude-only JSONL recordings (phase=None, degraded).
        jsonl = sorted(d.glob("*.csi.jsonl"))
        if not jsonl:
            raise FileNotFoundError(
                f"no *.adr018 or *.csi.jsonl captures in {d}"
            )
        log.warning(
            "no I/Q (*.adr018) captures found; falling back to amplitude-only "
            ".csi.jsonl (phase=None). Phase loss will be disabled (DDD I1)."
        )
        return cls._from_jsonl(jsonl, room, device=device)

    @classmethod
    def _from_jsonl(
        cls, files: Sequence[Path], room: RoomConfig, *, device: str
    ) -> "CsiMeasurementDataset":
        tx_pos = room.tx_pose.position
        rx_ids = room.rx_node_ids()
        rx_id = rx_ids[0] if rx_ids else room.tx_node_id
        node = room.nodes[rx_id]
        out: list[CsiMeasurement] = []
        for fp in files:
            for line in fp.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                amp = rec.get("subcarriers") or []
                if not amp:
                    continue
                n_sub = len(amp)
                # Amplitude-only -> zero-phase complex; flagged has_phase=False.
                h = torch.tensor(amp, dtype=torch.float32, device=device)
                h = torch.complex(h, torch.zeros_like(h)).unsqueeze(0)  # [1, sub]
                freqs = torch.tensor(
                    room.freqs_for_node(rx_id, n_sub),
                    dtype=torch.float32,
                    device=device,
                )
                out.append(
                    CsiMeasurement(
                        h=h,
                        tx_position=tx_pos,
                        rx_position=node.pose.position,
                        rx_orientation=node.pose.orientation,
                        freqs_hz=freqs,
                        timestamp_s=float(rec.get("timestamp", 0.0)),
                        node_id=rx_id,
                        has_phase=False,
                    )
                )
        if not out:
            raise ValueError("no usable rows in .csi.jsonl recordings")
        return cls(out, room)

    def split(self, holdout: float = 0.1) -> tuple[list[int], list[int]]:
        """Deterministic train/held-out index split (last ``holdout`` fraction)."""
        n = len(self.measurements)
        k = max(1, int(n * holdout))
        idx = list(range(n))
        return idx[: n - k], idx[n - k :]
