"""RFGS geometry & scene configuration (ADR-125, CSI Acquisition context).

Immutable scene description used by the CSI data loader and the forward
model: node poses (TX/RX geometry), room bounds, and the channel ->
subcarrier-frequency mapping needed to turn a WiFi channel into the
per-subcarrier center frequencies that the phase term of the RF forward
model depends on.

No torch import here -- this module is pure stdlib so it can be used by
tooling that does not have the ``rfgs`` extra installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

SPEED_OF_LIGHT = 299_792_458.0  # m/s

# ADR-018 binary CSI frame constants (mirror of pointcloud/src/parser.rs).
CSI_MAGIC_V1 = 0xC511_0001  # raw CSI
CSI_MAGIC_V6 = 0xC511_0006  # feature state
CSI_HEADER_SIZE = 20


def channel_center_freq_hz(channel: int) -> float:
    """Center frequency of a 2.4 GHz or 5 GHz WiFi channel, in Hz.

    2.4 GHz: ch 1..13 = 2412 + 5*(ch-1) MHz, ch 14 = 2484 MHz.
    5 GHz:   freq = 5000 + 5*ch MHz (ch >= 32).
    """
    if 1 <= channel <= 13:
        mhz = 2412 + 5 * (channel - 1)
    elif channel == 14:
        mhz = 2484
    elif channel >= 32:
        mhz = 5000 + 5 * channel
    else:
        raise ValueError(f"unsupported WiFi channel: {channel}")
    return mhz * 1e6


def subcarrier_frequencies(
    channel: int, n_subcarriers: int, bandwidth_mhz: float = 20.0
) -> list[float]:
    """Per-subcarrier center frequencies (Hz) for an OFDM channel.

    Subcarriers are spread symmetrically around the channel center with
    spacing ``bandwidth / n_subcarriers``. This is the frequency axis the
    forward model uses to compute the per-subcarrier phase ``exp(-j 2pi f tau)``.
    """
    if n_subcarriers <= 0:
        raise ValueError("n_subcarriers must be positive")
    f0 = channel_center_freq_hz(channel)
    spacing = (bandwidth_mhz * 1e6) / n_subcarriers
    half = (n_subcarriers - 1) / 2.0
    return [f0 + (k - half) * spacing for k in range(n_subcarriers)]


@dataclass(frozen=True)
class Pose:
    """Immutable pose: position (m) + unit-quaternion orientation (w,x,y,z)."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        n = math.sqrt(sum(c * c for c in self.orientation))
        if n < 1e-9:
            raise ValueError("pose orientation quaternion is degenerate")
        # Normalize without mutating frozen fields directly.
        object.__setattr__(
            self, "orientation", tuple(c / n for c in self.orientation)
        )


@dataclass(frozen=True)
class NodePose:
    """A sensing node's known pose + the channel it operates on."""

    node_id: int
    pose: Pose
    channel: int = 6
    n_antennas: int = 1


@dataclass(frozen=True)
class RoomConfig:
    """Immutable scene description bound to a CsiMeasurementDataset.

    ``tx_node_id`` selects the transmitter; every other node is treated as a
    receiver (GSRF's fixed-TX / many-RX assumption). For a full multistatic
    mesh, Phase 4 (ADR-125) iterates ``tx_node_id`` over all nodes.
    """

    nodes: Mapping[int, NodePose]
    bounds_min: tuple[float, float, float] = (-5.0, 0.0, -5.0)
    bounds_max: tuple[float, float, float] = (5.0, 3.0, 5.0)
    bandwidth_mhz: float = 20.0
    tx_node_id: int = 0

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("RoomConfig requires at least one node")
        if self.tx_node_id not in self.nodes:
            raise ValueError(
                f"tx_node_id {self.tx_node_id} not present in nodes {list(self.nodes)}"
            )

    @property
    def tx_pose(self) -> Pose:
        return self.nodes[self.tx_node_id].pose

    def rx_node_ids(self) -> list[int]:
        return [nid for nid in self.nodes if nid != self.tx_node_id]

    def freqs_for_node(self, node_id: int, n_subcarriers: int) -> list[float]:
        node = self.nodes[node_id]
        return subcarrier_frequencies(node.channel, n_subcarriers, self.bandwidth_mhz)

    @staticmethod
    def from_dict(d: Mapping) -> "RoomConfig":
        nodes: dict[int, NodePose] = {}
        for raw in d["nodes"]:
            nid = int(raw["node_id"])
            pos = tuple(float(x) for x in raw["position"])  # type: ignore[assignment]
            orient = tuple(
                float(x) for x in raw.get("orientation", (1.0, 0.0, 0.0, 0.0))
            )
            nodes[nid] = NodePose(
                node_id=nid,
                pose=Pose(position=pos, orientation=orient),  # type: ignore[arg-type]
                channel=int(raw.get("channel", 6)),
                n_antennas=int(raw.get("n_antennas", 1)),
            )
        return RoomConfig(
            nodes=nodes,
            bounds_min=tuple(d.get("bounds_min", (-5.0, 0.0, -5.0))),  # type: ignore[arg-type]
            bounds_max=tuple(d.get("bounds_max", (5.0, 3.0, 5.0))),  # type: ignore[arg-type]
            bandwidth_mhz=float(d.get("bandwidth_mhz", 20.0)),
            tx_node_id=int(d.get("tx_node_id", next(iter(nodes)))),
        )

    @staticmethod
    def load(path: str | Path) -> "RoomConfig":
        """Load a RoomConfig from a TOML file (stdlib ``tomllib``)."""
        import tomllib  # py3.11+

        with open(path, "rb") as f:
            return RoomConfig.from_dict(tomllib.load(f))
