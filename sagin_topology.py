"""Fixed 84-node Space-Air-Ground Integrated Network (SAGIN) substrate.

The topology is a static three-domain testbed: 12 LEO satellites, 4 HAPs,
8 UAVs, and 60 terrestrial edge/cloud nodes.  It deliberately keeps a similar
node count to the former RedBestel experiment while adding domain hierarchy,
heterogeneous resources, and delay-aware cross-domain links.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import graph_u

SPACE = range(0, 12)
HAPS = range(12, 16)
UAVS = range(16, 24)
GROUND = range(24, 84)


def _add_edge(edges: set[tuple[str, str]], left: int, right: int) -> None:
    """Store both orientations because graph_u.Graph neighbours are directed."""
    edges.add((str(left), str(right)))
    edges.add((str(right), str(left)))


def build_sagin(seed: int = 7) -> graph_u.Graph:
    """Build a connected, reproducible 84-node static SAGIN topology."""
    rng = random.Random(seed)
    edges: set[tuple[str, str]] = set()

    # Space: two 6-satellite orbital rings plus inter-plane links.
    for plane_start in (0, 6):
        for index in range(6):
            _add_edge(edges, plane_start + index, plane_start + (index + 1) % 6)
    for index in range(6):
        _add_edge(edges, index, index + 6)

    # Air: HAP backbone, UAV ring, and each HAP manages two UAVs.
    for hap in HAPS:
        _add_edge(edges, hap, 12 + ((hap - 12 + 1) % 4))
    for uav in UAVS:
        _add_edge(edges, uav, 16 + ((uav - 16 + 1) % 8))
        _add_edge(edges, 12 + ((uav - 16) // 2), uav)

    # Ground: six regional terrestrial chains, connected by a backbone.
    for region in range(6):
        start = 24 + region * 10
        for offset in range(9):
            _add_edge(edges, start + offset, start + offset + 1)
        if region:
            _add_edge(edges, 24 + (region - 1) * 10, start)

    # Cross-domain gateways: satellite-HAP, HAP-ground, and UAV-ground.
    for offset, hap in enumerate(HAPS):
        _add_edge(edges, hap, offset)
        _add_edge(edges, hap, offset + 6)
        for ground_offset in (1, 4, 7):
            _add_edge(edges, hap, 24 + offset * 15 + ground_offset)
    for offset, uav in enumerate(UAVS):
        _add_edge(edges, uav, 24 + (offset * 7) % 60)

    # Values are overwritten below; Parameters only initialize graph_u.Graph.
    parameters = graph_u.Parameters(1, 2, 1, 2, 0, 100, 0, 100, 1, 2, 1, 2, 1, 2)
    substrate = graph_u.Graph(84, edges, parameters)
    substrate.topology_name = "SAGIN-84"
    substrate.use_delay_routing = True
    substrate.node_domains = {}
    substrate.node_types = {}

    def node_profile(node: int) -> tuple[str, str, tuple[int, int], tuple[int, int]]:
        if node in SPACE:
            return "space", "leo_satellite", (250, 450), (10, 90)
        if node in HAPS:
            return "air", "hap", (450, 700), (10 + (node - 12) * 20, 60)
        if node in UAVS:
            return "air", "uav", (220, 420), (10 + (node - 16) * 10, 25)
        region, local = divmod(node - 24, 10)
        return "ground", "edge_server", (600, 1000), (local * 10, 5 + region * 15)

    for node in range(substrate.nodes):
        domain, node_type, cpu_range, position = node_profile(node)
        substrate.node_domains[node] = domain
        substrate.node_types[node] = node_type
        substrate.node_weights[node] = rng.randint(*cpu_range)
        substrate.node_pos[node] = position

    def link_profile(left: int, right: int) -> tuple[tuple[int, int], tuple[int, int]]:
        left_domain, right_domain = substrate.node_domains[left], substrate.node_domains[right]
        if left_domain == right_domain == "space":
            return (850, 1000), (15, 25)
        if {left_domain, right_domain} == {"space", "air"}:
            return (650, 850), (20, 35)
        if left_domain == right_domain == "air":
            return (350, 650), (5, 12)
        if {left_domain, right_domain} == {"air", "ground"}:
            return (300, 550), (3, 8)
        return (500, 800), (1, 4)

    unique_edges = sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges})
    for left, right in unique_edges:
        bandwidth_range, delay_range = link_profile(left, right)
        bandwidth, delay = rng.randint(*bandwidth_range), rng.randint(*delay_range)
        for key in ((str(left), str(right)), (str(right), str(left))):
            substrate.edge_weights[key] = bandwidth
            substrate.delay[key] = delay
    return substrate


def for_automate(req_no: int = 5):
    """Compatibility entry point used by automate.generateSubstrate."""
    return build_sagin(), []


def save_topology(output: str | Path = "1_sagin.pickle", seed: int = 7) -> Path:
    output = Path(output)
    with output.open("wb") as stream:
        pickle.dump(build_sagin(seed), stream)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the fixed 84-node SAGIN substrate.")
    parser.add_argument("--output", default="1_sagin.pickle")
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    print(f"Saved {save_topology(arguments.output, arguments.seed)}")
