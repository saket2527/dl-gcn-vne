"""Fixed 76-node Space-Air-Ground Integrated Network (SAGIN) substrate.

The topology is a static three-domain testbed: 12 LEO satellites, 4 HAPs, and
60 terrestrial edge/cloud nodes. It provides a HAP-only air layer with domain
hierarchy, heterogeneous resources, and delay-aware cross-domain links.
"""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import graph_u

SPACE = range(0, 12)
AIR = range(12, 16)
GROUND = range(16, 76)


def _add_edge(edges: set[tuple[str, str]], left: int, right: int) -> None:
    """Store both orientations because graph_u.Graph neighbours are directed."""
    edges.add((str(left), str(right)))
    edges.add((str(right), str(left)))


def build_sagin(seed: int = 7) -> graph_u.Graph:
    """Build a connected, reproducible 76-node static SAGIN topology."""
    rng = random.Random(seed)
    edges: set[tuple[str, str]] = set()

    # Space: two 6-satellite orbital rings plus inter-plane links.
    for plane_start in (0, 6):
        for index in range(6):
            _add_edge(edges, plane_start + index, plane_start + (index + 1) % 6)
    for index in range(6):
        _add_edge(edges, index, index + 6)

    # Air: HAP backbone.
    for hap in AIR:
        _add_edge(edges, hap, 12 + ((hap - 12 + 1) % 4))

    # Ground: six regional terrestrial chains, connected by a backbone.
    for region in range(6):
        start = 16 + region * 10
        for offset in range(9):
            _add_edge(edges, start + offset, start + offset + 1)
        if region:
            _add_edge(edges, 16 + (region - 1) * 10, start)

    # Cross-domain gateways: satellite-HAP and HAP-ground.
    for offset, hap in enumerate(AIR):
        _add_edge(edges, hap, offset)
        _add_edge(edges, hap, offset + 6)
        for ground_offset in (1, 4, 7):
            _add_edge(edges, hap, 16 + offset * 15 + ground_offset)
    
    # Values are overwritten below; Parameters only initialize graph_u.Graph.
    parameters = graph_u.Parameters(1, 2, 1, 2, 0, 100, 0, 100, 1, 2, 1, 2, 1, 2)
    substrate = graph_u.Graph(76, edges, parameters)
    substrate.topology_name = "SAGIN-76"
    substrate.use_delay_routing = True
    substrate.node_domains = {}
    substrate.node_types = {}

    def node_profile(node: int) -> tuple[str, str, tuple[int, int], tuple[int, int]]:
        if node in SPACE:
            return "space", "leo_satellite", (250, 450), (10, 90)
        if node in AIR:
            return "air", "hap", (450, 700), (10 + (node - 12) * 20, 60)
        region, local = divmod(node - 16, 10)
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
