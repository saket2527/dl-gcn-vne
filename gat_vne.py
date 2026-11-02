"""Inference and evaluation for the PPO-trained GAT-VNE policy."""

from __future__ import annotations

import copy
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

from dl_vne import VNEEnvironment, get_device, unique_bandwidth, undirected_edges
from gat_policy import GATPolicy

DEFAULT_CHECKPOINT = Path(__file__).with_name("gat_vne.pth")


def load_policy(
    checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
) -> GATPolicy:
    device_obj = get_device(device)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"GAT-VNE checkpoint not found: {checkpoint_path}. Run train_gat.py first."
        )
    state = torch.load(checkpoint_path, map_location=device_obj)
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    policy = GATPolicy(
        feature_dim=int(metadata.get("feature_dim", 3)),
        hidden_dim=int(metadata.get("hidden_dim", 64)),
        context_dim=int(metadata.get("context_dim", 4)),
        heads=int(metadata.get("heads", 4)),
    ).to(device_obj)
    model_state = (
        state["model_state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state
        else state
    )
    policy.load_state_dict(model_state)
    return policy.eval()


def main(
    substrate: Any | None = None,
    vne_list: list[Any] | None = None,
    checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate GAT-VNE and return metrics compatible with the batch runner."""
    started = time.perf_counter()
    if substrate is None or vne_list is None:
        import config
        substrate = substrate if substrate is not None else getattr(config, "substrate", None)
        vne_list = vne_list if vne_list is not None else getattr(config, "vne_list", None)
    if substrate is None or not hasattr(substrate, "nodes"):
        from dl_vne import _load_substrate
        substrate = _load_substrate(Path(__file__).with_name("1_sagin.pickle"))
    if not vne_list:
        from vne_u import create_vne
        vne_list = create_vne(no_requests=10)

    original = copy.deepcopy(substrate)
    environment = VNEEnvironment(copy.deepcopy(substrate), device)
    policy = load_policy(checkpoint, device)
    accepted = 0
    revenue = total_cost = total_hops = 0.0
    total_nodes = sum(request.nodes for request in vne_list)
    total_links = sum(len(list(undirected_edges(request))) for request in vne_list)
    with torch.no_grad():
        for request in vne_list:
            success, _, info = environment.embed_request(request, policy, sample=False)
            if success:
                accepted += 1
                revenue += info["revenue"]
                total_cost += info["node_cost"] + info["edge_cost"]
                total_hops += info["hops"]

    elapsed = timedelta(seconds=time.perf_counter() - started)
    used_nodes = sum(
        environment.substrate.node_weights[i] != original.node_weights[i]
        for i in range(substrate.nodes)
    )
    used_links = sum(
        _edge_weight(environment.substrate, left, right)
        != _edge_weight(original, left, right)
        for left, right in undirected_edges(original)
    )
    pre_resource = sum(float(value) for value in original.node_weights.values()) + unique_bandwidth(original)
    post_resource = (
        sum(float(value) for value in environment.substrate.node_weights.values())
        + unique_bandwidth(environment.substrate)
    )
    node_utilization = [
        (float(original.node_weights[node]) - float(environment.substrate.node_weights[node]))
        / float(original.node_weights[node])
        for node in range(substrate.nodes)
        if original.node_weights[node]
        and environment.substrate.node_weights[node] != original.node_weights[node]
    ]
    link_utilization = [
        (_edge_weight(original, left, right) - _edge_weight(environment.substrate, left, right))
        / _edge_weight(original, left, right)
        for left, right in undirected_edges(original)
        if _edge_weight(original, left, right)
        and _edge_weight(environment.substrate, left, right)
        != _edge_weight(original, left, right)
    ]
    return {
        "revenue": revenue,
        "total_cost": total_cost,
        "accepted": accepted,
        "total_request": len(vne_list),
        "pre_resource": pre_resource,
        "post_resource": post_resource,
        "No_of_Links_used": used_links,
        "No_of_Nodes_used": used_nodes,
        "avg_path": total_hops / max(1, total_links),
        "avg_exec": elapsed,
        "total_nodes": total_nodes,
        "total_links": total_links,
        "avg_bw": 100 * sum(link_utilization) / max(1, len(link_utilization)),
        "avg_crb": 100 * sum(node_utilization) / max(1, len(node_utilization)),
        "avg_link": 100 * used_links / max(1, len(list(undirected_edges(original)))),
        "avg_node": 100 * used_nodes / max(1, substrate.nodes),
    }


def _edge_weight(graph: Any, left: int, right: int) -> float:
    for key in ((str(left), str(right)), (left, right)):
        if key in graph.edge_weights:
            return float(graph.edge_weights[key])
    raise KeyError(f"No bandwidth entry for substrate edge ({left}, {right})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained GAT-VNE policy.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    print(main(checkpoint=args.checkpoint))
