"""Inference and transactional evaluation for DL-GCN-VNE.

The module is intentionally independent of NetworkX: it works with the
project's existing ``graph_u.Graph`` objects and their pickle files.
"""

from __future__ import annotations

import copy
import os
import pickle
import random
import statistics
import time
from datetime import timedelta
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable

import torch

from gcn_policy import GCNPolicy

DEFAULT_CHECKPOINT = Path(__file__).with_name("gcn_drl_vne.pth")


def get_device(device: str | torch.device | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _node_id(node: Any) -> int:
    return int(node)


def _edge_weight(graph: Any, left: int, right: int) -> float:
    for key in ((str(left), str(right)), (left, right)):
        if key in graph.edge_weights:
            return float(graph.edge_weights[key])
    raise KeyError(f"No bandwidth entry for substrate edge ({left}, {right})")


def _set_edge_weight(graph: Any, left: int, right: int, value: float) -> None:
    for key in ((str(left), str(right)), (left, right)):
        if key in graph.edge_weights:
            graph.edge_weights[key] = value
            return
    raise KeyError(f"No bandwidth entry for substrate edge ({left}, {right})")


def _edge_delay(graph: Any, left: int, right: int) -> float:
    """Return link delay; legacy substrates fall back to unit hop cost."""
    for key in ((str(left), str(right)), (left, right)):
        if hasattr(graph, "delay") and key in graph.delay:
            return float(graph.delay[key])
    return 1.0


def undirected_edges(graph: Any) -> Iterable[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    for left, right in graph.edge_weights:
        edge = tuple(sorted((_node_id(left), _node_id(right))))
        if edge not in seen:
            seen.add(edge)
            yield edge


def unique_bandwidth(graph: Any) -> float:
    return sum(_edge_weight(graph, left, right) for left, right in undirected_edges(graph))


class VNEEnvironment:
    """Mutable substrate environment with all-or-nothing request embedding."""

    def __init__(self, substrate: Any, device: str | torch.device | None = None) -> None:
        self.substrate = substrate
        self.device = get_device(device)
        self.initial_cpu = {int(k): float(v) for k, v in substrate.node_weights.items()}
        self.initial_bandwidth = unique_bandwidth(substrate)
        self.initial_resource = sum(self.initial_cpu.values()) + self.initial_bandwidth

    def adjacency_and_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = self.substrate.nodes
        adjacency = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        bandwidth_sums = torch.zeros(n, dtype=torch.float32, device=self.device)
        delay_sums = torch.zeros(n, dtype=torch.float32, device=self.device)
        for left, right in undirected_edges(self.substrate):
            bw = _edge_weight(self.substrate, left, right)
            adjacency[left, right] = adjacency[right, left] = 1.0
            bandwidth_sums[left] += bw
            bandwidth_sums[right] += bw
            delay = _edge_delay(self.substrate, left, right)
            delay_sums[left] += delay
            delay_sums[right] += delay
        available_cpu = torch.tensor(
            [float(self.substrate.node_weights[i]) for i in range(n)], dtype=torch.float32, device=self.device
        )
        degree = adjacency.sum(dim=1)
        # Feature scaling is per current substrate state, preserving dynamics.
        def normalise(value: torch.Tensor) -> torch.Tensor:
            return value / value.max().clamp_min(1.0)
        features = [normalise(available_cpu), normalise(bandwidth_sums), normalise(degree)]
        if hasattr(self.substrate, "node_domains"):
            features.append(normalise(delay_sums))
            domain_index = {"space": 0, "air": 1, "ground": 2}
            indices = torch.tensor(
                [domain_index[self.substrate.node_domains[i]] for i in range(n)], device=self.device
            )
            features.extend([(indices == index).float() for index in range(3)])
        features = torch.stack(features, dim=1)
        with_self_loops = adjacency + torch.eye(n, device=self.device)
        inv_sqrt = with_self_loops.sum(dim=1).clamp_min(1.0).pow(-0.5)
        normalized_adjacency = inv_sqrt[:, None] * with_self_loops * inv_sqrt[None, :]
        return features, normalized_adjacency, available_cpu

    def _shortest_feasible_path(self, source: int, target: int, demand: float) -> list[int]:
        """Dijkstra route on residual-bandwidth-feasible links, minimizing delay for SAGIN."""
        if source == target:
            return [source]
        parent: dict[int, int | None] = {source: None}
        distance: dict[int, int] = {source: 0}
        queue: list[tuple[int, int]] = [(0, source)]
        while queue:
            current_distance, current = heappop(queue)
            if current_distance != distance[current]:
                continue
            if current == target:
                path = [target]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            for neighbor in self.substrate.neighbours.get(current, set()):
                neighbor = _node_id(neighbor)
                if _edge_weight(self.substrate, current, neighbor) < demand:
                    continue
                link_cost = _edge_delay(self.substrate, current, neighbor) if getattr(self.substrate, "use_delay_routing", False) else 1
                candidate = current_distance + link_cost
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    parent[neighbor] = current
                    heappush(queue, (candidate, neighbor))
        return []

    def _feasible_distances(self, target: int, demand: float) -> list[float]:
        """One Dijkstra traversal gives all candidates' feasible hop distances."""
        distances = [float("inf")] * self.substrate.nodes
        distances[target] = 0.0
        queue: list[tuple[float, int]] = [(0.0, target)]
        while queue:
            current_distance, current = heappop(queue)
            if current_distance != distances[current]:
                continue
            for neighbor in self.substrate.neighbours.get(current, set()):
                neighbor = _node_id(neighbor)
                if _edge_weight(self.substrate, current, neighbor) < demand:
                    continue
                link_cost = _edge_delay(self.substrate, current, neighbor) if getattr(self.substrate, "use_delay_routing", False) else 1.0
                candidate = current_distance + link_cost
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heappush(queue, (candidate, neighbor))
        return distances

    def _candidate_routing_features(
        self, request: Any, virtual_node: int, placement: dict[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate routing cost from every candidate to mapped VNR neighbours.

        A candidate is infeasible when it cannot reach an already mapped
        virtual neighbour on a residual-bandwidth-feasible path.  The returned
        cost is demand-weighted hop count, normalized across candidates.
        """
        mapped_neighbours: list[tuple[int, float]] = []
        for left, right, bandwidth in self._virtual_edges(request):
            if left == virtual_node and right in placement:
                mapped_neighbours.append((placement[right], bandwidth))
            elif right == virtual_node and left in placement:
                mapped_neighbours.append((placement[left], bandwidth))
        costs = torch.zeros(self.substrate.nodes, dtype=torch.float32, device=self.device)
        feasible = torch.ones(self.substrate.nodes, dtype=torch.bool, device=self.device)
        for mapped_node, bandwidth in mapped_neighbours:
            distances = self._feasible_distances(mapped_node, bandwidth)
            for candidate, distance in enumerate(distances):
                if distance == float("inf"):
                    feasible[candidate] = False
                else:
                    costs[candidate] += distance * bandwidth
        costs = torch.where(feasible, costs, torch.full_like(costs, float(self.substrate.nodes)))
        return costs / costs.max().clamp_min(1.0), feasible

    @staticmethod
    def _virtual_edges(request: Any) -> Iterable[tuple[int, int, float]]:
        seen: set[tuple[int, int]] = set()
        for left, right in request.edge_weights:
            left_i, right_i = _node_id(left), _node_id(right)
            edge = tuple(sorted((left_i, right_i)))
            if edge not in seen:
                seen.add(edge)
                yield edge[0], edge[1], _edge_weight(request, edge[0], edge[1])

    def rollback(self, cpu_changes: list[tuple[int, float]], bw_changes: list[tuple[int, int, float]]) -> None:
        for node, amount in cpu_changes:
            self.substrate.node_weights[node] += amount
        for left, right, amount in bw_changes:
            _set_edge_weight(self.substrate, left, right, _edge_weight(self.substrate, left, right) + amount)
            _set_edge_weight(self.substrate, right, left, _edge_weight(self.substrate, right, left) + amount)

    def embed_request(self, request: Any, policy: GCNPolicy, *, sample: bool = False) -> tuple[bool, list[torch.Tensor], dict[str, Any]]:
        """Place and route one request, rolling back fully on every failure."""
        cpu_changes: list[tuple[int, float]] = []
        bw_changes: list[tuple[int, int, float]] = []
        log_probs: list[torch.Tensor] = []
        transitions: list[dict[str, torch.Tensor]] = []
        placement: dict[int, int] = {}
        used = torch.zeros(self.substrate.nodes, dtype=torch.bool, device=self.device)
        # Place virtual hubs first: they have the greatest influence on path cost.
        def priority(virtual_node: int) -> tuple[int, float, float]:
            incident = [bandwidth for left, right, bandwidth in self._virtual_edges(request)
                        if left == virtual_node or right == virtual_node]
            return len(incident), sum(incident), float(request.node_weights[virtual_node])
        order = sorted(range(request.nodes), key=priority, reverse=True)
        for virtual_node in order:
            demand = float(request.node_weights[virtual_node])
            features, adjacency, available_cpu = self.adjacency_and_features()
            if features.shape[1] != policy.feature_dim:
                raise ValueError(
                    f"Policy expects {policy.feature_dim} substrate features but topology provides {features.shape[1]}. "
                    "Train a new checkpoint for this topology."
                )
            route_cost, route_feasible = self._candidate_routing_features(request, virtual_node, placement)
            mask = policy.feasibility_mask(available_cpu, demand, used) & route_feasible
            virtual_degree = sum(
                left == virtual_node or right == virtual_node for left, right, _ in self._virtual_edges(request)
            )
            adjacent_bandwidth = sum(
                bandwidth for left, right, bandwidth in self._virtual_edges(request)
                if left == virtual_node or right == virtual_node
            )
            full_context = torch.stack((
                torch.full((self.substrate.nodes,), demand / max(self.initial_cpu.values(), default=1.0), device=self.device),
                torch.full((self.substrate.nodes,), virtual_degree / max(1, request.nodes - 1), device=self.device),
                torch.full((self.substrate.nodes,), adjacent_bandwidth / max(1.0, unique_bandwidth(request)), device=self.device),
                route_cost,
            ), dim=1)
            # Checkpoints trained before routing-aware features retain their old CPU-only input.
            context = full_context[:, :policy.context_dim]
            logits = policy(features, adjacency, context, mask)
            if not torch.isfinite(logits).any():
                self.rollback(cpu_changes, bw_changes)
                return False, log_probs, {"reason": "insufficient_cpu", "transitions": transitions}
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample() if sample else torch.argmax(logits)
            log_probs.append(distribution.log_prob(action))
            transitions.append({
                "features": features.detach(), "adjacency": adjacency.detach(),
                "context": context.detach(), "mask": mask.detach(), "action": action.detach(),
                "old_log_prob": distribution.log_prob(action).detach(),
            })
            substrate_node = int(action.item())
            self.substrate.node_weights[substrate_node] -= demand
            cpu_changes.append((substrate_node, demand))
            used[substrate_node] = True
            placement[virtual_node] = substrate_node

        hop_count = 0
        total_delay = 0.0
        for left, right, demand in self._virtual_edges(request):
            path = self._shortest_feasible_path(placement[left], placement[right], demand)
            if not path:
                self.rollback(cpu_changes, bw_changes)
                return False, log_probs, {"reason": "path_not_found", "placement": placement, "transitions": transitions}
            hop_count += len(path) - 1
            for source, target in zip(path, path[1:]):
                total_delay += _edge_delay(self.substrate, source, target)
                _set_edge_weight(self.substrate, source, target, _edge_weight(self.substrate, source, target) - demand)
                _set_edge_weight(self.substrate, target, source, _edge_weight(self.substrate, target, source) - demand)
                bw_changes.append((source, target, demand))

        node_cost = sum(amount for _, amount in cpu_changes)
        edge_cost = sum(amount for _, _, amount in bw_changes)
        revenue = node_cost + sum(demand for _, _, demand in self._virtual_edges(request))
        return True, log_probs, {
            "placement": placement, "node_cost": node_cost, "edge_cost": edge_cost,
            "revenue": revenue, "hops": hop_count, "total_delay": total_delay, "transitions": transitions,
        }


def load_policy(checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT, device: str | torch.device | None = None) -> GCNPolicy:
    device = get_device(device)
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"DL-GCN-VNE checkpoint not found: {checkpoint}. Run train_drl.py first.")
    state = torch.load(checkpoint, map_location=device)
    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    policy = GCNPolicy(
        feature_dim=int(metadata.get("feature_dim", 3)),
        hidden_dim=int(metadata.get("hidden_dim", 64)),
        context_dim=int(metadata.get("context_dim", 1)),
    ).to(device)
    policy.load_state_dict(
        state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state,
        strict=False,
    )
    return policy.eval()


def _load_substrate(path: str | os.PathLike[str]) -> Any:
    path = Path(path)
    if not path.exists() and path.name == "1_sagin.pickle":
        from sagin_topology import save_topology
        save_topology(path)
    with open(path, "rb") as stream:
        return pickle.load(stream)


def main(
    substrate: Any | None = None,
    vne_list: list[Any] | None = None,
    checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate a trained policy and return the baseline-compatible metrics."""
    started = time.perf_counter()
    if substrate is None or vne_list is None:
        import config
        substrate = substrate if substrate is not None else getattr(config, "substrate", None)
        vne_list = vne_list if vne_list is not None else getattr(config, "vne_list", None)
    if substrate is None or not hasattr(substrate, "nodes"):
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
    used_nodes = sum(environment.substrate.node_weights[i] != original.node_weights[i] for i in range(substrate.nodes))
    used_links = sum(
        _edge_weight(environment.substrate, left, right) != _edge_weight(original, left, right)
        for left, right in undirected_edges(original)
    )
    pre_resource = sum(float(v) for v in original.node_weights.values()) + unique_bandwidth(original)
    post_resource = sum(float(v) for v in environment.substrate.node_weights.values()) + unique_bandwidth(environment.substrate)
    node_util = [
        (float(original.node_weights[i]) - float(environment.substrate.node_weights[i])) / float(original.node_weights[i])
        for i in range(substrate.nodes) if environment.substrate.node_weights[i] != original.node_weights[i] and original.node_weights[i]
    ]
    link_util = [
        (_edge_weight(original, l, r) - _edge_weight(environment.substrate, l, r)) / _edge_weight(original, l, r)
        for l, r in undirected_edges(original) if _edge_weight(environment.substrate, l, r) != _edge_weight(original, l, r) and _edge_weight(original, l, r)
    ]
    return {
        "revenue": revenue, "total_cost": total_cost, "accepted": accepted, "total_request": len(vne_list),
        "pre_resource": pre_resource, "post_resource": post_resource, "No_of_Links_used": used_links,
        "No_of_Nodes_used": used_nodes, "avg_path": total_hops / max(1, total_links),
        "avg_exec": elapsed, "total_nodes": total_nodes, "total_links": total_links,
        "avg_bw": 100 * sum(link_util) / max(1, len(link_util)),
        "avg_crb": 100 * sum(node_util) / max(1, len(node_util)),
        "avg_link": 100 * used_links / max(1, len(list(undirected_edges(original)))),
        "avg_node": 100 * used_nodes / max(1, substrate.nodes),
    }


def benchmark(
    seeds: Iterable[int] = range(10), checkpoint: str | os.PathLike[str] = DEFAULT_CHECKPOINT,
    requests_per_seed: int = 10, device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic policy decisions on reproducible VNR workloads."""
    from vne_u import create_vne
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        random.seed(seed)
        runs.append(main(vne_list=create_vne(no_requests=requests_per_seed), checkpoint=checkpoint, device=device))
    rc_values = [100.0 * result["revenue"] / result["total_cost"] if result["total_cost"] else 0.0 for result in runs]
    return {
        "runs": runs,
        "mean_rc": statistics.mean(rc_values),
        "std_rc": statistics.stdev(rc_values) if len(rc_values) > 1 else 0.0,
        "mean_accepted": statistics.mean(result["accepted"] for result in runs),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained DL-GCN-VNE policy.")
    parser.add_argument("--benchmark-seeds", type=int, default=0, help="evaluate this many fixed seeds instead of one batch")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()
    print(benchmark(range(args.benchmark_seeds), args.checkpoint) if args.benchmark_seeds else main(checkpoint=args.checkpoint))
