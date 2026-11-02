"""Focused runtime checks for the GAT-VNE policy and embedding environment."""

from __future__ import annotations

import unittest

import torch

import graph_u
from dl_vne import VNEEnvironment
from gat_policy import GATPolicy


def _graph(
    nodes: int,
    edges: tuple[tuple[int, int], ...],
    cpu: tuple[float, ...],
    bandwidth: tuple[float, ...],
) -> graph_u.Graph:
    directed_edges = {
        (str(left), str(right))
        for source, target in edges
        for left, right in ((source, target), (target, source))
    }
    parameters = graph_u.Parameters(1, 2, 1, 2, 0, 100, 0, 100, 1, 2, 1, 2, 1, 2)
    graph = graph_u.Graph(nodes, directed_edges, parameters)
    graph.node_weights.update(dict(enumerate(cpu)))
    for (left, right), capacity in zip(edges, bandwidth):
        graph.edge_weights[(str(left), str(right))] = capacity
        graph.edge_weights[(str(right), str(left))] = capacity
    return graph


class GATVNETest(unittest.TestCase):
    def test_attention_policy_masks_infeasible_nodes(self) -> None:
        policy = GATPolicy(feature_dim=3, hidden_dim=16, context_dim=4, heads=4)
        features = torch.rand(4, 3)
        adjacency = torch.eye(4)
        adjacency[0, 1] = adjacency[1, 0] = 1
        mask = torch.tensor([True, False, True, False])

        logits = policy(features, adjacency, torch.zeros(4, 4), mask)

        self.assertEqual(logits.shape, (4,))
        self.assertTrue(torch.isfinite(logits[mask]).all())
        self.assertTrue(torch.isneginf(logits[~mask]).all())

    def test_gat_policy_embeds_feasible_request(self) -> None:
        substrate = _graph(
            nodes=4,
            edges=((0, 1), (1, 2), (2, 3), (3, 0)),
            cpu=(100, 100, 100, 100),
            bandwidth=(100, 100, 100, 100),
        )
        request = _graph(
            nodes=2,
            edges=((0, 1),),
            cpu=(20, 20),
            bandwidth=(10,),
        )
        environment = VNEEnvironment(substrate, device="cpu")
        policy = GATPolicy(feature_dim=3, hidden_dim=16, context_dim=4, heads=4)

        success, _, info = environment.embed_request(request, policy, sample=False)

        self.assertTrue(success, info)
        self.assertEqual(len(info["placement"]), 2)
        self.assertEqual(len(info["route_paths"]), 1)


if __name__ == "__main__":
    unittest.main()
