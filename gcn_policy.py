"""GCN actor used by the DL-GCN-VNE placement policy."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """A dependency-free GCN layer: ``A_hat @ (X @ W)``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, features: Tensor, normalized_adjacency: Tensor) -> Tensor:
        return normalized_adjacency @ self.linear(features)


class GCNPolicy(nn.Module):
    """Actor network for placing one virtual node on a substrate node.

    ``features`` is one row per substrate node containing available CPU,
    adjacent available bandwidth, and degree centrality.  The virtual-node CPU
    demand is deliberately supplied separately, allowing the same substrate
    encoding to score different virtual nodes.
    """

    def __init__(self, feature_dim: int = 3, hidden_dim: int = 64, context_dim: int = 4) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.gcn1 = GraphConvolution(feature_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def encode(self, features: Tensor, normalized_adjacency: Tensor) -> Tensor:
        embedding = F.relu(self.gcn1(features, normalized_adjacency))
        return F.relu(self.gcn2(embedding, normalized_adjacency))

    def _context_matrix(self, features: Tensor, virtual_features: Tensor | float) -> Tensor:
        context = torch.as_tensor(virtual_features, dtype=features.dtype, device=features.device)
        if context.ndim == 0:
            context = context.reshape(1, 1).expand(features.shape[0], 1)
        elif context.ndim == 1:
            context = context.reshape(1, -1).expand(features.shape[0], -1)
        if context.shape != (features.shape[0], self.context_dim):
            raise ValueError(f"Expected virtual context shape ({features.shape[0]}, {self.context_dim}), got {tuple(context.shape)}")
        return context

    def forward(
        self,
        features: Tensor,
        normalized_adjacency: Tensor,
        virtual_features: Tensor | float,
        feasible_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one placement logit per substrate node.

        Invalid choices receive ``-inf``.  If all choices are invalid the
        caller can reliably detect it with ``torch.isfinite(logits).any()``.
        """
        embedding = self.encode(features, normalized_adjacency)
        context = self._context_matrix(features, virtual_features)
        logits = self.actor(torch.cat((embedding, context), dim=-1)).squeeze(-1)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask.to(device=logits.device, dtype=torch.bool), -torch.inf)
        return logits

    def evaluate_action(
        self, features: Tensor, normalized_adjacency: Tensor, virtual_features: Tensor,
        feasible_mask: Tensor, action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return action log-probability, state value and entropy for PPO."""
        embedding = self.encode(features, normalized_adjacency)
        context = self._context_matrix(features, virtual_features)
        logits = self.actor(torch.cat((embedding, context), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~feasible_mask.to(device=logits.device, dtype=torch.bool), -torch.inf)
        distribution = torch.distributions.Categorical(logits=logits)
        summary = torch.cat((embedding.mean(dim=0), context.mean(dim=0)), dim=0)
        return distribution.log_prob(action), self.critic(summary).squeeze(-1), distribution.entropy()

    @staticmethod
    def feasibility_mask(available_cpu: Tensor, cpu_demand: float, used_nodes: Tensor) -> Tensor:
        """Nodes need enough CPU and cannot host two nodes of one VNR."""
        return (available_cpu >= float(cpu_demand)) & ~used_nodes.to(dtype=torch.bool)
