"""Graph-attention actor used by the GAT-VNE placement policy."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Multi-head additive graph attention over the substrate adjacency."""

    def __init__(self, in_features: int, out_features: int, heads: int = 4) -> None:
        super().__init__()
        if out_features % heads:
            raise ValueError("out_features must be divisible by the number of attention heads")
        self.heads = heads
        self.head_dim = out_features // heads
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        node_count = features.shape[0]
        projected = self.linear(features).view(node_count, self.heads, self.head_dim)
        source_score = (projected * self.attention_source).sum(dim=-1)
        target_score = (projected * self.attention_target).sum(dim=-1)
        scores = F.leaky_relu(
            source_score[:, None, :] + target_score[None, :, :], negative_slope=0.2
        )
        connected = adjacency > 0
        scores = scores.masked_fill(~connected.unsqueeze(-1), -torch.inf)
        attention = torch.softmax(scores, dim=1)
        aggregated = torch.einsum("ijh,jhd->ihd", attention, projected)
        return aggregated.reshape(node_count, -1) + self.bias


class GATPolicy(nn.Module):
    """PPO actor-critic that scores substrate placements with graph attention."""

    def __init__(
        self, feature_dim: int = 3, hidden_dim: int = 64, context_dim: int = 4, heads: int = 4
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by the number of attention heads")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.heads = heads
        self.gat1 = GraphAttentionLayer(feature_dim, hidden_dim, heads)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, heads)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def encode(self, features: Tensor, adjacency: Tensor) -> Tensor:
        embedding = F.elu(self.gat1(features, adjacency))
        return F.elu(self.gat2(embedding, adjacency))

    def _context_matrix(self, features: Tensor, virtual_features: Tensor | float) -> Tensor:
        context = torch.as_tensor(virtual_features, dtype=features.dtype, device=features.device)
        if context.ndim == 0:
            context = context.reshape(1, 1).expand(features.shape[0], 1)
        elif context.ndim == 1:
            context = context.reshape(1, -1).expand(features.shape[0], -1)
        if context.shape != (features.shape[0], self.context_dim):
            raise ValueError(
                f"Expected virtual context shape ({features.shape[0]}, {self.context_dim}), "
                f"got {tuple(context.shape)}"
            )
        return context

    def forward(
        self,
        features: Tensor,
        normalized_adjacency: Tensor,
        virtual_features: Tensor | float,
        feasible_mask: Tensor | None = None,
    ) -> Tensor:
        embedding = self.encode(features, normalized_adjacency)
        context = self._context_matrix(features, virtual_features)
        logits = self.actor(torch.cat((embedding, context), dim=-1)).squeeze(-1)
        if feasible_mask is not None:
            logits = logits.masked_fill(
                ~feasible_mask.to(device=logits.device, dtype=torch.bool), -torch.inf
            )
        return logits

    def evaluate_action(
        self,
        features: Tensor,
        normalized_adjacency: Tensor,
        virtual_features: Tensor,
        feasible_mask: Tensor,
        action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embedding = self.encode(features, normalized_adjacency)
        context = self._context_matrix(features, virtual_features)
        logits = self.actor(torch.cat((embedding, context), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(
            ~feasible_mask.to(device=logits.device, dtype=torch.bool), -torch.inf
        )
        distribution = torch.distributions.Categorical(logits=logits)
        summary = torch.cat((embedding.mean(dim=0), context.mean(dim=0)), dim=0)
        return distribution.log_prob(action), self.critic(summary).squeeze(-1), distribution.entropy()

    @staticmethod
    def feasibility_mask(available_cpu: Tensor, cpu_demand: float, used_nodes: Tensor) -> Tensor:
        return (available_cpu >= float(cpu_demand)) & ~used_nodes.to(dtype=torch.bool)
