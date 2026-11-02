"""Train GAT-VNE with PPO on dynamically generated VNR batches."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch

from dl_vne import VNEEnvironment, get_device
from gat_policy import GATPolicy
from train_drl import load_substrate, reward_for
from vne_u import create_vne

DEFAULT_CHECKPOINT = Path(__file__).with_name("gat_vne.pth")


def train(
    substrate_path: str | Path = "1_sagin.pickle",
    episodes: int = 300,
    requests_per_episode: int = 30,
    learning_rate: float = 3e-4,
    hidden_dim: int = 64,
    heads: int = 4,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
    seed: int = 7,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
) -> GATPolicy:
    if episodes <= 0 or requests_per_episode <= 0 or ppo_epochs <= 0:
        raise ValueError("episodes, requests_per_episode, and ppo_epochs must be positive")
    if hidden_dim % heads:
        raise ValueError("hidden_dim must be divisible by heads")

    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = get_device(device)
    substrate_template = load_substrate(substrate_path)
    feature_dim = 7 if hasattr(substrate_template, "node_domains") else 3
    policy = GATPolicy(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        context_dim=4,
        heads=heads,
    ).to(device_obj)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    policy.train()

    for episode in range(1, episodes + 1):
        environment = VNEEnvironment(copy.deepcopy(substrate_template), device_obj)
        transitions: list[dict[str, torch.Tensor]] = []
        rewards: list[float] = []
        accepted = 0
        for request in create_vne(no_requests=requests_per_episode):
            success, _, info = environment.embed_request(request, policy, sample=True)
            request_transitions = info["transitions"]
            transitions.extend(request_transitions)
            rewards.extend([reward_for(success, info)] * len(request_transitions))
            accepted += int(success)
        if not transitions:
            continue

        returns = torch.tensor(rewards, dtype=torch.float32, device=device_obj)
        old_log_probs = torch.stack([item["old_log_prob"] for item in transitions])
        for _ in range(ppo_epochs):
            log_probs, values, entropies = [], [], []
            for item in transitions:
                log_prob, value, entropy = policy.evaluate_action(
                    item["features"], item["adjacency"], item["context"],
                    item["mask"], item["action"],
                )
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
            log_probs_tensor = torch.stack(log_probs)
            values_tensor = torch.stack(values)
            advantages = returns - values_tensor.detach()
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp_min(1e-6)
            ratio = (log_probs_tensor - old_log_probs).exp()
            clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            actor_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = torch.nn.functional.mse_loss(values_tensor, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * torch.stack(entropies).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

        if episode == 1 or episode % 25 == 0 or episode == episodes:
            print(
                f"episode={episode:04d} accepted={accepted}/{requests_per_episode} "
                f"loss={loss.item():.4f}"
            )

    checkpoint_path = Path(checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "metadata": {
                "hidden_dim": hidden_dim,
                "feature_dim": feature_dim,
                "context_dim": 4,
                "heads": heads,
                "episodes": episodes,
            },
        },
        checkpoint_path,
    )
    return policy.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate-pickle", dest="substrate_path", default="1_sagin.pickle")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--requests-per-episode", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to CUDA when available")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    train(**vars(parse_args()))
