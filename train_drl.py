"""Train DL-GCN-VNE with PPO on dynamically generated VNR batches."""

from __future__ import annotations

import argparse
import copy
import pickle
import random
from pathlib import Path

import torch

from dl_vne import DEFAULT_CHECKPOINT, VNEEnvironment, get_device
from gcn_policy import GCNPolicy
from vne_u import create_vne


def load_substrate(path: str | Path):
    path = Path(path)
    if not path.exists() and not path.is_absolute():
        path = Path(__file__).with_name(path.name)
    if not path.exists() and path.name == "1_sagin.pickle":
        from sagin_topology import save_topology
        save_topology(path)
    with open(path, "rb") as stream:
        return pickle.load(stream)


def reward_for(success: bool, info: dict) -> float:
    """Reward successful R/C embeddings and discourage rejection and long paths."""
    if not success:
        return -1.0

    rc_ratio = info["revenue"] / max(info["node_cost"] + info["edge_cost"], 1e-8)
    return 8.0 * rc_ratio - 0.35 * info["hops"] - 0.01 * float(info.get("total_delay", 0.0))


def train(
    substrate_path: str | Path = "1_sagin.pickle",
    episodes: int = 300,
    requests_per_episode: int = 30,
    learning_rate: float = 3e-4,
    hidden_dim: int = 64,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
    seed: int = 7,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
) -> GCNPolicy:
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = get_device(device)
    substrate_template = load_substrate(substrate_path)
    feature_dim = 7 if hasattr(substrate_template, "node_domains") else 3
    policy = GCNPolicy(feature_dim=feature_dim, hidden_dim=hidden_dim, context_dim=4).to(device_obj)
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
        old_log_probs = torch.stack([transition["old_log_prob"] for transition in transitions])
        # PPO performs several clipped actor/critic updates over this episode.
        for _ in range(ppo_epochs):
            new_log_probs, values, entropies = [], [], []
            for transition in transitions:
                log_prob, value, entropy = policy.evaluate_action(
                    transition["features"], transition["adjacency"], transition["context"],
                    transition["mask"], transition["action"],
                )
                new_log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
            new_log_probs_tensor = torch.stack(new_log_probs)
            values_tensor = torch.stack(values)
            advantages = returns - values_tensor.detach()
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
            ratio = (new_log_probs_tensor - old_log_probs).exp()
            clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            actor_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = torch.nn.functional.mse_loss(values_tensor, returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * torch.stack(entropies).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
        if episode == 1 or episode % 25 == 0 or episode == episodes:
            print(f"episode={episode:04d} accepted={accepted}/{requests_per_episode} loss={loss.item():.4f}")

    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "metadata": {"hidden_dim": hidden_dim, "feature_dim": feature_dim, "context_dim": 4},
    }, checkpoint)
    return policy.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate-pickle", dest="substrate_path", default="1_sagin.pickle")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--requests-per-episode", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to CUDA when available")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
