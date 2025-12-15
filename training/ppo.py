from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import math


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    epochs: int = 3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95
    batch_size: int = 128


def compute_advantages(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def ppo_update(net: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Dict[str, torch.Tensor], config: PPOConfig) -> Tuple[float, float, float]:
    obs_tensors = batch["obs"]
    actions = batch["actions"]
    old_logprobs = batch["logprobs"]
    advantages = batch["advantages"]
    returns = batch["returns"]

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    dataset_size = actions.size(0)
    idx = torch.randperm(dataset_size)
    for start in range(0, dataset_size, config.batch_size):
        end = start + config.batch_size
        b = idx[start:end]
        obs_batch = {k: v[b] for k, v in obs_tensors.items()}
        actions_b = actions[b]
        old_logprobs_b = old_logprobs[b]
        adv_b = advantages[b]
        ret_b = returns[b]
        logits, values = net(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        logprobs = dist.log_prob(actions_b)
        ratio = torch.exp(logprobs - old_logprobs_b)
        clip_adv = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * adv_b
        policy_loss = -(torch.min(ratio * adv_b, clip_adv)).mean()
        value_loss = F.mse_loss(values, ret_b)
        entropy = dist.entropy().mean()
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
        optimizer.step()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()
    steps = math.ceil(dataset_size / config.batch_size)
    return total_policy_loss / steps, total_value_loss / steps, total_entropy / steps
