import numpy as np
import torch

from agents.deterministicBaseline import baseline_policy
from agents.networks import PolicyValueNet, obs_to_tensor
from env.pokerEnv import PokerEnv


def run_head_to_head(policy: PolicyValueNet, hands: int = 1000, device: torch.device | None = None) -> float:
    env = PokerEnv()
    device = device or torch.device("cpu")
    rewards = []
    for _ in range(hands):
        obs, mask = env.reset()
        done = False
        final_reward = 0.0
        while not done:
            player = env.state.current_player  # type: ignore
            if player == 0:
                obs_t = obs_to_tensor(obs, device)
                mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = policy.act(obs_t, mask_t)
            else:
                action = baseline_policy(obs, mask)
            obs, reward, done, info = env.step(action)
            if done:
                final_reward = info.get("rewards", [0.0, 0.0])[0]
                break
            mask = env.legal_action_mask()
        rewards.append(final_reward)
    ev_bb100 = np.mean(rewards) * 100.0 / env.stack_bb
    return float(ev_bb100)


def bootstrap_ci(data: list[float], alpha: float = 0.05, iters: int = 1000) -> tuple[float, float]:
    samples = []
    n = len(data)
    for _ in range(iters):
        resample = np.random.choice(data, n, replace=True)
        samples.append(np.mean(resample))
    lower = np.percentile(samples, alpha / 2 * 100)
    upper = np.percentile(samples, (1 - alpha / 2) * 100)
    return float(lower), float(upper)
