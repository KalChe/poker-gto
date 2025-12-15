from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import optim

from agents.deterministicBaseline import baseline_policy
from agents.networks import PolicyValueNet, obs_to_tensor
from env.pokerEnv import PokerEnv
from training.ppo import PPOConfig, compute_advantages, ppo_update


class SelfPlayTrainer:
    def __init__(self, device: torch.device, config: PPOConfig, history_len: int = 32):
        self.device = device
        self.config = config
        self.env = PokerEnv(history_len=history_len)
        self.net = PolicyValueNet(history_len=history_len).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-4)
        self.opponent_buffer: List[Dict] = []

    def select_opponent(self) -> Tuple[str, PolicyValueNet | None]:
        p = np.random.rand()
        if p < 0.4:
            return "self", None
        if p < 0.8 and self.opponent_buffer:
            state_dict = deepcopy(np.random.choice(self.opponent_buffer))
            opp = PolicyValueNet().to(self.device)
            opp.load_state_dict(state_dict)
            opp.eval()
            return "buffer", opp
        return "baseline", None

    def collect_batch(self, episodes: int = 32) -> Dict[str, torch.Tensor]:
        obs_list = []
        actions = []
        logprobs = []
        values = []
        rewards = []
        dones = []
        for _ in range(episodes):
            opponent_type, opp_net = self.select_opponent()
            obs, mask = self.env.reset()
            traj_obs = []
            traj_actions = []
            traj_logprobs = []
            traj_values = []
            traj_rewards = []
            traj_dones = []
            done = False
            final_reward = 0.0
            while not done:
                player = self.env.state.current_player  # type: ignore
                if player == 0:
                    obs_t = obs_to_tensor(obs, self.device)
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, logprob, value = self.net.act(obs_t, mask_t)
                    traj_obs.append({k: v.detach() for k, v in obs_t.items()})
                    traj_actions.append(action)
                    traj_logprobs.append(logprob.detach())
                    traj_values.append(value.detach())
                else:
                    if opponent_type == "baseline":
                        action = baseline_policy(obs, mask)
                    elif opponent_type == "self":
                        with torch.no_grad():
                            opp_obs = obs_to_tensor(obs, self.device)
                            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                            action, _, _ = self.net.act(opp_obs, mask_t)
                    else:
                        with torch.no_grad():
                            opp_obs = obs_to_tensor(obs, self.device)
                            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                            action, _, _ = opp_net.act(opp_obs, mask_t)  # type: ignore
                obs, reward, done, info = self.env.step(action)
                if player == 0:
                    traj_rewards.append(0.0)
                    traj_dones.append(float(done))
                if done:
                    if "rewards" in info:
                        final_reward = info["rewards"][0]
                    break
                mask = self.env.legal_action_mask()
            if traj_rewards:
                traj_rewards[-1] = final_reward
                traj_dones[-1] = 1.0
                traj_values.append(torch.tensor([0.0], device=self.device))
                obs_list.extend(traj_obs)
                actions.extend(traj_actions)
                logprobs.extend(traj_logprobs)
                values.extend(traj_values)
                rewards.extend(traj_rewards)
                dones.extend(traj_dones)
        if not actions:
            return {}
        obs_stacked = {k: torch.cat([o[k] for o in obs_list], dim=0) for k in obs_list[0].keys()}
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        logprobs_t = torch.stack(logprobs)
        values_t = torch.stack(values)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        adv, rets = compute_advantages(rewards_t, values_t, dones_t, self.config.gamma, self.config.lam)
        batch = {
            "obs": obs_stacked,
            "actions": actions_t,
            "logprobs": logprobs_t,
            "advantages": adv.to(self.device),
            "returns": rets.to(self.device),
        }
        return batch

    def train_step(self, episodes: int = 64) -> Dict[str, float]:
        batch = self.collect_batch(episodes)
        if not batch:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        policy_loss, value_loss, entropy = ppo_update(self.net, self.optimizer, batch, self.config)
        self.opponent_buffer.append(deepcopy(self.net.state_dict()))
        if len(self.opponent_buffer) > 16:
            self.opponent_buffer.pop(0)
        return {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}

    def estimate_exploitability(self, br_episodes: int = 32, br_updates: int = 3) -> float:
        br = PolicyValueNet().to(self.device)
        br_opt = optim.Adam(br.parameters(), lr=3e-4)
        br_config = PPOConfig()
        rewards_accum = []
        for _ in range(br_updates):
            obs_list = []
            actions = []
            logprobs = []
            values = []
            rewards = []
            dones = []
            for _ in range(br_episodes):
                obs, mask = self.env.reset()
                traj_obs = []
                traj_actions = []
                traj_logprobs = []
                traj_values = []
                traj_rewards = []
                traj_dones = []
                done = False
                final_reward = 0.0
                while not done:
                    player = self.env.state.current_player  # type: ignore
                    if player == 0:
                        with torch.no_grad():
                            opp_obs = obs_to_tensor(obs, self.device)
                            mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                            action_main, _, _ = self.net.act(opp_obs, mask_t)
                        obs, reward, done, info = self.env.step(action_main)
                        if done:
                            final_reward = info.get("rewards", [0.0, 0.0])[0]
                            break
                        mask = self.env.legal_action_mask()
                        continue
                    br_obs = obs_to_tensor(obs, self.device)
                    mask_t = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
                    action, logprob, value = br.act(br_obs, mask_t)
                    traj_obs.append({k: v.detach() for k, v in br_obs.items()})
                    traj_actions.append(action)
                    traj_logprobs.append(logprob.detach())
                    traj_values.append(value.detach())
                    obs, reward, done, info = self.env.step(action)
                    traj_rewards.append(0.0)
                    traj_dones.append(float(done))
                    if done:
                        final_reward = info.get("rewards", [0.0, 0.0])[1]
                        break
                    mask = self.env.legal_action_mask()
                if traj_rewards:
                    traj_rewards[-1] = final_reward
                    traj_dones[-1] = 1.0
                    traj_values.append(torch.tensor([0.0], device=self.device))
                    obs_list.extend(traj_obs)
                    actions.extend(traj_actions)
                    logprobs.extend(traj_logprobs)
                    values.extend(traj_values)
                    rewards.extend(traj_rewards)
                    dones.extend(traj_dones)
                    rewards_accum.append(final_reward)
            if not actions:
                continue
            obs_stacked = {k: torch.cat([o[k] for o in obs_list], dim=0) for k in obs_list[0].keys()}
            actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
            logprobs_t = torch.stack(logprobs)
            values_t = torch.stack(values)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
            adv, rets = compute_advantages(rewards_t, values_t, dones_t, br_config.gamma, br_config.lam)
            batch = {
                "obs": obs_stacked,
                "actions": actions_t,
                "logprobs": logprobs_t,
                "advantages": adv.to(self.device),
                "returns": rets.to(self.device),
            }
            ppo_update(br, br_opt, batch, br_config)
        if not rewards_accum:
            return 0.0
        return float(np.mean(rewards_accum))
