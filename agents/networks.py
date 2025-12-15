from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class HistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 4, d_model: int = 64, nhead: int = 4, num_layers: int = 2, max_len: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model, max_len)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.proj(history)
        x = self.pos(x)
        out = self.encoder(x)
        pooled = out.mean(dim=1)
        return pooled


class PolicyValueNet(nn.Module):
    def __init__(self, history_len: int = 32, hidden: int = 256, action_dim: int = 6):
        super().__init__()
        self.card_proj = nn.Linear(52, 64)
        self.board_proj = nn.Linear(52, 64)
        self.street_proj = nn.Linear(4, 8)
        self.scalar_proj = nn.Linear(2, 16)
        self.history_encoder = HistoryEncoder(max_len=history_len)
        self.trunk = nn.Sequential(
            nn.Linear(64 + 64 + 8 + 16 + 64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        hole = obs["hole_one_hot"]
        board = obs["board_one_hot"]
        street = obs["street"]
        scalar = torch.cat([obs["pot"], obs["effective_stack"]], dim=-1)
        history = obs["history"]
        hole_f = F.relu(self.card_proj(hole))
        board_f = F.relu(self.board_proj(board))
        street_f = F.relu(self.street_proj(street))
        scalar_f = F.relu(self.scalar_proj(scalar))
        hist_f = F.relu(self.history_encoder(history))
        x = torch.cat([hole_f, board_f, street_f, scalar_f, hist_f], dim=-1)
        trunk = self.trunk(x)
        logits = self.policy_head(trunk)
        value = self.value_head(trunk).squeeze(-1)
        return logits, value

    def act(self, obs: Dict[str, torch.Tensor], legal_mask: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        mask = legal_mask + 1e-9
        masked_logits = logits + torch.log(mask)
        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), logprob, value


def obs_to_tensor(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    import numpy as np

    return {
        "hole_one_hot": torch.tensor(obs["hole_one_hot"], dtype=torch.float32, device=device).unsqueeze(0),
        "board_one_hot": torch.tensor(obs["board_one_hot"], dtype=torch.float32, device=device).unsqueeze(0),
        "pot": torch.tensor(obs["pot"], dtype=torch.float32, device=device).unsqueeze(0),
        "effective_stack": torch.tensor(obs["effective_stack"], dtype=torch.float32, device=device).unsqueeze(0),
        "street": torch.tensor(obs["street"], dtype=torch.float32, device=device).unsqueeze(0),
        "history": torch.tensor(obs["history"], dtype=torch.float32, device=device).unsqueeze(0),
    }
