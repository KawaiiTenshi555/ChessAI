"""
DQN — Deep Q-Network (Mnih et al., 2015)

Architecture:
  QNetwork: MLP — input: flattened observation (1088 floats)
                   hidden layers: configurable (default [512, 256])
                   output: Q-values for all 4096 actions

Training:
  - Experience replay buffer (deque)
  - Separate target network, synced every target_update_freq steps
  - Epsilon-greedy exploration with legal-action masking
  - Adam optimizer + gradient clipping
"""

import io
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP that maps a flattened observation to Q-values over all actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent with experience replay and a target network.

    Hyperparameters
    ---------------
    lr               : float — Adam learning rate
    gamma            : float — discount factor
    epsilon          : float — initial exploration rate
    epsilon_min      : float — minimum exploration rate
    epsilon_decay    : float — multiplicative decay per episode
    buffer_size      : int   — maximum replay buffer capacity
    batch_size       : int   — mini-batch size for updates
    target_update_freq: int  — steps between target network syncs
    hidden_sizes     : list  — widths of the hidden layers
    """

    DEFAULT_CONFIG = {
        "lr":                1e-4,
        "gamma":             0.99,
        "epsilon":           1.0,
        "epsilon_min":       0.05,
        "epsilon_decay":     0.9995,
        "buffer_size":       50000,
        "batch_size":        256,
        "target_update_freq": 500,
        "hidden_sizes":      [512, 256],
    }

    def __init__(self, action_space_size: int, observation_shape: tuple,
                 config: Optional[dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(action_space_size, observation_shape, cfg)

        self.lr                 = cfg["lr"]
        self.gamma              = cfg["gamma"]
        self.epsilon            = cfg["epsilon"]
        self.epsilon_min        = cfg["epsilon_min"]
        self.epsilon_decay      = cfg["epsilon_decay"]
        self.buffer_size        = int(cfg["buffer_size"])
        self.batch_size         = int(cfg["batch_size"])
        self.target_update_freq = int(cfg["target_update_freq"])
        self.hidden_sizes       = list(cfg["hidden_sizes"])

        self.obs_dim  = int(np.prod(observation_shape))   # 8*8*17 = 1088
        self.n_actions = action_space_size                 # 4096

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_net     = QNetwork(self.obs_dim, self.n_actions, self.hidden_sizes).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.n_actions, self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # Replay buffer : liste circulaire pour O(1) en accès aléatoire
        self._buffer: list = []
        self._buf_pos: int = 0   # position d'écriture circulaire

        # Internal step counter (separate from training_steps which is managed by train())
        self._update_steps: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Flatten obs and convert to float32 tensor on device."""
        return torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device)

    def _legal_mask(self, legal_actions: List[int]) -> torch.Tensor:
        """Return an additive mask: 0 for legal actions, -inf for illegal ones."""
        mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
        if legal_actions:
            mask[legal_actions] = 0.0
        return mask

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """Epsilon-greedy with legal-action masking."""
        if not legal_actions:
            return 0

        if random.random() < self.epsilon:
            return int(random.choice(legal_actions))

        obs_t = self._obs_to_tensor(obs).unsqueeze(0)   # [1, obs_dim]
        with torch.no_grad():
            q_values = self.q_net(obs_t).squeeze(0)     # [n_actions]
        q_values = q_values + self._legal_mask(legal_actions)
        return int(q_values.argmax().item())

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        legal_next_actions: Optional[List[int]] = None,
    ) -> Optional[float]:
        """
        Store transition in replay buffer and, when ready, run one gradient step.

        Returns the training loss (MSE) or 0.0 if the buffer is not large enough yet.
        """
        entry = (
            obs.flatten().astype(np.float32),
            int(action),
            float(reward),
            next_obs.flatten().astype(np.float32),
            bool(done),
            list(legal_next_actions) if legal_next_actions else [],
        )
        if len(self._buffer) < self.buffer_size:
            self._buffer.append(entry)
        else:
            self._buffer[self._buf_pos] = entry
        self._buf_pos = (self._buf_pos + 1) % self.buffer_size

        if len(self._buffer) < self.batch_size:
            return 0.0

        return self._train_step()

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        """Decay epsilon at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_config(self) -> dict:
        return {
            "lr":                self.lr,
            "gamma":             self.gamma,
            "epsilon":           self.epsilon,
            "epsilon_min":       self.epsilon_min,
            "epsilon_decay":     self.epsilon_decay,
            "buffer_size":       self.buffer_size,
            "batch_size":        self.batch_size,
            "target_update_freq": self.target_update_freq,
            "hidden_sizes":      self.hidden_sizes,
        }

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(self) -> float:
        """Sample a mini-batch from the buffer and perform one gradient update."""
        batch = random.sample(self._buffer, self.batch_size)
        obs_b, act_b, rew_b, next_obs_b, done_b, next_legal_b = zip(*batch)

        # torch.from_numpy évite une copie mémoire vs torch.tensor(np.array(...))
        obs_t      = torch.from_numpy(np.stack(obs_b)).to(self.device, non_blocking=True)
        next_obs_t = torch.from_numpy(np.stack(next_obs_b)).to(self.device, non_blocking=True)
        act_t      = torch.tensor(act_b,  dtype=torch.long,    device=self.device)
        rew_t      = torch.tensor(rew_b,  dtype=torch.float32, device=self.device)
        done_t     = torch.tensor(done_b, dtype=torch.float32, device=self.device)

        # Current Q-values: Q(s, a) for the taken actions
        q_values = self.q_net(obs_t)                        # [B, n_actions]
        q_sa = q_values.gather(1, act_t.unsqueeze(1)).squeeze(1)  # [B]

        # Target Q-values: r + γ * max_a' Q_target(s', a')  (0 if done)
        with torch.no_grad():
            next_q = self.target_net(next_obs_t)            # [B, n_actions]
            # Build legal mask matrix in one shot [B, n_actions]
            legal_mask = torch.full(
                (self.batch_size, self.n_actions), float('-inf'), device=self.device
            )
            for i, legal in enumerate(next_legal_b):
                if legal:
                    legal_mask[i, legal] = 0.0
                else:
                    legal_mask[i] = 0.0  # terminal — all actions allowed (zeroed by done_t)
            next_q = next_q + legal_mask
            max_next_q = next_q.max(dim=1).values           # [B]
            targets = rew_t + self.gamma * max_next_q * (1.0 - done_t)

        loss = nn.functional.mse_loss(q_sa, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._update_steps += 1

        # Sync target network periodically
        if self._update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_extra_state(self) -> dict:
        buf = io.BytesIO()
        torch.save({
            "q_net":         self.q_net.state_dict(),
            "target_net":    self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "epsilon":       self.epsilon,
            "_update_steps": self._update_steps,
            "_buf_pos":      self._buf_pos,
        }, buf)
        return {"torch_state": buf.getvalue()}

    def _set_extra_state(self, state: dict) -> None:
        if "torch_state" not in state:
            return
        buf = io.BytesIO(state["torch_state"])
        checkpoint = torch.load(buf, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon       = checkpoint.get("epsilon", self.epsilon)
        self._update_steps = checkpoint.get("_update_steps", 0)
        self._buf_pos      = checkpoint.get("_buf_pos", 0)

    # ------------------------------------------------------------------
    # Web UI compatibility
    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        """Returns the current replay buffer size (used by the web UI)."""
        return len(self._buffer)
