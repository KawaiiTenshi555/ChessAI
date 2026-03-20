"""
REINFORCE — Monte Carlo Policy Gradient (Williams, 1992)

The policy is a stochastic MLP that outputs action logits.
At each step the agent samples an action from the masked categorical
distribution and stores its log-probability.  At episode end, the
stored log-probs are multiplied by discounted returns and averaged to
form the policy-gradient loss.

Optional variance-reduction baseline: subtract the mean return from
each discounted return before computing the loss.
"""

import io
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from ..base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """MLP policy: observation → action logits."""

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

class REINFORCEAgent(BaseAgent):
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.

    Exploration is implicit: the stochastic policy always samples from
    the action distribution — epsilon is kept at 0 purely for interface
    compatibility with the web UI and tabular agents.

    Hyperparameters
    ---------------
    lr           : float — Adam learning rate
    gamma        : float — discount factor
    epsilon      : float — always 0 (not used)
    hidden_sizes : list  — MLP hidden layer widths
    baseline     : bool  — subtract mean return for variance reduction
    """

    DEFAULT_CONFIG = {
        "lr":           3e-4,
        "gamma":        0.99,
        "epsilon":      0.0,
        "hidden_sizes": [512, 256],
        "baseline":     True,
    }

    def __init__(self, action_space_size: int, observation_shape: tuple,
                 config: Optional[dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(action_space_size, observation_shape, cfg)

        self.lr           = cfg["lr"]
        self.gamma        = cfg["gamma"]
        self.epsilon      = float(cfg["epsilon"])   # 0.0 — kept for UI compat
        self.hidden_sizes = list(cfg["hidden_sizes"])
        self.baseline     = bool(cfg["baseline"])

        self.obs_dim   = int(np.prod(observation_shape))
        self.n_actions = action_space_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy    = PolicyNetwork(self.obs_dim, self.n_actions, self.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Episode buffers — cleared in on_episode_end()
        self._log_probs: List[torch.Tensor] = []
        self._rewards:   List[float]        = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Sample an action from the masked stochastic policy.

        Illegal actions receive -inf logit so they get zero probability.
        The sampled action's log-probability is appended to _log_probs.
        """
        obs_t  = self._obs_to_tensor(obs).unsqueeze(0)   # [1, obs_dim]
        logits = self.policy(obs_t).squeeze(0)            # [n_actions]

        # Mask illegal actions
        mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
        if legal_actions:
            mask[legal_actions] = 0.0
        masked_logits = logits + mask

        dist   = Categorical(logits=masked_logits)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return int(action.item())

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
        Buffer the reward; the actual gradient update happens in on_episode_end().
        Returns 0.0 to satisfy the BaseAgent interface.
        """
        self._rewards.append(float(reward))
        return 0.0

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        """
        Compute discounted returns, form the REINFORCE loss, and update the policy.
        Clears episode buffers afterwards.
        """
        if not self._log_probs:
            return

        # Compute discounted returns G_t = Σ γ^k r_{t+k}
        returns = []
        G = 0.0
        for r in reversed(self._rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Variance-reduction baseline: subtract the mean return
        if self.baseline:
            returns_t = returns_t - returns_t.mean()

        # Stack log-probs into a [T] tensor
        log_probs_t = torch.stack(self._log_probs)   # [T]

        # Policy-gradient loss: -E[log π(a|s) * G_t]
        loss = -(log_probs_t * returns_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Clear episode buffers
        self._log_probs = []
        self._rewards   = []

    def get_config(self) -> dict:
        return {
            "lr":           self.lr,
            "gamma":        self.gamma,
            "epsilon":      self.epsilon,
            "hidden_sizes": self.hidden_sizes,
            "baseline":     self.baseline,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_extra_state(self) -> dict:
        buf = io.BytesIO()
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, buf)
        return {"torch_state": buf.getvalue()}

    def _set_extra_state(self, state: dict) -> None:
        if "torch_state" not in state:
            return
        buf        = io.BytesIO(state["torch_state"])
        checkpoint = torch.load(buf, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    # ------------------------------------------------------------------
    # Web UI compatibility
    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        """REINFORCE has no Q-table; returns 0 for web UI compatibility."""
        return 0
