"""
PPO — Proximal Policy Optimization (Schulman et al., 2017)

Architecture:
  ActorCriticNetwork: shared MLP trunk (obs_dim → 512 → 256)
                      actor head  → n_actions logits
                      critic head → 1 state value

Training:
  - Collects a rollout of `rollout_steps` environment steps
  - Computes GAE (Generalized Advantage Estimation) advantages
  - Performs `ppo_epochs` passes over the rollout using PPO-clip objective
  - Mini-batch size: `batch_size`  (random shuffle each epoch)
  - Gradient clipping: max norm 0.5

The rollout buffer is reset after every update, so episodes can span
multiple rollouts without any issues.
"""

import io
import random as pyrandom
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

class ActorCriticNetwork(nn.Module):
    """
    Shared-trunk MLP with separate actor and critic heads.

    Forward returns:
        logits  — [batch, n_actions]  (raw action scores)
        values  — [batch, 1]          (state value estimates)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk       = nn.Sequential(*layers)
        self.actor_head  = nn.Linear(in_dim, n_actions)
        self.critic_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        logits   = self.actor_head(features)
        values   = self.critic_head(features)
        return logits, values


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """
    PPO agent with GAE advantage estimation and clip-objective updates.

    Exploration is implicit via the stochastic policy; epsilon is kept
    at 0 for interface compatibility only.

    Hyperparameters
    ---------------
    lr            : float — Adam learning rate
    gamma         : float — discount factor
    gae_lambda    : float — GAE smoothing parameter (λ)
    clip_eps      : float — PPO clipping radius (ε)
    entropy_coef  : float — entropy bonus coefficient
    value_coef    : float — value-loss coefficient
    ppo_epochs    : int   — gradient passes per rollout
    batch_size    : int   — mini-batch size within each epoch
    rollout_steps : int   — steps to collect before each update
    epsilon       : float — always 0 (not used for exploration)
    hidden_sizes  : list  — shared trunk hidden layer widths
    """

    DEFAULT_CONFIG = {
        "lr":            3e-4,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "clip_eps":      0.2,
        "entropy_coef":  0.01,
        "value_coef":    0.5,
        "ppo_epochs":    4,
        "batch_size":    256,
        "rollout_steps": 512,
        "epsilon":       0.0,
        "hidden_sizes":  [512, 256],
    }

    def __init__(self, action_space_size: int, observation_shape: tuple,
                 config: Optional[dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(action_space_size, observation_shape, cfg)

        self.lr            = cfg["lr"]
        self.gamma         = cfg["gamma"]
        self.gae_lambda    = cfg["gae_lambda"]
        self.clip_eps      = cfg["clip_eps"]
        self.entropy_coef  = cfg["entropy_coef"]
        self.value_coef    = cfg["value_coef"]
        self.ppo_epochs    = int(cfg["ppo_epochs"])
        self.batch_size    = int(cfg["batch_size"])
        self.rollout_steps = int(cfg["rollout_steps"])
        self.epsilon       = float(cfg["epsilon"])
        self.hidden_sizes  = list(cfg["hidden_sizes"])

        self.obs_dim   = int(np.prod(observation_shape))
        self.n_actions = action_space_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ac_net    = ActorCriticNetwork(self.obs_dim, self.n_actions, self.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=self.lr)

        # Rollout buffer — lists grown step-by-step, cleared after each update
        self._rb_obs:         List[np.ndarray]       = []
        self._rb_actions:     List[int]              = []
        self._rb_log_probs:   List[torch.Tensor]     = []
        self._rb_rewards:     List[float]            = []
        self._rb_values:      List[torch.Tensor]     = []
        self._rb_dones:       List[bool]             = []
        self._rb_legal_masks: List[List[int]]        = []

        self._last_loss: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device)

    def _build_mask(self, legal_actions: List[int]) -> torch.Tensor:
        """Return additive logit mask: 0 for legal, -inf for illegal."""
        mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
        if legal_actions:
            mask[legal_actions] = 0.0
        return mask

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Forward pass through actor-critic, sample action, store rollout data.
        """
        obs_t  = self._obs_to_tensor(obs).unsqueeze(0)        # [1, obs_dim]
        with torch.no_grad():
            logits, value = self.ac_net(obs_t)                # [1,A], [1,1]
        logits = logits.squeeze(0)                            # [A]
        value  = value.squeeze(0)                             # [1]

        mask           = self._build_mask(legal_actions)
        masked_logits  = logits + mask
        dist           = Categorical(logits=masked_logits)
        action         = dist.sample()
        log_prob       = dist.log_prob(action)

        # Store step data in rollout buffer
        self._rb_obs.append(obs.flatten().astype(np.float32))
        self._rb_actions.append(int(action.item()))
        self._rb_log_probs.append(log_prob.detach())
        self._rb_values.append(value.detach())
        self._rb_legal_masks.append(list(legal_actions))

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
        Store reward and done flag. Trigger _ppo_update() when rollout is full.
        """
        self._rb_rewards.append(float(reward))
        self._rb_dones.append(bool(done))

        if len(self._rb_rewards) >= self.rollout_steps:
            # Bootstrap value for the last state
            next_obs_t = self._obs_to_tensor(next_obs).unsqueeze(0)
            with torch.no_grad():
                _, next_value = self.ac_net(next_obs_t)
            self._last_loss = self._ppo_update(float(next_value.item()), done)
            return self._last_loss

        return 0.0

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self, bootstrap_value: float, terminal: bool) -> float:
        """
        Compute GAE advantages, then iterate ppo_epochs times over shuffled
        mini-batches performing PPO-clip + value + entropy updates.
        """
        T = len(self._rb_rewards)
        if T == 0:
            return 0.0

        # ---- Build tensors from rollout buffer ----
        obs_arr     = torch.tensor(np.array(self._rb_obs),      dtype=torch.float32, device=self.device)  # [T, obs_dim]
        actions_arr = torch.tensor(self._rb_actions,            dtype=torch.long,    device=self.device)  # [T]
        old_log_probs = torch.stack(self._rb_log_probs)          # [T]
        values_arr  = torch.cat(self._rb_values).squeeze(-1)    # [T]
        rewards_arr = torch.tensor(self._rb_rewards,            dtype=torch.float32, device=self.device)  # [T]
        dones_arr   = torch.tensor(self._rb_dones,              dtype=torch.float32, device=self.device)  # [T]
        legal_masks = self._rb_legal_masks                       # list of T lists

        # ---- GAE advantage estimation ----
        advantages  = torch.zeros(T, dtype=torch.float32, device=self.device)
        last_gae    = 0.0
        next_val    = 0.0 if terminal else bootstrap_value

        for t in reversed(range(T)):
            next_v     = values_arr[t + 1].item() if t + 1 < T else next_val
            delta      = rewards_arr[t] + self.gamma * next_v * (1.0 - dones_arr[t]) - values_arr[t]
            last_gae   = float(delta) + self.gamma * self.gae_lambda * (1.0 - float(dones_arr[t])) * last_gae
            advantages[t] = last_gae

        returns = advantages + values_arr    # [T]

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # ---- PPO epochs ----
        total_loss = 0.0
        n_updates  = 0
        indices    = list(range(T))

        for _ in range(self.ppo_epochs):
            pyrandom.shuffle(indices)
            for start in range(0, T, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if not batch_idx:
                    continue

                # Batch slices
                b_obs      = obs_arr[batch_idx]
                b_actions  = actions_arr[batch_idx]
                b_old_lp   = old_log_probs[batch_idx]
                b_adv      = advantages[batch_idx]
                b_returns  = returns[batch_idx]
                b_masks    = [legal_masks[i] for i in batch_idx]

                # Recompute log_probs and values from current network
                logits_b, values_b = self.ac_net(b_obs)    # [B,A], [B,1]
                values_b = values_b.squeeze(-1)             # [B]

                # Apply legal masks (per sample)
                masked_logits_b = logits_b.clone()
                for i, legal in enumerate(b_masks):
                    m = self._build_mask(legal)
                    masked_logits_b[i] = masked_logits_b[i] + m

                dist_b    = Categorical(logits=masked_logits_b)
                new_lp    = dist_b.log_prob(b_actions)      # [B]
                entropy   = dist_b.entropy().mean()         # scalar

                # PPO ratio and clipped surrogate
                ratio  = torch.exp(new_lp - b_old_lp)
                surr1  = ratio * b_adv
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * nn.functional.mse_loss(values_b, b_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += float(loss.item())
                n_updates  += 1

        # Clear rollout buffer
        self._rb_obs         = []
        self._rb_actions     = []
        self._rb_log_probs   = []
        self._rb_rewards     = []
        self._rb_values      = []
        self._rb_dones       = []
        self._rb_legal_masks = []

        return total_loss / max(n_updates, 1)

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        """
        PPO updates happen mid-rollout (inside update()), so no action is
        needed at episode boundaries.  Rollout buffers are NOT cleared here
        because a rollout can span multiple episodes.
        """

    def get_config(self) -> dict:
        return {
            "lr":            self.lr,
            "gamma":         self.gamma,
            "gae_lambda":    self.gae_lambda,
            "clip_eps":      self.clip_eps,
            "entropy_coef":  self.entropy_coef,
            "value_coef":    self.value_coef,
            "ppo_epochs":    self.ppo_epochs,
            "batch_size":    self.batch_size,
            "rollout_steps": self.rollout_steps,
            "epsilon":       self.epsilon,
            "hidden_sizes":  self.hidden_sizes,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_extra_state(self) -> dict:
        buf = io.BytesIO()
        torch.save({
            "ac_net":    self.ac_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, buf)
        return {"torch_state": buf.getvalue()}

    def _set_extra_state(self, state: dict) -> None:
        if "torch_state" not in state:
            return
        buf        = io.BytesIO(state["torch_state"])
        checkpoint = torch.load(buf, map_location=self.device)
        self.ac_net.load_state_dict(checkpoint["ac_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    # ------------------------------------------------------------------
    # Web UI compatibility
    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        """PPO has no Q-table; returns 0 for web UI compatibility."""
        return 0
