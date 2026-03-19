"""
BaseAgent — abstract interface that every RL agent must implement.

All agents receive:
  - An observation (numpy array from ChessEnv)
  - A list of valid actions (integers)

And must be able to:
  - Select an action
  - Update their parameters from experience
  - Train over multiple episodes
  - Save / load their model
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all chess RL agents.

    Subclasses must implement:
        select_action, update, get_config

    Subclasses should override train, save, load as needed.
    """

    def __init__(self, action_space_size: int, observation_shape: tuple, config: dict = None):
        """
        Parameters
        ----------
        action_space_size : int
            Total number of possible actions (4096 for chess).
        observation_shape : tuple
            Shape of the observation array, e.g. (8, 8, 17).
        config : dict
            Hyperparameters for the agent.
        """
        self.action_space_size = action_space_size
        self.observation_shape = observation_shape
        self.config = config or {}

        # Training statistics (updated during train())
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_steps: int = 0

    # ------------------------------------------------------------------
    # Core interface — must be implemented by every agent
    # ------------------------------------------------------------------

    @abstractmethod
    def select_action(self, observation: np.ndarray, legal_actions: List[int]) -> int:
        """
        Choose an action given the current observation.

        Parameters
        ----------
        observation  : np.ndarray — current state from the environment
        legal_actions: list[int]  — valid actions at this step

        Returns
        -------
        int : chosen action
        """

    @abstractmethod
    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        legal_next_actions: Optional[List[int]] = None,
    ) -> Optional[float]:
        """
        Update the agent's parameters based on one transition.

        Returns the training loss (or None if not applicable).
        """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return a dict of hyperparameters and settings."""

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        """
        Hook called at the end of each episode.
        Override to implement epsilon decay, buffer flushes, etc.
        """

    # ------------------------------------------------------------------
    # Training loop — can be overridden for off-policy / batched updates
    # ------------------------------------------------------------------

    def train(self, env, n_episodes: int, verbose: bool = False) -> Dict[str, Any]:
        """
        Run `n_episodes` of training against `env`.

        Returns a stats dict: {episode_rewards, episode_lengths, ...}.
        """
        self.episode_rewards = []
        self.episode_lengths = []

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0.0
            ep_length = 0

            while not done:
                legal_actions = info.get("legal_actions", list(range(self.action_space_size)))
                action = self.select_action(obs, legal_actions)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                next_legal = next_info.get("legal_actions", []) if not done else []
                loss = self.update(obs, action, reward, next_obs, done, next_legal)

                obs = next_obs
                info = next_info
                ep_reward += reward
                ep_length += 1
                self.training_steps += 1

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)
            self.on_episode_end(ep, ep_reward, ep_length)

            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                avg_r = np.mean(self.episode_rewards[-100:])
                print(f"[{self.__class__.__name__}] Episode {ep+1}/{n_episodes} "
                      f"| Avg reward (last 100): {avg_r:.3f} "
                      f"| Length: {ep_length}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_steps": self.training_steps,
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
        }

    # ------------------------------------------------------------------
    # Persistence — override for models with weights
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        import pickle
        state = {
            "config": self.get_config(),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_steps": self.training_steps,
            **self._get_extra_state(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.episode_rewards = state.get("episode_rewards", [])
        self.episode_lengths = state.get("episode_lengths", [])
        self.training_steps = state.get("training_steps", 0)
        self._set_extra_state(state)

    def _get_extra_state(self) -> dict:
        """Override to include agent-specific data in save()."""
        return {}

    def _set_extra_state(self, state: dict) -> None:
        """Override to restore agent-specific data from load()."""

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.get_config()})"
