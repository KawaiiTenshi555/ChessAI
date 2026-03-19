"""
SARSA — State-Action-Reward-State-Action (on-policy TD(0))

Règle de mise à jour :
    Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]

où a' est choisi par la même politique ε-greedy que a (on-policy).
La Q-table est un dictionnaire creuse : les états non vus valent 0 par défaut.

Complexité mémoire : O(|états vus| × |actions|)
→ Adapté à des espaces d'états réduits ou en early training.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..base_agent import BaseAgent


class SarsaAgent(BaseAgent):
    """
    Agent SARSA on-policy TD(0) avec politique ε-greedy.

    Hyperparamètres
    ---------------
    alpha        : float  — taux d'apprentissage (0 < α ≤ 1)
    gamma        : float  — facteur de décompte  (0 < γ ≤ 1)
    epsilon      : float  — exploration initiale (0 ≤ ε ≤ 1)
    epsilon_min  : float  — exploration minimale
    epsilon_decay: float  — facteur de décroissance par épisode
    """

    DEFAULT_CONFIG = {
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
    }

    def __init__(self, action_space_size: int, observation_shape: tuple,
                 config: Optional[dict] = None):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(action_space_size, observation_shape, cfg)

        self.alpha         = cfg["alpha"]
        self.gamma         = cfg["gamma"]
        self.epsilon       = cfg["epsilon"]
        self.epsilon_min   = cfg["epsilon_min"]
        self.epsilon_decay = cfg["epsilon_decay"]

        # Q-table : state_key → {action → Q-value}
        self.Q: Dict[bytes, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Mémorise l'action a' choisie lors du update précédent (nécessaire pour on-policy)
        self._next_action: Optional[int] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(obs: np.ndarray) -> bytes:
        return obs.tobytes()

    def _epsilon_greedy(self, key: bytes, legal_actions: List[int]) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.choice(legal_actions))
        q = self.Q[key]
        return max(legal_actions, key=lambda a: q[a])

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Retourne l'action pré-choisie a' (stockée lors du update précédent)
        si elle existe, sinon en génère une nouvelle ε-greedy.
        """
        if self._next_action is not None:
            action = self._next_action
            self._next_action = None
            return action
        return self._epsilon_greedy(self._key(obs), legal_actions)

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool,
               legal_next_actions: Optional[List[int]] = None) -> float:
        """
        Mise à jour SARSA :
            TD_error = r + γ·Q(s', a') − Q(s, a)
            Q(s, a) ← Q(s, a) + α · TD_error
        """
        s  = self._key(obs)
        s2 = self._key(next_obs)

        if done or not legal_next_actions:
            q_next = 0.0
            self._next_action = None
        else:
            a2 = self._epsilon_greedy(s2, legal_next_actions)
            q_next = self.Q[s2][a2]
            self._next_action = a2  # réutilisé au prochain select_action

        td_error = reward + self.gamma * q_next - self.Q[s][action]
        self.Q[s][action] += self.alpha * td_error
        return float(abs(td_error))

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._next_action = None  # reset de sécurité entre épisodes

    def get_config(self) -> dict:
        return {
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

    def _get_extra_state(self) -> dict:
        return {"Q": {k: dict(v) for k, v in self.Q.items()}, "epsilon": self.epsilon}

    def _set_extra_state(self, state: dict) -> None:
        if "Q" in state:
            self.Q = defaultdict(lambda: defaultdict(float),
                                 {k: defaultdict(float, v) for k, v in state["Q"].items()})
        if "epsilon" in state:
            self.epsilon = state["epsilon"]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        """Nombre d'états distincts dans la Q-table."""
        return len(self.Q)
