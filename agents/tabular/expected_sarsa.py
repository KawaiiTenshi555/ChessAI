"""
Expected SARSA — on-policy TD(0) avec espérance sur la politique

Règle de mise à jour :
    Q(s, a) ← Q(s, a) + α · [r + γ · Σ_a' π(a'|s') · Q(s', a') − Q(s, a)]

où π est la politique ε-greedy :
    - Action gloutonne : probabilité (1 - ε) + ε / |A|
    - Autres actions   : probabilité ε / |A|

Par rapport à SARSA :
    + Variance réduite (on moyenne sur toutes les actions plutôt qu'en échantillonner une)
    + Convergence plus stable
    - Légèrement plus coûteux par step (calcul de l'espérance)

Par rapport à Q-Learning :
    ~ Même biais que Q-Learning si ε → 0
    + Meilleur compromis biais/variance en cours de training
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ..base_agent import BaseAgent


class ExpectedSarsaAgent(BaseAgent):
    """
    Agent Expected SARSA on-policy TD(0) avec politique ε-greedy.

    Hyperparamètres
    ---------------
    alpha        : float  — taux d'apprentissage
    gamma        : float  — facteur de décompte
    epsilon      : float  — exploration initiale
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

        self.Q: Dict[bytes, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

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

    def _expected_value(self, key: bytes, legal_actions: List[int]) -> float:
        """
        Calcule Σ_a π(a|s) · Q(s, a) sous la politique ε-greedy.

        Sous ε-greedy avec |A| actions légales :
            - chaque action reçoit un poids de base ε / |A|
            - l'action gloutonne reçoit un bonus de (1 − ε)
        """
        n = len(legal_actions)
        if n == 0:
            return 0.0

        q = self.Q[key]
        q_vals = [q[a] for a in legal_actions]

        greedy_idx = int(np.argmax(q_vals))
        base_prob  = self.epsilon / n
        # Probabilité de chaque action sous ε-greedy
        probs = [base_prob] * n
        probs[greedy_idx] += 1.0 - self.epsilon

        return float(np.dot(probs, q_vals))

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        return self._epsilon_greedy(self._key(obs), legal_actions)

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool,
               legal_next_actions: Optional[List[int]] = None) -> float:
        """
        Mise à jour Expected SARSA :
            expected_q = Σ_a' π(a'|s') · Q(s', a')
            TD_error   = r + γ · expected_q − Q(s, a)
            Q(s, a)   ← Q(s, a) + α · TD_error
        """
        s  = self._key(obs)
        s2 = self._key(next_obs)

        if done or not legal_next_actions:
            q_next = 0.0
        else:
            q_next = self._expected_value(s2, legal_next_actions)

        td_error = reward + self.gamma * q_next - self.Q[s][action]
        self.Q[s][action] += self.alpha * td_error
        return float(abs(td_error))

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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

    @property
    def q_table_size(self) -> int:
        return len(self.Q)
