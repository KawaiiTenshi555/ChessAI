"""
Monte Carlo — Every-Visit MC Control (on-policy)

Principe : accumuler les transitions de l'épisode complet, puis mettre
à jour la Q-table à partir des retours actualisés calculés en fin d'épisode.

Retour actualisé à l'étape t :
    G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γ^(T-t) · r_T

Mise à jour (every-visit) :
    Q(s, a) ← Q(s, a) + α · (G_t − Q(s, a))

Variantes implémentées via le paramètre `first_visit` :
    - every_visit (défaut) : mise à jour à chaque occurrence de (s, a)
    - first_visit          : mise à jour uniquement à la 1ère occurrence par épisode

Avantages vs TD :
    + Pas de biais de bootstrap (utilise le retour réel)
    + Adapté aux épisodes courts ou bien définis
Inconvénients :
    - Variance élevée (sensible aux récompenses tardives)
    - Pas de mise à jour pendant l'épisode → apprentissage plus lent
    - Pas adapté aux environnements continus (sans fin d'épisode)
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):
    """
    Agent Monte Carlo Every-Visit (ou First-Visit) avec politique ε-greedy.

    Hyperparamètres
    ---------------
    alpha        : float  — taux d'apprentissage (step-size fixe)
    gamma        : float  — facteur de décompte
    epsilon      : float  — exploration initiale
    epsilon_min  : float  — exploration minimale
    epsilon_decay: float  — facteur de décroissance par épisode
    first_visit  : bool   — si True, n'utilise que la 1ère visite de (s,a)
    """

    DEFAULT_CONFIG = {
        "alpha": 0.05,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "first_visit": False,
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
        self.first_visit   = cfg["first_visit"]

        self.Q: Dict[bytes, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

        # Buffer de l'épisode courant : liste de (state_key, action, reward)
        self._episode: List[Tuple[bytes, int, float]] = []

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

    def _flush_episode(self):
        """
        Calcule les retours G_t depuis la fin de l'épisode et met à jour Q.
        Appelé automatiquement en fin d'épisode via on_episode_end().
        """
        if self.first_visit:
            # Parcours inverse : on écrase systématiquement, de sorte que la
            # dernière écriture dans le dict temporaire correspond à t minimal
            # (1ʳᵉ visite en ordre chronologique, rencontrée en dernier en sens inverse).
            first_visit_returns: dict = {}
            G = 0.0
            for state_key, action, reward in reversed(self._episode):
                G = reward + self.gamma * G
                first_visit_returns[(state_key, action)] = G

            for (state_key, action), g in first_visit_returns.items():
                self.Q[state_key][action] += self.alpha * (g - self.Q[state_key][action])
        else:
            # Every-visit : mise à jour à chaque occurrence (parcours inverse)
            G = 0.0
            for state_key, action, reward in reversed(self._episode):
                G = reward + self.gamma * G
                self.Q[state_key][action] += self.alpha * (G - self.Q[state_key][action])

        self._episode.clear()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        return self._epsilon_greedy(self._key(obs), legal_actions)

    def update(self, obs: np.ndarray, action: int, reward: float,
               next_obs: np.ndarray, done: bool,
               legal_next_actions: Optional[List[int]] = None) -> Optional[float]:
        """
        Accumule la transition dans le buffer de l'épisode.
        La mise à jour réelle de Q se fait en fin d'épisode dans on_episode_end().
        """
        self._episode.append((self._key(obs), action, reward))
        return None  # pas de loss intermédiaire en Monte Carlo

    def on_episode_end(self, episode: int, reward: float, length: int) -> None:
        self._flush_episode()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_config(self) -> dict:
        return {
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "epsilon":       self.epsilon,
            "epsilon_min":   self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "first_visit":   self.first_visit,
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
