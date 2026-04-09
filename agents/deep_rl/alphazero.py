"""
AlphaZero-style agent for the local chess environment.

This implementation keeps the project architecture simple:
  - A policy-value MLP predicts move priors and a scalar outcome value
  - MCTS uses those priors to improve move selection
  - Training is done through self-play instead of the generic env loop

It is intentionally lightweight so it can run on the current project
without requiring a separate distributed self-play pipeline.
"""

from __future__ import annotations

import io
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chess_env.board import BLACK, QUEEN, WHITE, ChessBoard
from ..base_agent import BaseAgent


@dataclass
class MCTSNode:
    """One node in the Monte Carlo tree."""

    board: ChessBoard
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroNetwork(nn.Module):
    """Shared-trunk MLP with a policy head and a value head."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden

        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, n_actions)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        values = torch.tanh(self.value_head(features))
        return policy_logits, values


class AlphaZeroAgent(BaseAgent):
    """
    Lightweight AlphaZero-style agent.

    The agent uses:
      - canonical board encoding from the side-to-move perspective
      - PUCT-based MCTS for move selection
      - self-play to generate policy/value supervision targets
    """

    DEFAULT_CONFIG = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "hidden_sizes": [512, 256],
        "mcts_simulations": 32,
        "c_puct": 1.5,
        "temperature": 1.0,
        "temperature_drop_move": 12,
        "dirichlet_alpha": 0.30,
        "dirichlet_epsilon": 0.25,
        "replay_buffer_size": 2048,
        "training_batches_per_episode": 8,
        "policy_loss_coef": 1.0,
        "value_loss_coef": 1.0,
        "max_game_length": 256,
    }

    def __init__(
        self,
        action_space_size: int,
        observation_shape: tuple,
        config: Optional[dict] = None,
    ):
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(action_space_size, observation_shape, cfg)

        self.lr = float(cfg["lr"])
        self.weight_decay = float(cfg["weight_decay"])
        self.batch_size = int(cfg["batch_size"])
        self.hidden_sizes = list(cfg["hidden_sizes"])
        self.mcts_simulations = int(cfg["mcts_simulations"])
        self.c_puct = float(cfg["c_puct"])
        self.temperature = float(cfg["temperature"])
        self.temperature_drop_move = int(cfg["temperature_drop_move"])
        self.dirichlet_alpha = float(cfg["dirichlet_alpha"])
        self.dirichlet_epsilon = float(cfg["dirichlet_epsilon"])
        self.replay_buffer_size = int(cfg["replay_buffer_size"])
        self.training_batches_per_episode = int(cfg["training_batches_per_episode"])
        self.policy_loss_coef = float(cfg["policy_loss_coef"])
        self.value_loss_coef = float(cfg["value_loss_coef"])
        self.max_game_length = int(cfg["max_game_length"])

        # Kept for UI compatibility, but not exposed as a tunable hyperparameter.
        self.epsilon = 0.0

        self.obs_dim = int(np.prod(observation_shape))
        self.n_actions = action_space_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = AlphaZeroNetwork(self.obs_dim, self.n_actions, self.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self._replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(
            maxlen=self.replay_buffer_size
        )
        self._optimization_steps = 0
        self._last_total_loss = 0.0
        self._last_policy_loss = 0.0
        self._last_value_loss = 0.0

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _board_to_observation(self, board: ChessBoard) -> np.ndarray:
        """
        Canonical encoding from the side-to-move perspective.

        Channels:
          0-5   current player pieces
          6-11  opponent pieces
          12    side to move (always 1 in canonical form)
          13    en passant square
          14    current player castle kingside
          15    current player castle queenside
          16    opponent has any castling right
        """
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        current = board.turn
        opponent = -current

        for square in range(64):
            piece = board.get_piece(square)
            if piece == 0:
                continue

            row, col = ChessBoard.rc(square)
            if current == BLACK:
                row, col = 7 - row, 7 - col

            offset = 0 if board.color_of(piece) == current else 6
            channel = offset + abs(piece) - 1
            obs[row, col, channel] = 1.0

        obs[:, :, 12] = 1.0

        if board.en_passant_sq is not None:
            row, col = ChessBoard.rc(board.en_passant_sq)
            if current == BLACK:
                row, col = 7 - row, 7 - col
            obs[row, col, 13] = 1.0

        if board.castling_rights[current]["K"]:
            obs[:, :, 14] = 1.0
        if board.castling_rights[current]["Q"]:
            obs[:, :, 15] = 1.0
        if board.castling_rights[opponent]["K"] or board.castling_rights[opponent]["Q"]:
            obs[:, :, 16] = 1.0

        return obs

    @staticmethod
    def _legal_map(board: ChessBoard) -> Dict[int, object]:
        legal: Dict[int, object] = {}
        for move in board.get_legal_moves():
            action = move.to_action()
            if action not in legal or move.promotion == QUEEN:
                legal[action] = move
        return legal

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------

    def _evaluate_board(self, board: ChessBoard) -> Tuple[np.ndarray, float]:
        obs = self._board_to_observation(board)
        self.network.eval()
        with torch.no_grad():
            logits, value = self.network(self._obs_to_tensor(obs).unsqueeze(0))
        return logits.squeeze(0).detach().cpu().numpy(), float(value.item())

    def _expand_node(self, node: MCTSNode, add_root_noise: bool = False) -> float:
        result = node.board.get_result()
        if result is not None:
            node.is_expanded = True
            if result == 0:
                return 0.0
            return 1.0 if result == node.board.turn else -1.0

        legal_map = self._legal_map(node.board)
        if not legal_map:
            node.is_expanded = True
            return 0.0

        logits, value = self._evaluate_board(node.board)
        legal_actions = list(legal_map.keys())
        legal_logits = np.array([logits[action] for action in legal_actions], dtype=np.float64)
        legal_logits -= np.max(legal_logits)
        priors = np.exp(legal_logits)
        priors_sum = float(priors.sum())
        if priors_sum <= 0.0 or not np.isfinite(priors_sum):
            priors = np.ones(len(legal_actions), dtype=np.float64) / max(len(legal_actions), 1)
        else:
            priors /= priors_sum

        if add_root_noise and len(legal_actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            priors = (
                (1.0 - self.dirichlet_epsilon) * priors
                + self.dirichlet_epsilon * noise
            )

        for action, prior in zip(legal_actions, priors):
            child_board = node.board.copy()
            child_board._apply_move_unchecked(legal_map[action])
            node.children[action] = MCTSNode(board=child_board, prior=float(prior))

        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        best_action = -1
        best_child: Optional[MCTSNode] = None
        best_score = -float("inf")
        sqrt_visits = math.sqrt(node.visit_count + 1.0)

        for action, child in node.children.items():
            q_value = -child.value
            u_value = self.c_puct * child.prior * sqrt_visits / (1.0 + child.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS selection failed on a node without children.")
        return best_action, best_child

    @staticmethod
    def _backpropagate(path: List[MCTSNode], leaf_value: float) -> None:
        value = float(leaf_value)
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _run_mcts(self, board: ChessBoard, add_root_noise: bool) -> MCTSNode:
        root = MCTSNode(board=board.copy())
        self._expand_node(root, add_root_noise=add_root_noise)
        if not root.children:
            return root

        simulations = max(1, self.mcts_simulations)
        for _ in range(simulations):
            node = root
            path = [node]

            while node.children:
                _, node = self._select_child(node)
                path.append(node)
                if not node.is_expanded:
                    break

            leaf_value = self._expand_node(node, add_root_noise=False)
            self._backpropagate(path, leaf_value)

        return root

    def _root_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        policy = np.zeros(self.n_actions, dtype=np.float32)
        if not root.children:
            return policy

        actions = list(root.children.keys())
        visits = np.array(
            [root.children[action].visit_count for action in actions],
            dtype=np.float64,
        )

        if temperature <= 1e-6:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
        else:
            powered = np.power(np.maximum(visits, 0.0), 1.0 / temperature)
            denom = float(powered.sum())
            if denom <= 0.0 or not np.isfinite(denom):
                probs = np.ones_like(powered) / len(powered)
            else:
                probs = powered / denom

        policy[actions] = probs.astype(np.float32)
        return policy

    def _sample_action_from_policy(self, policy: np.ndarray, legal_actions: List[int]) -> int:
        if not legal_actions:
            return 0

        probs = np.array([policy[action] for action in legal_actions], dtype=np.float64)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return int(random.choice(legal_actions))

        probs /= total
        return int(np.random.choice(legal_actions, p=probs))

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_actions: List[int]) -> int:
        """
        Fallback action selector when only an observation is available.

        This path does not run MCTS because the board object is required for tree
        expansion. It still returns a legal move using the policy head.
        """
        if not legal_actions:
            return 0

        self.network.eval()
        with torch.no_grad():
            logits, _ = self.network(self._obs_to_tensor(obs).unsqueeze(0))

        masked_logits = torch.full((self.n_actions,), float("-inf"), device=self.device)
        masked_logits[legal_actions] = logits.squeeze(0)[legal_actions]
        return int(masked_logits.argmax().item())

    def select_move(self, board: ChessBoard, temperature: float = 0.0):
        legal_map = self._legal_map(board)
        if not legal_map:
            return None

        root = self._run_mcts(board, add_root_noise=False)
        policy = self._root_policy(root, temperature=temperature)
        action = self._sample_action_from_policy(policy, list(legal_map.keys()))
        return legal_map.get(action, legal_map[next(iter(legal_map))])

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        legal_next_actions: Optional[List[int]] = None,
    ) -> Optional[float]:
        # AlphaZero training is driven by self-play, not the generic step API.
        return 0.0

    def train(self, env, n_episodes: int, verbose: bool = False) -> Dict[str, object]:
        return self.train_self_play(n_episodes=n_episodes, progress_callback=None, verbose=verbose)

    def train_self_play(
        self,
        n_episodes: int,
        progress_callback: Optional[Callable[[dict], None]] = None,
        verbose: bool = False,
    ) -> Dict[str, object]:
        self.episode_rewards = []
        self.episode_lengths = []

        for episode in range(n_episodes):
            board = ChessBoard()
            examples: List[Tuple[np.ndarray, np.ndarray, int]] = []
            move_count = 0

            while board.get_result() is None and move_count < self.max_game_length:
                root = self._run_mcts(board, add_root_noise=True)
                legal_actions = list(root.children.keys())
                if not legal_actions:
                    break

                temperature = self.temperature if move_count < self.temperature_drop_move else 0.0
                policy = self._root_policy(root, temperature=temperature)
                action = self._sample_action_from_policy(policy, legal_actions)
                examples.append((self._board_to_observation(board), policy, board.turn))

                move = self._legal_map(board)[action]
                board._apply_move_unchecked(move)
                move_count += 1
                self.training_steps += 1

                if progress_callback is not None:
                    progress_callback(
                        {
                            "episodes_done": episode + 1,
                            "current_episode": episode + 1,
                            "total_episodes": n_episodes,
                            "board": board.copy(),
                            "last_move": move.to_uci(),
                        }
                    )

            result = board.get_result()
            if result is None:
                result = 0

            reward = 0.0 if result == 0 else (1.0 if result == WHITE else -1.0)
            self.episode_rewards.append(reward)
            self.episode_lengths.append(move_count)

            for obs, policy, player in examples:
                if result == 0:
                    value_target = 0.0
                else:
                    value_target = 1.0 if player == result else -1.0
                self._replay_buffer.append((obs.astype(np.float32), policy.astype(np.float32), value_target))

            losses = []
            for _ in range(max(1, self.training_batches_per_episode)):
                loss = self._train_step()
                if loss is not None:
                    losses.append(loss)

            if verbose and (episode + 1) % max(1, n_episodes // 10) == 0:
                avg_reward = float(np.mean(self.episode_rewards[-100:])) if self.episode_rewards else 0.0
                avg_loss = float(np.mean(losses)) if losses else 0.0
                print(
                    f"[AlphaZero] Episode {episode + 1}/{n_episodes} "
                    f"| Avg reward (last 100): {avg_reward:.3f} "
                    f"| Length: {move_count} | Loss: {avg_loss:.4f}"
                )

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_steps": self.training_steps,
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "last_loss": self._last_total_loss,
            "optimization_steps": self._optimization_steps,
        }

    def get_config(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "hidden_sizes": self.hidden_sizes,
            "mcts_simulations": self.mcts_simulations,
            "c_puct": self.c_puct,
            "temperature": self.temperature,
            "temperature_drop_move": self.temperature_drop_move,
            "dirichlet_alpha": self.dirichlet_alpha,
            "dirichlet_epsilon": self.dirichlet_epsilon,
            "replay_buffer_size": self.replay_buffer_size,
            "training_batches_per_episode": self.training_batches_per_episode,
            "policy_loss_coef": self.policy_loss_coef,
            "value_loss_coef": self.value_loss_coef,
            "max_game_length": self.max_game_length,
        }

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _train_step(self) -> Optional[float]:
        if not self._replay_buffer:
            return None

        batch_size = min(self.batch_size, len(self._replay_buffer))
        batch = random.sample(list(self._replay_buffer), batch_size)
        obs_batch, policy_batch, value_batch = zip(*batch)

        obs_t = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
        policy_t = torch.tensor(np.array(policy_batch), dtype=torch.float32, device=self.device)
        value_t = torch.tensor(value_batch, dtype=torch.float32, device=self.device)

        self.network.train()
        policy_logits, values = self.network(obs_t.view(batch_size, -1))
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
        value_loss = nn.functional.mse_loss(values.squeeze(-1), value_t)
        total_loss = self.policy_loss_coef * policy_loss + self.value_loss_coef * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
        self.optimizer.step()

        self._optimization_steps += 1
        self._last_policy_loss = float(policy_loss.item())
        self._last_value_loss = float(value_loss.item())
        self._last_total_loss = float(total_loss.item())
        return self._last_total_loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _get_extra_state(self) -> dict:
        buf = io.BytesIO()
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            buf,
        )
        return {
            "torch_state": buf.getvalue(),
            "replay_buffer": list(self._replay_buffer),
            "_optimization_steps": self._optimization_steps,
            "_last_total_loss": self._last_total_loss,
            "_last_policy_loss": self._last_policy_loss,
            "_last_value_loss": self._last_value_loss,
        }

    def _set_extra_state(self, state: dict) -> None:
        if "torch_state" in state:
            buf = io.BytesIO(state["torch_state"])
            checkpoint = torch.load(buf, map_location=self.device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        saved_buffer = state.get("replay_buffer", [])
        self._replay_buffer = deque(saved_buffer, maxlen=self.replay_buffer_size)
        self._optimization_steps = int(state.get("_optimization_steps", 0))
        self._last_total_loss = float(state.get("_last_total_loss", 0.0))
        self._last_policy_loss = float(state.get("_last_policy_loss", 0.0))
        self._last_value_loss = float(state.get("_last_value_loss", 0.0))

    # ------------------------------------------------------------------
    # Web UI compatibility
    # ------------------------------------------------------------------

    @property
    def q_table_size(self) -> int:
        return len(self._replay_buffer)
