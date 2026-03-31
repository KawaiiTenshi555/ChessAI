"""
ChessEnv — Gymnasium environment for chess.

Design:
  - The controlled agent always plays as `player_color` (WHITE by default).
  - After the agent's move, the opponent plays automatically using `opponent_policy`.
  - Observation: (8, 8, 17) float32 — see ChessBoard.get_observation().
  - Action space: Discrete(4096) encoded as from_sq * 64 + to_sq.
    Promotions default to queen; under-promotions are not distinguished in the
    flat action encoding (the engine automatically promotes to queen when
    a pawn reaches the back rank via that action).
  - Invalid actions receive a small penalty and a random legal move is played instead.
"""

import random
from typing import Callable, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .board import ChessBoard, Move, WHITE, BLACK, QUEEN


class ChessEnv(gym.Env):
    """
    Single-agent Gymnasium chess environment.

    Parameters
    ----------
    render_mode : str | None
        'ascii' or 'human' for terminal output, None for no rendering.
    opponent_policy : callable | None
        Function (obs, legal_moves, board) -> Move | int.
        Defaults to a uniform-random policy.
    player_color : int
        WHITE (1) or BLACK (-1). The agent controls this color.
    reward_shaping : bool
        If True, add a small material-balance bonus each step.
    terminal_win_reward : float
        Reward applied when the agent wins the game.
    terminal_loss_penalty : float
        Penalty applied when the agent loses the game.
    invalid_action_penalty : float
        Penalty applied when the agent submits an illegal action.
    """

    metadata = {"render_modes": ["ascii", "human"], "render_fps": 1}

    # Flat action space: from_sq ∈ [0,63], to_sq ∈ [0,63] → 4096 actions.
    # Promotion to queen is implicit when a pawn reaches the back rank.
    N_ACTIONS = 4096

    # Valeurs centipions normalisées (Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9)
    PIECE_VALUES = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        opponent_policy: Optional[Callable] = None,
        player_color: int = WHITE,
        reward_shaping: bool = False,
        capture_reward_scale: float = 0.0,
        loss_penalty_scale: float = 0.0,
        terminal_win_reward: float = 1.0,
        terminal_loss_penalty: float = 1.0,
        invalid_action_penalty: float = -0.05,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.opponent_policy: Callable = opponent_policy or self._random_policy
        self.player_color = player_color
        self.reward_shaping = reward_shaping
        self.capture_reward_scale = capture_reward_scale
        self.loss_penalty_scale = loss_penalty_scale
        self.terminal_win_reward = terminal_win_reward
        self.terminal_loss_penalty = terminal_loss_penalty
        self.invalid_action_penalty = invalid_action_penalty

        # Gymnasium interface
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8, 8, 17), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        self.board: ChessBoard = ChessBoard()
        self._game_over: bool = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.board = ChessBoard()
        self._game_over = False

        # If the agent plays black, let white (opponent) move first.
        if self.player_color == BLACK:
            self._play_opponent()

        return self._obs(), self._info()

    def step(self, action: int):
        if self._game_over:
            raise RuntimeError("Game is over — call reset() before stepping.")

        reward = 0.0
        legal_map = self._legal_map()

        # --- Agent's move ---
        mat_before = self._material_score() if self.reward_shaping else 0.0

        if action in legal_map:
            self.board._apply_move_unchecked(legal_map[action])
        else:
            # Invalid action: apply penalty, fall back to random legal move.
            reward += self.invalid_action_penalty
            if legal_map:
                self.board._apply_move_unchecked(random.choice(list(legal_map.values())))

        # Shaped reward for agent's move (capture or loss via promo capture)
        if self.reward_shaping:
            reward += self._capture_reward(self._material_score() - mat_before)

        # Check terminal after agent's move
        result = self.board.get_result()
        if result is not None:
            self._game_over = True
            reward += self._outcome_reward(result)
            if self.render_mode in ("ascii", "human"):
                self.render()
            return self._obs(), reward, True, False, self._info()

        # --- Opponent's move ---
        mat_before_opp = self._material_score() if self.reward_shaping else 0.0
        self._play_opponent()

        # Shaped reward for opponent's move (piece the agent lost)
        if self.reward_shaping:
            reward += self._capture_reward(self._material_score() - mat_before_opp)

        # Check terminal after opponent's move
        result = self.board.get_result()
        terminated = result is not None
        if terminated:
            self._game_over = True
            reward += self._outcome_reward(result)

        if self.render_mode in ("ascii", "human"):
            self.render()

        return self._obs(), reward, terminated, False, self._info()

    def render(self):
        if self.render_mode in ("ascii", "human"):
            print(self.board.render_ascii())

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    @staticmethod
    def move_to_action(move: Move) -> int:
        """Encode a Move into a flat integer action (from_sq * 64 + to_sq)."""
        return move.from_sq * 64 + move.to_sq

    @staticmethod
    def action_to_uci(action: int) -> str:
        files = "abcdefgh"
        from_sq, to_sq = action // 64, action % 64
        return (files[from_sq % 8] + str(from_sq // 8 + 1)
                + files[to_sq % 8] + str(to_sq // 8 + 1))

    def get_legal_actions(self) -> List[int]:
        """Return the list of valid flat actions for the current position."""
        return list(self._legal_map().keys())

    def action_to_move(self, action: int) -> Optional[Move]:
        """Resolve a flat action to a legal Move, or None if invalid."""
        return self._legal_map().get(action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _legal_map(self) -> dict:
        """Build {action_int: Move} for all legal moves in the current position."""
        legal: dict = {}
        for move in self.board.get_legal_moves():
            action = self.move_to_action(move)
            # Prefer queen-promotion over under-promotions for the same action key.
            if action not in legal or move.promotion == QUEEN:
                legal[action] = move
        return legal

    def _play_opponent(self):
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return
        obs = self._obs()
        result = self.opponent_policy(obs, legal_moves, self.board)
        if isinstance(result, Move):
            self.board._apply_move_unchecked(result)
        elif isinstance(result, int):
            lmap = self._legal_map()
            move = lmap.get(result, random.choice(legal_moves))
            self.board._apply_move_unchecked(move)
        else:
            raise ValueError(f"opponent_policy must return a Move or int, got {type(result)}")

    @staticmethod
    def _random_policy(obs: np.ndarray, legal_moves: List[Move], board: ChessBoard) -> Move:
        return random.choice(legal_moves)

    def _outcome_reward(self, result) -> float:
        if result == self.player_color:
            return self.terminal_win_reward
        if result == -self.player_color:
            return -self.terminal_loss_penalty
        return 0.0   # Draw

    def _material_score(self) -> float:
        """Material balance from the agent's point of view (in pawns)."""
        score = 0.0
        for sq in range(64):
            p = self.board.get_piece(sq)
            if p != 0:
                v = self.PIECE_VALUES[abs(p)]
                score += v if self.board.color_of(p) == self.player_color else -v
        return score

    def _capture_reward(self, delta: float) -> float:
        """
        Convert a material delta into a shaped reward.
        delta > 0 : agent captured a piece  → capture_reward_scale * delta
        delta < 0 : agent lost a piece      → -loss_penalty_scale * delta  (negative reward)
        """
        if not self.reward_shaping:
            return 0.0
        r = 0.0
        if delta > 0:
            r += self.capture_reward_scale * delta
        elif delta < 0:
            r -= self.loss_penalty_scale * (-delta)
        return r

    def _obs(self) -> np.ndarray:
        obs = self.board.get_observation()
        # Flip board so the agent always sees itself at the bottom.
        if self.player_color == BLACK:
            obs = obs[::-1, ::-1, :].copy()
        return obs

    def _info(self) -> dict:
        legal_moves = self.board.get_legal_moves()
        return {
            "turn": "white" if self.board.turn == WHITE else "black",
            "fullmove": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "in_check": self.board.is_in_check(self.board.turn),
            "legal_moves_uci": [m.to_uci() for m in legal_moves],
            "legal_actions": [self.move_to_action(m) for m in legal_moves],
            "result": self.board.get_result(),
        }
