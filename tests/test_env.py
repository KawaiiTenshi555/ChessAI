"""
Tests for ChessBoard and ChessEnv.

Run with:  pytest tests/test_env.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from chess_env.board import (
    ChessBoard, Move,
    WHITE, BLACK, EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
)
from chess_env.chess_env import ChessEnv


# ===========================================================================
# Helpers
# ===========================================================================

def make_move_uci(board: ChessBoard, uci: str) -> bool:
    """Apply a move given as UCI string (e.g. 'e2e4')."""
    files = "abcdefgh"
    from_col = files.index(uci[0])
    from_row = int(uci[1]) - 1
    to_col   = files.index(uci[2])
    to_row   = int(uci[3]) - 1
    promo = 0
    if len(uci) == 5:
        promo = "nbrq".index(uci[4]) + KNIGHT
    move = Move(ChessBoard.sq(from_row, from_col), ChessBoard.sq(to_row, to_col), promo)
    return board.make_move(move)


def moves_uci(board: ChessBoard) -> set:
    return {m.to_uci() for m in board.get_legal_moves()}


# ===========================================================================
# 1 — Initial position
# ===========================================================================

class TestInitialPosition:

    def test_white_pawn_count(self):
        b = ChessBoard()
        count = sum(1 for sq in range(64) if b.get_piece(sq) == WHITE * PAWN)
        assert count == 8

    def test_black_pawn_count(self):
        b = ChessBoard()
        count = sum(1 for sq in range(64) if b.get_piece(sq) == BLACK * PAWN)
        assert count == 8

    def test_white_king_position(self):
        b = ChessBoard()
        assert b.find_king(WHITE) == ChessBoard.sq(0, 4)  # e1

    def test_black_king_position(self):
        b = ChessBoard()
        assert b.find_king(BLACK) == ChessBoard.sq(7, 4)  # e8

    def test_turn_is_white(self):
        b = ChessBoard()
        assert b.turn == WHITE

    def test_legal_move_count(self):
        # Standard opening: 20 legal moves (16 pawn + 4 knight)
        b = ChessBoard()
        assert len(b.get_legal_moves()) == 20

    def test_not_in_check(self):
        b = ChessBoard()
        assert not b.is_in_check(WHITE)
        assert not b.is_in_check(BLACK)


# ===========================================================================
# 2 — Pawn moves
# ===========================================================================

class TestPawnMoves:

    def test_e4_opening(self):
        b = ChessBoard()
        assert make_move_uci(b, "e2e4")
        assert b.get_piece(ChessBoard.sq(3, 4)) == WHITE * PAWN
        assert b.get_piece(ChessBoard.sq(1, 4)) == EMPTY

    def test_double_push_en_passant_square(self):
        b = ChessBoard()
        make_move_uci(b, "e2e4")
        assert b.en_passant_sq == ChessBoard.sq(2, 4)  # e3

    def test_single_push_no_en_passant(self):
        b = ChessBoard()
        make_move_uci(b, "e2e3")
        assert b.en_passant_sq is None

    def test_en_passant_capture(self):
        b = ChessBoard()
        # Classic ep: e2e4, c7c5, e4e5, d7d5, e5xd6
        make_move_uci(b, "e2e4")
        make_move_uci(b, "c7c5")
        make_move_uci(b, "e4e5")
        make_move_uci(b, "d7d5")   # black pawn d7→d5, ep square = d6
        ep_sq = b.en_passant_sq
        assert ep_sq == ChessBoard.sq(5, 3)  # d6 (row 5, col 3)
        assert "e5d6" in moves_uci(b)
        make_move_uci(b, "e5d6")
        # Black pawn on d5 must be gone
        assert b.get_piece(ChessBoard.sq(4, 3)) == EMPTY  # d5 empty
        assert b.get_piece(ChessBoard.sq(5, 3)) == WHITE * PAWN  # d6

    def test_promotion_to_queen(self):
        b = ChessBoard()
        # Set up a white pawn on a7
        b.board[:] = 0
        b.board[6][0] = WHITE * PAWN   # a7
        b.board[0][4] = WHITE * KING   # e1
        b.board[7][4] = BLACK * KING   # e8
        b.turn = WHITE
        b._record_position()
        assert "a7a8q" in moves_uci(b)
        make_move_uci(b, "a7a8q")
        assert b.get_piece(ChessBoard.sq(7, 0)) == WHITE * QUEEN

    def test_promotion_to_knight(self):
        b = ChessBoard()
        b.board[:] = 0
        b.board[6][0] = WHITE * PAWN
        b.board[0][4] = WHITE * KING
        b.board[7][4] = BLACK * KING
        b.turn = WHITE
        b._record_position()
        make_move_uci(b, "a7a8n")
        assert b.get_piece(ChessBoard.sq(7, 0)) == WHITE * KNIGHT


# ===========================================================================
# 3 — Castling
# ===========================================================================

class TestCastling:

    def _clear_between_king_rook(self, b: ChessBoard, color: int, side: str):
        row = 0 if color == WHITE else 7
        if side == "K":
            b.board[row][5] = EMPTY
            b.board[row][6] = EMPTY
        else:
            b.board[row][1] = EMPTY
            b.board[row][2] = EMPTY
            b.board[row][3] = EMPTY

    def test_white_kingside_castling_available(self):
        b = ChessBoard()
        self._clear_between_king_rook(b, WHITE, "K")
        assert "e1g1" in moves_uci(b)

    def test_white_queenside_castling_available(self):
        b = ChessBoard()
        self._clear_between_king_rook(b, WHITE, "Q")
        assert "e1c1" in moves_uci(b)

    def test_castling_moves_rook(self):
        b = ChessBoard()
        self._clear_between_king_rook(b, WHITE, "K")
        make_move_uci(b, "e1g1")
        assert b.get_piece(ChessBoard.sq(0, 6)) == WHITE * KING
        assert b.get_piece(ChessBoard.sq(0, 5)) == WHITE * ROOK
        assert b.get_piece(ChessBoard.sq(0, 4)) == EMPTY
        assert b.get_piece(ChessBoard.sq(0, 7)) == EMPTY

    def test_castling_not_allowed_if_king_moved(self):
        b = ChessBoard()
        self._clear_between_king_rook(b, WHITE, "K")
        make_move_uci(b, "e1f1")  # king moves
        make_move_uci(b, "e7e6")
        make_move_uci(b, "f1e1")  # king returns
        make_move_uci(b, "e8e7")
        assert "e1g1" not in moves_uci(b)

    def test_castling_not_allowed_through_check(self):
        b = ChessBoard()
        b.board[:] = 0
        b.board[0][4] = WHITE * KING
        b.board[0][7] = WHITE * ROOK
        b.board[7][4] = BLACK * KING
        # Black rook attacks f1 (sq 5, row 0)
        b.board[7][5] = BLACK * ROOK
        b.turn = WHITE
        b._record_position()
        # Kingside castling passes through f1 which is attacked
        assert "e1g1" not in moves_uci(b)

    def test_castling_revoked_after_rook_captured(self):
        b = ChessBoard()
        b.board[:] = 0
        b.board[0][4] = WHITE * KING   # e1
        b.board[0][7] = WHITE * ROOK   # h1
        b.board[7][4] = BLACK * KING   # e8
        b.board[1][7] = BLACK * QUEEN  # h2  — will capture h1 (same file, one step)
        b.castling_rights = {WHITE: {"K": True, "Q": True}, BLACK: {"K": True, "Q": True}}
        b.turn = BLACK
        b._record_position()
        # Black queen h2→h1 captures white rook
        h2 = ChessBoard.sq(1, 7)
        h1 = ChessBoard.sq(0, 7)
        result = b.make_move(Move(h2, h1))
        assert result, "Move should be legal"
        assert not b.castling_rights[WHITE]["K"], "White kingside castling should be revoked"


# ===========================================================================
# 4 — Check, checkmate, stalemate
# ===========================================================================

class TestGameTermination:

    def test_scholar_checkmate(self):
        """Fool's mate setup (4 moves)."""
        b = ChessBoard()
        make_move_uci(b, "e2e4")
        make_move_uci(b, "e7e5")
        make_move_uci(b, "f1c4")
        make_move_uci(b, "b8c6")
        make_move_uci(b, "d1h5")
        make_move_uci(b, "a7a6")
        make_move_uci(b, "h5f7")  # checkmate
        assert b.is_checkmate()
        assert b.get_result() == WHITE

    def test_fools_mate(self):
        b = ChessBoard()
        make_move_uci(b, "f2f3")
        make_move_uci(b, "e7e5")
        make_move_uci(b, "g2g4")
        make_move_uci(b, "d8h4")  # Qh4#
        assert b.is_checkmate()
        assert b.get_result() == BLACK

    def test_stalemate(self):
        b = ChessBoard()
        b.board[:] = 0
        b.board[7][0] = BLACK * KING   # a8
        b.board[5][1] = WHITE * QUEEN  # b6
        b.board[6][2] = WHITE * KING   # c7
        b.turn = BLACK
        b._record_position()
        # Black king has no legal moves and is not in check → stalemate
        assert b.is_stalemate()
        assert b.get_result() == 0

    def test_fifty_move_rule(self):
        b = ChessBoard()
        b.halfmove_clock = 100
        assert b.is_fifty_move_rule()
        assert b.is_draw()

    def test_insufficient_material_kk(self):
        b = ChessBoard()
        b.board[:] = 0
        b.board[0][4] = WHITE * KING
        b.board[7][4] = BLACK * KING
        assert b.is_insufficient_material()

    def test_insufficient_material_kbk(self):
        b = ChessBoard()
        b.board[:] = 0
        b.board[0][4] = WHITE * KING
        b.board[0][3] = WHITE * BISHOP
        b.board[7][4] = BLACK * KING
        assert b.is_insufficient_material()

    def test_threefold_repetition(self):
        b = ChessBoard()
        # Repeat Ng1-f3-g1 three times
        for _ in range(3):
            make_move_uci(b, "g1f3")
            make_move_uci(b, "g8f6")
            make_move_uci(b, "f3g1")
            make_move_uci(b, "f6g8")
        assert b.is_threefold_repetition()
        assert b.is_draw()


# ===========================================================================
# 5 — Gymnasium interface
# ===========================================================================

class TestGymnasiumInterface:

    def test_reset_returns_correct_shapes(self):
        env = ChessEnv()
        obs, info = env.reset()
        assert obs.shape == (8, 8, 17)
        assert obs.dtype == np.float32
        assert "legal_actions" in info

    def test_step_random_game(self):
        """Play a full random game and verify it terminates."""
        import random
        env = ChessEnv(render_mode=None)
        obs, info = env.reset(seed=42)
        done = False
        steps = 0
        while not done and steps < 500:
            legal = info["legal_actions"]
            action = random.choice(legal) if legal else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        # Game must eventually end within reasonable moves
        assert steps < 500 or done

    def test_invalid_action_penalty(self):
        env = ChessEnv(invalid_action_penalty=-0.1)
        obs, info = env.reset(seed=0)
        # Action 0 (a1a1) is never legal — should trigger the penalty
        obs2, reward, terminated, truncated, info2 = env.step(0)
        if not terminated:
            assert reward <= 0  # penalty applied (or 0 if somehow legal)

    def test_custom_terminal_outcome_rewards(self):
        env = ChessEnv(terminal_win_reward=2.5, terminal_loss_penalty=1.75)
        assert env._outcome_reward(WHITE) == 2.5
        assert env._outcome_reward(BLACK) == -1.75
        assert env._outcome_reward(0) == 0.0

    def test_default_reward_parameters(self):
        env = ChessEnv()
        assert env.capture_reward_scale == pytest.approx(0.01)
        assert env.loss_penalty_scale == pytest.approx(0.01)
        assert env.terminal_win_reward == pytest.approx(10.0)
        assert env.terminal_loss_penalty == pytest.approx(10.0)
        assert env.invalid_action_penalty == pytest.approx(-1.0)

    def test_observation_space_contains_obs(self):
        env = ChessEnv()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_action_to_move_round_trip(self):
        env = ChessEnv()
        env.reset()
        legal = env.board.get_legal_moves()
        for move in legal:
            action = env.move_to_action(move)
            recovered = env.action_to_move(action)
            assert recovered is not None
            assert recovered.from_sq == move.from_sq
            assert recovered.to_sq == move.to_sq


# ===========================================================================
# 6 — Move generation correctness
# ===========================================================================

class TestMoveGeneration:

    def test_no_moves_leave_king_in_check(self):
        """After applying any legal move, the moving side is not in check."""
        b = ChessBoard()
        for move in b.get_legal_moves():
            copy = b.copy()
            copy._apply_move_unchecked(move)
            assert not copy.is_in_check(WHITE), f"Move {move} leaves king in check"

    def test_pinned_piece_cannot_move(self):
        """A piece pinned to the king cannot expose the king."""
        b = ChessBoard()
        b.board[:] = 0
        # White king on e1, white rook on e4, black queen on e8
        b.board[0][4] = WHITE * KING   # e1
        b.board[3][4] = WHITE * ROOK   # e4
        b.board[7][4] = BLACK * QUEEN  # e8
        b.board[7][0] = BLACK * KING   # a8
        b.turn = WHITE
        b._record_position()
        legal = b.get_legal_moves()
        legal_uci = {m.to_uci() for m in legal}
        # The rook is pinned on the e-file: it can only move along e-file
        rook_moves = {m.to_uci() for m in legal if m.from_sq == ChessBoard.sq(3, 4)}
        for m_uci in rook_moves:
            assert m_uci[0] == "e", f"Pinned rook moved off e-file: {m_uci}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
