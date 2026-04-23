import math
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from benchmark.stockfish import benchmark_agent_vs_stockfish, estimate_elo_from_match, estimate_elo_from_score

OBS_SHAPE = (8, 8, 17)
ACTION_SIZE = 4096
REPO_ROOT = Path(__file__).resolve().parent.parent


class FirstLegalAgent(BaseAgent):
    def select_action(self, observation: np.ndarray, legal_actions):
        return min(legal_actions)

    def select_move(self, board, temperature: float = 0.0):
        legal_moves = sorted(board.get_legal_moves(), key=lambda move: move.to_uci())
        return legal_moves[0] if legal_moves else None

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        legal_next_actions=None,
    ):
        return 0.0

    def get_config(self):
        return {}


def _write_fake_uci_engine(tmp_path: Path) -> Path:
    script = textwrap.dedent(
        f"""
        import sys

        sys.path.insert(0, {str(REPO_ROOT)!r})

        from chess_env.board import BISHOP, KNIGHT, QUEEN, ROOK, ChessBoard, Move

        def uci_to_move(board, uci):
            uci = uci.strip().lower()
            files = "abcdefgh"
            if len(uci) not in (4, 5):
                return None

            try:
                from_col = files.index(uci[0])
                from_row = int(uci[1]) - 1
                to_col = files.index(uci[2])
                to_row = int(uci[3]) - 1
            except (ValueError, IndexError):
                return None

            promotion = 0
            if len(uci) == 5:
                promotion = {{"n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN}}.get(uci[4], 0)
                if promotion == 0:
                    return None

            move = Move(ChessBoard.sq(from_row, from_col), ChessBoard.sq(to_row, to_col), promotion)
            legal = board.get_legal_moves()
            if move in legal:
                return move

            queen_move = Move(move.from_sq, move.to_sq, QUEEN)
            if queen_move in legal:
                return queen_move
            return None

        current_moves = []

        for raw in sys.stdin:
            line = raw.strip()
            if line == "uci":
                print("id name FakeStockfish")
                print("uciok")
                sys.stdout.flush()
            elif line == "isready":
                print("readyok")
                sys.stdout.flush()
            elif line.startswith("setoption "):
                continue
            elif line == "ucinewgame":
                current_moves = []
            elif line.startswith("position startpos"):
                if " moves " in line:
                    current_moves = line.split(" moves ", 1)[1].split()
                else:
                    current_moves = []
            elif line.startswith("go "):
                board = ChessBoard()
                valid = True
                for uci in current_moves:
                    move = uci_to_move(board, uci)
                    if move is None:
                        valid = False
                        break
                    board._apply_move_unchecked(move)

                if not valid:
                    print("bestmove 0000")
                else:
                    legal_moves = sorted(board.get_legal_moves(), key=lambda move: move.to_uci())
                    best = legal_moves[0].to_uci() if legal_moves else "0000"
                    print(f"bestmove {{best}}")
                sys.stdout.flush()
            elif line == "quit":
                break
        """
    )

    path = tmp_path / "fake_uci_engine.py"
    path.write_text(script, encoding="utf-8")
    return path


def test_estimate_elo_from_score_even_match():
    assert estimate_elo_from_score(0.5, 1500) == pytest.approx(1500.0)


def test_estimate_elo_from_match_uses_continuity_correction():
    estimated_elo, adjusted_score = estimate_elo_from_match(points=2.0, games=4, opponent_elo=1600)
    assert adjusted_score == pytest.approx(0.5)
    assert estimated_elo == pytest.approx(1600.0)


def test_benchmark_agent_vs_stockfish_runs_offline(tmp_path):
    fake_engine = _write_fake_uci_engine(tmp_path)
    agent = FirstLegalAgent(ACTION_SIZE, OBS_SHAPE)

    result = benchmark_agent_vs_stockfish(
        agent,
        stockfish_path=sys.executable,
        stockfish_args=[str(fake_engine)],
        n_games=4,
        stockfish_elo=1200,
        movetime_ms=1,
        max_plies=2,
    )

    assert result.games == 4
    assert result.games_as_white == 2
    assert result.games_as_black == 2
    assert result.wins == 0
    assert result.losses == 0
    assert result.draws == 4
    assert result.points == pytest.approx(2.0)
    assert result.score_rate == pytest.approx(0.5)
    assert result.estimated_elo == pytest.approx(1200.0)
    assert result.wl_record == "0/0"
    assert math.isclose(result.win_rate + result.loss_rate + result.draw_rate, 1.0)
