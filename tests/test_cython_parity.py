# tests/test_cython_parity.py
"""
Tests de parité entre la version Python (board.py) et la version Cython
(board_cy). Ces tests sont automatiquement ignorés si board_cy n'est pas
compilé.

Lancer avec :
    pytest tests/test_cython_parity.py -v
"""
import random
import pytest

from chess_env.board import ChessBoard as PyBoard, Move as PyMove

try:
    from chess_env.board_cy import (  # type: ignore[import]
        ChessBoard as CyBoard,
        Move as CyMove,
    )
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

# Tous les tests de ce module sont ignorés si Cython n'est pas compilé.
pytestmark = pytest.mark.skipif(
    not HAS_CYTHON,
    reason="board_cy not compiled — run: python setup.py build_ext --inplace",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _legal_uci_set(board):
    """Retourne l'ensemble des UCI des coups légaux."""
    return {m.to_uci() for m in board.get_legal_moves()}


def _pieces(board):
    """Retourne la liste des pièces sur les 64 cases."""
    return [board.get_piece(sq) for sq in range(64)]


def _apply_uci(board, uci: str):
    """Applique un coup UCI sur le board (Python ou Cython)."""
    legal = board.get_legal_moves()
    move = next((m for m in legal if m.to_uci() == uci), None)
    assert move is not None, f"Coup {uci!r} introuvable parmi {[m.to_uci() for m in legal]}"
    board._apply_move_unchecked(move)


def _sync_game(seed: int, n_moves: int = 50):
    """
    Joue n_moves coups aléatoires sur les deux boards en parallèle,
    retourne (py_board, cy_board).
    """
    rng = random.Random(seed)
    py = PyBoard()
    cy = CyBoard()

    for _ in range(n_moves):
        legal = py.get_legal_moves()
        if not legal:
            break
        uci = rng.choice(legal).to_uci()
        _apply_uci(py, uci)
        _apply_uci(cy, uci)

    return py, cy


# ---------------------------------------------------------------------------
# Tests de base
# ---------------------------------------------------------------------------

class TestInitialPosition:
    """Vérifie que les deux boards démarrent dans le même état."""

    def test_piece_count(self):
        py = PyBoard()
        cy = CyBoard()
        assert _pieces(py) == _pieces(cy)

    def test_turn(self):
        assert PyBoard().turn == CyBoard().turn

    def test_legal_moves_count(self):
        py = PyBoard()
        cy = CyBoard()
        assert len(py.get_legal_moves()) == 20
        assert len(cy.get_legal_moves()) == 20

    def test_legal_moves_set(self):
        py = PyBoard()
        cy = CyBoard()
        assert _legal_uci_set(py) == _legal_uci_set(cy)

    def test_observation_shape(self):
        import numpy as np
        obs = CyBoard().get_observation()
        assert obs.shape == (8, 8, 17)
        assert obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests de parité sur parties aléatoires
# ---------------------------------------------------------------------------

class TestRandomGames:
    """100 parties de 50 coups — vérifie l'ensemble des légaux à chaque étape."""

    @pytest.mark.parametrize("seed", range(100))
    def test_legal_moves_parity(self, seed):
        py, cy = _sync_game(seed, n_moves=50)
        assert _pieces(py) == _pieces(cy), f"seed={seed} : plateau différent"
        assert py.turn == cy.turn, f"seed={seed} : trait différent"
        assert _legal_uci_set(py) == _legal_uci_set(cy), f"seed={seed} : coups légaux différents"

    @pytest.mark.parametrize("seed", range(50))
    def test_halfmove_clock(self, seed):
        py, cy = _sync_game(seed, n_moves=30)
        assert py.halfmove_clock == cy.halfmove_clock

    @pytest.mark.parametrize("seed", range(50))
    def test_fullmove_number(self, seed):
        py, cy = _sync_game(seed, n_moves=30)
        assert py.fullmove_number == cy.fullmove_number


# ---------------------------------------------------------------------------
# Tests de positions spécifiques
# ---------------------------------------------------------------------------

class TestSpecialMoves:
    """Vérifie en passant, roque et promotion."""

    def test_en_passant(self):
        """Après 1.e4 e5 2.e5 d5 3.exd6 — la prise en passant doit être possible."""
        py = PyBoard()
        cy = CyBoard()
        for uci in ["e2e4", "e7e5", "e4e5", "d7d5"]:
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        assert "e5d6" in _legal_uci_set(py)
        assert _legal_uci_set(py) == _legal_uci_set(cy)

    def test_castling_kingside_white(self):
        """Après dégagement des pièces, O-O blanc doit être légal."""
        py = PyBoard()
        cy = CyBoard()
        # Dégager Cavalier et Fou côté roi blanc
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "f8e7"]:
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        assert "e1g1" in _legal_uci_set(py)
        assert _legal_uci_set(py) == _legal_uci_set(cy)

    def test_castling_queenside_white(self):
        """Dégagement côté dame — O-O-O blanc."""
        py = PyBoard()
        cy = CyBoard()
        for uci in ["d2d4", "d7d5", "b1c3", "b8c6", "c1f4", "c8f5",
                    "d1d2", "d8d7"]:
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        assert "e1c1" in _legal_uci_set(py)
        assert _legal_uci_set(py) == _legal_uci_set(cy)

    def test_promotion(self):
        """Vérifie que les 4 promotions sont générées."""
        py = PyBoard()
        cy = CyBoard()
        # Partie Réti rapide approchant la promotion (seed fixe)
        for uci in ["e2e4", "f7f5", "e4f5", "g7g5", "f5g6", "h7h6",
                    "g6g7", "h6h5", "g7g8q"]:
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        # La prise en passant g7g8 avec promotion dame doit exister sur les deux
        assert _legal_uci_set(py) == _legal_uci_set(cy)

    def test_promotion_choices(self):
        """Les 4 promotions (q/r/b/n) doivent être présentes avant le dernier coup."""
        py = PyBoard()
        cy = CyBoard()
        for uci in ["e2e4", "f7f5", "e4f5", "g7g5", "f5g6", "h7h6",
                    "g6g7", "h6h5"]:
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        py_set = _legal_uci_set(py)
        cy_set = _legal_uci_set(cy)
        for promo in ("g7g8q", "g7g8r", "g7g8b", "g7g8n"):
            assert promo in py_set, f"Promotion {promo} absente en Python"
            assert promo in cy_set, f"Promotion {promo} absente en Cython"


# ---------------------------------------------------------------------------
# Tests d'état de fin de partie
# ---------------------------------------------------------------------------

class TestGameEnd:
    """Échec et mat, nulle, pat."""

    def test_fool_checkmate(self):
        """Mat du berger (2 coups) : 1.f3 e5 2.g4 Dh4#"""
        py = PyBoard()
        cy = CyBoard()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        assert py.is_checkmate()
        assert cy.is_checkmate()
        assert py.get_result() == -1   # BLACK wins
        assert cy.get_result() == -1

    def test_stalemate(self):
        """Vérifie que is_stalemate() et is_draw() concordent."""
        # On simule un position de pat via une partie aléatoire longue
        # et on vérifie juste la cohérence Py ↔ Cy
        rng = random.Random(42)
        py = PyBoard()
        cy = CyBoard()
        for _ in range(200):
            legal = py.get_legal_moves()
            if not legal:
                break
            uci = rng.choice(legal).to_uci()
            _apply_uci(py, uci)
            _apply_uci(cy, uci)
        assert py.is_stalemate() == cy.is_stalemate()
        assert py.is_checkmate() == cy.is_checkmate()
        assert py.is_draw() == cy.is_draw()

    def test_fifty_move_rule(self):
        """Partie avec 50 coups sans pion ni capture → règle des 50 coups."""
        py, cy = _sync_game(seed=7, n_moves=120)
        assert py.is_fifty_move_rule() == cy.is_fifty_move_rule()

    def test_threefold_repetition(self):
        """Répétition triple — parité entre Python et Cython."""
        py = PyBoard()
        cy = CyBoard()
        # Aller-retour 3 fois avec les cavaliers (répétition garantie)
        for _ in range(3):
            for uci in ["g1f3", "g8f6", "f3g1", "f6g8"]:
                _apply_uci(py, uci)
                _apply_uci(cy, uci)
        assert py.is_threefold_repetition() == cy.is_threefold_repetition()
        assert py.is_draw() == cy.is_draw()

    def test_insufficient_material(self):
        """K vs K après captures successives."""
        py, cy = _sync_game(seed=99, n_moves=200)
        assert py.is_insufficient_material() == cy.is_insufficient_material()


# ---------------------------------------------------------------------------
# Tests de la propriété board (compat. numpy)
# ---------------------------------------------------------------------------

class TestBoardProperty:
    """Vérifie que la propriété board retourne bien un numpy (8,8)."""

    def test_board_shape(self):
        import numpy as np
        b = CyBoard()
        arr = b.board
        assert arr.shape == (8, 8)
        assert arr.dtype == np.int8

    def test_board_values_match(self):
        """Valeurs identiques entre Python et Cython."""
        py = PyBoard()
        cy = CyBoard()
        import numpy as np
        assert np.array_equal(py.board, cy.board)

    def test_board_after_moves(self):
        """Après quelques coups, les tableaux numpy concordent."""
        import numpy as np
        py, cy = _sync_game(seed=5, n_moves=15)
        assert np.array_equal(py.board, cy.board)


# ---------------------------------------------------------------------------
# Tests de copy()
# ---------------------------------------------------------------------------

class TestCopy:
    def test_copy_independence(self):
        """Modifier la copie ne doit pas affecter l'original."""
        cy = CyBoard()
        cy2 = cy.copy()
        _apply_uci(cy2, "e2e4")
        # Original inchangé
        assert _pieces(cy) != _pieces(cy2)

    def test_copy_parity_with_python(self):
        """copy() Cython et Python doivent produire le même état."""
        py, cy = _sync_game(seed=12, n_moves=20)
        py2 = py.copy()
        cy2 = cy.copy()
        assert _pieces(py2) == _pieces(cy2)
        assert _legal_uci_set(py2) == _legal_uci_set(cy2)
