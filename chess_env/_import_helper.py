# chess_env/_import_helper.py
# Centralise la logique de fallback Cython → Python.
#
# Utilisation depuis n'importe quel module du projet :
#
#   from chess_env._import_helper import (
#       ChessBoard, Move, WHITE, BLACK,
#       EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, ENGINE
#   )
#
# ENGINE vaut "cython" ou "python" selon ce qui est disponible.

try:
    from chess_env.board_cy import (  # type: ignore[import]
        ChessBoard, Move,
        WHITE, BLACK,
        EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    )
    ENGINE = "cython"
except ImportError:
    from chess_env.board import (
        ChessBoard, Move,
        WHITE, BLACK,
        EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    )
    ENGINE = "python"

__all__ = [
    "ChessBoard", "Move",
    "WHITE", "BLACK",
    "EMPTY", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING",
    "ENGINE",
]
