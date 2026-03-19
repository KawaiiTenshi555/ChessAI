"""
Serveur Flask pour l'interface web du jeu d'échecs.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, jsonify, request
from chess_env.board import ChessBoard, Move, WHITE, BLACK, QUEEN

app = Flask(__name__)

# --- État global (un seul jeu à la fois) ---
board = ChessBoard()
player_color = WHITE


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _board_state() -> dict:
    pieces = {}
    for sq in range(64):
        p = board.get_piece(sq)
        if p != 0:
            pieces[str(sq)] = int(p)

    result = board.get_result()
    result_str = None
    if result == WHITE:
        result_str = "white"
    elif result == BLACK:
        result_str = "black"
    elif result == 0:
        result_str = "draw"

    return {
        "pieces": pieces,
        "turn": "white" if board.turn == WHITE else "black",
        "player_color": "white" if player_color == WHITE else "black",
        "in_check": board.is_in_check(board.turn),
        "legal_moves": [m.to_uci() for m in board.get_legal_moves()],
        "result": result_str,
        "halfmove_clock": board.halfmove_clock,
        "fullmove": board.fullmove_number,
        "last_move": board.move_history[-1].to_uci() if board.move_history else None,
        "castling": {
            "wK": board.castling_rights[WHITE]["K"],
            "wQ": board.castling_rights[WHITE]["Q"],
            "bK": board.castling_rights[BLACK]["K"],
            "bQ": board.castling_rights[BLACK]["Q"],
        },
    }


def _uci_to_move(uci: str) -> Move | None:
    files = "abcdefgh"
    uci = uci.strip().lower()
    if len(uci) not in (4, 5):
        return None
    try:
        from_col = files.index(uci[0])
        from_row = int(uci[1]) - 1
        to_col   = files.index(uci[2])
        to_row   = int(uci[3]) - 1
    except (ValueError, IndexError):
        return None
    promo = 0
    if len(uci) == 5:
        promo_map = {"n": 2, "b": 3, "r": 4, "q": 5}
        promo = promo_map.get(uci[4], 0)
        if promo == 0:
            return None
    return Move(ChessBoard.sq(from_row, from_col), ChessBoard.sq(to_row, to_col), promo)


def _ai_move():
    """Fait jouer l'IA (aléatoire pour l'instant)."""
    legal = board.get_legal_moves()
    if legal:
        board._apply_move_unchecked(random.choice(legal))


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(_board_state())


@app.route("/api/legal_moves/<int:sq>")
def legal_moves_from(sq: int):
    moves = [m.to_uci() for m in board.get_legal_moves() if m.from_sq == sq]
    return jsonify({"moves": moves})


@app.route("/api/move", methods=["POST"])
def make_move():
    global board
    data = request.get_json(force=True)
    uci = data.get("uci", "")

    move = _uci_to_move(uci)
    if move is None:
        return jsonify({"error": "Format UCI invalide"}), 400

    legal = board.get_legal_moves()

    # Auto-promotion en dame si pas spécifiée
    if move not in legal:
        move_q = Move(move.from_sq, move.to_sq, QUEEN)
        if move_q in legal:
            move = move_q
        else:
            return jsonify({"error": "Coup illégal"}), 400

    board._apply_move_unchecked(move)

    # L'IA joue si la partie continue et c'est son tour
    if board.get_result() is None and board.turn != player_color:
        _ai_move()

    return jsonify(_board_state())


@app.route("/api/reset", methods=["POST"])
def reset():
    global board, player_color
    data = request.get_json(force=True) or {}
    color = data.get("color", "white")
    player_color = WHITE if color == "white" else BLACK
    board = ChessBoard()

    # Si le joueur est Noirs, l'IA joue d'abord
    if player_color == BLACK:
        _ai_move()

    return jsonify(_board_state())


def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)
