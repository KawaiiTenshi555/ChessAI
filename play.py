"""
Lanceur du jeu d'échecs.

Usage :
    python play.py                  # Interface web  (défaut)
    python play.py --terminal       # Terminal : Joueur (Blancs) vs IA aléatoire
    python play.py --terminal --color black   # Terminal : Joueur (Noirs)
    python play.py --terminal --both          # Terminal : deux IA s'affrontent
    python play.py --port 8080      # Web sur port personnalisé
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from chess_env.board import ChessBoard, Move, WHITE, BLACK


def parse_uci(uci: str, board: ChessBoard):
    """Convertit une chaîne UCI (ex: 'e2e4', 'a7a8q') en Move légal, ou None."""
    uci = uci.strip().lower()
    files = "abcdefgh"
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

    from_sq = ChessBoard.sq(from_row, from_col)
    to_sq   = ChessBoard.sq(to_row, to_col)
    move    = Move(from_sq, to_sq, promo)

    legal = board.get_legal_moves()
    # Si pas de promotion spécifiée et c'est un pion qui arrive en dernière rangée,
    # proposer la dame automatiquement.
    if move in legal:
        return move
    # Essayer promo dame par défaut
    move_q = Move(from_sq, to_sq, 5)
    if move_q in legal:
        print("  (promotion automatique en Dame)")
        return move_q
    return None


def print_legal_moves(board: ChessBoard):
    moves = sorted(m.to_uci() for m in board.get_legal_moves())
    cols = 10
    print("\n  Coups légaux :")
    for i in range(0, len(moves), cols):
        print("    " + "  ".join(moves[i:i+cols]))


def human_turn(board: ChessBoard) -> Move:
    while True:
        try:
            raw = input("\n  Votre coup (UCI, ex: e2e4) ou '?' pour la liste : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPartie abandonnée.")
            sys.exit(0)

        if raw == "q" or raw == "quit":
            print("Partie abandonnée.")
            sys.exit(0)
        if raw == "?":
            print_legal_moves(board)
            continue

        move = parse_uci(raw, board)
        if move is None:
            print(f"  Coup invalide : '{raw}'. Réessayez (ou '?' pour voir les coups légaux).")
            continue
        return move


def random_turn(board: ChessBoard) -> Move:
    return random.choice(board.get_legal_moves())


def play(player_color: int = WHITE, auto: bool = False):
    board = ChessBoard()
    color_name = {WHITE: "Blancs", BLACK: "Noirs"}

    if auto:
        print("\n=== Mode spectateur : IA aléatoire vs IA aléatoire ===\n")
    else:
        print(f"\n=== Vous jouez les {color_name[player_color]} ===")
        print("  Entrez les coups en notation UCI (ex: e2e4, g1f3, e1g1 pour le roque).")
        print("  Tapez '?' pour voir les coups légaux, 'quit' pour quitter.\n")

    while True:
        print("\n" + board.render_ascii())

        result = board.get_result()
        if result is not None:
            print("\n" + "=" * 40)
            if result == WHITE:
                print("  Résultat : les BLANCS gagnent !")
            elif result == BLACK:
                print("  Résultat : les NOIRS gagnent !")
            else:
                print("  Résultat : NULLE !")
            print("=" * 40)
            break

        current = board.turn
        print(f"\n  Tour {board.fullmove_number} — {'Blancs' if current == WHITE else 'Noirs'}")

        if auto:
            move = random_turn(board)
            print(f"  Coup joué : {move.to_uci()}")
            input("  [Entrée pour continuer]")
        elif current == player_color:
            move = human_turn(board)
            print(f"  Coup joué : {move.to_uci()}")
        else:
            move = random_turn(board)
            print(f"  Coup de l'IA : {move.to_uci()}")

        board._apply_move_unchecked(move)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jeu d'échecs — terminal ou web")
    parser.add_argument("--terminal", action="store_true",
                        help="Lancer en mode terminal (défaut: interface web)")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                        help="Couleur du joueur humain en mode terminal (default: white)")
    parser.add_argument("--both", action="store_true",
                        help="Terminal : deux IA s'affrontent")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port du serveur web (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Hôte du serveur web (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Ne pas ouvrir le navigateur automatiquement")
    args = parser.parse_args()

    if args.terminal:
        player_color = WHITE if args.color == "white" else BLACK
        play(player_color=player_color, auto=args.both)
    else:
        import webbrowser
        import threading
        from web.app import run as run_web

        url = f"http://{args.host}:{args.port}"
        print(f"\n  Chess IA — Interface Web")
        print(f"  Serveur démarré sur {url}")
        print(f"  Appuyez sur Ctrl+C pour arrêter.\n")

        if not args.no_browser:
            # Ouvre le navigateur après un court délai (laisse Flask démarrer)
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()

        run_web(host=args.host, port=args.port)
