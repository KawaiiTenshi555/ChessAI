"""
Chess board logic: piece representation, move generation, and game state detection.

Coordinate system:
  - Square 0 = a1 (bottom-left, white's perspective)
  - Square 63 = h8 (top-right)
  - row = sq // 8  (0 = rank 1, 7 = rank 8)
  - col = sq % 8   (0 = a-file, 7 = h-file)

Piece encoding:
  - Positive = White, Negative = Black
  - |piece| : 1=Pawn, 2=Knight, 3=Bishop, 4=Rook, 5=Queen, 6=King
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, NamedTuple

# --- Constants ---
EMPTY = 0
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6
WHITE, BLACK = 1, -1

PIECE_SYMBOLS = {
    PAWN: 'P', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K',
    -PAWN: 'p', -KNIGHT: 'n', -BISHOP: 'b', -ROOK: 'r', -QUEEN: 'q', -KING: 'k',
    EMPTY: '.',
}

INITIAL_BACK_RANK = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]


class _MoveUndo(NamedTuple):
    """Minimal state snapshot for push/pop during legal move checking."""
    turn: int
    cr_wk: bool; cr_wq: bool; cr_bk: bool; cr_bq: bool
    ep_sq: Optional[int]
    hmc: int; fmn: int
    captured: int
    ep_pawn_sq: Optional[int]  # sq of pawn removed by en passant
    rook_from: Optional[int]   # for undoing castling
    rook_to: Optional[int]


@dataclass
class Move:
    """Represents a single chess move."""
    from_sq: int   # 0–63
    to_sq: int     # 0–63
    promotion: int = 0  # 0 = no promotion, else KNIGHT/BISHOP/ROOK/QUEEN

    def to_action(self) -> int:
        """Encode as flat integer (used as Gymnasium action)."""
        return self.from_sq * 64 + self.to_sq

    @staticmethod
    def from_action(action: int) -> "Move":
        return Move(action // 64, action % 64)

    def to_uci(self) -> str:
        files = "abcdefgh"
        s = (files[self.from_sq % 8] + str(self.from_sq // 8 + 1)
             + files[self.to_sq % 8] + str(self.to_sq // 8 + 1))
        if self.promotion:
            s += "nbrq"[self.promotion - KNIGHT]
        return s

    def __eq__(self, other) -> bool:
        if not isinstance(other, Move):
            return False
        return (self.from_sq == other.from_sq
                and self.to_sq == other.to_sq
                and self.promotion == other.promotion)

    def __hash__(self) -> int:
        return hash((self.from_sq, self.to_sq, self.promotion))

    def __repr__(self) -> str:
        return f"Move({self.to_uci()})"


class ChessBoard:
    """
    Complete chess board with legal move generation and game state detection.

    Attributes:
        board           : numpy (8, 8) int8 — piece at each square
        turn            : WHITE or BLACK — whose turn it is
        castling_rights : {WHITE: {'K': bool, 'Q': bool}, BLACK: {...}}
        en_passant_sq   : int | None — target square (the square the pawn passes through)
        halfmove_clock  : int — for the 50-move rule
        fullmove_number : int
        position_history: dict[bytes -> int] — for threefold repetition
        move_history    : list[Move]
    """

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.turn = WHITE
        self.castling_rights: Dict[int, Dict[str, bool]] = {
            WHITE: {"K": True, "Q": True},
            BLACK: {"K": True, "Q": True},
        }
        self.en_passant_sq: Optional[int] = None
        self.halfmove_clock: int = 0
        self.fullmove_number: int = 1
        self.position_history: Dict[bytes, int] = {}
        self.move_history: List[Move] = []
        self._setup_initial_position()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_initial_position(self):
        for col, piece in enumerate(INITIAL_BACK_RANK):
            self.board[0][col] = piece    # White back rank
            self.board[7][col] = -piece   # Black back rank
        for col in range(8):
            self.board[1][col] = PAWN     # White pawns
            self.board[6][col] = -PAWN    # Black pawns
        self._record_position()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def sq(row: int, col: int) -> int:
        return row * 8 + col

    @staticmethod
    def rc(square: int) -> Tuple[int, int]:
        return square // 8, square % 8

    @staticmethod
    def on_board(row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def get_piece(self, square: int) -> int:
        r, c = self.rc(square)
        return int(self.board[r][c])

    def set_piece(self, square: int, piece: int):
        r, c = self.rc(square)
        self.board[r][c] = piece

    @staticmethod
    def color_of(piece: int) -> int:
        if piece > 0:
            return WHITE
        if piece < 0:
            return BLACK
        return 0

    # ------------------------------------------------------------------
    # Position hashing (for threefold repetition)
    # ------------------------------------------------------------------

    def _position_key(self) -> bytes:
        cr = self.castling_rights
        ep = self.en_passant_sq if self.en_passant_sq is not None else 255
        flags = bytes([
            1 if self.turn == WHITE else 0,
            1 if cr[WHITE]["K"] else 0,
            1 if cr[WHITE]["Q"] else 0,
            1 if cr[BLACK]["K"] else 0,
            1 if cr[BLACK]["Q"] else 0,
            ep,
        ])
        return self.board.tobytes() + flags

    def _record_position(self):
        key = self._position_key()
        self.position_history[key] = self.position_history.get(key, 0) + 1

    # ------------------------------------------------------------------
    # Board copy (used for legal-move check validation)
    # ------------------------------------------------------------------

    def copy(self) -> "ChessBoard":
        b = ChessBoard.__new__(ChessBoard)
        b.board = self.board.copy()
        b.turn = self.turn
        b.castling_rights = {
            WHITE: dict(self.castling_rights[WHITE]),
            BLACK: dict(self.castling_rights[BLACK]),
        }
        b.en_passant_sq = self.en_passant_sq
        b.halfmove_clock = self.halfmove_clock
        b.fullmove_number = self.fullmove_number
        b.position_history = dict(self.position_history)
        b.move_history = list(self.move_history)
        return b

    # ------------------------------------------------------------------
    # Move generation — pseudo-legal
    # ------------------------------------------------------------------

    def _pseudo_legal_moves(self, color: int) -> List[Move]:
        moves: List[Move] = []
        for sq in range(64):
            piece = self.get_piece(sq)
            if piece == 0 or self.color_of(piece) != color:
                continue
            pt = abs(piece)
            if pt == PAWN:
                moves.extend(self._pawn_moves(sq, color))
            elif pt == KNIGHT:
                moves.extend(self._knight_moves(sq, color))
            elif pt == BISHOP:
                moves.extend(self._sliding_moves(sq, color, [(1, 1), (1, -1), (-1, 1), (-1, -1)]))
            elif pt == ROOK:
                moves.extend(self._sliding_moves(sq, color, [(1, 0), (-1, 0), (0, 1), (0, -1)]))
            elif pt == QUEEN:
                moves.extend(self._sliding_moves(sq, color,
                    [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]))
            elif pt == KING:
                moves.extend(self._king_moves(sq, color))
        return moves

    def _pawn_moves(self, sq: int, color: int) -> List[Move]:
        moves: List[Move] = []
        row, col = self.rc(sq)
        direction = 1 if color == WHITE else -1
        start_row = 1 if color == WHITE else 6
        promo_row = 7 if color == WHITE else 0

        # Forward one square
        nr = row + direction
        if self.on_board(nr, col) and self.board[nr][col] == 0:
            if nr == promo_row:
                for promo in [QUEEN, ROOK, BISHOP, KNIGHT]:
                    moves.append(Move(sq, self.sq(nr, col), promo))
            else:
                moves.append(Move(sq, self.sq(nr, col)))
                # Forward two squares from starting rank
                if row == start_row:
                    nr2 = row + 2 * direction
                    if self.board[nr2][col] == 0:
                        moves.append(Move(sq, self.sq(nr2, col)))

        # Diagonal captures (normal + en passant)
        for dc in (-1, 1):
            nc = col + dc
            if not self.on_board(nr, nc):
                continue
            target_sq = self.sq(nr, nc)
            target = self.board[nr][nc]
            if target != 0 and self.color_of(target) != color:
                if nr == promo_row:
                    for promo in [QUEEN, ROOK, BISHOP, KNIGHT]:
                        moves.append(Move(sq, target_sq, promo))
                else:
                    moves.append(Move(sq, target_sq))
            elif target_sq == self.en_passant_sq:
                moves.append(Move(sq, target_sq))

        return moves

    def _knight_moves(self, sq: int, color: int) -> List[Move]:
        moves: List[Move] = []
        row, col = self.rc(sq)
        for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                       (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            nr, nc = row + dr, col + dc
            if not self.on_board(nr, nc):
                continue
            target = self.board[nr][nc]
            if target == 0 or self.color_of(target) != color:
                moves.append(Move(sq, self.sq(nr, nc)))
        return moves

    def _sliding_moves(self, sq: int, color: int, directions: list) -> List[Move]:
        moves: List[Move] = []
        row, col = self.rc(sq)
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            while self.on_board(nr, nc):
                target = self.board[nr][nc]
                if target == 0:
                    moves.append(Move(sq, self.sq(nr, nc)))
                elif self.color_of(target) != color:
                    moves.append(Move(sq, self.sq(nr, nc)))
                    break
                else:
                    break
                nr += dr
                nc += dc
        return moves

    def _king_moves(self, sq: int, color: int) -> List[Move]:
        moves: List[Move] = []
        row, col = self.rc(sq)
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1),
                       (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = row + dr, col + dc
            if not self.on_board(nr, nc):
                continue
            target = self.board[nr][nc]
            if target == 0 or self.color_of(target) != color:
                moves.append(Move(sq, self.sq(nr, nc)))
        moves.extend(self._castling_moves(color))
        return moves

    def _castling_moves(self, color: int) -> List[Move]:
        moves: List[Move] = []
        row = 0 if color == WHITE else 7
        rights = self.castling_rights[color]
        king_sq = self.sq(row, 4)
        opponent = -color

        # King must be on its starting square (handles custom board setups)
        if self.get_piece(king_sq) != color * KING:
            return moves

        # King must not currently be in check
        if self.is_square_attacked(king_sq, opponent):
            return moves

        # Kingside (O-O)
        if rights["K"]:
            if self.board[row][5] == 0 and self.board[row][6] == 0:
                if (not self.is_square_attacked(self.sq(row, 5), opponent)
                        and not self.is_square_attacked(self.sq(row, 6), opponent)):
                    moves.append(Move(king_sq, self.sq(row, 6)))

        # Queenside (O-O-O)
        if rights["Q"]:
            if (self.board[row][1] == 0 and self.board[row][2] == 0
                    and self.board[row][3] == 0):
                if (not self.is_square_attacked(self.sq(row, 3), opponent)
                        and not self.is_square_attacked(self.sq(row, 2), opponent)):
                    moves.append(Move(king_sq, self.sq(row, 2)))

        return moves

    # ------------------------------------------------------------------
    # Attack detection
    # ------------------------------------------------------------------

    def is_square_attacked(self, sq: int, by_color: int) -> bool:
        """Return True if `sq` is attacked by any piece of `by_color`."""
        row, col = self.rc(sq)

        # Pawn attacks
        # A pawn of `by_color` on (r, c) attacks (r+dir, c±1).
        # So sq (row, col) is attacked from (row - dir, col±1).
        pawn_dir = 1 if by_color == WHITE else -1
        pr = row - pawn_dir
        for dc in (-1, 1):
            pc = col + dc
            if self.on_board(pr, pc) and self.board[pr][pc] == by_color * PAWN:
                return True

        # Knight attacks
        for dr, dc in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                       (1, 2), (1, -2), (-1, 2), (-1, -2)]:
            nr, nc = row + dr, col + dc
            if self.on_board(nr, nc) and self.board[nr][nc] == by_color * KNIGHT:
                return True

        # Diagonal sliders (bishop / queen)
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = row + dr, col + dc
            while self.on_board(nr, nc):
                piece = self.board[nr][nc]
                if piece != 0:
                    if piece == by_color * BISHOP or piece == by_color * QUEEN:
                        return True
                    break
                nr += dr
                nc += dc

        # Straight sliders (rook / queen)
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = row + dr, col + dc
            while self.on_board(nr, nc):
                piece = self.board[nr][nc]
                if piece != 0:
                    if piece == by_color * ROOK or piece == by_color * QUEEN:
                        return True
                    break
                nr += dr
                nc += dc

        # King attacks (one square)
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1),
                       (1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = row + dr, col + dc
            if self.on_board(nr, nc) and self.board[nr][nc] == by_color * KING:
                return True

        return False

    def find_king(self, color: int) -> int:
        """Return the square of the king of `color`, or -1 if not found."""
        result = np.argwhere(self.board == color * KING)
        if len(result) == 0:
            return -1
        r, c = result[0]
        return int(r * 8 + c)

    def is_in_check(self, color: int) -> bool:
        king_sq = self.find_king(color)
        if king_sq == -1:
            return False
        return self.is_square_attacked(king_sq, -color)

    # ------------------------------------------------------------------
    # Push / pop for legal move checking (avoids board.copy())
    # ------------------------------------------------------------------

    def _push_legal(self, move: Move) -> _MoveUndo:
        """Apply move in-place for legality checking. Returns undo state."""
        from_row, from_col = move.from_sq >> 3, move.from_sq & 7
        to_row,   to_col   = move.to_sq   >> 3, move.to_sq   & 7
        piece    = int(self.board[from_row][from_col])
        pt       = abs(piece)
        color    = 1 if piece > 0 else -1
        captured = int(self.board[to_row][to_col])

        # Snapshot before modification
        saved_turn  = self.turn
        saved_ep    = self.en_passant_sq
        saved_hmc   = self.halfmove_clock
        saved_fmn   = self.fullmove_number
        saved_cr_wk = self.castling_rights[WHITE]["K"]
        saved_cr_wq = self.castling_rights[WHITE]["Q"]
        saved_cr_bk = self.castling_rights[BLACK]["K"]
        saved_cr_bq = self.castling_rights[BLACK]["Q"]

        ep_pawn_sq = None
        rook_from  = None
        rook_to    = None

        # Halfmove clock
        if pt == PAWN or captured != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # En passant capture
        if pt == PAWN and move.to_sq == self.en_passant_sq:
            ep_pawn_row = to_row - 1 if color == WHITE else to_row + 1
            ep_pawn_sq  = ep_pawn_row * 8 + to_col
            self.board[ep_pawn_row][to_col] = EMPTY

        # Move piece
        self.board[to_row][to_col]     = piece
        self.board[from_row][from_col] = EMPTY

        # Update en passant square
        if pt == PAWN and abs(to_row - from_row) == 2:
            self.en_passant_sq = ((from_row + to_row) >> 1) * 8 + from_col
        else:
            self.en_passant_sq = None

        # Castling: move rook
        if pt == KING:
            col_diff = to_col - from_col
            if col_diff == 2:
                rook_from = from_row * 8 + 7
                rook_to   = from_row * 8 + 5
                self.board[from_row][5] = self.board[from_row][7]
                self.board[from_row][7] = EMPTY
            elif col_diff == -2:
                rook_from = from_row * 8
                rook_to   = from_row * 8 + 3
                self.board[from_row][3] = self.board[from_row][0]
                self.board[from_row][0] = EMPTY
            self.castling_rights[color]["K"] = False
            self.castling_rights[color]["Q"] = False

        # Revoke castling when own rook moves
        if pt == ROOK:
            back_row = 0 if color == WHITE else 7
            if from_row == back_row:
                if from_col == 7:
                    self.castling_rights[color]["K"] = False
                elif from_col == 0:
                    self.castling_rights[color]["Q"] = False

        # Revoke castling when opponent rook captured
        if captured != EMPTY and abs(captured) == ROOK:
            cap_color = 1 if captured > 0 else -1
            back_row  = 0 if cap_color == WHITE else 7
            if to_row == back_row:
                if to_col == 7:
                    self.castling_rights[cap_color]["K"] = False
                elif to_col == 0:
                    self.castling_rights[cap_color]["Q"] = False

        # Promotion
        if move.promotion:
            self.board[to_row][to_col] = color * move.promotion

        # Switch turn
        if self.turn == BLACK:
            self.fullmove_number += 1
        self.turn = -self.turn

        return _MoveUndo(
            turn=saved_turn,
            cr_wk=saved_cr_wk, cr_wq=saved_cr_wq,
            cr_bk=saved_cr_bk, cr_bq=saved_cr_bq,
            ep_sq=saved_ep, hmc=saved_hmc, fmn=saved_fmn,
            captured=captured, ep_pawn_sq=ep_pawn_sq,
            rook_from=rook_from, rook_to=rook_to,
        )

    def _pop_legal(self, move: Move, undo: _MoveUndo) -> None:
        """Undo a move applied with _push_legal."""
        from_row, from_col = move.from_sq >> 3, move.from_sq & 7
        to_row,   to_col   = move.to_sq   >> 3, move.to_sq   & 7
        color = undo.turn  # color that originally made the move

        # Restore moved piece (pawn if promoted, else piece at to_sq)
        if move.promotion:
            self.board[from_row][from_col] = color * PAWN
        else:
            self.board[from_row][from_col] = self.board[to_row][to_col]
        self.board[to_row][to_col] = undo.captured

        # Restore en passant captured pawn
        if undo.ep_pawn_sq is not None:
            ep_row = undo.ep_pawn_sq >> 3
            ep_col = undo.ep_pawn_sq & 7
            self.board[ep_row][ep_col] = (-color) * PAWN

        # Restore castling rook
        if undo.rook_from is not None:
            rf_row = undo.rook_from >> 3; rf_col = undo.rook_from & 7
            rt_row = undo.rook_to   >> 3; rt_col = undo.rook_to   & 7
            self.board[rf_row][rf_col] = self.board[rt_row][rt_col]
            self.board[rt_row][rt_col] = EMPTY

        # Restore scalar state
        self.turn = undo.turn
        self.castling_rights[WHITE]["K"] = undo.cr_wk
        self.castling_rights[WHITE]["Q"] = undo.cr_wq
        self.castling_rights[BLACK]["K"] = undo.cr_bk
        self.castling_rights[BLACK]["Q"] = undo.cr_bq
        self.en_passant_sq  = undo.ep_sq
        self.halfmove_clock = undo.hmc
        self.fullmove_number = undo.fmn

    # ------------------------------------------------------------------
    # Legal move generation
    # ------------------------------------------------------------------

    def get_legal_moves(self) -> List[Move]:
        """Return all legal moves for the side to move."""
        color = self.turn
        legal: List[Move] = []
        for move in self._pseudo_legal_moves(color):
            undo = self._push_legal(move)
            if not self.is_in_check(color):
                legal.append(move)
            self._pop_legal(move, undo)
        return legal

    # ------------------------------------------------------------------
    # Apply move
    # ------------------------------------------------------------------

    def _apply_move_unchecked(self, move: Move):
        """Apply move without legality validation (internal use only)."""
        piece = self.get_piece(move.from_sq)
        pt = abs(piece)
        color = self.color_of(piece)
        from_row, from_col = self.rc(move.from_sq)
        to_row, to_col = self.rc(move.to_sq)
        captured = self.get_piece(move.to_sq)

        # Halfmove clock
        if pt == PAWN or captured != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Move the piece
        self.set_piece(move.to_sq, piece)
        self.set_piece(move.from_sq, EMPTY)

        # En passant capture
        if pt == PAWN and move.to_sq == self.en_passant_sq:
            ep_pawn_row = to_row - 1 if color == WHITE else to_row + 1
            self.board[ep_pawn_row][to_col] = EMPTY

        # Update en passant square
        if pt == PAWN and abs(to_row - from_row) == 2:
            self.en_passant_sq = self.sq((from_row + to_row) // 2, from_col)
        else:
            self.en_passant_sq = None

        # Castling: move the rook
        if pt == KING:
            col_diff = to_col - from_col
            if col_diff == 2:   # Kingside
                self.set_piece(self.sq(from_row, 5), self.get_piece(self.sq(from_row, 7)))
                self.set_piece(self.sq(from_row, 7), EMPTY)
            elif col_diff == -2:  # Queenside
                self.set_piece(self.sq(from_row, 3), self.get_piece(self.sq(from_row, 0)))
                self.set_piece(self.sq(from_row, 0), EMPTY)
            self.castling_rights[color]["K"] = False
            self.castling_rights[color]["Q"] = False

        # Revoke castling right when rook moves
        if pt == ROOK:
            back_row = 0 if color == WHITE else 7
            if from_row == back_row:
                if from_col == 7:
                    self.castling_rights[color]["K"] = False
                elif from_col == 0:
                    self.castling_rights[color]["Q"] = False

        # Revoke castling right when rook is captured
        if captured != EMPTY and abs(captured) == ROOK:
            cap_color = self.color_of(captured)
            back_row = 0 if cap_color == WHITE else 7
            if to_row == back_row:
                if to_col == 7:
                    self.castling_rights[cap_color]["K"] = False
                elif to_col == 0:
                    self.castling_rights[cap_color]["Q"] = False

        # Promotion
        if move.promotion:
            self.set_piece(move.to_sq, color * move.promotion)

        # Switch turn
        if self.turn == BLACK:
            self.fullmove_number += 1
        self.turn = -self.turn

        self._record_position()
        self.move_history.append(move)

    def make_move(self, move: Move) -> bool:
        """Apply a move after verifying it is legal. Returns True on success."""
        if move not in self.get_legal_moves():
            return False
        self._apply_move_unchecked(move)
        return True

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------

    def is_checkmate(self) -> bool:
        return self.is_in_check(self.turn) and len(self.get_legal_moves()) == 0

    def is_stalemate(self) -> bool:
        return not self.is_in_check(self.turn) and len(self.get_legal_moves()) == 0

    def is_fifty_move_rule(self) -> bool:
        return self.halfmove_clock >= 100  # 50 full moves = 100 half-moves

    def is_threefold_repetition(self) -> bool:
        return any(count >= 3 for count in self.position_history.values())

    def is_insufficient_material(self) -> bool:
        """K vs K, K+minor vs K, or K+B vs K+B (same-color bishops)."""
        pieces: Dict[int, int] = {}
        for sq in range(64):
            p = self.get_piece(sq)
            if p != 0:
                pieces[p] = pieces.get(p, 0) + 1

        non_kings = {p: v for p, v in pieces.items() if abs(p) != KING}

        if not non_kings:
            return True  # K vs K

        if len(non_kings) == 1:
            p, cnt = next(iter(non_kings.items()))
            if cnt == 1 and abs(p) in (KNIGHT, BISHOP):
                return True  # KNK or KBK

        if len(non_kings) == 2:
            types = list(non_kings.keys())
            # KBKB — both bishops
            if (abs(types[0]) == BISHOP and abs(types[1]) == BISHOP
                    and non_kings[types[0]] == 1 and non_kings[types[1]] == 1):
                # Same-color squares?
                sqs = [sq for sq in range(64) if abs(self.get_piece(sq)) == BISHOP]
                if len(sqs) == 2:
                    r0, c0 = self.rc(sqs[0])
                    r1, c1 = self.rc(sqs[1])
                    if (r0 + c0) % 2 == (r1 + c1) % 2:
                        return True

        return False

    def is_draw(self) -> bool:
        return (self.is_stalemate()
                or self.is_fifty_move_rule()
                or self.is_insufficient_material()
                or self.is_threefold_repetition())

    def get_result(self) -> Optional[int]:
        """
        Returns:
            None  : game still ongoing
            WHITE : white wins
            BLACK : black wins
            0     : draw
        """
        if self.is_checkmate():
            return -self.turn   # current player lost
        if self.is_draw():
            return 0
        return None

    # ------------------------------------------------------------------
    # Observation (for RL agents)
    # ------------------------------------------------------------------

    def get_observation(self) -> np.ndarray:
        """
        Returns a (8, 8, 17) float32 array:
          ch 0–5  : white pieces (P, N, B, R, Q, K)
          ch 6–11 : black pieces (p, n, b, r, q, k)
          ch 12   : turn (1.0 = white to move)
          ch 13   : en passant square (1.0 at the ep square)
          ch 14   : white can castle kingside
          ch 15   : white can castle queenside
          ch 16   : black can castle kingside / queenside (combined)
        """
        obs = np.zeros((8, 8, 17), dtype=np.float32)

        piece_channel = {
            WHITE * PAWN: 0,   WHITE * KNIGHT: 1, WHITE * BISHOP: 2,
            WHITE * ROOK: 3,   WHITE * QUEEN: 4,  WHITE * KING: 5,
            BLACK * PAWN: 6,   BLACK * KNIGHT: 7, BLACK * BISHOP: 8,
            BLACK * ROOK: 9,   BLACK * QUEEN: 10, BLACK * KING: 11,
        }

        for r in range(8):
            for c in range(8):
                p = int(self.board[r][c])
                if p in piece_channel:
                    obs[r, c, piece_channel[p]] = 1.0

        if self.turn == WHITE:
            obs[:, :, 12] = 1.0

        if self.en_passant_sq is not None:
            r, c = self.rc(self.en_passant_sq)
            obs[r, c, 13] = 1.0

        cr = self.castling_rights
        if cr[WHITE]["K"]:
            obs[:, :, 14] = 1.0
        if cr[WHITE]["Q"]:
            obs[:, :, 15] = 1.0
        if cr[BLACK]["K"] or cr[BLACK]["Q"]:
            obs[:, :, 16] = 1.0

        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_ascii(self) -> str:
        lines = ["  a b c d e f g h", "  ---------------"]
        for row in range(7, -1, -1):
            line = f"{row + 1}|"
            for col in range(8):
                line += PIECE_SYMBOLS[int(self.board[row][col])] + " "
            lines.append(line)
        lines.append("")
        lines.append(f"{'White' if self.turn == WHITE else 'Black'} to move  "
                     f"| Move {self.fullmove_number}  "
                     f"| Halfmove clock: {self.halfmove_clock}")
        if self.is_in_check(self.turn):
            lines.append("*** CHECK ***")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.render_ascii()
