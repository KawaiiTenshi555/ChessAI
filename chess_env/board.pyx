# chess_env/board.pyx
# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, nonecheck=False, initializedcheck=False
"""
Port Cython de chess_env/board.py.

Représentation interne : tableau C plat int8_t[64].
  sq = row * 8 + col   (row = sq >> 3,  col = sq & 7)

Encodage des pièces identique à board.py :
  positif = Blanc, négatif = Noir
  |valeur| : 1=Pion 2=Cavalier 3=Fou 4=Tour 5=Dame 6=Roi

API Python 100 % compatible avec board.py.
"""

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int8_t
from libc.string cimport memcpy

# ---------------------------------------------------------------------------
# Constantes Python (visibles depuis Python et ce module)
# ---------------------------------------------------------------------------
EMPTY  = 0
PAWN   = 1
KNIGHT = 2
BISHOP = 3
ROOK   = 4
QUEEN  = 5
KING   = 6
WHITE  = 1
BLACK  = -1

PIECE_SYMBOLS = {
    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
   -1: 'p',-2: 'n',-3: 'b',-4: 'r',-5: 'q',-6: 'k',
    0: '.',
}

INITIAL_BACK_RANK = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]

# ---------------------------------------------------------------------------
# Constantes C (compile-time DEF)
# ---------------------------------------------------------------------------
DEF CEMPTY  = 0
DEF CPAWN   = 1
DEF CKNIGHT = 2
DEF CBISHOP = 3
DEF CROOK   = 4
DEF CQUEEN  = 5
DEF CKING   = 6
DEF CWHITE  = 1
DEF CBLACK  = -1

# ---------------------------------------------------------------------------
# Tables précalculées au niveau module (cavalier)
# ---------------------------------------------------------------------------
cdef int KNIGHT_DESTS[64][8]
cdef int KNIGHT_N_DESTS[64]


def _init_tables():
    """Initialise les tables de coups de cavalier. Appelée au chargement."""
    cdef int sq, r, c, i, nr, nc, count
    cdef int dr[8]
    cdef int dc[8]
    dr[0]=2;  dr[1]=2;  dr[2]=-2; dr[3]=-2
    dr[4]=1;  dr[5]=1;  dr[6]=-1; dr[7]=-1
    dc[0]=1;  dc[1]=-1; dc[2]=1;  dc[3]=-1
    dc[4]=2;  dc[5]=-2; dc[6]=2;  dc[7]=-2

    for sq in range(64):
        r = sq >> 3
        c = sq & 7
        count = 0
        for i in range(8):
            nr = r + dr[i]
            nc = c + dc[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                KNIGHT_DESTS[sq][count] = nr * 8 + nc
                count += 1
        KNIGHT_N_DESTS[sq] = count


# ---------------------------------------------------------------------------
# Classe _MoveUndo
# ---------------------------------------------------------------------------
cdef class _MoveUndo:
    """État minimal pour push/pop pendant le test de légalité."""
    cdef int  turn
    cdef bint cr_wk, cr_wq, cr_bk, cr_bq
    cdef int  ep_sq        # -1 si absent
    cdef int  hmc, fmn
    cdef int  captured
    cdef int  ep_pawn_sq   # -1 si absent
    cdef int  rook_from    # -1 si absent
    cdef int  rook_to


# ---------------------------------------------------------------------------
# Classe Move
# ---------------------------------------------------------------------------
cdef class Move:
    """Représente un coup d'échecs — version Cython."""
    cdef public int from_sq
    cdef public int to_sq
    cdef public int promotion

    def __init__(self, int from_sq, int to_sq, int promotion=0):
        self.from_sq   = from_sq
        self.to_sq     = to_sq
        self.promotion = promotion

    def to_action(self):
        """Encode en entier plat (action Gymnasium)."""
        return self.from_sq * 64 + self.to_sq

    @staticmethod
    def from_action(int action):
        return Move(action // 64, action % 64)

    def to_uci(self):
        cdef str files = "abcdefgh"
        s = (files[self.from_sq & 7] + str((self.from_sq >> 3) + 1)
           + files[self.to_sq   & 7] + str((self.to_sq   >> 3) + 1))
        if self.promotion:
            # KNIGHT=2 → index 0, BISHOP=3→1, ROOK=4→2, QUEEN=5→3
            s += "nbrq"[self.promotion - CKNIGHT]
        return s

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_sq   == (<Move>other).from_sq
            and self.to_sq     == (<Move>other).to_sq
            and self.promotion == (<Move>other).promotion)

    def __hash__(self):
        return (self.from_sq << 12) | (self.to_sq << 6) | self.promotion

    def __repr__(self):
        return f"Move({self.to_uci()})"


# ---------------------------------------------------------------------------
# Classe ChessBoard
# ---------------------------------------------------------------------------
cdef class ChessBoard:
    """
    Moteur d'échecs complet — version Cython.

    Représentation interne : tableau C plat int8_t[64].
    API Python identique à chess_env/board.py.
    """

    # Plateau C plat
    cdef int8_t _board[64]

    # État de jeu
    cdef public int turn
    cdef int        en_passant_sq    # -1 si absent
    cdef public int halfmove_clock
    cdef public int fullmove_number

    # Droits de roque (4 bools C)
    cdef bint castle_wk, castle_wq, castle_bk, castle_bq

    # Historique Python (rarement accédé dans les boucles critiques)
    cdef public list move_history
    cdef dict        _position_history

    # ---------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------

    def __init__(self):
        cdef int sq
        for sq in range(64):
            self._board[sq] = CEMPTY
        self.turn            = CWHITE
        self.en_passant_sq   = -1
        self.halfmove_clock  = 0
        self.fullmove_number = 1
        self.castle_wk       = True
        self.castle_wq       = True
        self.castle_bk       = True
        self.castle_bq       = True
        self.move_history    = []
        self._position_history = {}
        self._setup_initial_position()

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------

    def _setup_initial_position(self):
        cdef int col, piece
        cdef int back_rank[8]
        back_rank[0]=CROOK; back_rank[1]=CKNIGHT; back_rank[2]=CBISHOP; back_rank[3]=CQUEEN
        back_rank[4]=CKING; back_rank[5]=CBISHOP; back_rank[6]=CKNIGHT; back_rank[7]=CROOK
        for col in range(8):
            piece = back_rank[col]
            self._board[col]      = <int8_t>piece     # rangée blanche (row 0)
            self._board[56 + col] = <int8_t>(-piece)  # rangée noire  (row 7)
            self._board[8  + col] = <int8_t>CPAWN     # pions blancs  (row 1)
            self._board[48 + col] = <int8_t>(-CPAWN)  # pions noirs   (row 6)
        self._record_position()

    # ---------------------------------------------------------------
    # Propriété board (compat. chess_env.py qui accède à board.board)
    # ---------------------------------------------------------------

    @property
    def board(self):
        """Retourne un numpy (8,8) int8 reconstruit depuis le tableau C."""
        cdef int sq
        cdef cnp.ndarray[cnp.int8_t, ndim=2] arr = np.zeros((8, 8), dtype=np.int8)
        for sq in range(64):
            arr[sq >> 3, sq & 7] = self._board[sq]
        return arr

    # ---------------------------------------------------------------
    # castling_rights (compat. dict board.py)
    # ---------------------------------------------------------------

    @property
    def castling_rights(self):
        return {
            CWHITE: {"K": bool(self.castle_wk), "Q": bool(self.castle_wq)},
            CBLACK: {"K": bool(self.castle_bk), "Q": bool(self.castle_bq)},
        }

    # ---------------------------------------------------------------
    # position_history (compat. board.py)
    # ---------------------------------------------------------------

    @property
    def position_history(self):
        return self._position_history

    # ---------------------------------------------------------------
    # Coordonnées
    # ---------------------------------------------------------------

    @staticmethod
    def sq(int row, int col):
        return row * 8 + col

    @staticmethod
    def rc(int square):
        return square >> 3, square & 7

    @staticmethod
    def on_board(int row, int col):
        return 0 <= row < 8 and 0 <= col < 8

    @staticmethod
    def color_of(int piece):
        if piece > 0:
            return CWHITE
        if piece < 0:
            return CBLACK
        return 0

    # ---------------------------------------------------------------
    # Accès C inline
    # ---------------------------------------------------------------

    cdef inline int8_t get_piece_c(self, int sq) noexcept:
        return self._board[sq]

    cdef inline void set_piece_c(self, int sq, int8_t piece) noexcept:
        self._board[sq] = piece

    def get_piece(self, int sq):
        return int(self._board[sq])

    def set_piece(self, int sq, int piece):
        self._board[sq] = <int8_t>piece

    # ---------------------------------------------------------------
    # Hachage de position (répétition triple)
    # ---------------------------------------------------------------

    def _position_key(self):
        """
        Produit les mêmes bytes que board.py (compat. checkpoints existants).
        """
        cdef int sq
        cdef int ep
        cdef cnp.ndarray[cnp.int8_t, ndim=2] arr
        ep = self.en_passant_sq if self.en_passant_sq >= 0 else 255
        arr = np.zeros((8, 8), dtype=np.int8)
        for sq in range(64):
            arr[sq >> 3, sq & 7] = self._board[sq]
        flags = bytes([
            1 if self.turn == CWHITE else 0,
            1 if self.castle_wk else 0,
            1 if self.castle_wq else 0,
            1 if self.castle_bk else 0,
            1 if self.castle_bq else 0,
            ep,
        ])
        return arr.tobytes() + flags

    def _record_position(self):
        key = self._position_key()
        self._position_history[key] = self._position_history.get(key, 0) + 1

    # ---------------------------------------------------------------
    # Copie
    # ---------------------------------------------------------------

    def copy(self):
        cdef ChessBoard b
        b = ChessBoard.__new__(ChessBoard)
        memcpy(b._board, self._board, 64 * sizeof(int8_t))
        b.turn              = self.turn
        b.en_passant_sq     = self.en_passant_sq
        b.halfmove_clock    = self.halfmove_clock
        b.fullmove_number   = self.fullmove_number
        b.castle_wk         = self.castle_wk
        b.castle_wq         = self.castle_wq
        b.castle_bk         = self.castle_bk
        b.castle_bq         = self.castle_bq
        b._position_history = dict(self._position_history)
        b.move_history      = list(self.move_history)
        return b

    # ---------------------------------------------------------------
    # Détection d'attaque — fonction la plus appelée
    # ---------------------------------------------------------------

    cdef bint is_square_attacked_c(self, int sq, int by_color) noexcept:
        """Retourne True si sq est attaqué par une pièce de by_color."""
        cdef int row, col, nr, nc, i, pr
        cdef int8_t piece
        cdef int pawn_dir
        cdef int8_t target_pawn, target_knight, target_bishop
        cdef int8_t target_rook, target_queen, target_king
        cdef int kdr[8]
        cdef int kdc[8]
        cdef int ddr[4]
        cdef int ddc[4]
        cdef int ldr[4]
        cdef int ldc[4]
        cdef int gdr[8]
        cdef int gdc[8]

        row = sq >> 3
        col = sq & 7
        pawn_dir      = 1 if by_color == CWHITE else -1
        target_pawn   = <int8_t>(by_color * CPAWN)
        target_knight = <int8_t>(by_color * CKNIGHT)
        target_bishop = <int8_t>(by_color * CBISHOP)
        target_rook   = <int8_t>(by_color * CROOK)
        target_queen  = <int8_t>(by_color * CQUEEN)
        target_king   = <int8_t>(by_color * CKING)

        # --- Pions ---
        pr = row - pawn_dir
        if 0 <= pr < 8:
            if col > 0 and self._board[pr * 8 + col - 1] == target_pawn:
                return True
            if col < 7 and self._board[pr * 8 + col + 1] == target_pawn:
                return True

        # --- Cavaliers ---
        kdr[0]=2;  kdr[1]=2;  kdr[2]=-2; kdr[3]=-2
        kdr[4]=1;  kdr[5]=1;  kdr[6]=-1; kdr[7]=-1
        kdc[0]=1;  kdc[1]=-1; kdc[2]=1;  kdc[3]=-1
        kdc[4]=2;  kdc[5]=-2; kdc[6]=2;  kdc[7]=-2
        for i in range(8):
            nr = row + kdr[i]
            nc = col + kdc[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self._board[nr * 8 + nc] == target_knight:
                    return True

        # --- Sliders diagonaux (fou / dame) ---
        ddr[0]=1;  ddr[1]=1;  ddr[2]=-1; ddr[3]=-1
        ddc[0]=1;  ddc[1]=-1; ddc[2]=1;  ddc[3]=-1
        for i in range(4):
            nr = row + ddr[i]
            nc = col + ddc[i]
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = self._board[nr * 8 + nc]
                if piece != CEMPTY:
                    if piece == target_bishop or piece == target_queen:
                        return True
                    break
                nr += ddr[i]
                nc += ddc[i]

        # --- Sliders droits (tour / dame) ---
        ldr[0]=1;  ldr[1]=-1; ldr[2]=0;  ldr[3]=0
        ldc[0]=0;  ldc[1]=0;  ldc[2]=1;  ldc[3]=-1
        for i in range(4):
            nr = row + ldr[i]
            nc = col + ldc[i]
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = self._board[nr * 8 + nc]
                if piece != CEMPTY:
                    if piece == target_rook or piece == target_queen:
                        return True
                    break
                nr += ldr[i]
                nc += ldc[i]

        # --- Roi ---
        gdr[0]=1;  gdr[1]=1;  gdr[2]=1;  gdr[3]=0
        gdr[4]=0;  gdr[5]=-1; gdr[6]=-1; gdr[7]=-1
        gdc[0]=1;  gdc[1]=0;  gdc[2]=-1; gdc[3]=1
        gdc[4]=-1; gdc[5]=1;  gdc[6]=0;  gdc[7]=-1
        for i in range(8):
            nr = row + gdr[i]
            nc = col + gdc[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self._board[nr * 8 + nc] == target_king:
                    return True

        return False

    def is_square_attacked(self, int sq, int by_color):
        return bool(self.is_square_attacked_c(sq, by_color))

    # ---------------------------------------------------------------
    # Recherche du roi
    # ---------------------------------------------------------------

    cdef int _find_king_c(self, int color) noexcept:
        cdef int sq
        cdef int8_t target
        target = <int8_t>(color * CKING)
        for sq in range(64):
            if self._board[sq] == target:
                return sq
        return -1

    def find_king(self, int color):
        return self._find_king_c(color)

    def is_in_check(self, int color):
        cdef int king_sq
        king_sq = self._find_king_c(color)
        if king_sq == -1:
            return False
        return bool(self.is_square_attacked_c(king_sq, -color))

    # ---------------------------------------------------------------
    # Push / pop légal (sans copie du plateau)
    # ---------------------------------------------------------------

    cdef _MoveUndo _push_legal(self, Move move) noexcept:
        # Toutes les déclarations locales en tête (règle Cython)
        cdef int from_sq, to_sq, from_row, from_col, to_row, to_col
        cdef int pt, color, ep_pawn_row, col_diff, back_row, cap_color, cap_back
        cdef int8_t piece, captured
        cdef _MoveUndo undo

        from_sq  = move.from_sq
        to_sq    = move.to_sq
        from_row = from_sq >> 3
        from_col = from_sq & 7
        to_row   = to_sq   >> 3
        to_col   = to_sq   & 7
        piece    = self._board[from_sq]
        pt       = piece if piece > 0 else -piece
        color    = 1 if piece > 0 else -1
        captured = self._board[to_sq]

        undo = _MoveUndo()
        undo.turn       = self.turn
        undo.cr_wk      = self.castle_wk
        undo.cr_wq      = self.castle_wq
        undo.cr_bk      = self.castle_bk
        undo.cr_bq      = self.castle_bq
        undo.ep_sq      = self.en_passant_sq
        undo.hmc        = self.halfmove_clock
        undo.fmn        = self.fullmove_number
        undo.captured   = captured
        undo.ep_pawn_sq = -1
        undo.rook_from  = -1
        undo.rook_to    = -1

        # Horloge des demi-coups
        if pt == CPAWN or captured != CEMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Prise en passant
        if pt == CPAWN and to_sq == self.en_passant_sq:
            ep_pawn_row = to_row - 1 if color == CWHITE else to_row + 1
            undo.ep_pawn_sq = ep_pawn_row * 8 + to_col
            self._board[undo.ep_pawn_sq] = CEMPTY

        # Déplacement
        self._board[to_sq]   = piece
        self._board[from_sq] = CEMPTY

        # Mise à jour en passant
        if pt == CPAWN and (to_row - from_row) * (to_row - from_row) == 4:
            self.en_passant_sq = ((from_row + to_row) >> 1) * 8 + from_col
        else:
            self.en_passant_sq = -1

        # Roque : déplacement de la tour
        if pt == CKING:
            col_diff = to_col - from_col
            if col_diff == 2:    # petit roque
                undo.rook_from = from_row * 8 + 7
                undo.rook_to   = from_row * 8 + 5
                self._board[from_row * 8 + 5] = self._board[from_row * 8 + 7]
                self._board[from_row * 8 + 7] = CEMPTY
            elif col_diff == -2:  # grand roque
                undo.rook_from = from_row * 8
                undo.rook_to   = from_row * 8 + 3
                self._board[from_row * 8 + 3] = self._board[from_row * 8]
                self._board[from_row * 8]     = CEMPTY
            if color == CWHITE:
                self.castle_wk = False
                self.castle_wq = False
            else:
                self.castle_bk = False
                self.castle_bq = False

        # Révoque si la tour bouge
        if pt == CROOK:
            back_row = 0 if color == CWHITE else 7
            if from_row == back_row:
                if from_col == 7:
                    if color == CWHITE:
                        self.castle_wk = False
                    else:
                        self.castle_bk = False
                elif from_col == 0:
                    if color == CWHITE:
                        self.castle_wq = False
                    else:
                        self.castle_bq = False

        # Révoque si la tour adverse est capturée
        if captured != CEMPTY and (captured == CROOK or captured == -CROOK):
            cap_color = 1 if captured > 0 else -1
            cap_back  = 0 if cap_color == CWHITE else 7
            if to_row == cap_back:
                if to_col == 7:
                    if cap_color == CWHITE:
                        self.castle_wk = False
                    else:
                        self.castle_bk = False
                elif to_col == 0:
                    if cap_color == CWHITE:
                        self.castle_wq = False
                    else:
                        self.castle_bq = False

        # Promotion
        if move.promotion:
            self._board[to_sq] = <int8_t>(color * move.promotion)

        # Changement de trait
        if self.turn == CBLACK:
            self.fullmove_number += 1
        self.turn = -self.turn

        return undo

    cdef void _pop_legal(self, Move move, _MoveUndo undo) noexcept:
        cdef int from_sq, to_sq, from_row, from_col, to_row, to_col
        cdef int color

        from_sq  = move.from_sq
        to_sq    = move.to_sq
        from_row = from_sq >> 3
        from_col = from_sq & 7
        to_row   = to_sq   >> 3
        to_col   = to_sq   & 7
        color    = undo.turn   # couleur qui avait joué

        # Restaurer la pièce déplacée
        if move.promotion:
            self._board[from_sq] = <int8_t>(color * CPAWN)
        else:
            self._board[from_sq] = self._board[to_sq]
        self._board[to_sq] = <int8_t>undo.captured

        # Restaurer le pion capturé en passant
        if undo.ep_pawn_sq >= 0:
            self._board[undo.ep_pawn_sq] = <int8_t>((-color) * CPAWN)

        # Restaurer la tour du roque
        if undo.rook_from >= 0:
            self._board[undo.rook_from] = self._board[undo.rook_to]
            self._board[undo.rook_to]   = CEMPTY

        # Restaurer l'état scalaire
        self.turn            = undo.turn
        self.castle_wk       = undo.cr_wk
        self.castle_wq       = undo.cr_wq
        self.castle_bk       = undo.cr_bk
        self.castle_bq       = undo.cr_bq
        self.en_passant_sq   = undo.ep_sq
        self.halfmove_clock  = undo.hmc
        self.fullmove_number = undo.fmn

    # ---------------------------------------------------------------
    # Génération pseudo-légale
    # ---------------------------------------------------------------

    def _pseudo_legal_moves(self, int color):
        """Wrapper Python pour compat. avec board.py."""
        return self._pseudo_legal_moves_c(color)

    cdef list _pseudo_legal_moves_c(self, int color):
        cdef list moves
        cdef int sq
        cdef int8_t piece
        cdef int pt

        moves = []
        for sq in range(64):
            piece = self._board[sq]
            if piece == CEMPTY:
                continue
            if color == CWHITE and piece < 0:
                continue
            if color == CBLACK and piece > 0:
                continue
            pt = piece if piece > 0 else -piece
            if pt == CPAWN:
                self._pawn_moves_c(sq, color, moves)
            elif pt == CKNIGHT:
                self._knight_moves_c(sq, color, moves)
            elif pt == CBISHOP:
                self._bishop_moves_c(sq, color, moves)
            elif pt == CROOK:
                self._rook_moves_c(sq, color, moves)
            elif pt == CQUEEN:
                self._queen_moves_c(sq, color, moves)
            elif pt == CKING:
                self._king_moves_c(sq, color, moves)

        return moves

    # --- Pions ---

    cdef void _pawn_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int row, col, direction, start_row, promo_row
        cdef int nr, nr2, nc, target_sq, dc
        cdef int8_t target

        row       = sq >> 3
        col       = sq & 7
        direction = 1 if color == CWHITE else -1
        start_row = 1 if color == CWHITE else 6
        promo_row = 7 if color == CWHITE else 0

        # Avance d'une case
        nr = row + direction
        if 0 <= nr < 8 and self._board[nr * 8 + col] == CEMPTY:
            if nr == promo_row:
                moves.append(Move(sq, nr * 8 + col, CQUEEN))
                moves.append(Move(sq, nr * 8 + col, CROOK))
                moves.append(Move(sq, nr * 8 + col, CBISHOP))
                moves.append(Move(sq, nr * 8 + col, CKNIGHT))
            else:
                moves.append(Move(sq, nr * 8 + col))
                if row == start_row:
                    nr2 = row + 2 * direction
                    if self._board[nr2 * 8 + col] == CEMPTY:
                        moves.append(Move(sq, nr2 * 8 + col))

        # Captures diagonales (+ en passant)
        for dc in (-1, 1):
            nc = col + dc
            if nc < 0 or nc > 7:
                continue
            if nr < 0 or nr > 7:
                continue
            target_sq = nr * 8 + nc
            target    = self._board[target_sq]
            if target != CEMPTY:
                if (color == CWHITE and target < 0) or (color == CBLACK and target > 0):
                    if nr == promo_row:
                        moves.append(Move(sq, target_sq, CQUEEN))
                        moves.append(Move(sq, target_sq, CROOK))
                        moves.append(Move(sq, target_sq, CBISHOP))
                        moves.append(Move(sq, target_sq, CKNIGHT))
                    else:
                        moves.append(Move(sq, target_sq))
            elif target_sq == self.en_passant_sq:
                moves.append(Move(sq, target_sq))

    # --- Cavaliers (tables précalculées) ---

    cdef void _knight_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int i, dest
        cdef int8_t target
        for i in range(KNIGHT_N_DESTS[sq]):
            dest   = KNIGHT_DESTS[sq][i]
            target = self._board[dest]
            if target == CEMPTY:
                moves.append(Move(sq, dest))
            elif (color == CWHITE and target < 0) or (color == CBLACK and target > 0):
                moves.append(Move(sq, dest))

    # --- Sliders génériques ---

    cdef void _sliding_moves_c(self, int sq, int color,
                                int* drs, int* dcs, int n_dirs,
                                list moves) noexcept:
        cdef int i, nr, nc, row, col
        cdef int8_t target

        row = sq >> 3
        col = sq & 7
        for i in range(n_dirs):
            nr = row + drs[i]
            nc = col + dcs[i]
            while 0 <= nr < 8 and 0 <= nc < 8:
                target = self._board[nr * 8 + nc]
                if target == CEMPTY:
                    moves.append(Move(sq, nr * 8 + nc))
                elif (color == CWHITE and target < 0) or (color == CBLACK and target > 0):
                    moves.append(Move(sq, nr * 8 + nc))
                    break
                else:
                    break
                nr += drs[i]
                nc += dcs[i]

    cdef void _bishop_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int drs[4]
        cdef int dcs[4]
        drs[0]=1;  drs[1]=1;  drs[2]=-1; drs[3]=-1
        dcs[0]=1;  dcs[1]=-1; dcs[2]=1;  dcs[3]=-1
        self._sliding_moves_c(sq, color, drs, dcs, 4, moves)

    cdef void _rook_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int drs[4]
        cdef int dcs[4]
        drs[0]=1;  drs[1]=-1; drs[2]=0;  drs[3]=0
        dcs[0]=0;  dcs[1]=0;  dcs[2]=1;  dcs[3]=-1
        self._sliding_moves_c(sq, color, drs, dcs, 4, moves)

    cdef void _queen_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int drs[8]
        cdef int dcs[8]
        drs[0]=1;  drs[1]=1;  drs[2]=-1; drs[3]=-1
        drs[4]=1;  drs[5]=-1; drs[6]=0;  drs[7]=0
        dcs[0]=1;  dcs[1]=-1; dcs[2]=1;  dcs[3]=-1
        dcs[4]=0;  dcs[5]=0;  dcs[6]=1;  dcs[7]=-1
        self._sliding_moves_c(sq, color, drs, dcs, 8, moves)

    # --- Roi ---

    cdef void _king_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int row, col, nr, nc, i
        cdef int8_t target
        cdef int drs[8]
        cdef int dcs[8]

        row = sq >> 3
        col = sq & 7
        drs[0]=1;  drs[1]=1;  drs[2]=1;  drs[3]=0
        drs[4]=0;  drs[5]=-1; drs[6]=-1; drs[7]=-1
        dcs[0]=1;  dcs[1]=0;  dcs[2]=-1; dcs[3]=1
        dcs[4]=-1; dcs[5]=1;  dcs[6]=0;  dcs[7]=-1

        for i in range(8):
            nr = row + drs[i]
            nc = col + dcs[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                target = self._board[nr * 8 + nc]
                if target == CEMPTY:
                    moves.append(Move(sq, nr * 8 + nc))
                elif (color == CWHITE and target < 0) or (color == CBLACK and target > 0):
                    moves.append(Move(sq, nr * 8 + nc))

        self._castling_moves_c(sq, color, moves)

    cdef void _castling_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int row, king_sq, opponent

        row      = 0 if color == CWHITE else 7
        king_sq  = row * 8 + 4
        opponent = -color

        # Le roi doit être sur sa case initiale
        if self._board[king_sq] != <int8_t>(color * CKING):
            return

        # Le roi ne doit pas être en échec
        if self.is_square_attacked_c(king_sq, opponent):
            return

        if color == CWHITE:
            if self.castle_wk:
                if (self._board[row * 8 + 5] == CEMPTY and
                        self._board[row * 8 + 6] == CEMPTY):
                    if (not self.is_square_attacked_c(row * 8 + 5, opponent) and
                            not self.is_square_attacked_c(row * 8 + 6, opponent)):
                        moves.append(Move(king_sq, row * 8 + 6))
            if self.castle_wq:
                if (self._board[row * 8 + 1] == CEMPTY and
                        self._board[row * 8 + 2] == CEMPTY and
                        self._board[row * 8 + 3] == CEMPTY):
                    if (not self.is_square_attacked_c(row * 8 + 3, opponent) and
                            not self.is_square_attacked_c(row * 8 + 2, opponent)):
                        moves.append(Move(king_sq, row * 8 + 2))
        else:
            if self.castle_bk:
                if (self._board[row * 8 + 5] == CEMPTY and
                        self._board[row * 8 + 6] == CEMPTY):
                    if (not self.is_square_attacked_c(row * 8 + 5, opponent) and
                            not self.is_square_attacked_c(row * 8 + 6, opponent)):
                        moves.append(Move(king_sq, row * 8 + 6))
            if self.castle_bq:
                if (self._board[row * 8 + 1] == CEMPTY and
                        self._board[row * 8 + 2] == CEMPTY and
                        self._board[row * 8 + 3] == CEMPTY):
                    if (not self.is_square_attacked_c(row * 8 + 3, opponent) and
                            not self.is_square_attacked_c(row * 8 + 2, opponent)):
                        moves.append(Move(king_sq, row * 8 + 2))

    # ---------------------------------------------------------------
    # Coups légaux
    # ---------------------------------------------------------------

    def get_legal_moves(self):
        cdef list pseudo, legal
        cdef Move move
        cdef _MoveUndo undo
        cdef int king_sq
        cdef int mover = self.turn   # sauvé AVANT le push (qui change self.turn)

        pseudo = self._pseudo_legal_moves_c(mover)
        legal  = []

        for move in pseudo:
            undo    = self._push_legal(move)
            king_sq = self._find_king_c(mover)              # roi du joueur qui a bougé
            if king_sq == -1 or not self.is_square_attacked_c(king_sq, -mover):
                legal.append(move)
            self._pop_legal(move, undo)
        return legal

    # ---------------------------------------------------------------
    # Application d'un coup (avec mise à jour historique)
    # ---------------------------------------------------------------

    def _apply_move_unchecked(self, Move move):
        """Applique un coup sans vérification de légalité (usage interne)."""
        cdef int from_sq, to_sq, from_row, from_col, to_row, to_col
        cdef int pt, color, ep_pawn_row, col_diff, back_row, cap_color, cap_back
        cdef int8_t piece, captured

        from_sq  = move.from_sq
        to_sq    = move.to_sq
        from_row = from_sq >> 3
        from_col = from_sq & 7
        to_row   = to_sq   >> 3
        to_col   = to_sq   & 7
        piece    = self._board[from_sq]
        pt       = piece if piece > 0 else -piece
        color    = 1 if piece > 0 else -1
        captured = self._board[to_sq]

        # Horloge des demi-coups
        if pt == CPAWN or captured != CEMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Déplacement
        self._board[to_sq]   = piece
        self._board[from_sq] = CEMPTY

        # Prise en passant
        if pt == CPAWN and to_sq == self.en_passant_sq:
            ep_pawn_row = to_row - 1 if color == CWHITE else to_row + 1
            self._board[ep_pawn_row * 8 + to_col] = CEMPTY

        # Mise à jour en passant
        if pt == CPAWN and (to_row - from_row) * (to_row - from_row) == 4:
            self.en_passant_sq = ((from_row + to_row) >> 1) * 8 + from_col
        else:
            self.en_passant_sq = -1

        # Roque : déplacement de la tour
        if pt == CKING:
            col_diff = to_col - from_col
            if col_diff == 2:
                self._board[from_row * 8 + 5] = self._board[from_row * 8 + 7]
                self._board[from_row * 8 + 7] = CEMPTY
            elif col_diff == -2:
                self._board[from_row * 8 + 3] = self._board[from_row * 8]
                self._board[from_row * 8]     = CEMPTY
            if color == CWHITE:
                self.castle_wk = False
                self.castle_wq = False
            else:
                self.castle_bk = False
                self.castle_bq = False

        # Révoque si la tour bouge
        if pt == CROOK:
            back_row = 0 if color == CWHITE else 7
            if from_row == back_row:
                if from_col == 7:
                    if color == CWHITE:
                        self.castle_wk = False
                    else:
                        self.castle_bk = False
                elif from_col == 0:
                    if color == CWHITE:
                        self.castle_wq = False
                    else:
                        self.castle_bq = False

        # Révoque si tour adverse capturée
        if captured != CEMPTY and (captured == CROOK or captured == -CROOK):
            cap_color = 1 if captured > 0 else -1
            cap_back  = 0 if cap_color == CWHITE else 7
            if to_row == cap_back:
                if to_col == 7:
                    if cap_color == CWHITE:
                        self.castle_wk = False
                    else:
                        self.castle_bk = False
                elif to_col == 0:
                    if cap_color == CWHITE:
                        self.castle_wq = False
                    else:
                        self.castle_bq = False

        # Promotion
        if move.promotion:
            self._board[to_sq] = <int8_t>(color * move.promotion)

        # Changement de trait
        if self.turn == CBLACK:
            self.fullmove_number += 1
        self.turn = -self.turn

        self._record_position()
        self.move_history.append(move)

    def make_move(self, Move move):
        """Applique un coup après vérification. Retourne True si légal."""
        legal = self.get_legal_moves()
        for m in legal:
            if m == move:
                self._apply_move_unchecked(move)
                return True
        return False

    # ---------------------------------------------------------------
    # État de jeu
    # ---------------------------------------------------------------

    def is_checkmate(self):
        return self.is_in_check(self.turn) and len(self.get_legal_moves()) == 0

    def is_stalemate(self):
        return not self.is_in_check(self.turn) and len(self.get_legal_moves()) == 0

    def is_fifty_move_rule(self):
        return self.halfmove_clock >= 100

    def is_threefold_repetition(self):
        return any(count >= 3 for count in self._position_history.values())

    def is_insufficient_material(self):
        """K vs K, K+mineur vs K, ou K+F vs K+F (mêmes cases)."""
        cdef int sq
        cdef int8_t p
        pieces = {}
        for sq in range(64):
            p = self._board[sq]
            if p != CEMPTY:
                pieces[int(p)] = pieces.get(int(p), 0) + 1

        non_kings = {p: v for p, v in pieces.items() if abs(p) != CKING}

        if not non_kings:
            return True  # K vs K

        if len(non_kings) == 1:
            p, cnt = next(iter(non_kings.items()))
            if cnt == 1 and abs(p) in (CKNIGHT, CBISHOP):
                return True

        if len(non_kings) == 2:
            types = list(non_kings.keys())
            if (abs(types[0]) == CBISHOP and abs(types[1]) == CBISHOP
                    and non_kings[types[0]] == 1 and non_kings[types[1]] == 1):
                sqs = [sq for sq in range(64) if abs(self._board[sq]) == CBISHOP]
                if len(sqs) == 2:
                    r0 = sqs[0] >> 3; c0 = sqs[0] & 7
                    r1 = sqs[1] >> 3; c1 = sqs[1] & 7
                    if (r0 + c0) % 2 == (r1 + c1) % 2:
                        return True

        return False

    def is_draw(self):
        return (self.is_stalemate()
                or self.is_fifty_move_rule()
                or self.is_insufficient_material()
                or self.is_threefold_repetition())

    def get_result(self):
        """
        Retourne :
            None  : partie en cours
            WHITE : victoire blanche
            BLACK : victoire noire
            0     : nulle
        """
        if self.is_checkmate():
            return -self.turn
        if self.is_draw():
            return 0
        return None

    # ---------------------------------------------------------------
    # Observation Gymnasium (8, 8, 17) float32
    # ---------------------------------------------------------------

    def get_observation(self):
        cdef cnp.ndarray[cnp.float32_t, ndim=3] obs
        cdef int sq, row, col, ch
        cdef int8_t piece

        obs = np.zeros((8, 8, 17), dtype=np.float32)

        for sq in range(64):
            piece = self._board[sq]
            if piece == CEMPTY:
                continue
            row = sq >> 3
            col = sq & 7
            if piece > 0:
                ch = piece - 1          # canaux 0-5 : blancs
            else:
                ch = (-piece - 1) + 6   # canaux 6-11 : noirs
            obs[row, col, ch] = 1.0

        if self.turn == CWHITE:
            obs[:, :, 12] = 1.0

        if self.en_passant_sq >= 0:
            obs[self.en_passant_sq >> 3, self.en_passant_sq & 7, 13] = 1.0

        if self.castle_wk:
            obs[:, :, 14] = 1.0
        if self.castle_wq:
            obs[:, :, 15] = 1.0
        if self.castle_bk or self.castle_bq:
            obs[:, :, 16] = 1.0

        return obs

    # ---------------------------------------------------------------
    # Rendu
    # ---------------------------------------------------------------

    def render_ascii(self):
        cdef int row, col
        lines = ["  a b c d e f g h", "  ---------------"]
        for row in range(7, -1, -1):
            line = f"{row + 1}|"
            for col in range(8):
                line += PIECE_SYMBOLS[int(self._board[row * 8 + col])] + " "
            lines.append(line)
        lines.append("")
        lines.append(f"{'White' if self.turn == CWHITE else 'Black'} to move  "
                     f"| Move {self.fullmove_number}  "
                     f"| Halfmove clock: {self.halfmove_clock}")
        if self.is_in_check(self.turn):
            lines.append("*** CHECK ***")
        return "\n".join(lines)

    def __repr__(self):
        return self.render_ascii()


# ---------------------------------------------------------------------------
# Initialisation des tables au chargement du module
# ---------------------------------------------------------------------------
_init_tables()
