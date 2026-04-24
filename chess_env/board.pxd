# chess_env/board.pxd — Déclarations Cython pour board.pyx
# Permet aux autres modules .pyx d'importer les types via :
#   from chess_env.board_cy cimport ChessBoard, Move, _MoveUndo

from libc.stdint cimport int8_t


cdef class Move:
    cdef public int from_sq
    cdef public int to_sq
    cdef public int promotion


cdef class _MoveUndo:
    cdef int  turn
    cdef bint cr_wk, cr_wq, cr_bk, cr_bq
    cdef int  ep_sq        # -1 si pas d'en passant
    cdef int  hmc, fmn
    cdef int  captured
    cdef int  ep_pawn_sq   # -1 si pas de prise en passant
    cdef int  rook_from    # -1 si pas de roque
    cdef int  rook_to


cdef class ChessBoard:
    # Plateau plat C — sq = row*8 + col
    cdef int8_t _board[64]

    # État de jeu
    cdef public int turn
    cdef int        en_passant_sq    # -1 si absent
    cdef int        halfmove_clock
    cdef int        fullmove_number

    # Droits de roque
    cdef bint castle_wk, castle_wq, castle_bk, castle_bq

    # Historique (objets Python — rarement accédés en boucle critique)
    cdef public list move_history
    cdef dict        _position_history

    # Accès C inline
    cdef inline int8_t get_piece_c(self, int sq) noexcept
    cdef inline void   set_piece_c(self, int sq, int8_t piece) noexcept

    # Détection d'attaque / roi
    cdef bint is_square_attacked_c(self, int sq, int by_color) noexcept
    cdef int  _find_king_c(self, int color) noexcept

    # Push / pop légal (pour filtrer les pseudo-légaux)
    cdef _MoveUndo _push_legal(self, Move move) noexcept
    cdef void      _pop_legal(self, Move move, _MoveUndo undo) noexcept
