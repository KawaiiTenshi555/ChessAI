# Plan de migration C/Cython — Moteur d'échecs ChessIA

## Contexte et motivation

### Pourquoi ce projet est limité par le CPU

Le profiling de l'entraînement RL révèle que le goulot d'étranglement n'est **pas** le réseau de neurones mais le moteur d'échecs Python pur :

| Opération | Temps mesuré | Processeur |
|---|---|---|
| `get_legal_moves()` | 0.31 ms/appel | CPU — Python pur |
| `env.step()` complet | ~2 ms/step | CPU — Python pur |
| Forward pass réseau (batch=256) | ~0.17 ms | GPU |
| `get_observation()` | 0.015 ms | CPU — numpy |

`get_legal_moves()` est appelé ~30 fois par step d'entraînement (génération pseudo-légale + filtrage par échec). Le GPU reste inactif la majeure partie du temps car la collecte d'expériences est entièrement liée au CPU.

### Objectif

Réécrire le moteur d'échecs (`chess_env/board.py`) en Cython pour obtenir un code compilé en C, tout en conservant **exactement la même API Python** — les agents, l'environnement Gymnasium, la web app et les tests continuent de fonctionner sans modification.

---

## Comparaison des approches

| Approche | Effort | Gain move gen | Maintenance | Recommandation |
|---|---|---|---|---|
| **Cython typé** | Moyen (1–2 sem.) | 10–30× | Facile | ✅ Phase 1 |
| **Extension C pure** (CPython API) | Élevé (3–4 sem.) | 30–80× | Difficile | Non recommandé |
| **Pybind11** (C++) | Élevé (3–4 sem.) | 30–80× | Moyen | Alternative à C pur |
| **Bitboards en Cython** | Très élevé (4–6 sem.) | 100–300× | Moyen | ✅ Phase 2 (optionnel) |
| **python-chess** (lib externe) | Faible (2–3 j.) | 2–5× | Très facile | Alternative rapide |

---

## Architecture cible

```
chess_env/
├── board.py          # Version Python originale (conservée comme fallback)
├── board.pyx         # Version Cython (nouveau)
├── board.pxd         # Déclarations d'interface Cython (nouveau)
├── chess_env.py      # Inchangé — importe board_cy si disponible, sinon board
└── __init__.py

docs/
└── plan_cython.md    # Ce fichier

setup.py              # Script de compilation Cython (nouveau)
```

---

## Phase 0 — Infrastructure (½ journée)

### 0.1 Dépendances

```bash
pip install cython
# Sur Windows : Visual Studio Build Tools requis
# Sur Linux/Mac : gcc ou clang suffisent
```

Vérifier que la compilation fonctionne :

```bash
python -c "import Cython; print(Cython.__version__)"
```

### 0.2 Créer `setup.py`

```python
# setup.py
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="chess_env.board_cy",
        sources=["chess_env/board.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=(
            ["/O2", "/GL"] if os.name == "nt"
            else ["-O3", "-march=native", "-ffast-math"]
        ),
        extra_link_args=["/LTCG"] if os.name == "nt" else [],
    )
]

setup(
    name="ChessIA",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,       # pas de vérif. d'index → vitesse
            "wraparound": False,        # pas d'index négatifs → vitesse
            "cdivision": True,          # division C (pas de check ZeroDivision)
            "nonecheck": False,         # pas de vérif. None sur cdef classes
            "initializedcheck": False,  # pas de vérif. d'init. memoryview
        },
        annotate=True,                  # génère board.pyx.html pour inspecter
    ),
)
```

Commande de compilation :

```bash
python setup.py build_ext --inplace
```

> **Note Windows** : il faut Visual Studio Build Tools 2022 avec le workload "Desktop development with C++". Téléchargeable gratuitement sur visualstudio.microsoft.com.

### 0.3 Adaptateur d'import dans `chess_env/chess_env.py`

Ajouter en haut du fichier, **avant** tout import de `board` :

```python
try:
    from chess_env.board_cy import ChessBoard, Move, WHITE, BLACK, QUEEN
    from chess_env.board_cy import PAWN, KNIGHT, BISHOP, ROOK, KING, EMPTY
    _ENGINE = "cython"
except ImportError:
    from chess_env.board import ChessBoard, Move, WHITE, BLACK, QUEEN
    from chess_env.board import PAWN, KNIGHT, BISHOP, ROOK, KING, EMPTY
    _ENGINE = "python"
```

Faire de même dans tous les fichiers qui importent de `chess_env.board` directement :
- `web/app.py`
- `benchmark/stockfish.py`
- `agents/deep_rl/alphazero.py`

---

## Phase 1 — Port de `board.py` vers `board.pyx` (3–5 jours)

### 1.1 Structure générale du fichier `.pyx`

```cython
# chess_env/board.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdint cimport int8_t, int32_t, uint64_t
from libc.string cimport memcpy

# Constantes (identiques à board.py, visibles en Python et Cython)
EMPTY  = 0
PAWN   = 1; KNIGHT = 2; BISHOP = 3; ROOK = 4; QUEEN = 5; KING = 6
WHITE  = 1; BLACK  = -1

# Déclaration C des constantes pour usage interne rapide
DEF CEMPTY  = 0
DEF CPAWN   = 1; DEF CKNIGHT = 2; DEF CBISHOP = 3
DEF CROOK   = 4; DEF CQUEEN  = 5; DEF CKING   = 6
DEF CWHITE  = 1; DEF CBLACK  = -1
```

### 1.2 Classe `Move` en Cython

```cython
cdef class Move:
    """Représente un coup d'échecs — version Cython avec attributs C."""
    cdef public int from_sq
    cdef public int to_sq
    cdef public int promotion

    def __init__(self, int from_sq, int to_sq, int promotion=0):
        self.from_sq   = from_sq
        self.to_sq     = to_sq
        self.promotion = promotion

    def to_action(self):
        return self.from_sq * 64 + self.to_sq

    @staticmethod
    def from_action(int action):
        return Move(action // 64, action % 64)

    def to_uci(self):
        cdef str files = "abcdefgh"
        s = (files[self.from_sq & 7] + str((self.from_sq >> 3) + 1)
           + files[self.to_sq   & 7] + str((self.to_sq   >> 3) + 1))
        if self.promotion:
            s += "nbrq"[self.promotion - CPAWN - 1]
        return s

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.from_sq   == other.from_sq
            and self.to_sq     == other.to_sq
            and self.promotion == other.promotion)

    def __hash__(self):
        return (self.from_sq << 12) | (self.to_sq << 6) | self.promotion

    def __repr__(self):
        return f"Move({self.to_uci()})"
```

### 1.3 Classe `ChessBoard` — représentation interne

**Changement clé** : remplacer `np.zeros((8,8), dtype=np.int8)` par un tableau C plat `int8_t[64]`.

```cython
cdef class ChessBoard:
    # Plateau : tableau C plat, accès board[sq] en O(1) garanti
    cdef int8_t  _board[64]

    # État de jeu — tous des entiers C, pas d'objets Python
    cdef public int   turn
    cdef int          en_passant_sq    # -1 si pas d'en passant
    cdef int          halfmove_clock
    cdef int          fullmove_number

    # Droits de roque — 4 booléens C
    cdef bint castle_wk, castle_wq, castle_bk, castle_bq

    # Historique — reste en Python (accédé rarement)
    cdef public list  move_history
    cdef dict         _position_history

    def __init__(self):
        cdef int sq
        for sq in range(64):
            self._board[sq] = CEMPTY
        self.turn             = CWHITE
        self.en_passant_sq    = -1
        self.halfmove_clock   = 0
        self.fullmove_number  = 1
        self.castle_wk        = True
        self.castle_wq        = True
        self.castle_bk        = True
        self.castle_bq        = True
        self.move_history     = []
        self._position_history = {}
        self._setup_initial_position()
```

### 1.4 Accès au plateau — fonctions `cdef inline`

```cython
    cdef inline int8_t get_piece_c(self, int sq) noexcept nogil:
        return self._board[sq]

    cdef inline void set_piece_c(self, int sq, int8_t piece) noexcept nogil:
        self._board[sq] = piece

    # Wrapper Python pour compat. API existante
    def get_piece(self, int sq):
        return int(self._board[sq])

    def set_piece(self, int sq, int piece):
        self._board[sq] = <int8_t>piece
```

> `noexcept nogil` : permet à Cython de générer du code C sans gestion d'exceptions ni GIL — maximum de vitesse pour les boucles internes.

### 1.5 Détection d'attaque — fonction critique

C'est la fonction la plus appelée (plusieurs fois par move légal). Doit être `cdef` pure :

```cython
    cdef bint is_square_attacked_c(self, int sq, int by_color) noexcept:
        """Retourne True si sq est attaqué par une pièce de by_color."""
        cdef int row = sq >> 3, col = sq & 7
        cdef int nr, nc, dr, dc, i
        cdef int8_t piece
        cdef int pawn_dir = 1 if by_color == CWHITE else -1
        cdef int target_pawn = <int8_t>(by_color * CPAWN)
        cdef int target_knight = <int8_t>(by_color * CKNIGHT)
        cdef int target_bishop = <int8_t>(by_color * CBISHOP)
        cdef int target_rook   = <int8_t>(by_color * CROOK)
        cdef int target_queen  = <int8_t>(by_color * CQUEEN)
        cdef int target_king   = <int8_t>(by_color * CKING)

        # Attaques de pion
        cdef int pr = row - pawn_dir
        if 0 <= pr < 8:
            if col > 0 and self._board[pr * 8 + col - 1] == target_pawn:
                return True
            if col < 7 and self._board[pr * 8 + col + 1] == target_pawn:
                return True

        # Attaques de cavalier — 8 sauts fixes
        cdef int[8] knight_dr = [2, 2, -2, -2, 1, 1, -1, -1]
        cdef int[8] knight_dc = [1, -1, 1, -1, 2, -2, 2, -2]
        for i in range(8):
            nr = row + knight_dr[i]
            nc = col + knight_dc[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self._board[nr * 8 + nc] == target_knight:
                    return True

        # Sliders diagonaux (fou/dame)
        cdef int[4] diag_dr = [1, 1, -1, -1]
        cdef int[4] diag_dc = [1, -1, 1, -1]
        for i in range(4):
            nr = row + diag_dr[i]
            nc = col + diag_dc[i]
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = self._board[nr * 8 + nc]
                if piece != CEMPTY:
                    if piece == target_bishop or piece == target_queen:
                        return True
                    break
                nr += diag_dr[i]
                nc += diag_dc[i]

        # Sliders droits (tour/dame)
        cdef int[4] line_dr = [1, -1, 0, 0]
        cdef int[4] line_dc = [0, 0, 1, -1]
        for i in range(4):
            nr = row + line_dr[i]
            nc = col + line_dc[i]
            while 0 <= nr < 8 and 0 <= nc < 8:
                piece = self._board[nr * 8 + nc]
                if piece != CEMPTY:
                    if piece == target_rook or piece == target_queen:
                        return True
                    break
                nr += line_dr[i]
                nc += line_dc[i]

        # Roi
        cdef int[8] king_dr = [1, 1, 1, 0, 0, -1, -1, -1]
        cdef int[8] king_dc = [1, 0, -1, 1, -1, 1, 0, -1]
        for i in range(8):
            nr = row + king_dr[i]
            nc = col + king_dc[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                if self._board[nr * 8 + nc] == target_king:
                    return True

        return False

    # Wrapper Python
    def is_square_attacked(self, int sq, int by_color):
        return self.is_square_attacked_c(sq, by_color)
```

### 1.6 Structure `_MoveUndo` en Cython

```cython
cdef class _MoveUndo:
    """État minimal pour push/pop — attributs C purs."""
    cdef int   turn
    cdef bint  cr_wk, cr_wq, cr_bk, cr_bq
    cdef int   ep_sq
    cdef int   hmc, fmn
    cdef int   captured
    cdef int   ep_pawn_sq    # -1 si pas d'en passant
    cdef int   rook_from     # -1 si pas de roque
    cdef int   rook_to
```

### 1.7 `_push_legal` / `_pop_legal` typés

```cython
    cdef _MoveUndo _push_legal(self, Move move) noexcept:
        cdef int from_sq = move.from_sq, to_sq = move.to_sq
        cdef int from_row = from_sq >> 3, from_col = from_sq & 7
        cdef int to_row   = to_sq   >> 3, to_col   = to_sq   & 7
        cdef int8_t piece    = self._board[from_sq]
        cdef int    pt       = piece if piece > 0 else -piece
        cdef int    color    = 1 if piece > 0 else -1
        cdef int8_t captured = self._board[to_sq]

        cdef _MoveUndo undo = _MoveUndo()
        undo.turn     = self.turn
        undo.cr_wk    = self.castle_wk;  undo.cr_wq = self.castle_wq
        undo.cr_bk    = self.castle_bk;  undo.cr_bq = self.castle_bq
        undo.ep_sq    = self.en_passant_sq
        undo.hmc      = self.halfmove_clock
        undo.fmn      = self.fullmove_number
        undo.captured = captured
        undo.ep_pawn_sq = -1
        undo.rook_from  = -1
        undo.rook_to    = -1

        # ... (logique identique à board.py mais avec accès tableau C)
        return undo

    cdef void _pop_legal(self, Move move, _MoveUndo undo) noexcept:
        # ... restauration rapide depuis undo
        pass
```

### 1.8 `get_legal_moves` final

```cython
    def get_legal_moves(self):
        cdef list pseudo = self._pseudo_legal_moves_c(self.turn)
        cdef list legal  = []
        cdef Move move
        cdef _MoveUndo undo

        for move in pseudo:
            undo = self._push_legal(move)
            if not self.is_square_attacked_c(self._find_king_c(self.turn), -self.turn):
                legal.append(move)
            self._pop_legal(move, undo)
        return legal

    cdef int _find_king_c(self, int color) noexcept:
        cdef int sq
        cdef int8_t target = <int8_t>(color * CKING)
        for sq in range(64):
            if self._board[sq] == target:
                return sq
        return -1
```

---

## Phase 2 — Génération pseudo-légale typée (2–3 jours)

### 2.1 Tables de mouvements précalculées

Précalculer en dehors de la classe les sauts de cavalier et les directions de pions — 0 calcul à l'exécution :

```cython
# Initialisé une seule fois au chargement du module
cdef int[64][8] KNIGHT_DESTS   # pour chaque case : cases attaquées (≤8)
cdef int[64]    KNIGHT_N_DESTS # nombre de destinations valides

def _init_knight_table():
    cdef int sq, r, c, i, nr, nc, count
    cdef int[8] dr = [2, 2, -2, -2, 1, 1, -1, -1]
    cdef int[8] dc = [1, -1, 1, -1, 2, -2, 2, -2]
    for sq in range(64):
        r = sq >> 3; c = sq & 7; count = 0
        for i in range(8):
            nr = r + dr[i]; nc = c + dc[i]
            if 0 <= nr < 8 and 0 <= nc < 8:
                KNIGHT_DESTS[sq][count] = nr * 8 + nc
                count += 1
        KNIGHT_N_DESTS[sq] = count

_init_knight_table()
```

### 2.2 Génération pseudo-légale des cavaliers

```cython
    cdef void _knight_moves_c(self, int sq, int color, list moves) noexcept:
        cdef int i, dest
        cdef int8_t target
        for i in range(KNIGHT_N_DESTS[sq]):
            dest   = KNIGHT_DESTS[sq][i]
            target = self._board[dest]
            if target == CEMPTY or (target > 0) != (color > 0):
                moves.append(Move(sq, dest))
```

### 2.3 Sliders (fou, tour, dame) avec early exit

```cython
    cdef void _sliding_moves_c(self, int sq, int color,
                               int* drs, int* dcs, int n_dirs,
                               list moves) noexcept:
        cdef int i, nr, nc
        cdef int8_t target
        cdef int row = sq >> 3, col = sq & 7
        for i in range(n_dirs):
            nr = row + drs[i]; nc = col + dcs[i]
            while 0 <= nr < 8 and 0 <= nc < 8:
                target = self._board[nr * 8 + nc]
                if target == CEMPTY:
                    moves.append(Move(sq, nr * 8 + nc))
                elif (target > 0) != (color > 0):   # capture adverse
                    moves.append(Move(sq, nr * 8 + nc))
                    break
                else:
                    break                             # bloqué par pièce amie
                nr += drs[i]; nc += dcs[i]
```

---

## Phase 3 — Observation Gymnasium (½ journée)

```cython
    def get_observation(self):
        """Encode le plateau en tenseur (8, 8, 17) float32."""
        cdef cnp.ndarray[cnp.float32_t, ndim=3] obs = np.zeros((8, 8, 17), dtype=np.float32)
        cdef int sq, row, col, ch
        cdef int8_t piece

        for sq in range(64):
            piece = self._board[sq]
            if piece == CEMPTY:
                continue
            row = sq >> 3; col = sq & 7
            if piece > 0:
                ch = piece - 1         # canaux 0-5 : blancs
            else:
                ch = (-piece - 1) + 6  # canaux 6-11 : noirs
            obs[row, col, ch] = 1.0

        # Canal 12 : trait
        if self.turn == CWHITE:
            obs[:, :, 12] = 1.0

        # Canal 13 : en passant
        if self.en_passant_sq >= 0:
            obs[self.en_passant_sq >> 3, self.en_passant_sq & 7, 13] = 1.0

        # Canaux 14-16 : droits de roque
        if self.castle_wk: obs[:, :, 14] = 1.0
        if self.castle_wq: obs[:, :, 15] = 1.0
        if self.castle_bk or self.castle_bq: obs[:, :, 16] = 1.0

        return obs
```

---

## Phase 4 — Copie du plateau (pour AlphaZero MCTS) (½ journée)

```cython
    def copy(self):
        cdef ChessBoard b = ChessBoard.__new__(ChessBoard)
        # Copie mémoire directe — bien plus rapide que dict/list Python
        memcpy(b._board, self._board, 64 * sizeof(int8_t))
        b.turn             = self.turn
        b.en_passant_sq    = self.en_passant_sq
        b.halfmove_clock   = self.halfmove_clock
        b.fullmove_number  = self.fullmove_number
        b.castle_wk        = self.castle_wk
        b.castle_wq        = self.castle_wq
        b.castle_bk        = self.castle_bk
        b.castle_bq        = self.castle_bq
        b._position_history = dict(self._position_history)
        b.move_history      = list(self.move_history)
        return b
```

---

## Phase 5 — Tests et validation (1 jour)

### 5.1 Script de validation de parité

```python
# tests/test_cython_parity.py
"""Vérifie que ChessBoard Cython et Python donnent les mêmes résultats."""
import pytest
from chess_env.board    import ChessBoard as ChessBoardPy
from chess_env.board_cy import ChessBoard as ChessBoardCy

def compare_boards(board_py, board_cy):
    assert [board_py.get_piece(sq) for sq in range(64)] == \
           [board_cy.get_piece(sq) for sq in range(64)]
    assert board_py.turn == board_cy.turn
    assert {m.to_uci() for m in board_py.get_legal_moves()} == \
           {m.to_uci() for m in board_cy.get_legal_moves()}

@pytest.fixture(params=range(50))
def random_game(request):
    import random; random.seed(request.param)
    py = ChessBoardPy(); cy = ChessBoardCy()
    for _ in range(40):
        moves_py = py.get_legal_moves()
        if not moves_py: break
        uci = random.choice(moves_py).to_uci()
        # Appliquer le même coup aux deux
        for board, cls in [(py, ChessBoardPy), (cy, ChessBoardCy)]:
            legal = board.get_legal_moves()
            m = next(m for m in legal if m.to_uci() == uci)
            board._apply_move_unchecked(m)
    return py, cy

def test_parity(random_game):
    compare_boards(*random_game)
```

### 5.2 Benchmark comparatif

```python
# tools/benchmark_engine.py
import time
from chess_env.board    import ChessBoard as Py
from chess_env.board_cy import ChessBoard as Cy

def bench(cls, n=2000):
    board = cls()
    import random
    start = time.perf_counter()
    for _ in range(n):
        moves = board.get_legal_moves()
        if not moves: board = cls(); continue
        board._apply_move_unchecked(random.choice(moves))
    return time.perf_counter() - start

t_py = bench(Py)
t_cy = bench(Cy)
print(f"Python  : {t_py*1000:.1f} ms / 2000 positions")
print(f"Cython  : {t_cy*1000:.1f} ms / 2000 positions")
print(f"Speedup : {t_py/t_cy:.1f}x")
```

---

## Phase 6 (optionnel) — Bitboards pour aller plus loin

Cette phase remplace le tableau `int8_t[64]` par **12 entiers 64 bits** — c'est l'architecture de tous les moteurs modernes (Stockfish, Leela Zero, etc.).

### Principe

Un bitboard est un `uint64_t` où chaque bit représente une case du plateau :

```
Bit 0  = a1    Bit 7  = h1
Bit 8  = a2    Bit 63 = h8
```

Un `1` signifie "il y a une pièce de ce type sur cette case".

```cython
cdef class ChessBoardBB:
    # 6 types × 2 couleurs = 12 bitboards
    cdef uint64_t bb_pawns_w, bb_knights_w, bb_bishops_w
    cdef uint64_t bb_rooks_w, bb_queens_w,  bb_kings_w
    cdef uint64_t bb_pawns_b, bb_knights_b, bb_bishops_b
    cdef uint64_t bb_rooks_b, bb_queens_b,  bb_kings_b

    # Unions précalculées
    cdef uint64_t occupied_w    # toutes les pièces blanches
    cdef uint64_t occupied_b    # toutes les pièces noires
    cdef uint64_t occupied      # toutes les pièces
```

### Tables précalculées (initialisées une seule fois)

```cython
# Attaques de cavalier pour chaque case — 1 opération bit à bit à l'exécution
cdef uint64_t[64] KNIGHT_ATTACKS
cdef uint64_t[64] KING_ATTACKS
cdef uint64_t[2][64] PAWN_ATTACKS   # [color][sq]

# Magic bitboards pour les sliders (technique avancée)
# Permet de calculer les attaques de fou/tour en O(1) avec une table
cdef uint64_t[64] BISHOP_MAGIC
cdef uint64_t[64] ROOK_MAGIC
cdef uint64_t[64][512]  BISHOP_ATTACKS   # indexé par magic
cdef uint64_t[64][4096] ROOK_ATTACKS
```

### Génération de coups en O(1) pour les pièces simples

```cython
    cdef uint64_t knight_attacks_bb(self, int sq) noexcept:
        # Une seule lecture de table — ~1 cycle CPU
        return KNIGHT_ATTACKS[sq] & ~self.occupied_w   # filtre pièces amies

    cdef void gen_knight_moves(self, list moves) noexcept:
        cdef uint64_t knights = self.bb_knights_w
        cdef uint64_t attacks
        cdef int from_sq, to_sq
        while knights:
            from_sq = _bit_scan(knights)    # BSF (Bit Scan Forward)
            attacks = KNIGHT_ATTACKS[from_sq] & ~self.occupied_w
            while attacks:
                to_sq = _bit_scan(attacks)
                moves.append(Move(from_sq, to_sq))
                attacks &= attacks - 1      # clear lowest bit
            knights &= knights - 1
```

### Gains attendus avec les bitboards

| Opération | Python pur | Cython typé | Bitboards Cython |
|---|---|---|---|
| `get_legal_moves()` | 0.31 ms | ~0.03 ms | ~0.002 ms |
| `is_square_attacked()` | ~0.05 ms | ~0.005 ms | ~0.0005 ms |
| `board.copy()` | ~0.02 ms | ~0.001 ms | ~0.001 ms |
| Steps entraînement/s | ~500 | ~5 000 | ~50 000+ |

---

## Récapitulatif des phases

| Phase | Durée estimée | Gain entraînement | Priorité |
|---|---|---|---|
| **0** — Infrastructure | ½ jour | — | ✅ Obligatoire |
| **1** — Board + push/pop Cython | 3–4 jours | 10–20× move gen | ✅ Prioritaire |
| **2** — Pseudo-légaux typés | 2–3 jours | +50% supplémentaire | ✅ Prioritaire |
| **3** — Observation Gymnasium | ½ jour | marginal | ✅ Facile |
| **4** — board.copy() C | ½ jour | utile pour AlphaZero | ✅ Facile |
| **5** — Tests parité + bench | 1 jour | — | ✅ Obligatoire |
| **6** — Bitboards | 4–6 semaines | ×10 supplémentaire | ⬜ Long terme |

**Total phases 0–5 : 8–10 jours de travail, gain attendu ~10–30× sur `get_legal_moves`, ~5–10× sur la vitesse globale d'entraînement.**

---

## Points d'attention

### Compilation sur Windows
Visual Studio Build Tools est requis. Sans eux, `python setup.py build_ext` échoue avec `error: Microsoft Visual C++ 14.0 or greater is required`.

```
winget install Microsoft.VisualStudio.2022.BuildTools
```
Puis cocher "Desktop development with C++" dans le Visual Studio Installer.

### GIL et threads
Les fonctions `cdef ... noexcept nogil` peuvent libérer le GIL — utile pour les futures parallélisations de l'MCTS d'AlphaZero avec des threads Python.

### Débogage
En cas d'erreur de segfault (accès mémoire hors limites) :
```python
# Recompiler temporairement avec les vérifications activées
compiler_directives={"boundscheck": True, "wraparound": True}
```

### Distribution
Pour partager le projet sans imposer Cython à d'autres :
```python
# Le fallback dans chess_env.py garantit que la version Python fonctionne
# si le module Cython n'est pas compilé
try:
    from chess_env.board_cy import ChessBoard
except ImportError:
    from chess_env.board import ChessBoard
```

---

## Ressources

- [Documentation Cython](https://cython.readthedocs.io/en/latest/)
- [Cython Best Practices](https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html)
- [Magic Bitboards — Chess Programming Wiki](https://www.chessprogramming.org/Magic_Bitboards)
- [CPython Extension Modules](https://docs.python.org/3/extending/extending.html)
