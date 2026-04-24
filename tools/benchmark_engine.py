#!/usr/bin/env python3
"""
tools/benchmark_engine.py
Benchmark comparatif Python vs Cython sur 1000 positions d'échecs.

Utilisation :
    python tools/benchmark_engine.py
    python tools/benchmark_engine.py --positions 2000
    python tools/benchmark_engine.py --warmup 100
"""
import argparse
import random
import time
import sys
import os

# S'assure que la racine du projet est dans sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chess_env.board import ChessBoard as PyBoard

try:
    from chess_env.board_cy import ChessBoard as CyBoard  # type: ignore[import]
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False


# ---------------------------------------------------------------------------
# Génération du corpus de positions
# ---------------------------------------------------------------------------

def generate_positions(cls, n_positions: int, seed: int = 42):
    """
    Joue des parties aléatoires et collecte n_positions plateaux.
    Retourne une liste de copies de ChessBoard.
    """
    rng = random.Random(seed)
    boards = []
    board = cls()

    while len(boards) < n_positions:
        legal = board.get_legal_moves()
        if not legal:
            board = cls()
            continue
        board._apply_move_unchecked(rng.choice(legal))
        boards.append(board.copy())

    return boards


# ---------------------------------------------------------------------------
# Fonction de benchmark
# ---------------------------------------------------------------------------

def bench_legal_moves(boards: list, label: str) -> dict:
    """
    Mesure le temps de get_legal_moves() sur une liste de plateaux pré-calculés.
    Retourne un dict avec les statistiques.
    """
    n = len(boards)
    times = []

    for board in boards:
        t0 = time.perf_counter()
        board.get_legal_moves()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    total_ms   = sum(times) * 1000.0
    mean_us    = (sum(times) / n) * 1_000_000.0
    min_us     = min(times) * 1_000_000.0
    max_us     = max(times) * 1_000_000.0

    # Médiane
    sorted_times = sorted(times)
    median_us = sorted_times[n // 2] * 1_000_000.0

    return {
        "label":     label,
        "n":         n,
        "total_ms":  total_ms,
        "mean_us":   mean_us,
        "median_us": median_us,
        "min_us":    min_us,
        "max_us":    max_us,
    }


def print_stats(stats: dict):
    label = stats["label"]
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  Positions testées : {stats['n']}")
    print(f"  Temps total       : {stats['total_ms']:.1f} ms")
    print(f"  Moyenne           : {stats['mean_us']:.1f} µs / appel")
    print(f"  Médiane           : {stats['median_us']:.1f} µs / appel")
    print(f"  Min               : {stats['min_us']:.1f} µs")
    print(f"  Max               : {stats['max_us']:.1f} µs")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python vs Cython pour get_legal_moves()"
    )
    parser.add_argument(
        "--positions", type=int, default=1000,
        help="Nombre de positions à tester (défaut : 1000)"
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Positions d'échauffement ignorées (défaut : 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine aléatoire (défaut : 42)"
    )
    args = parser.parse_args()

    n          = args.positions
    n_warmup   = args.warmup
    seed       = args.seed
    n_total    = n + n_warmup

    print(f"\nChessIA — Benchmark moteur d'échecs")
    print(f"Positions : {n} (+ {n_warmup} échauffement)")
    print(f"Graine    : {seed}")

    # --- Python ---
    print("\nGénération du corpus Python…", end="", flush=True)
    py_positions = generate_positions(PyBoard, n_total, seed=seed)
    print(f" {len(py_positions)} positions OK")

    # Échauffement (ignoré dans les stats)
    for b in py_positions[:n_warmup]:
        b.get_legal_moves()

    stats_py = bench_legal_moves(py_positions[n_warmup:], "Python pur (board.py)")
    print_stats(stats_py)

    # --- Cython ---
    if not HAS_CYTHON:
        print("\n" + "=" * 50)
        print("  Cython non disponible.")
        print("  Pour compiler : python setup.py build_ext --inplace")
        print("=" * 50)
        return

    print("\nGénération du corpus Cython…", end="", flush=True)
    cy_positions = generate_positions(CyBoard, n_total, seed=seed)
    print(f" {len(cy_positions)} positions OK")

    # Échauffement
    for b in cy_positions[:n_warmup]:
        b.get_legal_moves()

    stats_cy = bench_legal_moves(cy_positions[n_warmup:], "Cython typé (board_cy)")
    print_stats(stats_cy)

    # --- Comparaison ---
    speedup = stats_py["mean_us"] / stats_cy["mean_us"]
    speedup_med = stats_py["median_us"] / stats_cy["median_us"]

    print(f"\n{'=' * 50}")
    print(f"  Speedup moyen    : {speedup:.1f}×")
    print(f"  Speedup médian   : {speedup_med:.1f}×")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
