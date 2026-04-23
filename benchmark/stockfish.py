"""
Benchmark local d'un agent contre Stockfish via le protocole UCI.

Usage typique:
    python -m benchmark.stockfish --agent alphazero --games 20 --stockfish-path C:\\path\\to\\stockfish.exe
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from agents.base_agent import BaseAgent
from chess_env.board import BISHOP, BLACK, KNIGHT, QUEEN, ROOK, WHITE, ChessBoard, Move

ACTION_SIZE = 4096
OBS_SHAPE = (8, 8, 17)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


class BenchmarkError(RuntimeError):
    """Raised when the local benchmark cannot be executed."""


@dataclass
class BenchmarkResult:
    games: int
    games_as_white: int
    games_as_black: int
    wins: int
    losses: int
    draws: int
    points: float
    win_rate: float
    loss_rate: float
    draw_rate: float
    score_rate: float
    adjusted_score: float
    win_loss_ratio: float
    wl_record: str
    estimated_elo: float
    stockfish_elo: int

    def to_dict(self) -> dict:
        ratio = self.win_loss_ratio
        if math.isinf(ratio):
            ratio = None
        return {
            "games": self.games,
            "games_as_white": self.games_as_white,
            "games_as_black": self.games_as_black,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "points": self.points,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
            "score_rate": self.score_rate,
            "adjusted_score": self.adjusted_score,
            "win_loss_ratio": ratio,
            "wl_record": self.wl_record,
            "estimated_elo": self.estimated_elo,
            "stockfish_elo": self.stockfish_elo,
        }


def _agent_classes() -> Dict[str, type]:
    from agents.deep_rl import AlphaZeroAgent, DQNAgent, PPOAgent
    from agents.policy_gradient import REINFORCEAgent
    from agents.tabular import ExpectedSarsaAgent, MonteCarloAgent, QLearningAgent, SarsaAgent

    return {
        "alphazero": AlphaZeroAgent,
        "dqn": DQNAgent,
        "expected_sarsa": ExpectedSarsaAgent,
        "monte_carlo": MonteCarloAgent,
        "ppo": PPOAgent,
        "q_learning": QLearningAgent,
        "reinforce": REINFORCEAgent,
        "sarsa": SarsaAgent,
    }


def load_agent(
    agent_name: str,
    checkpoint_path: Optional[str | os.PathLike[str]] = None,
    *,
    strict_checkpoint: bool = True,
) -> BaseAgent:
    """Instantiate an agent and load its checkpoint from disk."""
    classes = _agent_classes()
    agent_cls = classes.get(agent_name)
    if agent_cls is None:
        supported = ", ".join(sorted(classes))
        raise ValueError(f"Agent inconnu '{agent_name}'. Agents supportés: {supported}.")

    path = Path(checkpoint_path) if checkpoint_path is not None else MODELS_DIR / f"{agent_name}.pkl"
    config = None

    if path.exists():
        with path.open("rb") as handle:
            state = pickle.load(handle)
        saved_config = state.get("config")
        if isinstance(saved_config, dict):
            config = saved_config
    elif strict_checkpoint:
        raise FileNotFoundError(
            f"Checkpoint introuvable pour '{agent_name}': {path}. "
            "Enregistre d'abord le modèle ou fournis --checkpoint."
        )

    agent = agent_cls(ACTION_SIZE, OBS_SHAPE, config=config)
    if path.exists():
        agent.load(str(path))
    return agent


def discover_stockfish_path(
    candidates: Optional[Sequence[str | os.PathLike[str]]] = None,
) -> Optional[str]:
    """Try to locate a local Stockfish binary without network access."""
    search_paths: List[Path] = []

    if candidates:
        search_paths.extend(Path(candidate) for candidate in candidates)

    for executable_name in ("stockfish", "stockfish.exe"):
        resolved = shutil.which(executable_name)
        if resolved:
            search_paths.append(Path(resolved))

    search_paths.extend(
        [
            REPO_ROOT / "stockfish.exe",
            REPO_ROOT / "stockfish",
            REPO_ROOT / "benchmark" / "stockfish.exe",
            REPO_ROOT / "benchmark" / "stockfish",
            REPO_ROOT / "engines" / "stockfish.exe",
            REPO_ROOT / "engines" / "stockfish",
        ]
    )

    seen: set[Path] = set()
    for path in search_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return str(path)
    return None


def estimate_elo_from_score(score: float, opponent_elo: float) -> float:
    """
    Convert an expected score in (0, 1) into an Elo estimate.

    Formula:
        score = 1 / (1 + 10 ** ((R_opp - R_agent) / 400))
    """
    if not 0.0 < score < 1.0:
        raise ValueError("Le score doit être strictement compris entre 0 et 1.")
    return float(opponent_elo - 400.0 * math.log10((1.0 / score) - 1.0))


def estimate_elo_from_match(points: float, games: int, opponent_elo: float) -> tuple[float, float]:
    """
    Estimate Elo from a finite match score with a small continuity correction.

    Returns:
        (estimated_elo, adjusted_score)
    """
    if games <= 0:
        raise ValueError("Le nombre de parties doit être > 0.")
    adjusted_score = (float(points) + 0.5) / (float(games) + 1.0)
    return estimate_elo_from_score(adjusted_score, opponent_elo), adjusted_score


class UCIEngine:
    """Minimal line-based UCI wrapper for local engines such as Stockfish."""

    def __init__(
        self,
        command: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        cwd: Optional[str | os.PathLike[str]] = None,
    ):
        if isinstance(command, (str, os.PathLike)):
            self.command = [str(command)]
        else:
            self.command = [str(part) for part in command]
        self.cwd = str(cwd) if cwd is not None else None
        self.process: Optional[subprocess.Popen[str]] = None

    def __enter__(self) -> "UCIEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def start(self) -> None:
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.cwd,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise BenchmarkError(f"Impossible de lancer le moteur UCI: {' '.join(self.command)}") from exc

        self._send("uci")
        self._read_until(lambda line: line == "uciok", "uciok")
        self._send("isready")
        self._read_until(lambda line: line == "readyok", "readyok")

    def close(self) -> None:
        if self.process is None:
            return

        try:
            if self.process.poll() is None:
                self._send("quit")
                self.process.communicate(timeout=2)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
        finally:
            self.process = None

    def _send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise BenchmarkError("Le moteur UCI n'est pas démarré.")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_line(self) -> str:
        if self.process is None or self.process.stdout is None:
            raise BenchmarkError("Le moteur UCI n'est pas démarré.")
        line = self.process.stdout.readline()
        if line == "":
            return_code = self.process.poll()
            raise BenchmarkError(f"Le moteur UCI s'est arrêté de façon inattendue (code {return_code}).")
        return line.strip()

    def _read_until(self, predicate, expected_label: str) -> str:
        transcript: List[str] = []
        while True:
            line = self._read_line()
            transcript.append(line)
            if predicate(line):
                return line
            if len(transcript) > 2000:
                joined = "\n".join(transcript[-10:])
                raise BenchmarkError(
                    f"Réponse UCI inattendue, '{expected_label}' non reçu. Dernières lignes:\n{joined}"
                )

    def set_option(self, name: str, value: str | int | float | bool) -> None:
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        self._send(f"setoption name {name} value {rendered}")

    def sync(self) -> None:
        self._send("isready")
        self._read_until(lambda line: line == "readyok", "readyok")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self.sync()

    def bestmove(
        self,
        moves: Sequence[str],
        *,
        movetime_ms: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> str:
        if moves:
            self._send(f"position startpos moves {' '.join(moves)}")
        else:
            self._send("position startpos")

        if depth is not None:
            self._send(f"go depth {int(depth)}")
        else:
            effective_movetime = 50 if movetime_ms is None else int(movetime_ms)
            self._send(f"go movetime {effective_movetime}")

        line = self._read_until(lambda current: current.startswith("bestmove "), "bestmove")
        parts = line.split()
        if len(parts) < 2:
            raise BenchmarkError(f"Réponse bestmove invalide: {line}")
        return parts[1]


def _uci_to_move(board: ChessBoard, uci: str) -> Optional[Move]:
    uci = uci.strip().lower()
    if len(uci) not in (4, 5):
        return None

    files = "abcdefgh"
    try:
        from_col = files.index(uci[0])
        from_row = int(uci[1]) - 1
        to_col = files.index(uci[2])
        to_row = int(uci[3]) - 1
    except (ValueError, IndexError):
        return None

    promotion = 0
    if len(uci) == 5:
        promo_map = {"n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN}
        promotion = promo_map.get(uci[4], 0)
        if promotion == 0:
            return None

    move = Move(ChessBoard.sq(from_row, from_col), ChessBoard.sq(to_row, to_col), promotion)
    legal_moves = board.get_legal_moves()
    if move in legal_moves:
        return move

    queen_promotion = Move(move.from_sq, move.to_sq, QUEEN)
    if queen_promotion in legal_moves:
        return queen_promotion
    return None


def _agent_observation(board: ChessBoard, agent_color: int) -> np.ndarray:
    obs = board.get_observation()
    if agent_color == BLACK:
        return obs[::-1, ::-1, :].copy()
    return obs


def _select_agent_move(
    agent: BaseAgent,
    board: ChessBoard,
    *,
    agent_color: int,
    temperature: float = 0.0,
) -> Move:
    legal_moves = board.get_legal_moves()
    if not legal_moves:
        raise BenchmarkError("Aucun coup légal disponible pour l'agent.")

    if hasattr(agent, "select_move"):
        try:
            move = agent.select_move(board, temperature=temperature)
        except TypeError:
            move = agent.select_move(board)
        if move in legal_moves:
            return move
        raise BenchmarkError("select_move() a renvoyé un coup illégal pendant le benchmark.")

    obs = _agent_observation(board, agent_color)
    legal_actions = [move.to_action() for move in legal_moves]

    epsilon_backup = getattr(agent, "epsilon", None)
    list_backups = {
        attr: list(value)
        for attr, value in vars(agent).items()
        if attr.startswith("_") and isinstance(value, list)
    }

    try:
        if epsilon_backup is not None:
            agent.epsilon = 0.0
        action = agent.select_action(obs, legal_actions)
    finally:
        if epsilon_backup is not None:
            agent.epsilon = epsilon_backup
        for attr, value in list_backups.items():
            setattr(agent, attr, value)

    legal_map = {move.to_action(): move for move in legal_moves}
    move = legal_map.get(action)
    if move is None:
        raise BenchmarkError(f"L'agent a renvoyé une action illégale pendant le benchmark: {action}")
    return move


def _play_one_game(
    agent: BaseAgent,
    engine: UCIEngine,
    *,
    agent_color: int,
    movetime_ms: Optional[int],
    depth: Optional[int],
    max_plies: int,
    agent_temperature: float,
) -> int:
    board = ChessBoard()
    plies = 0

    while board.get_result() is None and plies < max_plies:
        if board.turn == agent_color:
            move = _select_agent_move(
                agent,
                board,
                agent_color=agent_color,
                temperature=agent_temperature,
            )
        else:
            bestmove_uci = engine.bestmove(
                [move.to_uci() for move in board.move_history],
                movetime_ms=movetime_ms,
                depth=depth,
            )
            if bestmove_uci in {"0000", "(none)"}:
                break

            move = _uci_to_move(board, bestmove_uci)
            if move is None:
                raise BenchmarkError(f"Coup UCI invalide renvoyé par Stockfish: {bestmove_uci}")

        board._apply_move_unchecked(move)
        plies += 1

    result = board.get_result()
    if result is None:
        return 0
    return result


def benchmark_agent_vs_stockfish(
    agent: BaseAgent,
    stockfish_path: Optional[str | os.PathLike[str]] = None,
    *,
    stockfish_args: Optional[Sequence[str | os.PathLike[str]]] = None,
    n_games: int = 10,
    stockfish_elo: int = 1200,
    movetime_ms: Optional[int] = 50,
    depth: Optional[int] = None,
    max_plies: int = 512,
    agent_temperature: float = 0.0,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run a fully local benchmark against Stockfish and estimate the agent Elo."""
    if n_games <= 0:
        raise ValueError("n_games doit être > 0.")
    if max_plies <= 0:
        raise ValueError("max_plies doit être > 0.")

    resolved_stockfish = (
        str(stockfish_path)
        if stockfish_path is not None
        else discover_stockfish_path()
    )
    if resolved_stockfish is None:
        raise BenchmarkError(
            "Stockfish introuvable localement. Fournis un chemin explicite via --stockfish-path "
            "ou place l'exécutable dans le projet."
        )

    command: List[str] = [resolved_stockfish]
    if stockfish_args:
        command.extend(str(arg) for arg in stockfish_args)

    wins = 0
    losses = 0
    draws = 0
    games_as_white = 0
    games_as_black = 0

    with UCIEngine(command) as engine:
        engine.set_option("UCI_LimitStrength", True)
        engine.set_option("UCI_Elo", int(stockfish_elo))
        engine.set_option("Threads", 1)
        engine.set_option("Hash", 16)
        engine.sync()

        for game_index in range(n_games):
            agent_color = WHITE if game_index % 2 == 0 else BLACK
            if agent_color == WHITE:
                games_as_white += 1
            else:
                games_as_black += 1

            engine.new_game()
            result = _play_one_game(
                agent,
                engine,
                agent_color=agent_color,
                movetime_ms=movetime_ms,
                depth=depth,
                max_plies=max_plies,
                agent_temperature=agent_temperature,
            )

            if result == agent_color:
                wins += 1
                outcome = "win"
            elif result == -agent_color:
                losses += 1
                outcome = "loss"
            else:
                draws += 1
                outcome = "draw"

            if verbose:
                side = "white" if agent_color == WHITE else "black"
                print(f"[{game_index + 1}/{n_games}] agent={side} -> {outcome}")

    points = float(wins) + 0.5 * float(draws)
    score_rate = points / float(n_games)
    estimated_elo, adjusted_score = estimate_elo_from_match(points, n_games, stockfish_elo)
    win_loss_ratio = float("inf") if losses == 0 and wins > 0 else (
        0.0 if losses == 0 else wins / float(losses)
    )

    return BenchmarkResult(
        games=n_games,
        games_as_white=games_as_white,
        games_as_black=games_as_black,
        wins=wins,
        losses=losses,
        draws=draws,
        points=points,
        win_rate=wins / float(n_games),
        loss_rate=losses / float(n_games),
        draw_rate=draws / float(n_games),
        score_rate=score_rate,
        adjusted_score=adjusted_score,
        win_loss_ratio=win_loss_ratio,
        wl_record=f"{wins}/{losses}",
        estimated_elo=estimated_elo,
        stockfish_elo=int(stockfish_elo),
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local d'un agent contre Stockfish")
    parser.add_argument(
        "--agent",
        required=True,
        choices=sorted(_agent_classes()),
        help="Nom de l'agent à charger depuis models/<agent>.pkl",
    )
    parser.add_argument(
        "--checkpoint",
        help="Chemin vers le checkpoint .pkl de l'agent (par défaut: models/<agent>.pkl).",
    )
    parser.add_argument(
        "--stockfish-path",
        help="Chemin local vers l'exécutable Stockfish. Auto-détection si absent.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Nombre de parties à jouer (alternance blanc/noir).",
    )
    parser.add_argument(
        "--stockfish-elo",
        type=int,
        default=1200,
        help="Niveau Elo demandé à Stockfish via UCI_Elo.",
    )
    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=50,
        help="Temps de réflexion de Stockfish par coup en millisecondes.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Profondeur fixe pour Stockfish. Si fournie, remplace --movetime-ms.",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=512,
        help="Coupe de sécurité en demi-coups avant de déclarer la partie nulle.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température de sélection côté agent si select_move(board, temperature=...) est supporté.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Affiche le résultat de chaque partie pendant le benchmark.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    agent = load_agent(args.agent, checkpoint_path=args.checkpoint)
    result = benchmark_agent_vs_stockfish(
        agent,
        stockfish_path=args.stockfish_path,
        n_games=args.games,
        stockfish_elo=args.stockfish_elo,
        movetime_ms=args.movetime_ms,
        depth=args.depth,
        max_plies=args.max_plies,
        agent_temperature=args.temperature,
        verbose=args.verbose,
    )

    ratio_display = "inf" if math.isinf(result.win_loss_ratio) else f"{result.win_loss_ratio:.3f}"
    print(f"Agent: {args.agent}")
    print(f"Stockfish Elo: {result.stockfish_elo}")
    print(f"Parties: {result.games} (white={result.games_as_white}, black={result.games_as_black})")
    print(f"W/L/D: {result.wins}/{result.losses}/{result.draws}")
    print(f"Ratio W/L: {ratio_display}")
    print(f"Score: {result.points:.1f}/{result.games} ({result.score_rate:.1%})")
    print(f"Elo estimé: {result.estimated_elo:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
