"""
Serveur Flask pour l'interface web du jeu d'échecs.
Supporte plusieurs agents IA (Random + agents tabulaires entraînés).
"""

import random
import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, jsonify, request
from chess_env.board import ChessBoard, Move, WHITE, BLACK, QUEEN

app = Flask(__name__)

# ------------------------------------------------------------------
# État global
# ------------------------------------------------------------------

board          = ChessBoard()
player_color   = WHITE
current_agent  = None   # None = random | instance de BaseAgent
agent_name     = "random"
training_status = {"running": False, "progress": 0, "total": 0, "episodes_done": 0}

# Registre des agents disponibles (instanciés à la demande)
_agent_registry: dict = {}


def _get_or_create_agent(name: str):
    """Retourne un agent existant ou en crée un nouveau."""
    if name in _agent_registry:
        return _agent_registry[name]

    from agents.tabular import SarsaAgent, QLearningAgent, ExpectedSarsaAgent, MonteCarloAgent
    ACTION_SIZE = 4096
    OBS_SHAPE   = (8, 8, 17)
    agents_map = {
        "sarsa":          SarsaAgent(ACTION_SIZE, OBS_SHAPE),
        "q_learning":     QLearningAgent(ACTION_SIZE, OBS_SHAPE),
        "expected_sarsa": ExpectedSarsaAgent(ACTION_SIZE, OBS_SHAPE),
        "monte_carlo":    MonteCarloAgent(ACTION_SIZE, OBS_SHAPE),
    }
    if name in agents_map:
        _agent_registry[name] = agents_map[name]
        return _agent_registry[name]
    return None


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

    agent_info = None
    if current_agent is not None:
        agent_info = {
            "name":          agent_name,
            "epsilon":       round(getattr(current_agent, "epsilon", 1.0), 3),
            "q_table_size":  getattr(current_agent, "q_table_size", 0),
            "training_steps": current_agent.training_steps,
        }

    return {
        "pieces":        pieces,
        "turn":          "white" if board.turn == WHITE else "black",
        "player_color":  "white" if player_color == WHITE else "black",
        "in_check":      board.is_in_check(board.turn),
        "legal_moves":   [m.to_uci() for m in board.get_legal_moves()],
        "result":        result_str,
        "halfmove_clock": board.halfmove_clock,
        "fullmove":      board.fullmove_number,
        "last_move":     board.move_history[-1].to_uci() if board.move_history else None,
        "agent":         agent_info,
        "training":      dict(training_status),
    }


def _uci_to_move(uci: str):
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
    """Fait jouer l'IA (agent entraîné ou random)."""
    import numpy as np
    legal = board.get_legal_moves()
    if not legal:
        return

    if current_agent is not None:
        obs = board.get_observation()
        legal_actions = [m.from_sq * 64 + m.to_sq for m in legal]
        action = current_agent.select_action(obs, legal_actions)
        # Retrouver le Move correspondant
        legal_map = {m.from_sq * 64 + m.to_sq: m for m in legal}
        move = legal_map.get(action, random.choice(legal))
        board._apply_move_unchecked(move)
    else:
        board._apply_move_unchecked(random.choice(legal))


# ------------------------------------------------------------------
# Routes — jeu
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
    data = request.get_json(force=True) or {}
    uci = data.get("uci", "")

    move = _uci_to_move(uci)
    if move is None:
        return jsonify({"error": f"Format UCI invalide : '{uci}'"}), 400

    legal = board.get_legal_moves()
    if move not in legal:
        move_q = Move(move.from_sq, move.to_sq, QUEEN)
        if move_q in legal:
            move = move_q
        else:
            legal_uci = [m.to_uci() for m in legal]
            return jsonify({"error": f"Coup illégal : {uci}", "legal": legal_uci}), 400

    board._apply_move_unchecked(move)

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

    if player_color == BLACK:
        _ai_move()

    return jsonify(_board_state())


# ------------------------------------------------------------------
# Routes — agents
# ------------------------------------------------------------------

@app.route("/api/agents")
def list_agents():
    agents = [
        {"id": "random",         "label": "Aléatoire",      "tier": 0},
        {"id": "sarsa",          "label": "SARSA",           "tier": 1},
        {"id": "q_learning",     "label": "Q-Learning",      "tier": 1},
        {"id": "expected_sarsa", "label": "Expected SARSA",  "tier": 1},
        {"id": "monte_carlo",    "label": "Monte Carlo",     "tier": 1},
    ]
    current = agent_name
    infos = {}
    for a in agents:
        aid = a["id"]
        if aid in _agent_registry:
            ag = _agent_registry[aid]
            infos[aid] = {
                "epsilon":        round(getattr(ag, "epsilon", 1.0), 3),
                "training_steps": ag.training_steps,
                "q_table_size":   getattr(ag, "q_table_size", 0),
            }
    return jsonify({"agents": agents, "current": current, "infos": infos})


@app.route("/api/select_agent", methods=["POST"])
def select_agent():
    global current_agent, agent_name
    data = request.get_json(force=True) or {}
    name = data.get("agent", "random")

    if name == "random":
        current_agent = None
        agent_name = "random"
    else:
        agent = _get_or_create_agent(name)
        if agent is None:
            return jsonify({"error": f"Agent inconnu : {name}"}), 400
        current_agent = agent
        agent_name = name

    return jsonify({"selected": agent_name})


@app.route("/api/train", methods=["POST"])
def train_agent():
    """Lance l'entraînement de l'agent courant dans un thread séparé."""
    global training_status

    if current_agent is None:
        return jsonify({"error": "Sélectionne d'abord un agent (pas random)"}), 400
    if training_status["running"]:
        return jsonify({"error": "Entraînement déjà en cours"}), 400

    data = request.get_json(force=True) or {}
    n_episodes = int(data.get("episodes", 100))

    def _train():
        from chess_env.chess_env import ChessEnv
        training_status["running"]      = True
        training_status["total"]        = n_episodes
        training_status["progress"]     = 0
        training_status["episodes_done"] = 0

        env = ChessEnv(render_mode=None)
        agent = current_agent

        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                legal = info.get("legal_actions", [])
                if not legal:
                    break
                action = agent.select_action(obs, legal)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                next_legal = next_info.get("legal_actions", []) if not done else []
                agent.update(obs, action, reward, next_obs, done, next_legal)
                obs, info = next_obs, next_info
                agent.training_steps += 1

            agent.on_episode_end(ep, 0.0, 0)
            training_status["progress"]      = int((ep + 1) / n_episodes * 100)
            training_status["episodes_done"] = ep + 1

        training_status["running"] = False

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"started": True, "episodes": n_episodes, "agent": agent_name})


@app.route("/api/training_status")
def get_training_status():
    info = dict(training_status)
    if current_agent is not None:
        info["epsilon"]        = round(getattr(current_agent, "epsilon", 1.0), 3)
        info["training_steps"] = current_agent.training_steps
        info["q_table_size"]   = getattr(current_agent, "q_table_size", 0)
    return jsonify(info)


# ------------------------------------------------------------------
# Lancement
# ------------------------------------------------------------------

def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)
