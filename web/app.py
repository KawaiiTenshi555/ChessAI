"""
Serveur Flask pour l'interface web du jeu d'échecs.
Supporte plusieurs agents IA (Random + agents tabulaires entraînés).
"""

import random
import sys
import os
import pickle
import threading
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, jsonify, request
from chess_env.board import ChessBoard, Move, WHITE, BLACK, QUEEN

app = Flask(__name__)

ACTION_SIZE = 4096
OBS_SHAPE = (8, 8, 17)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
STRUCTURAL_CONFIG_KEYS = {
    "dqn": {"buffer_size", "hidden_sizes"},
    "reinforce": {"hidden_sizes"},
    "ppo": {"hidden_sizes"},
}

# ------------------------------------------------------------------
# État global
# ------------------------------------------------------------------

board          = ChessBoard()
player_color   = WHITE
current_agent  = None   # None = random | instance de BaseAgent
agent_name     = "random"
training_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "episodes_done": 0,
    "agent": None,
    "error": None,
    "checkpoint_saved": False,
    "checkpoint_path": None,
}

# Registre des agents disponibles (instanciés à la demande)
_agent_registry: dict = {}
_training_state_lock = threading.Lock()


def _agent_classes():
    from agents.tabular import SarsaAgent, QLearningAgent, ExpectedSarsaAgent, MonteCarloAgent
    from agents.deep_rl import DQNAgent, PPOAgent
    from agents.policy_gradient import REINFORCEAgent

    return {
        "sarsa": SarsaAgent,
        "q_learning": QLearningAgent,
        "expected_sarsa": ExpectedSarsaAgent,
        "monte_carlo": MonteCarloAgent,
        "dqn": DQNAgent,
        "reinforce": REINFORCEAgent,
        "ppo": PPOAgent,
    }


def _agent_default_config(name: str) -> dict:
    agent_cls = _agent_classes().get(name)
    defaults = getattr(agent_cls, "DEFAULT_CONFIG", {}) if agent_cls else {}
    normalized = {}
    for key, value in defaults.items():
        normalized[key] = list(value) if isinstance(value, list) else value
    return normalized


def _agent_state_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.pkl"


def _peek_saved_config(name: str):
    path = _agent_state_path(name)
    if not path.exists():
        return None

    try:
        with path.open("rb") as f:
            state = pickle.load(f)
    except Exception:
        return None

    config = state.get("config")
    return dict(config) if isinstance(config, dict) else None


def _create_agent(name: str, config: dict | None = None):
    agent_cls = _agent_classes().get(name)
    if agent_cls is None:
        return None
    return agent_cls(ACTION_SIZE, OBS_SHAPE, config=config)


def _save_agent_checkpoint(name: str, agent) -> Path:
    path = _agent_state_path(name)
    agent.save(str(path))
    return path


def _get_or_create_agent(name: str):
    """Retourne un agent existant ou en crée un nouveau."""
    if name in _agent_registry:
        _normalize_agent_config(name, _agent_registry[name])
        return _agent_registry[name]

    agent = _create_agent(name, config=_peek_saved_config(name))
    if agent is None:
        return None

    checkpoint_path = _agent_state_path(name)
    if checkpoint_path.exists():
        try:
            agent.load(str(checkpoint_path))
        except Exception:
            agent = _create_agent(name)

    _normalize_agent_config(name, agent)
    _agent_registry[name] = agent
    return agent


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _checkpoint_info(name: str) -> dict:
    path = _agent_state_path(name)
    return {
        "exists": path.exists(),
        "path": str(path),
    }


def _mean_reward(agent) -> float:
    rewards = getattr(agent, "episode_rewards", [])
    if not rewards:
        return 0.0
    return float(sum(rewards) / len(rewards))


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Valeur booléenne invalide: {value}")
    return bool(value)


def _coerce_config_value(template, value):
    if isinstance(template, bool):
        return _coerce_bool(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, list):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, list):
            raise ValueError("Une liste est attendue")
        return [int(v) for v in value]
    return value


def _apply_hyperparam_updates(agent, updates: dict) -> None:
    for key, value in updates.items():
        setattr(agent, key, value)
        agent.config[key] = value

    if "lr" in updates and hasattr(agent, "optimizer"):
        for group in agent.optimizer.param_groups:
            group["lr"] = float(updates["lr"])


def _normalize_agent_config(name: str, agent) -> None:
    defaults = _agent_default_config(name)
    if not defaults:
        return

    normalized = {}
    for key, template in defaults.items():
        if not hasattr(agent, key):
            continue

        try:
            normalized[key] = _coerce_config_value(template, getattr(agent, key))
        except (ValueError, TypeError):
            normalized[key] = list(template) if isinstance(template, list) else template

    _apply_hyperparam_updates(agent, normalized)


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
        checkpoint = _checkpoint_info(agent_name)
        agent_info = {
            "name":          agent_name,
            "epsilon":       round(getattr(current_agent, "epsilon", 1.0), 3),
            "q_table_size":  getattr(current_agent, "q_table_size", 0),
            "training_steps": current_agent.training_steps,
            "mean_reward":   _mean_reward(current_agent),
            "checkpoint_exists": checkpoint["exists"],
            "checkpoint_path": checkpoint["path"],
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

        # Sauvegarder tous les buffers liste privés de l'agent (policy-gradient,
        # PPO rollout, etc.) pour ne pas polluer l'entraînement pendant une partie.
        _buf_backup = {
            attr: list(val)
            for attr, val in vars(current_agent).items()
            if isinstance(val, list) and attr.startswith("_")
        }

        action = current_agent.select_action(obs, legal_actions)

        # Restaurer les buffers (on annule l'effet de select_action)
        for attr, val in _buf_backup.items():
            setattr(current_agent, attr, val)

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
        {"id": "dqn",            "label": "DQN",             "tier": 4},
        {"id": "reinforce",      "label": "REINFORCE",       "tier": 3},
        {"id": "ppo",            "label": "PPO",             "tier": 4},
    ]
    current = agent_name
    infos = {}
    for a in agents:
        aid = a["id"]
        if aid in _agent_registry:
            ag = _agent_registry[aid]
            checkpoint = _checkpoint_info(aid)
            infos[aid] = {
                "epsilon":        round(getattr(ag, "epsilon", 1.0), 3),
                "training_steps": ag.training_steps,
                "q_table_size":   getattr(ag, "q_table_size", 0),
                "mean_reward":    _mean_reward(ag),
                "checkpoint_exists": checkpoint["exists"],
                "checkpoint_path": checkpoint["path"],
            }
    return jsonify({"agents": agents, "current": current, "infos": infos})


@app.route("/api/select_agent", methods=["POST"])
def select_agent():
    global current_agent, agent_name
    if training_status["running"]:
        return jsonify({"error": "Impossible de changer d'agent pendant l'entraînement"}), 400

    data = request.get_json(force=True) or {}
    name = data.get("agent", "random")

    if name == "random":
        current_agent = None
        agent_name = "random"
        checkpoint = None
    else:
        agent = _get_or_create_agent(name)
        if agent is None:
            return jsonify({"error": f"Agent inconnu : {name}"}), 400
        current_agent = agent
        agent_name = name
        checkpoint = _checkpoint_info(name)

    return jsonify({
        "selected": agent_name,
        "checkpoint_exists": checkpoint["exists"] if checkpoint else False,
        "checkpoint_path": checkpoint["path"] if checkpoint else None,
    })


@app.route("/api/train", methods=["POST"])
def train_agent():
    """Lance l'entraînement de l'agent courant dans un thread séparé."""
    global training_status

    if current_agent is None:
        return jsonify({"error": "Sélectionne d'abord un agent (pas random)"}), 400

    data = request.get_json(force=True) or {}
    n_episodes           = int(data.get("episodes", 100))
    reward_shaping       = bool(data.get("reward_shaping", False))
    capture_reward_scale = float(data.get("capture_reward_scale", 0.1))
    loss_penalty_scale   = float(data.get("loss_penalty_scale", 0.1))
    terminal_win_reward  = float(data.get("terminal_win_reward", 1.0))
    terminal_loss_penalty = float(data.get("terminal_loss_penalty", 1.0))

    viz_delay = max(0.0, float(data.get("viz_delay", 0))) / 1000.0  # ms → s
    selected_agent_name = agent_name
    agent = current_agent
    _normalize_agent_config(selected_agent_name, agent)

    with _training_state_lock:
        if training_status["running"]:
            return jsonify({"error": "Entraînement déjà en cours"}), 400

        training_status["running"] = True
        training_status["total"] = n_episodes
        training_status["progress"] = 0
        training_status["episodes_done"] = 0
        training_status["board"] = {}
        training_status["train_last_move"] = None
        training_status["agent"] = selected_agent_name
        training_status["error"] = None
        training_status["checkpoint_saved"] = False
        training_status["checkpoint_path"] = None

    def _train():
        import time
        from chess_env.chess_env import ChessEnv

        try:
            env = ChessEnv(
                render_mode=None,
                reward_shaping=reward_shaping,
                capture_reward_scale=capture_reward_scale,
                loss_penalty_scale=loss_penalty_scale,
                terminal_win_reward=terminal_win_reward,
                terminal_loss_penalty=terminal_loss_penalty,
            )

            for ep in range(n_episodes):
                obs, info = env.reset()
                done = False
                ep_reward = 0.0
                ep_length = 0

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
                    ep_reward += reward
                    ep_length += 1
                    agent.training_steps += 1

                    # Snapshot pour la visualisation live
                    bd = env.board
                    training_status["board"] = {
                        str(sq): int(bd.get_piece(sq))
                        for sq in range(64) if bd.get_piece(sq) != 0
                    }
                    training_status["train_last_move"] = (
                        bd.move_history[-1].to_uci() if bd.move_history else None
                    )

                    if viz_delay > 0:
                        time.sleep(viz_delay)

                agent.episode_rewards.append(ep_reward)
                agent.episode_lengths.append(ep_length)
                agent.on_episode_end(ep, ep_reward, ep_length)
                training_status["progress"] = int((ep + 1) / n_episodes * 100)
                training_status["episodes_done"] = ep + 1

            agent.finalize_training()
            checkpoint_path = _save_agent_checkpoint(selected_agent_name, agent)
            training_status["checkpoint_saved"] = True
            training_status["checkpoint_path"] = str(checkpoint_path)
        except Exception as exc:
            training_status["error"] = str(exc)
        finally:
            with _training_state_lock:
                training_status["running"] = False
                training_status["board"] = {}
                training_status["train_last_move"] = None

    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"started": True, "episodes": n_episodes, "agent": agent_name})


@app.route("/api/training_status")
def get_training_status():
    info = dict(training_status)
    if current_agent is not None:
        checkpoint = _checkpoint_info(agent_name)
        info["epsilon"]        = round(getattr(current_agent, "epsilon", 1.0), 3)
        info["training_steps"] = current_agent.training_steps
        info["q_table_size"]   = getattr(current_agent, "q_table_size", 0)
        info["mean_reward"]    = _mean_reward(current_agent)
        info["checkpoint_exists"] = checkpoint["exists"]
        info["checkpoint_path"] = checkpoint["path"]
    return jsonify(info)


@app.route("/api/hyperparams", methods=["GET"])
def get_hyperparams():
    if current_agent is None:
        return jsonify({"error": "Aucun agent sélectionné"}), 400
    return jsonify({
        "agent": agent_name,
        "params": current_agent.get_config(),
        "checkpoint": _checkpoint_info(agent_name),
    })


@app.route("/api/hyperparams", methods=["POST"])
def set_hyperparams():
    if current_agent is None:
        return jsonify({"error": "Aucun agent sélectionné"}), 400
    if training_status["running"]:
        return jsonify({"error": "Entraînement en cours, attends la fin"}), 400

    data = request.get_json(force=True) or {}
    config = current_agent.get_config()
    schema = _agent_default_config(agent_name)

    unsupported = sorted(
        key for key in data
        if key in STRUCTURAL_CONFIG_KEYS.get(agent_name, set()) and data.get(key) != config.get(key)
    )
    if unsupported:
        keys = ", ".join(unsupported)
        return jsonify({
            "error": f"Les paramètres {keys} nécessitent de recréer l'agent; modification refusée."
        }), 400

    updates = {}
    errors = {}
    for key, val in data.items():
        if key not in config:
            continue
        try:
            template = schema.get(key, config[key])
            updates[key] = _coerce_config_value(template, val)
        except (ValueError, TypeError) as exc:
            errors[key] = str(exc)

    if errors:
        return jsonify({"error": "Paramètres invalides", "details": errors}), 400

    _apply_hyperparam_updates(current_agent, updates)
    _normalize_agent_config(agent_name, current_agent)
    checkpoint_path = _save_agent_checkpoint(agent_name, current_agent)

    return jsonify({
        "params": current_agent.get_config(),
        "checkpoint_saved": True,
        "checkpoint_path": str(checkpoint_path),
    })


# ------------------------------------------------------------------
# Lancement
# ------------------------------------------------------------------

def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)
