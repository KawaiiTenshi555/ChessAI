"""
Serveur Flask pour l'interface web du jeu d'échecs.
Supporte plusieurs agents IA (Random + agents tabulaires entraînés).
"""

import random
import sys
import os
import pickle
import threading
import time
from datetime import datetime
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
    "alphazero": {"hidden_sizes", "replay_buffer_size"},
}

# ------------------------------------------------------------------
# État global
# ------------------------------------------------------------------

board          = ChessBoard()
player_color   = WHITE
current_agent  = None   # None = random | instance de BaseAgent
agent_name     = "random"

# Registre des agents disponibles (instanciés à la demande)
_agent_registry: dict = {}

# ------------------------------------------------------------------
# Multi-training : sessions identifiées par ID (plusieurs par agent)
# ------------------------------------------------------------------

_training_sessions: dict  = {}          # session_id -> status dict
_sessions_lock            = threading.Lock()   # protège uniquement les écritures dans le dict

def _make_session_id(agent_name: str) -> str:
    """Génère un ID unique : agent_<HHMMSS_ms>."""
    ts = datetime.now().strftime("%H%M%S_%f")[:9]   # HHMMSSms
    return f"{agent_name}_{ts}"


def _device_label(device) -> str:
    """Retourne un libellé lisible pour un torch.device."""
    import torch
    d = str(device)
    if d == "cpu":
        return "CPU"
    if d.startswith("cuda"):
        idx = int(d.split(":")[-1]) if ":" in d else 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "GPU"
        return f"GPU {idx} — {name}"
    return d


def _global_device_info() -> dict:
    """Résumé global du dispositif de calcul disponible."""
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu", "label": "CPU", "gpu_count": 0, "gpus": []}
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index":  i,
            "name":   props.name,
            "memory": round(props.total_memory / 1024 ** 3, 1),
        })
    return {
        "device":    f"cuda:{torch.cuda.current_device()}",
        "label":     _device_label(f"cuda:{torch.cuda.current_device()}"),
        "gpu_count": torch.cuda.device_count(),
        "gpus":      gpus,
        "cuda_version": torch.version.cuda,
    }

def _empty_session(agent_name: str, session_id: str = "") -> dict:
    return {
        "session_id":       session_id or agent_name,
        "running":          False,
        "progress":         0,
        "total":            0,
        "episodes_done":    0,
        "agent":            agent_name,
        "error":            None,
        "checkpoint_saved": False,
        "checkpoint_path":  None,
        "board":            {},
        "train_last_move":  None,
        "start_time":       None,
    }

def _latest_session_for(agent_name: str) -> dict:
    """Retourne la session la plus récente (running en priorité) pour cet agent."""
    sessions = [s for s in _training_sessions.values() if s["agent"] == agent_name]
    if not sessions:
        return _empty_session(agent_name)
    # Priorité aux sessions en cours, sinon la plus récente par start_time
    running = [s for s in sessions if s["running"]]
    if running:
        return max(running, key=lambda s: s.get("start_time") or 0)
    return max(sessions, key=lambda s: s.get("start_time") or 0)

# Alias backward-compat utilisé dans _board_state et hyperparams
def _session_status(agent_name: str) -> dict:
    return _latest_session_for(agent_name)


def _agent_classes():
    from agents.tabular import SarsaAgent, QLearningAgent, ExpectedSarsaAgent, MonteCarloAgent
    from agents.deep_rl import AlphaZeroAgent, DQNAgent, PPOAgent
    from agents.policy_gradient import REINFORCEAgent

    return {
        "sarsa": SarsaAgent,
        "q_learning": QLearningAgent,
        "expected_sarsa": ExpectedSarsaAgent,
        "monte_carlo": MonteCarloAgent,
        "dqn": DQNAgent,
        "alphazero": AlphaZeroAgent,
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


def _agent_capabilities(name: str) -> dict:
    return {
        "reward_shaping": name not in {"alphazero"},
        "tree_search": name in {"alphazero"},
        "self_play": name != "random",
    }


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
    if "weight_decay" in updates and hasattr(agent, "optimizer"):
        for group in agent.optimizer.param_groups:
            group["weight_decay"] = float(updates["weight_decay"])


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
        "training":      _session_status(agent_name),
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


def _obs_for_color(board: ChessBoard, color: int):
    obs = board.get_observation()
    if color == BLACK:
        obs = obs[::-1, ::-1, :].copy()
    return obs


def _select_agent_move_for_board(agent, board: ChessBoard, legal_moves=None, preserve_buffers: bool = False):
    legal = legal_moves if legal_moves is not None else board.get_legal_moves()
    if not legal:
        return None

    if hasattr(agent, "select_move"):
        move = agent.select_move(board, temperature=0.0)
        return move if move in legal else random.choice(legal)

    legal_actions = [m.from_sq * 64 + m.to_sq for m in legal]
    obs = _obs_for_color(board, board.turn)

    backups = {}
    if preserve_buffers:
        backups = {
            attr: list(val)
            for attr, val in vars(agent).items()
            if isinstance(val, list) and attr.startswith("_")
        }

    action = agent.select_action(obs, legal_actions)

    if preserve_buffers:
        for attr, val in backups.items():
            setattr(agent, attr, val)

    legal_map = {m.from_sq * 64 + m.to_sq: m for m in legal}
    return legal_map.get(action, random.choice(legal))


def _self_play_policy(agent):
    def _policy(_obs, legal_moves, board):
        return _select_agent_move_for_board(
            agent,
            board,
            legal_moves=legal_moves,
            preserve_buffers=True,
        )

    return _policy


def _ai_move():
    """Fait jouer l'IA (agent entraîné ou random)."""
    legal = board.get_legal_moves()
    if not legal:
        return

    if current_agent is not None:
        move = _select_agent_move_for_board(
            current_agent,
            board,
            legal_moves=legal,
            preserve_buffers=True,
        )
        board._apply_move_unchecked(move if move in legal else random.choice(legal))
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
        {"id": "random",         "label": "Aléatoire",      "tier": 0, **_agent_capabilities("random")},
        {"id": "sarsa",          "label": "SARSA",           "tier": 1, **_agent_capabilities("sarsa")},
        {"id": "q_learning",     "label": "Q-Learning",      "tier": 1, **_agent_capabilities("q_learning")},
        {"id": "expected_sarsa", "label": "Expected SARSA",  "tier": 1, **_agent_capabilities("expected_sarsa")},
        {"id": "monte_carlo",    "label": "Monte Carlo",     "tier": 1, **_agent_capabilities("monte_carlo")},
        {"id": "dqn",            "label": "DQN",             "tier": 4, **_agent_capabilities("dqn")},
        {"id": "alphazero",      "label": "AlphaZero",       "tier": 5, **_agent_capabilities("alphazero")},
        {"id": "reinforce",      "label": "REINFORCE",       "tier": 3, **_agent_capabilities("reinforce")},
        {"id": "ppo",            "label": "PPO",             "tier": 4, **_agent_capabilities("ppo")},
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
    # Les sessions d'entraînement tournent sur des instances isolées —
    # on peut changer l'agent de jeu à tout moment.

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
        "capabilities": _agent_capabilities(agent_name),
    })


def _build_session_agent(target_name: str, ref_agent=None):
    """Crée une instance d'agent indépendante pour une session d'entraînement.

    Priorité config : ref_agent (registre ou courant) > checkpoint > défauts.
    Charge les poids du checkpoint si disponible, puis retourne une instance
    isolée qui ne partage aucun état avec l'agent utilisé pour jouer.
    """
    # Config de référence (ordre de priorité)
    if ref_agent is None:
        ref_agent = _agent_registry.get(target_name)
    config = ref_agent.get_config() if ref_agent is not None else (_peek_saved_config(target_name) or {})

    session_agent = _create_agent(target_name, config=config)
    if session_agent is None:
        return None

    # Charger les poids du checkpoint principal si disponible
    checkpoint_path = _agent_state_path(target_name)
    if checkpoint_path.exists():
        try:
            session_agent.load(str(checkpoint_path))
        except Exception:
            pass  # démarrer à zéro si le checkpoint est corrompu

    _normalize_agent_config(target_name, session_agent)
    return session_agent


@app.route("/api/train", methods=["POST"])
def train_agent():
    """Lance une nouvelle session d'entraînement dans un thread dédié.

    Chaque appel crée une session indépendante avec sa propre instance
    d'agent — plusieurs sessions du même agent peuvent coexister.

    Paramètres JSON :
      agent    : nom de l'agent (défaut = agent courant)
      episodes : nombre d'épisodes
      viz_delay: délai visualisation en ms
      reward_shaping + paramètres associés
    """
    data = request.get_json(force=True) or {}
    target_name = data.get("agent", agent_name)

    if target_name == "random":
        return jsonify({"error": "Impossible d'entraîner l'agent aléatoire"}), 400

    # Référence à l'agent de jeu courant (pour copier la config et recharger après)
    playing_ref = _agent_registry.get(target_name) or (current_agent if target_name == agent_name else None)

    # Instance isolée pour cette session (config copiée de l'agent courant)
    session_agent = _build_session_agent(target_name, ref_agent=playing_ref)
    if session_agent is None:
        return jsonify({"error": f"Agent inconnu : {target_name}"}), 400

    n_episodes            = int(data.get("episodes", 100))
    viz_delay             = max(0.0, float(data.get("viz_delay", 0))) / 1000.0
    capabilities          = _agent_capabilities(target_name)
    reward_shaping        = bool(data.get("reward_shaping", False)) if capabilities["reward_shaping"] else False
    capture_reward_scale  = float(data.get("capture_reward_scale", 0.01)) if capabilities["reward_shaping"] else 0.0
    loss_penalty_scale    = float(data.get("loss_penalty_scale", 0.01)) if capabilities["reward_shaping"] else 0.0
    terminal_win_reward   = float(data.get("terminal_win_reward", 10.0)) if capabilities["reward_shaping"] else 10.0
    terminal_loss_penalty = float(data.get("terminal_loss_penalty", 10.0)) if capabilities["reward_shaping"] else 10.0

    session_id = _make_session_id(target_name)
    device_str   = str(getattr(session_agent, "device", "cpu"))
    device_lbl   = _device_label(device_str)
    session: dict = {
        "session_id":       session_id,
        "running":          True,
        "progress":         0,
        "total":            n_episodes,
        "episodes_done":    0,
        "agent":            target_name,
        "device":           device_str,
        "device_label":     device_lbl,
        "error":            None,
        "checkpoint_saved": False,
        "checkpoint_path":  None,
        "board":            {},
        "train_last_move":  None,
        "start_time":       time.time(),
    }
    with _sessions_lock:
        _training_sessions[session_id] = session

    def _train():
        from chess_env.chess_env import ChessEnv
        try:
            if hasattr(session_agent, "train_self_play"):
                def _progress(payload: dict) -> None:
                    bd = payload.get("board")
                    if bd is not None:
                        session["board"] = {
                            str(sq): int(bd.get_piece(sq))
                            for sq in range(64) if bd.get_piece(sq) != 0
                        }
                    session["train_last_move"] = payload.get("last_move")
                    current_ep = int(payload.get("current_episode", payload.get("episodes_done", 0)))
                    session["progress"]      = int(current_ep / max(n_episodes, 1) * 100)
                    session["episodes_done"] = current_ep
                    if viz_delay > 0:
                        time.sleep(viz_delay)

                session_agent.train_self_play(n_episodes=n_episodes, progress_callback=_progress)
                session["progress"]      = 100
                session["episodes_done"] = n_episodes
            else:
                for ep in range(n_episodes):
                    env = ChessEnv(
                        render_mode=None,
                        opponent_policy=_self_play_policy(session_agent),
                        player_color=WHITE if ep % 2 == 0 else BLACK,
                        reward_shaping=reward_shaping,
                        capture_reward_scale=capture_reward_scale,
                        loss_penalty_scale=loss_penalty_scale,
                        terminal_win_reward=terminal_win_reward,
                        terminal_loss_penalty=terminal_loss_penalty,
                    )
                    obs, info = env.reset()
                    done = False
                    ep_reward = 0.0
                    ep_length = 0

                    while not done:
                        legal = info.get("legal_actions", [])
                        if not legal:
                            break
                        action = session_agent.select_action(obs, legal)
                        next_obs, reward, terminated, truncated, next_info = env.step(action)
                        done = terminated or truncated
                        next_legal = next_info.get("legal_actions", []) if not done else []
                        session_agent.update(obs, action, reward, next_obs, done, next_legal)
                        obs, info = next_obs, next_info
                        ep_reward += reward
                        ep_length += 1
                        session_agent.training_steps += 1

                        bd = env.board
                        session["board"] = {
                            str(sq): int(bd.get_piece(sq))
                            for sq in range(64) if bd.get_piece(sq) != 0
                        }
                        session["train_last_move"] = (
                            bd.move_history[-1].to_uci() if bd.move_history else None
                        )
                        if viz_delay > 0:
                            time.sleep(viz_delay)

                    session_agent.episode_rewards.append(ep_reward)
                    session_agent.episode_lengths.append(ep_length)
                    session_agent.on_episode_end(ep, ep_reward, ep_length)
                    session["progress"]      = int((ep + 1) / n_episodes * 100)
                    session["episodes_done"] = ep + 1

                session_agent.finalize_training()

            # Sauvegarde : checkpoint principal (écrase) + checkpoint de session
            main_path = _save_agent_checkpoint(target_name, session_agent)
            session_path = MODELS_DIR / f"{session_id}.pkl"
            session_agent.save(str(session_path))
            session["checkpoint_saved"] = True
            session["checkpoint_path"]  = str(main_path)
            session["session_checkpoint"] = str(session_path)

            # Recharger les poids dans l'instance de jeu (registre ou current_agent direct)
            if playing_ref is not None:
                try:
                    playing_ref.load(str(main_path))
                except Exception:
                    pass
        except Exception as exc:
            import traceback
            session["error"] = str(exc)
            session["traceback"] = traceback.format_exc()
        finally:
            session["running"]         = False
            session["board"]           = {}
            session["train_last_move"] = None

    threading.Thread(target=_train, daemon=True, name=f"train-{session_id}").start()
    return jsonify({"started": True, "episodes": n_episodes, "agent": target_name, "session_id": session_id})


@app.route("/api/training_status")
def get_training_status():
    """Statut de la session la plus récente pour l'agent courant (compat. UI)."""
    info = dict(_latest_session_for(agent_name))
    if current_agent is not None:
        checkpoint = _checkpoint_info(agent_name)
        info["epsilon"]           = round(getattr(current_agent, "epsilon", 1.0), 3)
        info["training_steps"]    = current_agent.training_steps
        info["q_table_size"]      = getattr(current_agent, "q_table_size", 0)
        info["mean_reward"]       = _mean_reward(current_agent)
        info["checkpoint_exists"] = checkpoint["exists"]
        info["checkpoint_path"]   = checkpoint["path"]
    return jsonify(info)


@app.route("/api/training_sessions")
def get_training_sessions():
    """Toutes les sessions d'entraînement triées par date de démarrage."""
    sessions = {}
    with _sessions_lock:
        snapshot = dict(_training_sessions)
    for sid, sess in snapshot.items():
        s = {k: v for k, v in sess.items() if k not in ("board", "train_last_move", "traceback")}
        sessions[sid] = s
    return jsonify({"sessions": sessions})


@app.route("/api/hyperparams", methods=["GET"])
def get_hyperparams():
    if current_agent is None:
        return jsonify({"error": "Aucun agent sélectionné"}), 400
    return jsonify({
        "agent": agent_name,
        "params": current_agent.get_config(),
        "checkpoint": _checkpoint_info(agent_name),
        "capabilities": _agent_capabilities(agent_name),
    })


@app.route("/api/hyperparams", methods=["POST"])
def set_hyperparams():
    if current_agent is None:
        return jsonify({"error": "Aucun agent sélectionné"}), 400
    # Les sessions d'entraînement étant isolées, on peut modifier les hyperparams
    # de l'agent de jeu indépendamment.

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


@app.route("/api/device_info")
def device_info():
    """Informations sur le dispositif de calcul (CPU / GPU)."""
    return jsonify(_global_device_info())


# ------------------------------------------------------------------
# Lancement
# ------------------------------------------------------------------

def run(host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
    app.run(host=host, port=port, debug=debug)
