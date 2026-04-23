import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import web.app as web_app
from agents.deep_rl import AlphaZeroAgent
from agents.policy_gradient import PPOAgent
from chess_env.board import ChessBoard


OBS_SHAPE = (8, 8, 17)
ACTION_SIZE = 4096


@pytest.fixture(autouse=True)
def isolate_web_state(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "MODELS_DIR", tmp_path)
    web_app._agent_registry.clear()
    web_app._training_sessions.clear()
    web_app.current_agent = None
    web_app.agent_name = "random"
    yield
    web_app._agent_registry.clear()
    web_app._training_sessions.clear()
    web_app.current_agent = None
    web_app.agent_name = "random"


def test_get_or_create_agent_loads_saved_q_learning_checkpoint():
    agent = web_app._create_agent("q_learning")
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    key = obs.tobytes()
    agent.Q[key][123] = 4.5
    agent.training_steps = 42
    agent.epsilon = 0.25

    checkpoint_path = web_app._save_agent_checkpoint("q_learning", agent)
    assert checkpoint_path.exists()

    web_app._agent_registry.clear()
    loaded = web_app._get_or_create_agent("q_learning")

    assert loaded.training_steps == 42
    assert abs(loaded.epsilon - 0.25) < 1e-9
    assert abs(loaded.Q[key][123] - 4.5) < 1e-9


def test_dqn_checkpoint_recreates_agent_with_saved_config():
    agent = web_app._create_agent("dqn", config={"hidden_sizes": [64], "batch_size": 8})
    web_app._save_agent_checkpoint("dqn", agent)

    web_app._agent_registry.clear()
    loaded = web_app._get_or_create_agent("dqn")

    assert loaded.hidden_sizes == [64]
    assert loaded.batch_size == 8
    assert loaded.q_net.net[0].out_features == 64


def test_alphazero_checkpoint_recreates_agent_with_saved_config():
    agent = web_app._create_agent(
        "alphazero",
        config={
            "hidden_sizes": [32],
            "mcts_simulations": 4,
            "batch_size": 4,
            "replay_buffer_size": 32,
            "training_batches_per_episode": 1,
            "max_game_length": 4,
        },
    )
    web_app._save_agent_checkpoint("alphazero", agent)

    web_app._agent_registry.clear()
    loaded = web_app._get_or_create_agent("alphazero")

    assert loaded.hidden_sizes == [32]
    assert loaded.mcts_simulations == 4
    assert loaded.batch_size == 4


def test_set_hyperparams_persists_runtime_safe_updates():
    web_app.current_agent = web_app._create_agent("dqn")
    web_app.agent_name = "dqn"
    client = web_app.app.test_client()

    response = client.post("/api/hyperparams", json={
        "batch_size": 32,
        "target_update_freq": 250,
        "lr": 2e-4,
    })

    assert response.status_code == 200
    assert web_app.current_agent.batch_size == 32
    assert web_app.current_agent.target_update_freq == 250
    assert abs(web_app.current_agent.lr - 2e-4) < 1e-12

    web_app._agent_registry.clear()
    loaded = web_app._get_or_create_agent("dqn")
    assert loaded.batch_size == 32
    assert loaded.target_update_freq == 250
    assert abs(loaded.lr - 2e-4) < 1e-12


def test_alphazero_hyperparams_persist_runtime_safe_updates():
    web_app.current_agent = web_app._create_agent(
        "alphazero",
        config={
            "hidden_sizes": [32],
            "mcts_simulations": 4,
            "batch_size": 4,
            "replay_buffer_size": 32,
            "training_batches_per_episode": 1,
            "max_game_length": 4,
        },
    )
    web_app.agent_name = "alphazero"
    client = web_app.app.test_client()

    response = client.post("/api/hyperparams", json={
        "mcts_simulations": 8,
        "c_puct": 2.1,
        "weight_decay": 5e-4,
    })

    assert response.status_code == 200
    assert web_app.current_agent.mcts_simulations == 8
    assert web_app.current_agent.c_puct == pytest.approx(2.1)
    assert web_app.current_agent.weight_decay == pytest.approx(5e-4)

    web_app._agent_registry.clear()
    loaded = web_app._get_or_create_agent("alphazero")
    assert loaded.mcts_simulations == 8
    assert loaded.c_puct == pytest.approx(2.1)
    assert loaded.weight_decay == pytest.approx(5e-4)


def test_board_state_exposes_mean_reward_for_selected_agent():
    web_app.current_agent = web_app._create_agent("q_learning")
    web_app.agent_name = "q_learning"
    web_app.current_agent.episode_rewards = [1.0, -0.5, 0.5]

    state = web_app._board_state()

    assert "agent" in state
    assert state["agent"]["mean_reward"] == pytest.approx((1.0 - 0.5 + 0.5) / 3.0)


def test_set_hyperparams_rejects_structural_dqn_updates():
    web_app.current_agent = web_app._create_agent("dqn")
    web_app.agent_name = "dqn"
    client = web_app.app.test_client()

    response = client.post("/api/hyperparams", json={"hidden_sizes": [128, 64]})

    assert response.status_code == 400
    data = response.get_json()
    assert "recréer l'agent" in data["error"]


def test_select_agent_exposes_alphazero_capabilities():
    client = web_app.app.test_client()

    response = client.post("/api/select_agent", json={"agent": "alphazero"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["selected"] == "alphazero"
    assert data["capabilities"]["self_play"] is True
    assert data["capabilities"]["reward_shaping"] is False


def test_set_hyperparams_uses_canonical_schema_for_polluted_ppo_agent():
    web_app.current_agent = web_app._create_agent("ppo")
    web_app.agent_name = "ppo"
    web_app.current_agent.ppo_epochs = 4.0
    web_app.current_agent.batch_size = 32.0
    web_app.current_agent.rollout_steps = 128.0
    client = web_app.app.test_client()

    response = client.post("/api/hyperparams", json={
        "ppo_epochs": 6,
        "batch_size": 64,
        "rollout_steps": 256,
    })

    assert response.status_code == 200
    assert isinstance(web_app.current_agent.ppo_epochs, int)
    assert isinstance(web_app.current_agent.batch_size, int)
    assert isinstance(web_app.current_agent.rollout_steps, int)
    assert web_app.current_agent.ppo_epochs == 6
    assert web_app.current_agent.batch_size == 64
    assert web_app.current_agent.rollout_steps == 256


def test_normalize_agent_config_recovers_float_polluted_ppo_fields():
    agent = web_app._create_agent("ppo")
    agent.ppo_epochs = 3.0
    agent.batch_size = 16.0
    agent.rollout_steps = 96.0

    web_app._normalize_agent_config("ppo", agent)

    assert agent.ppo_epochs == 3
    assert agent.batch_size == 16
    assert agent.rollout_steps == 96
    assert isinstance(agent.ppo_epochs, int)
    assert isinstance(agent.batch_size, int)
    assert isinstance(agent.rollout_steps, int)


def test_train_route_allows_multiple_sessions_same_agent(monkeypatch):
    """Deux lancements consécutifs du même agent créent deux sessions distinctes."""
    web_app.current_agent = web_app._create_agent("ppo")
    web_app.agent_name = "ppo"
    client = web_app.app.test_client()

    original_thread = web_app.threading.Thread

    class DummyThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            return None  # ne démarre pas vraiment

    monkeypatch.setattr(web_app.threading, "Thread", DummyThread)

    first = client.post("/api/train", json={"episodes": 1})
    second = client.post("/api/train", json={"episodes": 1})

    assert first.status_code == 200
    assert second.status_code == 200  # deux sessions autorisées

    # Deux sessions distinctes dans le registre
    ppo_sessions = [s for s in web_app._training_sessions.values() if s["agent"] == "ppo"]
    assert len(ppo_sessions) == 2

    monkeypatch.setattr(web_app.threading, "Thread", original_thread)


def test_web_train_route_updates_episode_rewards(monkeypatch):
    web_app.current_agent = web_app._create_agent("q_learning")
    web_app.agent_name = "q_learning"
    client = web_app.app.test_client()

    original_thread = web_app.threading.Thread

    class ImmediateThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            if self._target is not None:
                self._target()

    monkeypatch.setattr(web_app.threading, "Thread", ImmediateThread)

    response = client.post("/api/train", json={"episodes": 1})

    assert response.status_code == 200
    assert len(web_app.current_agent.episode_rewards) == 1
    assert len(web_app.current_agent.episode_lengths) == 1
    status = client.get("/api/training_status").get_json()
    assert status["mean_reward"] == pytest.approx(web_app.current_agent.episode_rewards[0])

    monkeypatch.setattr(web_app.threading, "Thread", original_thread)


def test_self_play_policy_preserves_ppo_buffers():
    agent = PPOAgent(
        ACTION_SIZE,
        OBS_SHAPE,
        config={"hidden_sizes": [32], "batch_size": 4, "rollout_steps": 32, "ppo_epochs": 1},
    )
    board = ChessBoard()
    policy = web_app._self_play_policy(agent)

    move = policy(None, board.get_legal_moves(), board)

    assert move in board.get_legal_moves()
    assert len(agent._rb_obs) == 0
    assert len(agent._rb_actions) == 0
    assert len(agent._rb_log_probs) == 0
    assert len(agent._rb_rewards) == 0


def test_web_train_route_runs_alphazero_self_play(monkeypatch):
    web_app.current_agent = AlphaZeroAgent(
        ACTION_SIZE,
        OBS_SHAPE,
        config={
            "hidden_sizes": [32],
            "batch_size": 4,
            "mcts_simulations": 2,
            "replay_buffer_size": 32,
            "training_batches_per_episode": 1,
            "max_game_length": 4,
            "temperature_drop_move": 2,
        },
    )
    web_app.agent_name = "alphazero"
    client = web_app.app.test_client()

    original_thread = web_app.threading.Thread

    class ImmediateThread:
        def __init__(self, target=None, daemon=None, name=None):
            self._target = target
            self.daemon = daemon

        def start(self):
            if self._target is not None:
                self._target()

    monkeypatch.setattr(web_app.threading, "Thread", ImmediateThread)

    response = client.post("/api/train", json={
        "episodes": 1,
        "reward_shaping": True,
        "capture_reward_scale": 0.25,
    })

    assert response.status_code == 200
    assert len(web_app.current_agent.episode_rewards) == 1
    assert web_app.current_agent.q_table_size > 0
    status = client.get("/api/training_status").get_json()
    assert status["checkpoint_saved"] is True
    assert status["episodes_done"] == 1

    monkeypatch.setattr(web_app.threading, "Thread", original_thread)


def test_ppo_finalize_training_flushes_partial_rollout():
    agent = PPOAgent(
        ACTION_SIZE,
        OBS_SHAPE,
        config={"hidden_sizes": [32], "batch_size": 4, "rollout_steps": 32, "ppo_epochs": 1},
    )
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    legal_actions = [0, 1, 2]

    for _ in range(3):
        action = agent.select_action(obs, legal_actions)
        loss = agent.update(obs, action, 1.0, obs, False, legal_actions)
        assert loss == 0.0

    assert len(agent._rb_rewards) == 3
    agent.finalize_training()

    assert len(agent._rb_rewards) == 0
    assert len(agent._rb_obs) == 0
    assert isinstance(agent._last_loss, float)


def test_ppo_finalize_training_handles_single_step_rollout():
    agent = PPOAgent(
        ACTION_SIZE,
        OBS_SHAPE,
        config={"hidden_sizes": [32], "batch_size": 4, "rollout_steps": 32, "ppo_epochs": 1},
    )
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)
    legal_actions = [0, 1, 2]

    action = agent.select_action(obs, legal_actions)
    loss = agent.update(obs, action, 1.0, obs, True, [])
    assert loss == 0.0

    agent.finalize_training()

    assert len(agent._rb_rewards) == 0
    assert len(agent._rb_obs) == 0
    assert isinstance(agent._last_loss, float)
