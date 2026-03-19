"""
Tests des agents Tier 1 (méthodes tabulaires).

Run with:  pytest tests/test_agents.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from chess_env.chess_env import ChessEnv
from agents.tabular import SarsaAgent, QLearningAgent, ExpectedSarsaAgent, MonteCarloAgent

# Forme de l'observation et taille du l'espace d'action
OBS_SHAPE    = (8, 8, 17)
ACTION_SIZE  = 4096

# Nombre d'épisodes pour les smoke-tests (court, juste pour vérifier que ça tourne)
SMOKE_EPISODES = 5


# ===========================================================================
# Fixtures
# ===========================================================================

def make_env(seed: int = 0) -> ChessEnv:
    env = ChessEnv(render_mode=None)
    env.reset(seed=seed)
    return env


def make_agents():
    return {
        "sarsa":          SarsaAgent(ACTION_SIZE, OBS_SHAPE),
        "q_learning":     QLearningAgent(ACTION_SIZE, OBS_SHAPE),
        "expected_sarsa": ExpectedSarsaAgent(ACTION_SIZE, OBS_SHAPE),
        "monte_carlo":    MonteCarloAgent(ACTION_SIZE, OBS_SHAPE),
    }


# ===========================================================================
# 1 — Interface commune
# ===========================================================================

class TestBaseInterface:

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_select_action_returns_legal_action(self, agent_name):
        agents = make_agents()
        agent = agents[agent_name]
        env = make_env()
        obs, info = env.reset(seed=42)
        legal = info["legal_actions"]
        action = agent.select_action(obs, legal)
        assert action in legal, f"{agent_name}: action {action} not in legal actions"

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_update_returns_loss_or_none(self, agent_name):
        agents = make_agents()
        agent = agents[agent_name]
        env = make_env()
        obs, info = env.reset(seed=0)
        legal = info["legal_actions"]
        action = agent.select_action(obs, legal)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_legal = next_info["legal_actions"] if not (terminated or truncated) else []
        loss = agent.update(obs, action, reward, next_obs, terminated or truncated, next_legal)
        assert loss is None or isinstance(loss, float)

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_get_config_returns_dict(self, agent_name):
        agents = make_agents()
        cfg = agents[agent_name].get_config()
        assert isinstance(cfg, dict)
        assert "alpha" in cfg
        assert "gamma" in cfg
        assert "epsilon" in cfg

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_repr(self, agent_name):
        agents = make_agents()
        r = repr(agents[agent_name])
        assert agent_name.replace("_", "").lower() in r.lower().replace("_", "")


# ===========================================================================
# 2 — Entraînement (smoke tests)
# ===========================================================================

class TestTraining:

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_train_completes(self, agent_name):
        agents = make_agents()
        agent = agents[agent_name]
        env = make_env()
        stats = agent.train(env, n_episodes=SMOKE_EPISODES)
        assert "episode_rewards" in stats
        assert len(stats["episode_rewards"]) == SMOKE_EPISODES

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_training_steps_count(self, agent_name):
        agents = make_agents()
        agent = agents[agent_name]
        env = make_env()
        agent.train(env, n_episodes=SMOKE_EPISODES)
        assert agent.training_steps > 0

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_epsilon_decays_after_training(self, agent_name):
        agents = make_agents()
        agent = agents[agent_name]
        initial_epsilon = agent.epsilon
        env = make_env()
        agent.train(env, n_episodes=SMOKE_EPISODES)
        assert agent.epsilon <= initial_epsilon

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa"])
    def test_q_table_populated_after_training(self, agent_name):
        """Après entraînement, la Q-table doit contenir des entrées."""
        agents = make_agents()
        agent = agents[agent_name]
        env = make_env()
        agent.train(env, n_episodes=SMOKE_EPISODES)
        assert agent.q_table_size > 0

    def test_monte_carlo_episode_buffer_cleared_after_training(self):
        agent = MonteCarloAgent(ACTION_SIZE, OBS_SHAPE)
        env = make_env()
        agent.train(env, n_episodes=SMOKE_EPISODES)
        assert len(agent._episode) == 0, "Buffer MC non vidé après entraînement"

    def test_monte_carlo_q_table_populated(self):
        agent = MonteCarloAgent(ACTION_SIZE, OBS_SHAPE)
        env = make_env()
        agent.train(env, n_episodes=SMOKE_EPISODES)
        assert agent.q_table_size > 0


# ===========================================================================
# 3 — Comportement spécifique à chaque algorithme
# ===========================================================================

class TestAlgorithmSpecific:

    def test_sarsa_on_policy_next_action_used(self):
        """SARSA doit réutiliser l'action a' stockée lors du update."""
        agent = SarsaAgent(ACTION_SIZE, OBS_SHAPE)
        env = make_env()
        obs, info = env.reset(seed=1)
        legal = info["legal_actions"]
        action = agent.select_action(obs, legal)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_legal = next_info["legal_actions"] if not (terminated or truncated) else []
        if next_legal:
            agent.update(obs, action, reward, next_obs, False, next_legal)
            stored = agent._next_action
            assert stored is None or stored in next_legal

    def test_sarsa_next_action_cleared_on_done(self):
        agent = SarsaAgent(ACTION_SIZE, OBS_SHAPE)
        dummy_obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        agent.update(dummy_obs, 0, 1.0, dummy_obs, done=True, legal_next_actions=[])
        assert agent._next_action is None

    def test_qlearning_uses_max_not_sampled(self):
        """Q-Learning doit utiliser le max sur les actions légales, pas un échantillon."""
        agent = QLearningAgent(ACTION_SIZE, OBS_SHAPE, config={"alpha": 1.0, "epsilon": 0.0})
        s  = np.zeros(OBS_SHAPE, dtype=np.float32)
        s2 = np.ones(OBS_SHAPE,  dtype=np.float32) * 0.5

        # Pré-remplir Q(s2, .) avec des valeurs connues
        key2 = s2.tobytes()
        agent.Q[key2][10] = 5.0   # meilleure action
        agent.Q[key2][20] = 1.0

        # Après update, Q(s, 0) doit se rapprocher de 0 + γ * 5.0
        agent.update(s, 0, 0.0, s2, done=False, legal_next_actions=[10, 20])
        expected = agent.gamma * 5.0
        assert abs(agent.Q[s.tobytes()][0] - expected) < 1e-6

    def test_expected_sarsa_expected_value_computation(self):
        """Vérifie le calcul de l'espérance sous ε-greedy."""
        agent = ExpectedSarsaAgent(ACTION_SIZE, OBS_SHAPE,
                                   config={"alpha": 1.0, "epsilon": 0.5})
        s  = np.zeros(OBS_SHAPE, dtype=np.float32)
        s2 = np.ones(OBS_SHAPE,  dtype=np.float32) * 0.5

        key2 = s2.tobytes()
        # Deux actions : 0 (valeur 4.0) et 1 (valeur 2.0) → greedy = 0
        agent.Q[key2][0] = 4.0
        agent.Q[key2][1] = 2.0
        # ε=0.5, |A|=2 : prob(greedy)=0.5+0.25=0.75, prob(autre)=0.25
        # E = 0.75*4.0 + 0.25*2.0 = 3.0 + 0.5 = 3.5
        expected_q = 0.75 * 4.0 + 0.25 * 2.0
        target = 0.0 + agent.gamma * expected_q

        agent.update(s, 5, 0.0, s2, done=False, legal_next_actions=[0, 1])
        assert abs(agent.Q[s.tobytes()][5] - target) < 1e-5

    def test_monte_carlo_first_visit_vs_every_visit(self):
        """First-visit ne doit mettre à jour qu'une fois par (s,a) par épisode."""
        agent_fv = MonteCarloAgent(ACTION_SIZE, OBS_SHAPE,
                                   config={"alpha": 1.0, "gamma": 1.0, "first_visit": True})
        agent_ev = MonteCarloAgent(ACTION_SIZE, OBS_SHAPE,
                                   config={"alpha": 1.0, "gamma": 1.0, "first_visit": False})

        # Épisode synthétique : (s, a=0, r=1), (s, a=0, r=1) — même (s,a) deux fois
        obs = np.zeros(OBS_SHAPE, dtype=np.float32)
        key = obs.tobytes()

        for agent in (agent_fv, agent_ev):
            agent._episode = [(key, 0, 1.0), (key, 0, 1.0)]
            agent._flush_episode()

        # First-visit : G=2.0 (r1 + γ·r2 = 1+1=2), appliqué une seule fois
        # Q(s,0) = 0 + 1.0*(2-0) = 2.0
        assert abs(agent_fv.Q[key][0] - 2.0) < 1e-6

        # Every-visit : deux mises à jour successives (sens inverse)
        # Étape 2 (idx 1) : G=1.0 → Q = 0 + 1*(1-0) = 1.0
        # Étape 1 (idx 0) : G=2.0 → Q = 1.0 + 1*(2-1.0) = 2.0
        assert abs(agent_ev.Q[key][0] - 2.0) < 1e-6


# ===========================================================================
# 4 — Save / Load
# ===========================================================================

class TestPersistence:

    @pytest.mark.parametrize("agent_name", ["sarsa", "q_learning", "expected_sarsa", "monte_carlo"])
    def test_save_load_roundtrip(self, tmp_path, agent_name):
        agents = make_agents()
        agent = agents[agent_name]
        env = make_env()
        agent.train(env, n_episodes=3)

        path = str(tmp_path / f"{agent_name}.pkl")
        agent.save(path)

        # Charger dans un agent vierge
        agents2 = make_agents()
        agent2 = agents2[agent_name]
        agent2.load(path)

        assert agent2.training_steps == agent.training_steps
        assert abs(agent2.epsilon - agent.epsilon) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
