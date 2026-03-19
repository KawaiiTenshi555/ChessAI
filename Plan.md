# Plan du Projet : Chess IA avec Gymnasium

## Vue d'ensemble

Projet de jeu d'échecs complet en Python avec environnement Gymnasium, plusieurs agents de reinforcement learning, et un système de benchmark/comparaison.

---

## Phase 1 — Environnement de jeu (Gymnasium)

### 1.1 Structure du projet

```
ChessIA/
├── chess_env/
│   ├── __init__.py
│   ├── chess_env.py          # Environnement Gymnasium principal
│   ├── board.py              # Logique du plateau d'échecs
│   ├── pieces.py             # Définition des pièces et mouvements
│   ├── rules.py              # Règles : roque, en passant, promotion, échec/mat
│   └── renderer.py           # Rendu visuel (pygame ou ASCII)
├── agents/
│   ├── __init__.py
│   ├── base_agent.py         # Classe abstraite commune à tous les agents
│   ├── tabular/
│   │   ├── sarsa.py
│   │   ├── q_learning.py
│   │   ├── expected_sarsa.py
│   │   └── monte_carlo.py
│   ├── model_based/
│   │   └── dyna_q.py
│   ├── policy_gradient/
│   │   ├── reinforce.py
│   │   ├── a2c.py
│   │   └── a3c.py
│   ├── deep_rl/
│   │   ├── dqn.py
│   │   ├── ddpg.py
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   └── td3.py
│   └── offline_rl/
│       ├── lcts.py
│       └── cql.py
├── benchmark/
│   ├── __init__.py
│   ├── runner.py             # Lancement des matchs entre agents
│   ├── metrics.py            # Calcul des métriques (win rate, ELO, etc.)
│   ├── report_generator.py   # Génération du compte rendu final
│   └── plots.py              # Graphiques et visualisations
├── configs/
│   ├── env_config.yaml       # Paramètres de l'environnement
│   └── agents_config.yaml    # Hyperparamètres par agent
├── models/                   # Poids sauvegardés des agents entraînés
├── results/                  # Résultats bruts des benchmarks
├── reports/                  # Compte rendus générés
├── tests/
│   ├── test_env.py
│   ├── test_agents.py
│   └── test_benchmark.py
├── docs/
│   └── agents/               # Documentation de chaque agent
├── requirements.txt
└── README.md
```

### 1.2 Environnement Gymnasium (`chess_env.py`)

- Hérite de `gymnasium.Env`
- **Espace d'observation** : représentation du plateau
  - Option A : vecteur 8×8×12 (one-hot par type de pièce et couleur)
  - Option B : vecteur plat 768 + features supplémentaires (tour, roque dispo, en passant)
- **Espace d'action** : actions discrètes — encodage UCI (de-à) ou index sur ~4672 coups possibles (convention AlphaZero)
- **Récompenses** :
  - +1 victoire, -1 défaite, 0 nulle
  - Récompenses intermédiaires optionnelles (matériel, contrôle du centre, etc.)
- **Modes** : self-play, vs agent aléatoire, vs agent fixe
- **Terminaison** : échec et mat, pat, règle des 50 coups, répétition de position, insuffisance de matériel
- Support de `render()` en mode ASCII et optionnellement pygame

### 1.3 Règles complètes à implémenter

- Tous les mouvements de pièces (y compris le cavalier, le fou, etc.)
- Roque (grand et petit) avec conditions (roi/tour non bougés, pas en échec)
- Prise en passant
- Promotion de pion (choix de la pièce)
- Détection d'échec, d'échec et mat, de pat
- Règle des 50 coups
- Répétition de position (nulle par triple répétition)

---

## Phase 2 — Agents de Reinforcement Learning

Les agents sont classés du plus simple au plus avancé.

### Tier 1 — Méthodes tabulaires (baselines)

> Adaptées à des espaces réduits / versions simplifiées du jeu

| # | Agent | Type | Caractéristiques |
|---|-------|------|-----------------|
| 1 | **SARSA** | On-policy TD | Q(s,a) mis à jour avec l'action réellement prise |
| 2 | **Q-Learning** | Off-policy TD | Q(s,a) mis à jour avec le max futur |
| 3 | **Expected SARSA** | On-policy TD | Mise à jour avec l'espérance sur toutes les actions |
| 4 | **Monte Carlo** | MC | Mise à jour en fin d'épisode (returns complets) |

### Tier 2 — Méthodes basées sur un modèle

| # | Agent | Type | Caractéristiques |
|---|-------|------|-----------------|
| 5 | **Dyna-Q** | Model-based | Q-Learning + simulation interne (planification) |

### Tier 3 — Policy Gradient

| # | Agent | Type | Caractéristiques |
|---|-------|------|-----------------|
| 6 | **REINFORCE** | PG basique | Monte Carlo policy gradient, haute variance |
| 7 | **A2C** | Actor-Critic | Advantage Actor-Critic, synchrone |
| 8 | **A3C** | Actor-Critic | Version asynchrone multi-thread de A2C |

### Tier 4 — Deep RL (continuum et discret)

| # | Agent | Type | Caractéristiques |
|---|-------|------|-----------------|
| 9  | **DQN** | Value-based | Deep Q-Network, replay buffer, target network |
| 10 | **DDPG** | Actor-Critic off-policy | Actions continues, déterministe |
| 11 | **PPO** | Policy gradient | Proximal Policy Optimization, stable et efficace |
| 12 | **A2C/A3C** | Actor-Critic | (voir Tier 3) |
| 13 | **SAC** | Actor-Critic off-policy | Soft Actor-Critic, entropie maximale |
| 14 | **TD3** | Actor-Critic off-policy | Twin Delayed DDPG, améliore DDPG |

### Tier 5 — Offline / Advanced RL

| # | Agent | Type | Caractéristiques |
|---|-------|------|-----------------|
| 15 | **LCTS** | Tree Search | Learning-guided MCTS (combinaison RL + MCTS) |
| 16 | **CQL** | Offline RL | Conservative Q-Learning (apprentissage sur données fixes) |

### 2.1 Interface commune (`base_agent.py`)

Tous les agents implémentent :

```python
class BaseAgent:
    def select_action(self, observation, valid_actions) -> int
    def update(self, obs, action, reward, next_obs, done) -> None
    def train(self, env, n_episodes) -> dict
    def save(self, path: str) -> None
    def load(self, path: str) -> None
    def get_config(self) -> dict
```

### 2.2 Documentation par agent

Chaque agent aura un fichier `docs/agents/<nom_agent>.md` contenant :
- Description théorique
- Algorithme (pseudo-code)
- Hyperparamètres et leur effet
- Avantages / Inconvénients
- Références bibliographiques
- Résultats obtenus lors des benchmarks

---

## Phase 3 — Système de Benchmark

### 3.1 Modes de benchmark

1. **Agent vs Agent** : matchs directs entre deux agents entraînés
2. **Agent vs Random** : baseline simple
3. **Tournament Round-Robin** : tous les agents s'affrontent
4. **ELO League** : système de rating ELO continu

### 3.2 Métriques collectées

- Win rate / Draw rate / Loss rate
- Score ELO
- Durée moyenne des parties
- Nombre de coups moyen
- Temps de décision moyen par coup
- Courbe d'apprentissage (reward en fonction des épisodes d'entraînement)
- Stabilité de l'entraînement (variance des rewards)
- Efficacité computationnelle (GPU/CPU time)

### 3.3 Visualisations

- Courbes d'apprentissage par agent
- Heatmap des confrontations (matrice win rate)
- Classement ELO au fil du temps
- Comparaison des temps d'inférence
- Distribution des longueurs de parties

### 3.4 Génération du rapport

- Format Markdown + HTML optionnel
- Sections : résumé exécutif, classement, analyses par agent, conclusions
- Export CSV des données brutes
- Graphiques sauvegardés en PNG/SVG

---

## Phase 4 — Outils et Infrastructure

### 4.1 Configuration

- Fichiers YAML pour les hyperparamètres
- Logging avec niveaux (DEBUG, INFO, WARNING)
- Sauvegarde automatique des checkpoints

### 4.2 Reproductibilité

- Seeds fixes pour NumPy, random, PyTorch/TensorFlow
- Versionnage des configurations dans les résultats

### 4.3 Tests

- Tests unitaires de l'environnement (mouvements légaux, règles)
- Tests d'intégration agent-environnement
- Validation que chaque agent tourne sans erreur

---

## Phase 5 — Compte rendu final

### Structure du rapport

1. **Introduction** : objectifs, contexte
2. **Environnement** : description technique, espace d'état/action
3. **Agents** : description de chaque agent, hyperparamètres utilisés
4. **Résultats** :
   - Classement général
   - Analyse des forces/faiblesses
   - Comparaison théorie vs pratique
5. **Conclusions** : recommandations, agents les plus performants, pistes d'amélioration
6. **Annexes** : données brutes, configurations complètes

---

## Ordre de développement recommandé

```
1. chess_env/        → environnement complet + tests
2. base_agent.py     → interface commune
3. Tier 1 agents     → SARSA, Q-Learning, Expected SARSA, Monte Carlo
4. Tier 2 agents     → Dyna-Q
5. Tier 3 agents     → REINFORCE, A2C, A3C
6. Tier 4 agents     → DQN, PPO, SAC, TD3, DDPG
7. Tier 5 agents     → LCTS, CQL
8. benchmark/        → runner, métriques, visualisations
9. report_generator  → rapport final
```

---

## Stack technique

| Composant | Bibliothèque |
|-----------|-------------|
| Environnement | `gymnasium` |
| Deep Learning | `torch` (PyTorch) |
| Calcul numérique | `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Rendu pygame | `pygame` (optionnel) |
| Gestion config | `pyyaml` |
| Logging | `logging` (stdlib) |
| Tests | `pytest` |
| Rapports | `jinja2` (templates HTML) |

---

## Notes importantes

- Les agents tabulaires (Tier 1) nécessitent une **réduction de l'espace d'état** pour être viables sur les échecs complets. Envisager une version simplifiée (4×4 ou 6×6) pour ces agents, ou une représentation hachée.
- DDPG, SAC, TD3 sont conçus pour des **actions continues** — une adaptation pour l'espace discret des échecs sera nécessaire (ex. : action embedding ou reformulation).
- LCTS combine MCTS avec un réseau de neurones guidant la recherche (inspiré d'AlphaZero).
- CQL suppose un **dataset offline** de parties d'échecs — utiliser des datasets publics (ex. : Lichess game database).
