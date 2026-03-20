# DQN — Deep Q-Network

## Description

DQN (Mnih et al., 2015) est la première application réussie du deep learning au reinforcement learning à grande échelle. Il remplace la Q-table tabulaire par un **réseau de neurones** qui approxime Q(s, a) pour tous les états, permettant de traiter des espaces d'états très grands comme les échecs.

Deux innovations clés distinguent DQN du Q-Learning classique :
1. **Experience Replay** : les transitions (s, a, r, s') sont stockées dans un buffer circulaire et tirées aléatoirement pour l'entraînement, ce qui brise les corrélations temporelles.
2. **Target Network** : un second réseau, gelé et synchronisé périodiquement, génère les cibles de mise à jour, ce qui stabilise l'apprentissage.

## Architecture

```
Observation (8×8×17) → flatten → 1088 entrées
    ↓
Linear(1088 → 512) + ReLU
    ↓
Linear(512 → 256) + ReLU
    ↓
Linear(256 → 4096)   ← une Q-valeur par action possible
```

Le masquage des actions illégales est appliqué avant l'argmax : les actions interdites reçoivent −∞.

## Règle de mise à jour

```
y = r + γ · max_{a' légal} Q_target(s', a')   (si non terminal)
y = r                                           (si terminal)

L = MSE(Q_online(s, a), y)
∇θ ← Adam(L)
```

Le réseau cible est synchronisé toutes les `target_update_freq` étapes.

## Pseudo-code

```
Initialiser Q_online et Q_target (même architecture)
Initialiser replay buffer D (capacité N)

Pour chaque étape :
    a ← ε-greedy(Q_online, s, legal_actions)
    Exécuter a → observer r, s'
    Stocker (s, a, r, s', done, legal') dans D

    Si |D| ≥ batch_size :
        Tirer un mini-batch aléatoire de D
        Calculer les cibles y avec Q_target
        Minimiser MSE(Q_online(s,a), y) via Adam
        Clipper le gradient (norm ≤ 10)

    Toutes les target_update_freq étapes :
        Q_target ← Q_online

À chaque fin d'épisode :
    ε ← max(ε_min, ε × ε_decay)
```

## Hyperparamètres

| Paramètre           | Défaut   | Description |
|---------------------|----------|-------------|
| `lr`                | 1e-4     | Taux d'apprentissage Adam |
| `gamma`             | 0.99     | Décompte des récompenses futures |
| `epsilon`           | 1.0      | Exploration initiale (ε-greedy) |
| `epsilon_min`       | 0.05     | Plancher d'exploration |
| `epsilon_decay`     | 0.9995   | Multiplicateur par épisode |
| `buffer_size`       | 10 000   | Capacité du replay buffer |
| `batch_size`        | 64       | Taille du mini-batch pour chaque mise à jour |
| `target_update_freq`| 500      | Nombre de steps entre deux synchronisations du réseau cible |

## Avantages

- Gère les grands espaces d'états grâce à la généralisation du réseau
- Le replay buffer améliore l'efficacité des données (chaque transition peut être réutilisée)
- Convergence plus stable que Q-Learning brut grâce au target network
- Masquage des actions illégales intégré

## Inconvénients

- Lent à démarrer (le buffer doit être rempli avant les premières mises à jour)
- Nécessite beaucoup de RAM pour le buffer avec des observations (8,8,17)
- Overestimation des Q-valeurs (corrigé par Double DQN, non implémenté ici)
- Convergence sensible au choix du taux d'apprentissage

## Différences vs Q-Learning tabulaire

| | Q-Learning | DQN |
|---|---|---|
| Représentation Q | Table (états × actions) | Réseau de neurones |
| Généralisation | Aucune | Oui (états similaires → Q similaires) |
| Mémoire | Proportionnelle aux états vus | Fixe (buffer + poids réseau) |
| Replay | Non | Oui (expériences rejouées aléatoirement) |
| Stabilité | Instable sur grands espaces | Stable grâce au target network |

## Références

- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*, Nature
- Mnih et al. (2013) — *Playing Atari with Deep Reinforcement Learning*, arXiv:1312.5602
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, chap. 9–10
