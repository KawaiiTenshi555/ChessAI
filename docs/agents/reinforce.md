# REINFORCE — Monte Carlo Policy Gradient

## Description

REINFORCE (Williams, 1992) est le premier algorithme de **policy gradient** : plutôt que d'apprendre une fonction de valeur Q(s, a), il optimise directement les paramètres θ de la politique π_θ(a | s).

L'idée centrale est d'augmenter la probabilité des actions qui ont mené à de bons retours, et de diminuer celles qui ont mené à de mauvais retours. La mise à jour se fait **en fin d'épisode** à partir des retours réels complets (Monte Carlo), sans bootstrap.

## Architecture

```
Observation (8×8×17) → flatten → 1088 entrées
    ↓
Linear(1088 → 512) + ReLU
    ↓
Linear(512 → 256) + ReLU
    ↓
Linear(256 → 4096)   ← logits (scores bruts par action)
    ↓
Masquage actions illégales (−∞ sur les coups interdits)
    ↓
Softmax → distribution de probabilités sur les actions légales
    ↓
Categorical sample → action choisie
```

## Règle de mise à jour

Le gradient de la performance J(θ) est estimé par :

```
∇J(θ) ≈ Σ_t G_t · ∇ log π_θ(a_t | s_t)

où G_t = Σ_{k=t}^{T} γ^{k-t} · r_k   (retour actualisé à partir de t)
```

Avec **baseline** (réduction de variance) :

```
∇J(θ) ≈ Σ_t (G_t − b) · ∇ log π_θ(a_t | s_t)

où b = mean(G_0, G_1, ..., G_T)   (baseline simple = moyenne des retours)
```

## Pseudo-code

```
Initialiser π_θ (réseau de politique)

Pour chaque épisode :
    Générer une trajectoire τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)
    en suivant π_θ avec masquage des actions illégales

    Pour chaque t de 0 à T :
        Calculer G_t = Σ_{k=t}^{T} γ^{k-t} · r_k

    Si baseline :
        G_t ← G_t − mean(G)

    loss = −mean(log π_θ(a_t | s_t) · G_t)
    ∇θ ← Adam(loss)
    Clipper le gradient (norm ≤ 1.0)
```

## Hyperparamètres

| Paramètre    | Défaut | Description |
|--------------|--------|-------------|
| `lr`         | 3e-4   | Taux d'apprentissage Adam |
| `gamma`      | 0.99   | Décompte des récompenses futures |
| `baseline`   | True   | Soustraction de la moyenne des retours (réduit la variance) |
| `epsilon`    | 0.0    | Non utilisé (la politique est naturellement stochastique) |

## Avantages

- Simple à comprendre et à implémenter
- Politique stochastique native (pas besoin d'ε-greedy explicite)
- Pas de Q-table ou de réseau cible à gérer
- Masquage des actions illégales direct dans la distribution

## Inconvénients

- **Haute variance** : les retours Monte Carlo fluctuent beaucoup, surtout sur des parties longues
- **Efficacité des données faible** : on-policy, chaque trajectoire n'est utilisée qu'une fois
- Mise à jour uniquement en fin d'épisode → apprentissage lent sur parties longues
- Peut converger vers des politiques sous-optimales (minima locaux)

## Différences vs méthodes tabulaires

| | SARSA/Q-Learning | REINFORCE |
|---|---|---|
| Ce qu'on apprend | Fonction de valeur Q(s,a) | Politique π(a\|s) directement |
| Mise à jour | En cours d'épisode | En fin d'épisode |
| Exploration | ε-greedy explicite | Stochasticité naturelle de π |
| Scalabilité | Limitée (table) | Bonne (réseau de neurones) |
| Variance | Faible (TD) | Élevée (Monte Carlo) |

## Différences vs PPO

| | REINFORCE | PPO |
|---|---|---|
| Critique | Aucun (pas de value function) | Oui (réseau critic) |
| Avantage | Retour brut G_t | Avantage estimé par GAE |
| Stabilité | Faible | Élevée (clipping, epochs multiples) |
| Efficacité données | Faible (1 update/épisode) | Meilleure (K epochs par rollout) |

## Références

- Williams (1992) — *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, Machine Learning
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, chap. 13
- Schulman et al. (2015) — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438
