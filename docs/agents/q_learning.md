# Q-Learning

## Description

Q-Learning est un algorithme de Temporal Difference (TD) **off-policy** (Watkins, 1989).
Il apprend directement la politique optimale Q*, indépendamment de la politique de comportement utilisée pour collecter l'expérience.

## Règle de mise à jour

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') − Q(s, a)]
```

La cible utilise le **maximum** sur toutes les actions légales en s', pas l'action réellement choisie.
Cela permet une mise à jour off-policy : on peut apprendre la politique optimale tout en explorant.

## Pseudo-code

```
Initialiser Q(s, a) = 0 pour tous (s, a)
Pour chaque épisode :
    Initialiser s
    Pour chaque étape :
        Choisir a ← ε-greedy(Q, s, legal_actions)
        Exécuter a, observer r, s'
        Q(s, a) ← Q(s, a) + α · [r + γ · max_a'(Q(s', a')) − Q(s, a)]
        s ← s'
    Décroître ε
```

## Hyperparamètres

| Paramètre       | Défaut | Description |
|-----------------|--------|-------------|
| `alpha`         | 0.1    | Taux d'apprentissage |
| `gamma`         | 0.99   | Décompte des récompenses futures |
| `epsilon`       | 1.0    | Exploration initiale |
| `epsilon_min`   | 0.05   | Plancher d'exploration |
| `epsilon_decay` | 0.995  | Multiplicateur de décroissance par épisode |

## Avantages

- Converge vers la politique **optimale** (preuve théorique sous hypothèses classiques)
- Off-policy : peut théoriquement réutiliser de l'expérience passée
- Plus agressif et potentiellement plus rapide que SARSA
- Algorithme de référence incontournable en RL tabulaire

## Inconvénients

- **Maximisation bias** : `max` sur des Q estimés bruitées peut surestimer les valeurs
  → Solution : Double Q-Learning (deux tables Q pour décoréler sélection et évaluation)
- Peu adapté aux grands espaces (mémoire, pas de généralisation)
- Peut être instable avec un grand α et des récompenses éparses

## Note sur les échecs

Sur les échecs complets, la Q-table devient impraticable en mémoire.
Options pour contourner :
1. Réduire le plateau (4×4, 6×6)
2. Utiliser un hash de Zobrist comme clé (plus compact)
3. Passer à DQN (approximation par réseau de neurones)

## Références

- Watkins (1989) — *Learning from Delayed Rewards* (thèse)
- Watkins & Dayan (1992) — *Q-Learning*, Machine Learning 8
- Sutton & Barto (2018) — chap. 6.5
