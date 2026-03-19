# SARSA — State-Action-Reward-State-Action

## Description

SARSA est un algorithme de Temporal Difference (TD) **on-policy**. Il apprend une fonction de valeur d'action Q(s, a) en mettant à jour les valeurs en cours d'épisode, à chaque transition.

## Règle de mise à jour

```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]
```

où **a'** est choisi par la **même politique ε-greedy** utilisée pour choisir **a**.
C'est ce qui en fait un algorithme on-policy : la cible dépend du comportement futur effectif.

## Pseudo-code

```
Initialiser Q(s, a) = 0 pour tous (s, a)
Pour chaque épisode :
    Initialiser s
    Choisir a ← ε-greedy(Q, s, legal_actions)
    Pour chaque étape :
        Exécuter a, observer r, s'
        Choisir a' ← ε-greedy(Q, s', legal_actions')
        Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a') − Q(s, a)]
        s ← s' ; a ← a'
    Décroître ε
```

## Hyperparamètres

| Paramètre       | Défaut | Description |
|-----------------|--------|-------------|
| `alpha`         | 0.1    | Taux d'apprentissage — plus élevé = mise à jour plus agressive |
| `gamma`         | 0.99   | Décompte des récompenses futures |
| `epsilon`       | 1.0    | Exploration initiale (décroît à chaque épisode) |
| `epsilon_min`   | 0.05   | Plancher d'exploration |
| `epsilon_decay` | 0.995  | Multiplicateur de décroissance par épisode |

## Avantages

- Simple et garanties de convergence (sous conditions classiques de RL tabulaire)
- Apprend en cours d'épisode (pas besoin d'attendre la fin)
- La politique convergée tient compte de l'exploration (plus prudente que Q-Learning)

## Inconvénients

- On-policy : ne peut pas réutiliser des expériences passées (pas de replay buffer)
- Peu adapté aux grands espaces d'états (mémoire proportionnelle aux états visités)
- Convergence plus lente que Q-Learning dans les environnements où l'exploitation rapide est utile

## Différences vs Q-Learning

| | SARSA | Q-Learning |
|---|---|---|
| Type | On-policy | Off-policy |
| Cible | Q(s', a') où a' ∼ π | max_a' Q(s', a') |
| Politique apprise | Politique ε-greedy | Politique optimale |
| Comportement aux bords (falaises) | Prudent | Risqué |

## Références

- Rummery & Niranjan (1994) — *On-Line Q-Learning Using Connectionist Systems*
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, chap. 6.4
