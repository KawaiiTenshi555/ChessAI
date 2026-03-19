# Expected SARSA

## Description

Expected SARSA est une variante de SARSA qui remplace l'échantillon de Q(s', a') par
l'**espérance** de Q(s', ·) sous la politique courante. Cela réduit la variance sans
introduire le biais de maximisation de Q-Learning.

## Règle de mise à jour

```
Q(s, a) ← Q(s, a) + α · [r + γ · E_π[Q(s', a')] − Q(s, a)]
```

avec :
```
E_π[Q(s', a')] = Σ_a' π(a'|s') · Q(s', a')
```

Sous ε-greedy avec |A| actions légales :
```
π(a'|s') = ε/|A|              pour toutes les actions
          + (1 − ε)            pour l'action gloutonne uniquement
```

## Pseudo-code

```
Initialiser Q(s, a) = 0 pour tous (s, a)
Pour chaque épisode :
    Initialiser s
    Pour chaque étape :
        Choisir a ← ε-greedy(Q, s, legal_actions)
        Exécuter a, observer r, s'
        expected_q = Σ_a' π(a'|s') · Q(s', a')
        Q(s, a) ← Q(s, a) + α · [r + γ · expected_q − Q(s, a)]
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

## Comparaison des trois algorithmes TD

| Critère            | SARSA | Expected SARSA | Q-Learning |
|--------------------|-------|----------------|------------|
| Type               | On-policy | On-policy | Off-policy |
| Cible              | Q(s', a') | E[Q(s', ·)] | max Q(s', ·) |
| Variance           | Haute | **Faible** | Haute (biais) |
| Biais de maximisation | Non | Non | **Oui** |
| Coût par step      | O(1) | O(\|A\|) | O(\|A\|) |

## Avantages

- Meilleur compromis biais/variance que SARSA et Q-Learning
- Convergence théoriquement garantie (van Seijen et al., 2009)
- Pas de biais de maximisation contrairement à Q-Learning

## Inconvénients

- Légèrement plus coûteux par step (calcul de l'espérance sur |A| actions)
- Toujours limité par la taille de la Q-table pour les grands espaces

## Références

- van Seijen et al. (2009) — *A Theoretical and Empirical Analysis of Expected Sarsa*
- Sutton & Barto (2018) — chap. 6.6
