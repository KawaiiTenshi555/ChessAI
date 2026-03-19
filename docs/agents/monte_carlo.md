# Monte Carlo Control

## Description

Les méthodes Monte Carlo apprennent à partir d'**épisodes complets** en calculant les
retours réels observés. Contrairement aux méthodes TD, il n'y a pas de bootstrap : la
mise à jour utilise le retour vrai G_t et non une estimation Q(s', a').

## Retour actualisé

```
G_t = r_t + γ · r_{t+1} + γ² · r_{t+2} + ... + γ^(T−t) · r_T
```

## Règle de mise à jour (every-visit)

```
Q(s, a) ← Q(s, a) + α · (G_t − Q(s, a))
```

Appliqué à **chaque occurrence** de (s, a) dans l'épisode (every-visit),
ou uniquement à la **première** (first-visit, activé via `first_visit=True`).

## Pseudo-code

```
Initialiser Q(s, a) = 0
Pour chaque épisode :
    Générer épisode complet : s0, a0, r0, s1, a1, r1, ..., sT
    G = 0
    Pour t de T-1 à 0 (sens inverse) :
        G = r_t + γ · G
        [Si first_visit : ignorer si (s_t, a_t) déjà vu dans cet épisode]
        Q(s_t, a_t) ← Q(s_t, a_t) + α · (G − Q(s_t, a_t))
    Décroître ε
```

## Hyperparamètres

| Paramètre       | Défaut | Description |
|-----------------|--------|-------------|
| `alpha`         | 0.05   | Taux d'apprentissage (plus faible car retours à haute variance) |
| `gamma`         | 0.99   | Décompte des récompenses futures |
| `epsilon`       | 1.0    | Exploration initiale |
| `epsilon_min`   | 0.05   | Plancher d'exploration |
| `epsilon_decay` | 0.995  | Multiplicateur de décroissance par épisode |
| `first_visit`   | False  | Si True, n'utilise que la 1ère occurrence de (s, a) par épisode |

## First-Visit vs Every-Visit

| | First-Visit | Every-Visit |
|---|---|---|
| Biais | Non biaisé | Légèrement biaisé |
| Variance | Plus élevée | Plus faible |
| Convergence | Vers Q* | Vers Q* |
| Usage recommandé | Épisodes courts | Épisodes longs avec répétitions |

## Avantages

- **Aucun biais de bootstrap** : utilise les retours réels, pas des estimations
- Ne nécessite pas de modèle de transition (model-free)
- Fonctionne bien sur les épisodes terminaux bien définis (comme les échecs)

## Inconvénients

- **Haute variance** : les retours dépendent de l'entièreté de l'épisode futur
- Doit attendre la **fin de l'épisode** pour apprendre (pas de mise à jour en cours)
- Inefficace sur les longues parties (signal de récompense dilué par γ^T)
- Non applicable aux environnements continus (sans terminaison)

## Conseil pour les échecs

Les parties d'échecs peuvent durer plusieurs centaines de coups, ce qui amplifie
la variance de G_t. Pour améliorer les performances :
- Utiliser un `reward_shaping=True` dans ChessEnv (récompenses intermédiaires)
- Réduire `gamma` légèrement (0.95 plutôt que 0.99)
- Envisager des épisodes plus courts (contrainte de durée)

## Références

- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, chap. 5
- Barto & Duff (1994) — *Monte Carlo Inversion and Reinforcement Learning*
