# PPO — Proximal Policy Optimization

## Description

PPO (Schulman et al., 2017) est aujourd'hui l'un des algorithmes de reinforcement learning les plus utilisés en pratique. Il combine les avantages des méthodes actor-critic (faible variance grâce à un critique) avec une contrainte de proximité qui empêche des mises à jour trop grandes de la politique — rendant l'entraînement beaucoup plus stable que REINFORCE.

PPO utilise deux réseaux partagés :
- **Actor** : génère une distribution de probabilités sur les actions (politique π_θ)
- **Critic** : estime la valeur de l'état V(s) pour calculer les avantages

## Architecture

```
Observation (8×8×17) → flatten → 1088 entrées
    ↓
Linear(1088 → 512) + ReLU      ← tronc partagé
    ↓
Linear(512 → 256) + ReLU
         ↙              ↘
Actor head            Critic head
Linear(256 → 4096)    Linear(256 → 1)
  (logits d'action)     (valeur V(s))
```

Le masquage des actions illégales est appliqué avant le softmax dans la tête actor.

## Avantage Généralisé (GAE)

PPO utilise GAE (Schulman et al., 2016) pour estimer les avantages, interpolant entre TD(1) (haute variance) et TD(0) (biais élevé) :

```
δ_t   = r_t + γ · V(s_{t+1}) − V(s_t)   (erreur TD)
A_t   = Σ_{k=0}^{T-t} (γλ)^k · δ_{t+k}  (GAE avec λ ∈ [0,1])

λ = 0  →  A_t = δ_t          (faible variance, plus biaisé)
λ = 1  →  A_t = G_t − V(s_t) (haute variance, non biaisé)
```

## Objectif PPO (clipped surrogate)

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)   (ratio de probabilités)

L_CLIP = E_t [ min(r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t) ]

L_V    = MSE(V_θ(s_t), R_t)   (perte du critique)

L_ENT  = H[π_θ(· | s_t)]      (entropie pour l'exploration)

L_total = −L_CLIP + c_v · L_V − c_e · L_ENT
```

Le **clipping** empêche que le ratio `r_t` s'éloigne trop de 1, évitant les mises à jour déstabilisantes.

## Pseudo-code

```
Initialiser π_θ (actor-critic partagé)

Répéter :
    === Phase de collecte (rollout) ===
    Exécuter π_θ pendant rollout_steps étapes
    Stocker (s, a, log_π, r, V(s), done) pour chaque step

    === Phase de mise à jour ===
    Calculer les retours R_t et les avantages A_t (GAE)
    Normaliser A_t : (A_t − mean) / std

    Pour chaque epoch de 1 à ppo_epochs :
        Mélanger les données du rollout
        Pour chaque mini-batch :
            Recalculer log_π_θ(a | s) et V_θ(s) avec le réseau courant
            Calculer r_t = exp(log_π_θ − log_π_old)
            Calculer L_CLIP, L_V, L_ENT
            Minimiser L_total via Adam
            Clipper le gradient (norm ≤ 0.5)
```

## Hyperparamètres

| Paramètre       | Défaut | Description |
|-----------------|--------|-------------|
| `lr`            | 3e-4   | Taux d'apprentissage Adam |
| `gamma`         | 0.99   | Décompte des récompenses futures |
| `gae_lambda`    | 0.95   | Paramètre λ du GAE (0 = TD pur, 1 = MC pur) |
| `clip_eps`      | 0.2    | Seuil de clipping PPO (ε) |
| `entropy_coef`  | 0.01   | Poids du bonus d'entropie (encourage l'exploration) |
| `value_coef`    | 0.5    | Poids de la perte du critique |
| `ppo_epochs`    | 4      | Nombre d'epochs de mise à jour par rollout |
| `batch_size`    | 256    | Taille des mini-batches pendant les epochs |
| `rollout_steps` | 512    | Nombre de steps collectés avant chaque phase de mise à jour |

## Avantages

- Très stable grâce au clipping (pas d'effondrement de politique)
- Efficacité des données bien meilleure que REINFORCE (K epochs par rollout)
- Le critique réduit drastiquement la variance par rapport à Monte Carlo pur
- Gère bien les actions discrètes avec masquage illégal
- Bonus d'entropie intégré pour maintenir l'exploration

## Inconvénients

- On-policy : les données du rollout sont jetées après la mise à jour
- Plus de hyperparamètres à régler que DQN ou REINFORCE
- Le rollout doit être suffisamment long pour que le GAE soit précis
- Moins efficace en données que les méthodes off-policy (DQN, SAC)

## Comparaison globale

| | REINFORCE | DQN | PPO |
|---|---|---|---|
| Type | On-policy PG | Off-policy Value | On-policy Actor-Critic |
| Réseau | 1 (actor) | 1 (Q-network) | 1 (actor + critic partagés) |
| Variance | Très haute | Faible | Faible (GAE) |
| Stabilité | Faible | Moyenne | Haute (clipping) |
| Efficacité données | Faible | Haute (replay) | Moyenne (K epochs) |
| Adapté aux échecs | Moyen | Bon | Très bon |

## Références

- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*, arXiv:1707.06347
- Schulman et al. (2016) — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, chap. 13
