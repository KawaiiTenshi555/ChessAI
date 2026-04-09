# AlphaZero

## Vue d'ensemble

Cet agent combine :

- un réseau politique-valeur qui prédit les coups prometteurs
- une recherche MCTS guidée par ces prédictions
- un entraînement par self-play

L'implémentation du projet reste volontairement légère pour tourner dans l'environnement actuel, sans pipeline distribué.

## Hyperparamètres principaux

- `mcts_simulations` : nombre de simulations MCTS par coup
- `c_puct` : équilibre exploration / exploitation dans l'arbre
- `temperature` : diversité des coups joués en self-play
- `temperature_drop_move` : nombre de demi-coups avant jeu quasi-déterministe
- `dirichlet_alpha` : forme du bruit Dirichlet à la racine
- `dirichlet_epsilon` : poids du bruit Dirichlet
- `lr` : learning rate du réseau
- `weight_decay` : régularisation L2
- `batch_size` : taille des mini-batchs d'entraînement
- `training_batches_per_episode` : nombre de mises à jour après chaque partie
- `max_game_length` : borne de sécurité pour arrêter une partie trop longue

## Intégration projet

- Agent disponible côté backend sous `alphazero`
- Coup en partie joué via `select_move(board)`
- Entraînement web effectué en self-play
- Les options de reward shaping sont masquées dans l'UI quand AlphaZero est sélectionné

## Limites

- Réseau MLP simple, sans résidus ni convolutions
- Self-play local mono-processus
- Observation compacte adaptée à l'architecture actuelle du projet
