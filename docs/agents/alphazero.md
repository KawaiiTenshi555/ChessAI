# AlphaZero

## Vue d'ensemble

Cet agent combine :

- un reseau politique-valeur qui predit les coups prometteurs
- une recherche MCTS guidee par ces predictions
- un entrainement par self-play

L'implementation du projet reste volontairement legere pour tourner dans l'environnement actuel, sans pipeline distribue.

## Hyperparametres principaux

- `mcts_simulations` : nombre de simulations MCTS par coup
- `c_puct` : equilibre exploration / exploitation dans l'arbre
- `temperature` : diversite des coups joues en self-play
- `temperature_drop_move` : nombre de demi-coups avant jeu quasi-deterministe
- `dirichlet_alpha` : forme du bruit Dirichlet a la racine
- `dirichlet_epsilon` : poids du bruit Dirichlet
- `lr` : learning rate du reseau
- `weight_decay` : regularisation L2
- `batch_size` : taille des mini-batchs d'entrainement
- `training_batches_per_episode` : nombre de mises a jour apres chaque partie
- `max_game_length` : borne de securite pour arreter une partie trop longue

## Integration projet

- Agent disponible cote backend sous `alphazero`
- Coup en partie joue via `select_move(board)`
- Entrainement web effectue en self-play
- Les options de reward shaping sont masquees dans l'UI quand AlphaZero est selectionne

## Limites

- Reseau MLP simple, sans residus ni convolutions
- Self-play local mono-processus
- Observation compacte adaptee a l'architecture actuelle du projet

## Benchmark local contre Stockfish

Un benchmark local UCI est disponible via `benchmark/stockfish.py`.

Exemple :

```bash
python -m benchmark.stockfish --agent alphazero --games 20 --stockfish-elo 1200 --stockfish-path C:\path\to\stockfish.exe
```

Le benchmark :

- lance Stockfish en local via un sous-processus UCI
- alterne automatiquement les couleurs blanc / noir
- calcule le score `W/L/D`
- estime un Elo de l'agent a partir du score obtenu contre le niveau Elo demande a Stockfish

Sortie attendue :

- `W/L/D`
- `Ratio W/L`
- `Elo estime`

Notes :

- aucune connexion Internet n'est necessaire
- si `--stockfish-path` n'est pas fourni, le script essaie de detecter `stockfish` localement
- `--depth` peut etre utilise a la place de `--movetime-ms`
