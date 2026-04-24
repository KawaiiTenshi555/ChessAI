# README — Moteur Cython (board_cy)

Ce document explique comment compiler et utiliser la version Cython du moteur
d'échecs, ainsi que comment revenir au Python pur si nécessaire.

---

## Prérequis

### Python
```
pip install cython numpy
```

### Windows
Visual Studio Build Tools 2022 est requis (compilateur C MSVC).

```
winget install Microsoft.VisualStudio.2022.BuildTools
```

Dans le Visual Studio Installer, cocher **"Desktop development with C++"**.

### Linux / macOS
`gcc` ou `clang` suffisent (généralement déjà installés).

---

## Compilation

Depuis la racine du projet :

```bash
python setup.py build_ext --inplace
```

Cela génère un fichier `.pyd` (Windows) ou `.so` (Linux/macOS) dans
`chess_env/` (par exemple `chess_env/board_cy.cpython-311-win_amd64.pyd`).

Le script génère aussi `chess_env/board.pyx.html` — un fichier annoté qui
permet de visualiser les lignes lentes (fond jaune = appel Python, blanc =
code C pur).

---

## Vérification

```python
from chess_env._import_helper import ChessBoard, ENGINE
print(ENGINE)   # "cython" si compilé, "python" sinon
```

Ou directement :

```python
try:
    import chess_env.board_cy
    print("Cython actif")
except ImportError:
    print("Python pur (board_cy non compilé)")
```

---

## Utilisation

Le module `chess_env._import_helper` choisit automatiquement l'implémentation
disponible :

```python
from chess_env._import_helper import ChessBoard, Move, WHITE, BLACK, QUEEN

board = ChessBoard()
for move in board.get_legal_moves():
    print(move.to_uci())
```

Les agents, l'environnement Gymnasium (`ChessEnv`) et la web app continuent
de fonctionner sans modification grâce au fallback dans `chess_env.py`.

---

## Revenir au Python pur

Si vous souhaitez forcer le Python pur (débogage, CI sans compilateur) :

**Option 1** — Ne pas compiler (ne pas exécuter `setup.py`).

**Option 2** — Supprimer le `.pyd` / `.so` compilé :

```bash
# Windows
del chess_env\board_cy*.pyd

# Linux / macOS
rm chess_env/board_cy*.so
```

Le fallback dans `chess_env/_import_helper.py` utilisera alors `board.py`
automatiquement.

---

## Tests de parité

Pour vérifier que Cython et Python donnent exactement les mêmes résultats :

```bash
pytest tests/test_cython_parity.py -v
```

Les tests sont automatiquement ignorés (`skipif`) si `board_cy` n'est pas
compilé.

---

## Benchmark

```bash
python tools/benchmark_engine.py
python tools/benchmark_engine.py --positions 2000 --warmup 100
```

Exemple de résultat attendu sur un PC moderne :

```
Python pur : ~300 µs / appel
Cython     :  ~20 µs / appel
Speedup    :  ~15×
```

---

## Gains de performance attendus

| Opération           | Python pur | Cython typé |
|---------------------|-----------|-------------|
| `get_legal_moves()` | ~300 µs   | ~20–30 µs   |
| `env.step()`        | ~2 ms     | ~0.2 ms     |
| `board.copy()`      | ~20 µs    | ~1 µs       |
| Steps entraîn./s    | ~500      | ~4 000–8 000 |

---

## Notes Windows

- La DLL compilée est spécifique à la version de Python (ex. 3.11 x64).
- Si vous changez de version Python, recompilez.
- En cas d'erreur `Microsoft Visual C++ 14.0 or greater is required`,
  installez ou réparez les Build Tools.
- Vous pouvez vérifier la disponibilité du compilateur avec :
  ```
  python -c "import distutils.msvc9compiler"
  ```
  ou
  ```
  cl.exe
  ```
  depuis une "Developer Command Prompt for VS 2022".
