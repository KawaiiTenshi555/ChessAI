# setup.py — Script de compilation Cython pour le moteur d'échecs ChessIA
#
# Compilation :
#   python setup.py build_ext --inplace
#
# Sur Windows : Visual Studio Build Tools 2022 requis
# Sur Linux/Mac : gcc ou clang suffisent

import os
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("WARNING: Cython not found. Install it with: pip install cython")

if os.name == "nt":
    extra_compile_args = ["/O2", "/GL"]
    extra_link_args = ["/LTCG"]
else:
    extra_compile_args = ["-O3", "-march=native", "-ffast-math"]
    extra_link_args = []

extensions = [
    Extension(
        name="chess_env.board_cy",
        sources=["chess_env/board.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

if HAS_CYTHON:
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
            "initializedcheck": False,
        },
        annotate=True,  # génère board.pyx.html pour inspecter les zones lentes
    )
else:
    ext_modules = []

setup(
    name="ChessIA",
    version="1.0.0",
    ext_modules=ext_modules,
    python_requires=">=3.9",
)
