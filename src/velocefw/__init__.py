"""Veloce Framework.

A Python package for handling astronomical data and building reconstructing
frameworks. The package was developed during the analysis of VELOCE-III to
build a framework to reconstruct galactic classical Cepheids RV curves based on
the models published in VELOCE (Anderson, Viviani et al. 2024).

The package was then generalized to be used for similar analysis and expanding
the initial published framework, as well as to allow new users a rapid and
hassle-free access to the original VELOCE-III framework.

"""

from velocefw import model
from velocefw.read_fits import get_header, get_table

__all__ = ["get_header", "get_table", "model"]
