# veloce-framework (`velocefw`)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](tests/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](#roadmap)
[![Last commit](https://img.shields.io/github/last-commit/giordiviv/veloce-framework)](https://github.com/giordiviv/veloce-framework/commits/main)
[![A&A](https://img.shields.io/badge/A%26A-VELOCE%20III-b31b1b.svg)](https://doi.org/10.1051/0004-6361/202558106)
<!-- Add once available:
[![CI](https://github.com/giordiviv/veloce-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/giordiviv/veloce-framework/actions/workflows/ci.yml)
-->

**velocefw** is a lightweight Python framework for **building parametric models by
composition, tying or fixing their parameters, and fitting them to data**, with least
squares, maximum a posteriori (MAP) and differential-evolution fitters and data-driven
priors (multivariate KDEs and fast spline densities).

It was developed for **VELOCE III** (Viviani & Anderson, A&A 2026), which reconstructs the
radial-velocity (RV) curves of classical Cepheids from **as few as 3 sparse observations**,
using a principal-component basis learned from the Fourier fits of the VELOCE survey
(Anderson, Viviani et al. 2024). The core is generic, though: any model you can write as
`f(theta, x)` fits in.

## Features

- **Composable, stateless models.** Combine models with `+ - * / **`; the global parameter
  layout (order, unique names, masks) is built for you.
- **Tied and fixed parameters** through a `Parametrization` (identity, fixed values, shared
  groups).
- **Compiled models.** `compile_model` flattens the expression tree once for fast evaluation
  inside optimisers and MCMC loops, and the result is picklable for multiprocessing.
- **Fitters** with uniform results: `fit_least_squares`, `fit_map` and
  `differential_evolution_posterior` return a `SuccessfulFitResult` (parameters, errors,
  statistics) or a `FailedFitResult`.
- **Data-driven priors.** Fit a `KDEDensity` to a population, condition it, speed it up with a
  `SplineDensity`, and attach it to parameters with `ParameterPrior`.
- **FITS I/O helpers** (`get_header`, `get_table`) built on astropy.

## Installation

`velocefw` is not on PyPI yet. Install it from GitHub (Python ≥ 3.12):

```bash
pip install "git+https://github.com/giordiviv/veloce-framework.git"
pip install numpy scipy pandas astropy matplotlib statsmodels   # runtime dependencies
```

Development install:

```bash
git clone https://github.com/giordiviv/veloce-framework.git
cd veloce-framework
pip install -e ".[dev]"
pre-commit install
pytest
```

## Quickstart

Fit a synthetic Cepheid RV curve with a constant (systemic velocity) plus a two-harmonic
Fourier series:

```python
import numpy as np
import velocefw as vf
from velocefw.model import Constant, FourierSeries, calculate_phase

rng = np.random.default_rng(0)
t = np.sort(rng.uniform(0, 60, 40))                  # observation times [d]
phase = calculate_phase(t, period=5.366, epoch=0.0)
rv_err = np.full(t.size, 0.3)                        # [km/s]
rv = -15 + 12*np.cos(2*np.pi*phase) + 5*np.sin(4*np.pi*phase) + rng.normal(0, rv_err)

model = Constant(name="vgamma") + FourierSeries(nharm=2, name="FS")
print(model.layout.names)            # ['c', 'a1', 'b1', 'a2', 'b2']

result = vf.fit.fit_least_squares(model, x=phase, y=rv, yerr=rv_err)
print(result.theta_free)             # ≈ [-15, 12, 0, 0, 5]
print(result.reduced_chi2)           # ≈ 1.3
print(result.extra["theta_err"])     # 1σ uncertainties
```

### Writing your own model

Subclass `BaseModel`, declare the number of parameters, and implement `evaluate`:

```python
import numpy as np
from velocefw.model import BaseModel, FourierSeries

class Line(BaseModel):
    @property
    def n_params(self):
        return 2

    @property
    def param_names(self):
        return ["slope", "intercept"]

    def evaluate(self, theta, x, **kwargs):
        return theta[0] * np.asarray(x, float) + theta[1]

model = Line(name="trend") + FourierSeries(nharm=3)
print(model.layout.names)   # ['slope', 'intercept', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3']
```

### Data-driven priors

```python
from velocefw.prior import KDEDensity, SplineDensity, ParameterPrior

kde = KDEDensity(["log_P", "pc1"]).fit(train_df)    # population of known stars
cond = kde.conditional(log_P=1.2)                   # prior on pc1 for this period
prior = ParameterPrior(SplineDensity(cond), {"pc1": 1})
# pass `logprior=prior` to fit_map or differential_evolution_posterior
```

## Package layout

```
src/velocefw/
├── model/        stateless models, composition, parameter layout, parametrization, compilation
├── fit/          objective function, least-squares / MAP / differential-evolution fitters, results
├── prior/        densities (KDE, conditional KDE, spline, uniform) → log-prior callables
├── read_fits.py  FITS header / table readers (astropy)
└── plots_utils.py  plotting helpers (available as velocefw.plot)
notebooks/        the VELOCE III analysis that uses the framework
tests/            pytest suite
```

## Roadmap

`velocefw` is in **alpha** (`0.0.x`) and the API may change. Next steps:

- [ ] Move the VELOCE III PCA pulsation model from the notebooks into `velocefw.veloce3`, with a
      one-call `reconstruct(t, rv, rv_err, period, epoch)`.
- [ ] Ship the framework tables as package data, and the framework files on Zenodo (with a DOI).
- [ ] Continuous integration (GitHub Actions) on Python 3.12–3.14.
- [ ] Documentation website and a PyPI release.

## Citation

If you use `velocefw` or the VELOCE III framework, please cite:

> Viviani, G. & Anderson, R. I. (2026). *VELOCE III. Reconstructing Radial Velocity Curves of
> Classical Cepheids.* Astronomy & Astrophysics.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
