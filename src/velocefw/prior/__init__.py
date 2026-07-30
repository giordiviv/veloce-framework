"""Prior distributions for model parameters.

Typical three-step workflow:

1. **Density** — fit a KDE to data::

       kde = KDEDensity(["log_P", "pc1"]).fit(train_df)

2. **Condition / prepare** — slice or wrap for speed::

       cond = kde.conditional(log_P=1.2)   # ConditionalDensity over pc1
       fast = SplineDensity(cond)          # pre-computed for MCMC

3. **Prior** — connect to model parameters and compose::

       prior = ParameterPrior(fast, {"pc1": 1})
       log_prior = CombinedLogPrior([prior_pc, prior_phase])

"""

from velocefw.prior.density import (
    KDEConditionalDensity,
    KDEDensity,
    SplineDensity,
    UniformDensity,
)
from velocefw.prior.prior import CombinedLogPrior, ParameterPrior

__all__ = [
    "CombinedLogPrior",
    "KDEConditionalDensity",
    "KDEDensity",
    "ParameterPrior",
    "SplineDensity",
    "UniformDensity",
]
