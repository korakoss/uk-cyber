"""Cost-distribution fitting machinery (shared).

The survey records cost as a BAND, not a number. To turn bands into pounds we
fit a zero-inflated lognormal to the banded data by censored maximum likelihood
(each firm's cost is known only to lie in an interval [L,U)), then read pound
values off the fitted shape. Two readings are provided:

  band_conditional_mean(mu, sigma, band)
      E[X | L <= X < U] — the pound value of a firm KNOWN to be in a given band.
      Used by the semiparametric ("A") estimate, which keeps each firm's own
      observed band and only uses the model to place a value within it.

  bounded_cell_mean(fit)
      The single analytic mean single-incident cost of a whole cell, using the
      FITTED band probabilities (not the empirical counts). Used by the fully
      parametric ("B") estimate, which lets the model set both the band mix and
      the within-band value.

Band 10 is BOUNDED at £100k-£500k throughout (see data.BAND_BOUNDS): the survey
offered higher bands and no firm selected them, so no unbounded lognormal tail
is used here. (The catastrophic >£500k tail is a SEPARATE, explicit sensitivity,
not something this module manufactures.)

Provenance: distilled from the fit/deband code repeated across
src/estimation/national_simulation.py and size_type_cellmean_estimate.py.
"""

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from data import BAND_BOUNDS


@dataclass
class LognormalFit:
    p0: float        # P(band == 1), the no-cost (zero-inflation) mass
    mu: float        # lognormal mu, fit on the cost>0 firms
    sigma: float     # lognormal sigma
    n: int           # total firms with a valid band
    n_nonzero: int   # firms with cost > 0 (bands 2..10)


def _band_probs(mu, sigma, bands=range(2, 11)):
    """P(X in band b) under Lognormal(mu,sigma) for the given bands (bounded)."""
    return np.array([norm.cdf(np.log(BAND_BOUNDS[b][1]), mu, sigma)
                     - norm.cdf(np.log(max(BAND_BOUNDS[b][0], 1)), mu, sigma)
                     for b in bands])


def fit_lognormal(band_codes):
    """Censored-MLE zero-inflated lognormal on integer band codes (1..10).

    Returns a LognormalFit, or None if there are too few cost>0 observations
    (<5) to identify a shape.
    """
    b = np.asarray(band_codes, dtype=float)
    b = b[~np.isnan(b)].astype(int)
    n = len(b)
    if n == 0:
        return None
    counts = np.array([np.sum(b == k) for k in range(1, 11)], dtype=float)
    p0 = counts[0] / n
    cond = counts[1:]  # bands 2..10
    if cond.sum() < 5:
        return None

    def nll(params):
        mu, log_sigma = params
        pred = _band_probs(mu, np.exp(log_sigma))
        s = pred.sum()
        if s < 1e-9:
            return 1e10
        return -(cond * np.log(np.clip(pred / s, 1e-12, 1))).sum()

    best = None
    for mu0 in (5, 7, 9, 11):
        for ls0 in (0.5, 1.0, 1.5):
            r = minimize(nll, [mu0, ls0], method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000})
            if best is None or r.fun < best.fun:
                best = r
    return LognormalFit(p0=p0, mu=best.x[0], sigma=float(np.exp(best.x[1])),
                        n=n, n_nonzero=int(cond.sum()))


def band_conditional_mean(mu, sigma, band):
    """Analytic E[X | L <= X < U] for X ~ Lognormal(mu,sigma). Band 1 -> 0.
    Band 10 is bounded (U = £500k), so this is finite and capped."""
    if band == 1:
        return 0.0
    L, U = BAND_BOUNDS[band]
    a = (np.log(max(L, 1)) - mu) / sigma
    bU = (np.log(U) - mu) / sigma
    denom = norm.cdf(bU) - norm.cdf(a)
    if denom < 1e-12:
        return float(np.exp(mu + sigma ** 2 / 2))
    num = norm.cdf(bU - sigma) - norm.cdf(a - sigma)
    return float(np.exp(mu + sigma ** 2 / 2) * num / denom)


def bounded_cell_mean(fit: LognormalFit):
    """Single analytic mean single-incident cost of a cell (£), fully parametric.

    = (1 - p0) * sum_b [ P_fit(band b | cost>0) * E[X | band b] ]
    using the FITTED band probabilities and BOUNDED band conditional means.
    Includes the zero-cost mass via (1 - p0).
    """
    w = _band_probs(fit.mu, fit.sigma)
    w = w / w.sum()
    nonzero_mean = sum(wi * band_conditional_mean(fit.mu, fit.sigma, b)
                       for wi, b in zip(w, range(2, 11)))
    return (1 - fit.p0) * nonzero_mean
