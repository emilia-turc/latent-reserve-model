"""Parameter estimation: delay scan + L-BFGS-B on rescaled parameters.

Procedure (manuscript, Methods "Parameter estimation"):
  1. For every candidate delay (1-41 samples, keeping >= 60 % of the recording),
     minimise the objective with loose ODE tolerances from a common start.
  2. Refit the two best delays with tight tolerances, starting from their own
     solution, and keep the best.

The delay d (in samples) pairs P(t_i) with E(t_{i+d}): the MAP series is truncated
at the end and the EEG series at the beginning (``shift_series``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import minimize

from reserve_model.adjoint import LOOSE_TOL, TIGHT_TOL, cost_and_gradient
from reserve_model.io import Recording
from reserve_model.models import LAMBDA_NESTED_START, MODELS, ModelSpec

log = logging.getLogger(__name__)

DELAY_GRID = np.arange(1, 42)  # samples (1 sample = 15.87 s)
MIN_RETAINED_FRACTION = 0.6
N_REFINE = 2
MAXITER = 20000


@dataclass(frozen=True)
class FitResult:
    model: str
    params: np.ndarray
    delay: int  # samples
    cost: float
    converged: bool  # optimiser success flag of the retained fit


def shift_series(time, eeg, pam, delay):
    """Pair pam[i] with eeg[i + delay]; returns (time, eeg, pam) of length n - delay."""
    if delay <= 0 or delay >= len(time):
        raise ValueError("delay must be > 0 and < len(time)")
    return time[:-delay], eeg[delay:], pam[:-delay]


def _run_scaled(fun_and_grad, x0, bounds, max_restarts=1, **options):
    """L-BFGS-B on parameters divided by |x0|, so that the optimiser sees values ~ 1."""
    scale = np.where(x0 != 0, np.abs(x0), 1.0)

    def scaled(s):
        cost, grad = fun_and_grad(s * scale)
        return cost, grad * scale  # chain rule

    scaled_bounds = [(lo / sc, hi / sc) for (lo, hi), sc in zip(bounds, scale)]
    s = x0 / scale
    for _ in range(max_restarts + 1):
        res = minimize(
            scaled, s, jac=True, method="L-BFGS-B", bounds=scaled_bounds, options=options
        )
        if res.success or "ABNORMAL" not in str(res.message):
            break
        s = res.x  # restart once from the last iterate (resets the curvature memory)
    res.x = res.x * scale
    return res


def _fit_one_delay(delay, x0, bounds, model, rec: Recording, ftol, tol):
    time, eeg, pam = shift_series(rec.time, rec.eeg, rec.map, delay)
    fun_and_grad = partial(cost_and_gradient, time=time, eeg=eeg, pam=pam, model=model, tol=tol)
    return _run_scaled(
        lambda p: fun_and_grad(p), x0, bounds,
        maxiter=MAXITER, ftol=ftol, gtol=1e-5, maxls=50,
    )  # fmt: skip


def _finite_or_inf(x):
    return x if np.isfinite(x) else np.inf


def fit(
    rec: Recording,
    model: ModelSpec,
    init_guess=None,
    bounds=None,
    delay_values=None,
) -> FitResult:
    """Fit ``model`` to one recording, scanning the delay."""
    if not model.fitted:
        raise ValueError(f"Model '{model.name}' is simulated, not fitted.")
    x0 = model.initial_guess() if init_guess is None else np.asarray(init_guess, float).copy()
    bounds = model.bounds if bounds is None else bounds
    if len(x0) != len(bounds):
        raise ValueError(f"init_guess has {len(x0)} values but {len(bounds)} bounds.")

    n = len(rec)
    candidates = DELAY_GRID if delay_values is None else delay_values
    delays = [int(d) for d in candidates if 0 < d < n and (n - d) / n >= MIN_RETAINED_FRACTION]
    if not delays:
        raise ValueError(f"{rec.patient_id}: no admissible delay for {n} samples.")

    # Stage 1: delay scan, loose tolerances
    scan = {d: _fit_one_delay(d, x0, bounds, model, rec, 1e-6, LOOSE_TOL) for d in delays}
    best_d = min(scan, key=lambda d: _finite_or_inf(scan[d].fun))
    best = scan[best_d]

    # Stage 2: refine the best delays with tight tolerances
    for d in sorted(scan, key=lambda d: _finite_or_inf(scan[d].fun))[:N_REFINE]:
        res = _fit_one_delay(d, scan[d].x, bounds, model, rec, 1e-9, TIGHT_TOL)
        if _finite_or_inf(res.fun) < _finite_or_inf(best.fun):
            best, best_d = res, d

    log.info("%s [%s]: delay %d samples, cost %.4f", rec.patient_id, model.name, best_d, best.fun)
    return FitResult(model.name, best.x, best_d, float(best.fun), bool(best.success))


def fit_full_nested(rec: Recording, constant_fit: FitResult, half_width: int = 3) -> FitResult:
    """Full model, warm-started at the constant time-scale solution.

    lambda_R = lambda_E = 1 reproduces the constant time-scale model, and the bounds
    contain it, so the full model cannot fit worse because of the optimiser. The
    delay is searched within +/- ``half_width`` samples of the constant model's delay.
    """
    full = MODELS["full"]
    x0 = np.concatenate([constant_fit.params, LAMBDA_NESTED_START])
    delays = range(max(1, constant_fit.delay - half_width), constant_fit.delay + half_width + 1)
    return fit(rec, full, init_guess=x0, delay_values=delays)
