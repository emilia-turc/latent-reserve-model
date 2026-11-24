"""ODE models, their derivatives and parameter bounds.

Notation follows the manuscript (Eqs. R, E, FR, FE, taur_mod, taue_mod):

    tau_R(t) dR/dt = -R + F_R(P(t - delta)),    F_R(P) = sigma(gamma  (P - P50))
    tau_E(R) dE/dt = -E + E_inf + alpha (F_E(R) - 1/2),  F_E(R) = sigma(gamma2 (R - R50))

    tau_R(t) = tau_R [1 - F_R + lambda_R F_R]
    tau_E(R) = tau_E [F_E + lambda_E (1 - F_E)]

The delay ``delta`` is not an ODE parameter: it is implemented by shifting the
MAP and EEG series against each other (see ``fitting.shift_series``), so every
right-hand side below receives the already-delayed pressure P(t).

Parameter vectors are plain arrays whose order is given by ``ModelSpec.param_names``.
The derivatives follow S1 Appendix, "Adjoint sensitivity analysis".

Models
------
full                 11 parameters including the delay (k = 11)
constant_timescale   lambda_R = lambda_E = 1                (k = 9)
sigmoid              delayed sigmoid, first-order EEG only  (k = 6)
linearized           small gamma / large gamma2 limit; simulated, never fitted
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import expit

# ---------------------------------------------------------------------------
# Nonlinearities
# ---------------------------------------------------------------------------


def f_R(P, gamma, P50):
    """Recruitment of reserve by pressure: sigma(gamma (P - P50))."""
    return expit(gamma * (P - P50))


def f_E(R, gamma2, R50):
    """EEG nonlinearity of the reserve: sigma(gamma2 (R - R50))."""
    return expit(gamma2 * (R - R50))


def heaviside(x):
    return 1.0 * (x >= 0)


def p_onset(P50, R50, gamma):
    """Pressure at which the steady-state reserve reaches R50."""
    return P50 + np.log(R50 / (1.0 - R50)) / gamma


# ---------------------------------------------------------------------------
# Full model (and constant time-scale model, obtained with lambda_R = lambda_E = 1)
# ---------------------------------------------------------------------------

FULL_PARAMS = (
    "tau_R", "tau_E", "alpha", "E_inf", "P50", "R50",
    "gamma", "gamma2", "lambda_R", "lambda_E",
)  # fmt: skip


def _full_terms(R, E, P, params):
    """Quantities shared by the right-hand side and its derivatives."""
    tau_R, tau_E, alpha, E_inf, P50, R50, gamma, gamma2, lam_R, lam_E = params
    FR = f_R(P, gamma, P50)
    FE = f_E(R, gamma2, R50)
    tauR_mod = tau_R * ((1 - FR) + lam_R * FR)
    tauE_mod = tau_E * (FE + lam_E * (1 - FE))
    N = -E + E_inf + alpha * (FE - 0.5)
    return FR, FE, tauR_mod, tauE_mod, N


def full_rhs(t, y, P_of_t, params):
    R, E = y
    FR, _, tauR_mod, tauE_mod, N = _full_terms(R, E, P_of_t(t), params)
    return [(FR - R) / tauR_mod, N / tauE_mod]


def full_jacobian(t, y, P_of_t, params):
    """df/dy ('Jacobian with respect to the state')."""
    R, E = y
    _, tau_E, alpha, _, _, R50, _, gamma2, _, lam_E = params
    _, FE, tauR_mod, tauE_mod, N = _full_terms(R, E, P_of_t(t), params)
    c_E = alpha / tauE_mod - tau_E * (1 - lam_E) * N / tauE_mod**2
    df2_dR = c_E * gamma2 * FE * (1 - FE)
    return np.array([[-1.0 / tauR_mod, 0.0], [df2_dR, -1.0 / tauE_mod]])


def full_dparams(time, R, E, P, params):
    """df/dtheta along a trajectory, shape (2, n_params, len(time))."""
    tau_R, tau_E, alpha, _, P50, R50, gamma, gamma2, lam_R, lam_E = params
    FR, FE, tauR_mod, tauE_mod, N = _full_terms(R, E, P, params)
    A_R = FR - R
    c_R = 1.0 / tauR_mod - tau_R * (lam_R - 1) * A_R / tauR_mod**2
    c_E = alpha / tauE_mod - tau_E * (1 - lam_E) * N / tauE_mod**2
    dFR = FR * (1 - FR)
    dFE = FE * (1 - FE)

    d = np.zeros((2, len(FULL_PARAMS), len(time)))
    names = {n: i for i, n in enumerate(FULL_PARAMS)}
    d[0, names["tau_R"]] = -A_R / (tau_R * tauR_mod)
    d[0, names["lambda_R"]] = -tau_R * FR * A_R / tauR_mod**2
    d[0, names["gamma"]] = c_R * (P - P50) * dFR
    d[0, names["P50"]] = -gamma * c_R * dFR
    d[1, names["tau_E"]] = -N / (tau_E * tauE_mod)
    d[1, names["lambda_E"]] = -tau_E * (1 - FE) * N / tauE_mod**2
    d[1, names["alpha"]] = (FE - 0.5) / tauE_mod
    d[1, names["E_inf"]] = 1.0 / tauE_mod
    d[1, names["gamma2"]] = c_E * (R - R50) * dFE
    d[1, names["R50"]] = -gamma2 * c_E * dFE
    return d


# Constant time-scale model: the full model with lambda_R = lambda_E = 1.
CONSTANT_PARAMS = FULL_PARAMS[:8]
_ONES = np.array([1.0, 1.0])


def _with_unit_lambdas(params):
    return np.concatenate([np.asarray(params, dtype=float), _ONES])


def constant_rhs(t, y, P_of_t, params):
    return full_rhs(t, y, P_of_t, _with_unit_lambdas(params))


def constant_jacobian(t, y, P_of_t, params):
    return full_jacobian(t, y, P_of_t, _with_unit_lambdas(params))


def constant_dparams(time, R, E, P, params):
    return full_dparams(time, R, E, P, _with_unit_lambdas(params))[:, : len(CONSTANT_PARAMS), :]


# ---------------------------------------------------------------------------
# Delayed sigmoid model: first-order dynamics of E only (Eq. sigmoid)
# ---------------------------------------------------------------------------

SIGMOID_PARAMS = ("tau_E", "alpha", "E_inf", "P50", "gamma")


def sigmoid_rhs(t, y, P_of_t, params):
    """R is not dynamic in this model: its slot is pinned to its initial value."""
    E = y[1]
    tau_E, alpha, E_inf, P50, gamma = params
    F = f_R(float(P_of_t(t)), gamma, P50)
    return [0.0, (alpha * (F - 0.5) - (E - E_inf)) / tau_E]


def sigmoid_jacobian(t, y, P_of_t, params):
    return np.array([[0.0, 0.0], [0.0, -1.0 / params[0]]])


def sigmoid_dparams(time, R, E, P, params):
    tau_E, alpha, E_inf, P50, gamma = params
    F = f_R(P, gamma, P50)
    N = alpha * (F - 0.5) - (E - E_inf)
    dF = F * (1 - F)
    d = np.zeros((2, len(SIGMOID_PARAMS), len(time)))
    d[1, 0] = -N / tau_E**2
    d[1, 1] = (F - 0.5) / tau_E
    d[1, 2] = 1.0 / tau_E
    d[1, 3] = -alpha * gamma * dF / tau_E
    d[1, 4] = alpha * (P - P50) * dF / tau_E
    return d


# ---------------------------------------------------------------------------
# Linearized model (simulate only)
# ---------------------------------------------------------------------------


def linearized_rhs(t, y, P_of_t, params):
    """Eqs. R_linear, E_low, E_high of the manuscript (full-model parameters)."""
    R, E = y
    tau_R, tau_E, alpha, E_inf, P50, R50, gamma, _, lam_R, lam_E = params
    tau_eff = tau_R * (1 + lam_R) / 2
    H = heaviside(R - R50)
    tauE_mod = tau_E * (H + lam_E * (1 - H))
    dR = (0.5 + gamma / 4 * (float(P_of_t(t)) - P50) - R) / tau_eff
    dE = (-E + E_inf + alpha * (H - 0.5)) / tauE_mod
    return [dR, dE]


# ---------------------------------------------------------------------------
# Registry, bounds and starting point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    name: str
    param_names: tuple[str, ...]
    rhs: Callable
    jacobian: Callable | None  # None for models that are never fitted
    dparams: Callable | None
    bounds: tuple[tuple[float, float], ...]
    fitted: bool = True

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def k(self) -> int:
        """Number of free parameters, counting the delay (used in the BIC)."""
        return self.n_params + 1

    def index(self, name: str) -> int:
        return self.param_names.index(name)

    def initial_guess(self) -> np.ndarray:
        """Common starting vector (preliminary fit to one patient), see COMMON_START."""
        return np.array([COMMON_START[n] for n in self.param_names], dtype=float)


# Common starting parameter vector, obtained from a preliminary fit to one patient
# and used for all patients (reduced models) and as a warm start of the constant
# time-scale model. lambda_R and lambda_E start at the value that reduces the full
# model to the constant time-scale model (see FULL_NESTED_BOUNDS).
COMMON_START = {
    "tau_R": 2.203162,
    "tau_E": 617.2218,
    "alpha": 2.420048,
    "E_inf": 0.1430630,
    "P50": 83.79825,
    "R50": 0.1909968,
    "gamma": 0.1328911,
    "gamma2": 25.07695,
    "lambda_R": 0.3684495,
    "lambda_E": 0.1524404,
}

# Bounds (see Table of parameters in the manuscript)
_BOUNDS = {
    "tau_R": (1, 50),
    "tau_E": (150, 2000),
    "alpha": (0, 5),
    "E_inf": (-1, 3.0),
    "P50": (10, 95),
    "R50": (1e-3, 1),
    "gamma": (1e-3, 10.0),
    "gamma2": (1.0, 500.0),
    "lambda_R": (1e-3, 1.0),
    "lambda_E": (1e-3, 1.0),
}


def _bounds(names):
    return tuple(_BOUNDS[n] for n in names)


MODELS: dict[str, ModelSpec] = {
    "full": ModelSpec(
        "full", FULL_PARAMS, full_rhs, full_jacobian, full_dparams, _bounds(FULL_PARAMS)
    ),
    "constant_timescale": ModelSpec(
        "constant_timescale",
        CONSTANT_PARAMS,
        constant_rhs,
        constant_jacobian,
        constant_dparams,
        _bounds(CONSTANT_PARAMS),
    ),  # fmt: skip
    "sigmoid": ModelSpec(
        "sigmoid",
        SIGMOID_PARAMS,
        sigmoid_rhs,
        sigmoid_jacobian,
        sigmoid_dparams,
        _bounds(SIGMOID_PARAMS),
    ),  # fmt: skip
    "linearized": ModelSpec(
        "linearized",
        FULL_PARAMS,
        linearized_rhs,
        None,
        None,
        _bounds(FULL_PARAMS),
        fitted=False,
    ),  # fmt: skip
}

# The full model is fitted as an extension of the constant time-scale model:
# warm start at its solution with lambda_R = lambda_E = 1, which the bounds contain.
LAMBDA_NESTED_START = (1.0, 1.0)
