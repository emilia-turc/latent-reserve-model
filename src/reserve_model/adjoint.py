"""Objective function and its gradient by the adjoint method.

The objective is

    J = 1/N sum_i (E_i - E_data_i)^2
        + 1/2 (softmax(E) - max E_data)^2 + 3/2 (softmin(E) - min E_data)^2 ,

and its gradient is obtained from a single backward integration (S1 Appendix):

    dmu/dt = -(df/dy)^T mu - dL/dy,   mu(T) = 0,     dJ/dtheta = int mu^T df/dtheta dt.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson, solve_ivp

from reserve_model.models import ModelSpec

BETA = 200.0  # sharpness of the smooth extrema
W_MAX, W_MIN = 0.5, 1.5  # weights of the extrema penalties

# ODE solver tolerances: loose for the delay scan, tight for the final refinement.
LOOSE_TOL = {"rtol": 1e-5, "atol": 1e-8}
TIGHT_TOL = {"rtol": 1e-8, "atol": 1e-10}

FAILED_COST = 1e12  # returned when the forward solve fails or overflows


def softmax(E, beta=BETA):
    """Smooth maximum: exponentially weighted mean (S1 Appendix, Eq. smooth_extrema)."""
    w = np.exp(beta * (E - E.max()))
    return np.sum(E * w) / np.sum(w)


def softmin(E, beta=BETA):
    w = np.exp(-beta * (E - E.min()))
    return np.sum(E * w) / np.sum(w)


def dsoftmax(E, beta=BETA):
    """d softmax / d E_i."""
    w = np.exp(beta * (E - E.max()))
    Z = np.sum(w)
    return (w / Z) * (1 + beta * (E - np.sum(E * w) / Z))


def dsoftmin(E, beta=BETA):
    w = np.exp(-beta * (E - E.min()))
    Z = np.sum(w)
    return (w / Z) * (1 - beta * (E - np.sum(E * w) / Z))


def objective(E_sim, E_data):
    """Objective J and its derivative g_i = dJ/dE_i (S1 Appendix, Eq. gi)."""
    n = len(E_data)
    s_max, s_min = softmax(E_sim), softmin(E_sim)
    d_max = s_max - E_data.max()
    d_min = s_min - E_data.min()
    J = np.sum((E_sim - E_data) ** 2) / n + W_MAX * d_max**2 + W_MIN * d_min**2
    g = (
        2 * (E_sim - E_data) / n
        + 2 * W_MAX * d_max * dsoftmax(E_sim)
        + 2 * W_MIN * d_min * dsoftmin(E_sim)
    )
    return J, g


def initial_state(eeg):
    """R(t0) = 1 (full reserve); E(t0) = mean of the first ten samples."""
    return [1.0, float(np.mean(eeg[:10]))]


def cost_and_gradient(params, time, eeg, pam, model: ModelSpec, tol=LOOSE_TOL):
    """Objective value and gradient with respect to ``params``.

    ``pam`` is already aligned with ``eeg`` (the delay is applied by shifting the
    series, see ``fitting.shift_series``) and linearly interpolated between samples.
    """
    fail = (FAILED_COST, np.full(len(params), 1e2))

    def pam_of_t(t):
        return np.interp(t, time, pam)

    def forward_jac(t, y, pam_of_t, params):
        return model.jacobian(t, y, pam_of_t, params)

    sol = solve_ivp(
        model.rhs, (time[0], time[-1]), initial_state(eeg), method="BDF",
        jac=forward_jac, dense_output=True, args=(pam_of_t, params), **tol,
    )  # fmt: skip
    if not sol.success:
        return fail
    R_sim, E_sim = sol.sol(time)
    if not np.all(np.isfinite(E_sim)):
        return fail

    cost, g = objective(E_sim, eeg)
    if not np.isfinite(cost):
        return fail

    # Backward (adjoint) integration. The impulses g_i delta(t - t_i) are
    # represented by their piecewise-linear interpolation; the resulting gradient
    # is divided by the mean sampling interval (S1 Appendix, numerical details).
    def adjoint_rhs(t, mu):
        R, E = sol.sol(t)
        J = model.jacobian(t, [R, E], pam_of_t, params)
        return -J.T @ mu - np.array([0.0, np.interp(t, time, g)])

    def adjoint_jac(t, mu):
        R, E = sol.sol(t)
        return -model.jacobian(t, [R, E], pam_of_t, params).T

    sol_adj = solve_ivp(
        adjoint_rhs, (time[-1], time[0]), [0.0, 0.0], method="BDF",
        t_eval=time[::-1], jac=adjoint_jac, **tol,
    )  # fmt: skip
    mu = sol_adj.y[:, ::-1]  # (2, n), increasing time

    df_dtheta = model.dparams(time, R_sim, E_sim, pam, params)  # (2, p, n)
    integrand = np.einsum("st,spt->pt", mu, df_dtheta)
    grad = np.array([simpson(row, x=time) for row in integrand])
    grad /= np.mean(np.diff(time))
    return cost, grad
