"""Exactness of the Jacobians df/dy and df/dtheta against finite differences."""

import numpy as np
import pytest

from reserve_model.models import MODELS, f_R, p_onset

FITTED = ["full", "constant_timescale", "sigmoid"]


def _setup(name):
    model = MODELS[name]
    rng = np.random.default_rng(0)
    params = model.initial_guess() * np.exp(rng.normal(0, 0.1, model.n_params))
    P = lambda t: 82.0 + 0.0 * t  # noqa: E731
    return model, params, P


@pytest.mark.parametrize("name", FITTED)
def test_state_jacobian(name):
    model, params, P = _setup(name)
    y = np.array([0.4, 0.3])
    J = model.jacobian(0.0, y, P, params)
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fd = (
            np.array(model.rhs(0.0, y + e, P, params)) - np.array(model.rhs(0.0, y - e, P, params))
        ) / (2 * h)
        np.testing.assert_allclose(J[:, j], fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("name", FITTED)
def test_parameter_derivatives(name):
    model, params, _ = _setup(name)
    R, E, Pval = np.array([0.4]), np.array([0.3]), np.array([82.0])
    d = model.dparams(np.zeros(1), R, E, Pval, params)[:, :, 0]
    for i in range(model.n_params):
        h = 1e-6 * max(abs(params[i]), 1)
        up, dn = params.copy(), params.copy()
        up[i] += h
        dn[i] -= h
        fd = (
            np.array(model.rhs(0.0, [0.4, 0.3], lambda t: 82.0, up))
            - np.array(model.rhs(0.0, [0.4, 0.3], lambda t: 82.0, dn))
        ) / (2 * h)
        np.testing.assert_allclose(d[:, i], fd, rtol=1e-4, atol=1e-9, err_msg=model.param_names[i])


def test_constant_timescale_is_full_with_unit_lambdas():
    full, constant = MODELS["full"], MODELS["constant_timescale"]
    p8 = constant.initial_guess()
    P = lambda t: 75.0  # noqa: E731
    a = constant.rhs(0.0, [0.5, 0.2], P, p8)
    b = full.rhs(0.0, [0.5, 0.2], P, np.concatenate([p8, [1.0, 1.0]]))
    np.testing.assert_allclose(a, b)


def test_p_onset_is_where_steady_state_reserve_equals_R50():
    P50, R50, gamma = 84.0, 0.2, 0.13
    assert f_R(p_onset(P50, R50, gamma), gamma, P50) == pytest.approx(R50)
