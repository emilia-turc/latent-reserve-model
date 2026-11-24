"""The adjoint gradient agrees with finite differences of the objective."""

import numpy as np
import pytest

from reserve_model.adjoint import TIGHT_TOL, cost_and_gradient, dsoftmax, dsoftmin, softmax, softmin
from reserve_model.fitting import shift_series
from reserve_model.models import MODELS


def test_smooth_extrema_derivatives():
    E = np.random.default_rng(0).normal(size=15)
    for f, df in [(softmax, dsoftmax), (softmin, dsoftmin)]:
        fd = np.array(
            [(f(E + 1e-7 * np.eye(15)[i]) - f(E - 1e-7 * np.eye(15)[i])) / 2e-7 for i in range(15)]
        )
        np.testing.assert_allclose(df(E), fd, atol=1e-5)


@pytest.mark.parametrize("name", ["full", "constant_timescale", "sigmoid"])
def test_adjoint_gradient_matches_finite_differences(name, synthetic_recording):
    rec = synthetic_recording
    model = MODELS[name]
    time, eeg, pam = shift_series(rec.time, rec.eeg, rec.map, 4)
    p = model.initial_guess()

    def cost(q):
        return cost_and_gradient(q, time, eeg, pam, model, TIGHT_TOL)[0]

    _, grad = cost_and_gradient(p, time, eeg, pam, model, TIGHT_TOL)
    fd = np.array(
        [
            (cost(p + h * e) - cost(p - h * e)) / (2 * h)
            for e, h in zip(np.eye(len(p)), 1e-5 * np.maximum(np.abs(p), 1))
        ]
    )
    # The impulses of the adjoint equation are represented by piecewise-linear sources
    # (documented approximation), so agreement is in direction, not to machine precision.
    cosine = grad @ fd / (np.linalg.norm(grad) * np.linalg.norm(fd))
    assert cosine > 0.99
    big = np.abs(fd) > 0.05 * np.abs(fd).max()
    assert np.all(np.sign(grad[big]) == np.sign(fd[big]))
