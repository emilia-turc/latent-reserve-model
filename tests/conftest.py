import numpy as np
import pytest

from reserve_model.io import Recording


@pytest.fixture(scope="session")
def synthetic_recording():
    """80 samples (15.87 s): a MAP drop and a delayed, noisy sigmoid EEG response."""
    rng = np.random.default_rng(1)
    n, dt = 80, 15.873
    t = np.arange(n) * dt
    P = 90 - 30 * np.exp(-0.5 * ((t - t[30]) / 90.0) ** 2) + rng.normal(0, 1, n)
    F = 1 / (1 + np.exp(-0.15 * (P - 80)))
    E = np.zeros(n)
    E[0] = 0.7
    for i in range(1, n):
        target = 0.3 + 0.9 * (F[max(i - 4, 0)] - 0.5)
        E[i] = E[i - 1] + (target - E[i - 1]) * dt / 400
    E += rng.normal(0, 0.05, n)
    return Recording("synthetic", t, P, (E - E.mean()) / E.std())
