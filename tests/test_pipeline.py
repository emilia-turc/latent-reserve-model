"""End to end: the fit recovers the delay of a noise-free synthetic sigmoid response."""

import numpy as np

from reserve_model import metrics
from reserve_model.fitting import fit
from reserve_model.io import Recording
from reserve_model.models import MODELS
from reserve_model.simulation import FIT_CONSISTENT, PAPER, simulate


def _synthetic_from_model(delay):
    n, dt = 70, 15.873
    t = np.arange(n) * dt
    P = 90 - 28 * np.exp(-0.5 * ((t - t[35]) / 80.0) ** 2)  # flat for the first ~25 samples
    model = MODELS["sigmoid"]
    params = np.array([500.0, 1.0, 0.0, 80.0, 0.2])
    # Build E on the shifted window by simulating, then embed it into a full-length series.
    steady = params[2] + params[1] * (1 / (1 + np.exp(-params[4] * (P[0] - params[3]))) - 0.5)
    rec = Recording("s", t, P, np.full(n, steady))  # E(t0) = steady state at the initial MAP
    sim = simulate(rec, model, params, delay, FIT_CONSISTENT)
    eeg = np.concatenate([np.full(delay, sim.E[0]), sim.E])
    return Recording("s", t, P, eeg), model


def test_recovers_delay_and_fits_well():
    rec, model = _synthetic_from_model(delay=5)
    result = fit(rec, model, delay_values=[3, 4, 5, 6, 7])
    assert result.delay == 5
    sim = simulate(rec, model, result.params, result.delay, FIT_CONSISTENT)
    assert metrics.r2(sim.E_data, sim.E) > 0.99


def test_paper_and_fit_consistent_settings_differ_only_slightly(synthetic_recording):
    model = MODELS["sigmoid"]
    p = model.initial_guess()
    a = simulate(synthetic_recording, model, p, 4, PAPER)
    b = simulate(synthetic_recording, model, p, 4, FIT_CONSISTENT)
    assert len(a.E) == len(b.E) - 1
    assert np.max(np.abs(a.E - b.E[: len(a.E)])) < 0.05
