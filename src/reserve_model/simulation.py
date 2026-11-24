"""Forward simulation of a model on a recording, with the fitted parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from reserve_model.adjoint import TIGHT_TOL, initial_state
from reserve_model.fitting import shift_series
from reserve_model.io import Recording
from reserve_model.models import ModelSpec


@dataclass(frozen=True)
class Simulation:
    time: np.ndarray  # time axis of the shifted series (seconds)
    R: np.ndarray
    E: np.ndarray  # simulated alpha power
    E_data: np.ndarray  # empirical alpha power, same samples
    delay: int
    start_index: int  # index in the recording of the first simulated EEG sample (= delay)


@dataclass(frozen=True)
class SimulationSettings:
    """How a fitted model is simulated for evaluation.

    interpolation : interpolation of MAP between samples.
    tol           : solve_ivp tolerances.
    drop_last     : number of trailing samples of the fitted window left out.
    """

    interpolation: str
    tol: dict
    drop_last: int


# Settings that produced the figures and tables of the manuscript. They differ from
# the fitting procedure in three minor ways (cubic MAP interpolation, scipy's default
# tolerances, last sample of the window not simulated).
PAPER = SimulationSettings("cubic", {"rtol": 1e-3, "atol": 1e-6}, drop_last=1)

# Simulation exactly on the window and with the interpolation / tolerances of the
# final stage of the fit.
FIT_CONSISTENT = SimulationSettings("linear", TIGHT_TOL, drop_last=0)

SETTINGS = {"paper": PAPER, "fit-consistent": FIT_CONSISTENT}


def simulate(
    rec: Recording,
    model: ModelSpec,
    params,
    delay: int,
    settings: SimulationSettings = PAPER,
) -> Simulation:
    """Integrate ``model`` with ``params`` over the window fitted with ``delay``."""
    time, eeg, pam = shift_series(rec.time, rec.eeg, rec.map, delay)
    if settings.drop_last:
        time, eeg, pam = (a[: -settings.drop_last] for a in (time, eeg, pam))
    pam_of_t = interp1d(
        time, pam, kind=settings.interpolation, bounds_error=False, fill_value="extrapolate"
    )
    sol = solve_ivp(
        model.rhs, (time[0], time[-1]), initial_state(eeg), t_eval=time,
        method="BDF", args=(pam_of_t, params), **settings.tol,
    )  # fmt: skip
    if not sol.success:
        raise RuntimeError(f"{rec.patient_id}: simulation failed ({sol.message})")
    return Simulation(sol.t, sol.y[0], sol.y[1], eeg, delay, start_index=delay)
