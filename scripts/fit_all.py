"""Fit the models to every patient (one CSV per patient in ``--data``).

Order: constant time-scale and sigmoid models (independent), then the full model,
warm-started at the constant time-scale solution, then the linearized model, which is
only simulated with the full-model parameters.

    uv run python scripts/fit_all.py --data data/inputs --out results --jobs 4

Outputs, for each model in ``results/<model>/``: ``parameters.csv`` (patient, delay in
samples, cost, converged, parameters) and ``simulations/<patient>.csv`` (time, R, E,
E_data). Patients already present in ``parameters.csv`` are skipped, so the script can
be interrupted and restarted. Fitting takes minutes per patient and model.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from reserve_model.fitting import FitResult, fit, fit_full_nested
from reserve_model.io import Recording, load_recordings
from reserve_model.models import MODELS
from reserve_model.simulation import SETTINGS, simulate

log = logging.getLogger("fit_all")


# -- result tables -----------------------------------------------------------------


def load_fits(out: Path, model: str) -> dict[str, FitResult]:
    path = out / model / "parameters.csv"
    if not path.exists():
        return {}
    spec = MODELS[model]
    df = pd.read_csv(path)
    return {
        r.patient_id: FitResult(
            model,
            r[list(spec.param_names)].to_numpy(float),
            int(r.delay),
            r.cost,
            bool(r.converged),
        )
        for _, r in df.iterrows()
    }


def save_fit(out: Path, rec: Recording, fit_: FitResult, settings) -> None:
    spec = MODELS[fit_.model]
    sim = simulate(rec, spec, fit_.params, fit_.delay, settings)
    folder = out / fit_.model
    (folder / "simulations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": sim.time, "R": sim.R, "E": sim.E, "E_data": sim.E_data}).to_csv(
        folder / "simulations" / f"{rec.patient_id}.csv", index=False
    )
    row = {
        "patient_id": rec.patient_id,
        "delay": fit_.delay,
        "cost": fit_.cost,
        "converged": fit_.converged,
    }
    row.update(dict(zip(spec.param_names, fit_.params)))
    path = folder / "parameters.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


# -- workers -----------------------------------------------------------------------


def _fit_reduced(args):
    rec, model = args
    return rec.patient_id, fit(rec, MODELS[model])


def _fit_full(args):
    rec, constant_fit = args
    return rec.patient_id, fit_full_nested(rec, constant_fit)


def run_stage(pool, worker, jobs, recs, out, model, settings):
    done = load_fits(out, model)
    todo = [j for j in jobs if j[0].patient_id not in done]
    log.info("== %s: %d to fit, %d already done", model, len(todo), len(done))
    by_id = {r.patient_id: r for r in recs}
    for pid, result in pool.map(worker, todo):
        save_fit(out, by_id[pid], result, settings)
        done[pid] = result
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/inputs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--jobs", type=int, default=1, help="patients fitted in parallel")
    ap.add_argument(
        "--simulation",
        choices=SETTINGS,
        default="paper",
        help="settings of the evaluation simulations (see README)",
    )
    ap.add_argument("--patients", nargs="*", help="restrict to these patient ids")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    settings = SETTINGS[args.simulation]

    recs = load_recordings(args.data)
    if args.patients:
        recs = [r for r in recs if r.patient_id in args.patients]
    log.info("%d patients", len(recs))

    with ProcessPoolExecutor(args.jobs) as pool:
        constant = run_stage(
            pool,
            _fit_reduced,
            [(r, "constant_timescale") for r in recs],
            recs,
            args.out,
            "constant_timescale",
            settings,
        )
        run_stage(
            pool, _fit_reduced, [(r, "sigmoid") for r in recs], recs, args.out, "sigmoid", settings
        )
        full = run_stage(
            pool,
            _fit_full,
            [(r, constant[r.patient_id]) for r in recs if r.patient_id in constant],
            recs,
            args.out,
            "full",
            settings,
        )

    # Linearized model: simulated with the full-model parameters, never fitted.
    done = load_fits(args.out, "linearized")
    for rec in recs:
        if rec.patient_id in full and rec.patient_id not in done:
            f = full[rec.patient_id]
            save_fit(
                args.out, rec, FitResult("linearized", f.params, f.delay, np.nan, True), settings
            )
    log.info("done")


if __name__ == "__main__":
    main()
