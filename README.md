# reserve_model

Code accompanying the article *A delayed sigmoidal model with first-order dynamics describes EEG alpha-band responses to arterial pressure challenges during general anesthesia* (PLOS Computational Biology, 2026; DOI: [article DOI]).

A latent reserve model links mean arterial pressure (MAP) to the alpha-band power of the
simnplified frontal EEG under general anesthesia: pressure recruits a hidden reserve `R(t)` (first
ODE), which drives alpha power `E(t)` (second ODE). The package fits the model and three
simplified versions to the recordings of a patient, with the delay between MAP and EEG
estimated by a grid search and the other parameters by L-BFGS-B with gradients obtained
from the adjoint method (derivation: S4 Text of the article).

| name in code          | model                                                    | k (with delay) |
|-----------------------|----------------------------------------------------------|----------------|
| `full`                | full model, Eqs. (1)–(4) of the article                  | 11 |
| `constant_timescale`  | `lambda_R = lambda_E = 1`                                | 9  |
| `sigmoid`             | delayed sigmoid, first-order EEG dynamics only           | 6  |
| `linearized`          | small-`gamma`, large-`gamma2` limit; simulated, not fitted | –  |

Parameter names are the symbols of the article: `tau_R, tau_E, alpha, E_inf, P50, R50,
gamma, gamma2, lambda_R, lambda_E`. (Earlier versions of the code used `a, E_level, pam0,
R_cut` for `alpha, E_inf, P50, R50`, and `no_time_modulation` for `constant_timescale`.

## Installation

Python ≥ 3.11 (the article used 3.12) and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                      # core dependencies, versions pinned in uv.lock
uv sync --extra analysis     # + plotting/statistics used for the article's figures
uv run pytest                # unit tests, ~2 min
```

Versions used for the article: see `uv.lock` (numpy, scipy, pandas, pyarrow).

## Data

Raw clinical recordings cannot be shared (ethics committee restriction). `data/inputs/`
contains the **processed, anonymised signals that are the input of the fits and the source
of the figures**: one CSV per patient (`time_s`, `map_mmHg`, `alpha_z`), sampled every
15.87 s, cropped to the window around the MAP challenge, with time re-based to 0.
See `data/README.md`. `scripts/prepare_inputs.py` documents how they were produced from
the raw processed signals (it cannot be run without the restricted data).

## Reproducing the results

```bash
uv run python scripts/fit_all.py --data data/inputs --out results --jobs 4
uv run python scripts/summarize.py --results results
```

`fit_all.py` fits, per patient, the constant time-scale and sigmoid models from a common
starting vector, then the full model warm-started at the constant time-scale solution
(delay searched within ±3 samples of that solution), then simulates the linearized model
with the full-model parameters. Expect several hours for 56 patients on a laptop
(minutes per patient and model); use `--jobs` to parallelise and restart safely (finished
patients are skipped). `summarize.py` writes per-patient R², RMSE, bias, AIC/BIC, the pooled R²
and the number of patients for which each model has the lowest BIC (Table 2 of the article).

The optimisation is deterministic, but results can vary slightly across platforms and
library versions because the delay-scan loss is jagged for some patients.

## Layout

```
src/reserve_model/
  models.py         ODEs, Jacobians df/dy and df/dtheta (S1 Appendix), bounds, starting vector
  adjoint.py        objective (MSE + smooth extrema) and adjoint gradient
  fitting.py        delay scan, L-BFGS-B on rescaled parameters, nested full-model fit
  simulation.py     forward simulation of a fitted model
  preprocessing.py  standardisation, cropping, detrending
  metrics.py        R², bias, RMSE, AIC/BIC
  io.py             Recording and CSV helpers
scripts/            prepare_inputs.py, fit_all.py, summarize.py
tests/              Jacobians, adjoint gradient, preprocessing, end-to-end fit
```

Minimal example:

```python
from reserve_model.io import Recording
from reserve_model.models import MODELS
from reserve_model.fitting import fit
from reserve_model.simulation import simulate
from reserve_model import metrics

rec = Recording.from_csv("data/inputs/p1.csv")
res = fit(rec, MODELS["sigmoid"])  # delay in samples of 15.87 s
sim = simulate(rec, MODELS["sigmoid"], res.params, res.delay)
print(res.delay * 15.87 / 60, "min", metrics.r2(sim.E_data, sim.E))
```

## Citation and license

Please cite the article and the archived version of this software
([Zenodo DOI]); see `CITATION.cff`. MIT license (`LICENSE`).
