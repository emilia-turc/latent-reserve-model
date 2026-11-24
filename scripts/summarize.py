"""Per-patient metrics, pooled R2 and BIC comparison from ``results/``.

    uv run python scripts/summarize.py --results results
    uv run python scripts/summarize.py --results results --previous /path/to/old/simulations

``--previous DIR`` (optional, ``DIR/<model>/simulation_<patient>.parquet`` as written by
the earlier pipeline) compares the new simulated E with the stored one.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reserve_model import metrics
from reserve_model.models import MODELS

ORDER = ["full", "constant_timescale", "sigmoid", "linearized"]


def load_simulations(results: Path, model: str) -> dict[str, pd.DataFrame]:
    folder = results / model / "simulations"
    return {p.stem: pd.read_csv(p) for p in folder.glob("*.csv")} if folder.exists() else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--previous", type=Path, default=None)
    args = ap.parse_args()

    rows, pooled = [], []
    for name in ORDER:
        sims = load_simulations(args.results, name)
        if not sims:
            continue
        k = MODELS[name].k if name != "linearized" else MODELS["full"].k
        per_patient = pd.DataFrame(
            [
                {
                    "model": name,
                    "patient_id": pid,
                    **metrics.patient_metrics(s["E_data"], s["E"], k),
                }
                for pid, s in sims.items()
            ]
        )
        rows.append(per_patient)
        summary = metrics.summarize(per_patient)
        pooled.append(
            {
                "model": name,
                "k": k,
                **summary,
                **metrics.pooled_metrics(
                    {p: (s["E_data"].to_numpy(), s["E"].to_numpy()) for p, s in sims.items()}
                ),
            }
        )
    if not rows:
        raise SystemExit(f"No simulations found in {args.results}")

    per_patient = pd.concat(rows, ignore_index=True)
    table = pd.DataFrame(pooled)

    # Lowest BIC among the three fitted models, patient by patient
    fitted = per_patient[per_patient["model"].isin(["full", "constant_timescale", "sigmoid"])]
    wide = fitted.pivot(index="patient_id", columns="model", values="bic").dropna()
    wins = wide.idxmin(axis=1).value_counts()
    table["lowest_bic"] = (
        table["model"].map(wins).fillna(0).astype(int).where(table["model"] != "linearized")
    )

    per_patient.to_csv(args.results / "per_patient_metrics.csv", index=False)
    table.to_csv(args.results / "summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(table.round(3).to_string(index=False))

    if args.previous is not None:
        print(
            "\nComparison with the previously stored simulations (max |E_new - E_old|, per model):"
        )
        for name in ORDER:
            diffs = []
            for pid, s in load_simulations(args.results, name).items():
                old = args.previous / name / f"simulation_{pid}.parquet"
                if old.exists():
                    e_old = pd.read_parquet(old)["E"].to_numpy()
                    n = min(len(e_old), len(s))
                    diffs.append(np.max(np.abs(e_old[:n] - s["E"].to_numpy()[:n])))
            if diffs:
                print(
                    f"  {name}: {len(diffs)} patients, median {np.median(diffs):.2e}, max {np.max(diffs):.2e}"
                )


if __name__ == "__main__":
    main()
