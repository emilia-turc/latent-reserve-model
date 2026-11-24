"""Raw processed signals -> anonymised model inputs (``data/inputs/<patient>.csv``).

NOT runnable from the public repository: it needs the clinical recordings, which cannot
be shared (ethics committee). It documents how the shared files were produced.

Input, per patient: ``<raw-dir>/<patient>_processed.parquet`` with columns ``time`` (s),
``PA`` (MAP) and ``alpha_power_2``; and a CSV of analysis windows with columns
``patient_id, start, end`` (seconds), chosen visually around each MAP challenge before
any model fitting.

Output, per patient: ``time_s`` (re-based to 0 at the start of the window, so that no
recording date/time is released), ``map_mmHg``, ``alpha_z``.

    uv run python scripts/prepare_inputs.py --raw-dir RAW --windows windows.csv --out data/inputs
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from reserve_model.io import Recording
from reserve_model.preprocessing import preprocess_recording

log = logging.getLogger("prepare_inputs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/inputs"))
    ap.add_argument("--eeg-column", default="alpha_power_2")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    for _, w in pd.read_csv(args.windows).iterrows():
        pid = w["patient_id"]
        raw = pd.read_parquet(args.raw_dir / f"{pid}_processed.parquet")
        df = preprocess_recording(raw, w["start"], w["end"], args.eeg_column)
        Recording(
            pid,
            (df["time"] - df["time"].iloc[0]).to_numpy(),
            df["MAP"].to_numpy(),
            df["alpha"].to_numpy(),
        ).to_csv(args.out / f"{pid}.csv")
        log.info("%s: %d samples", pid, len(df))


if __name__ == "__main__":
    main()
