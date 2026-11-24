"""Recordings and result tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Recording:
    """One preprocessed patient recording, sampled every ~15.87 s.

    time : seconds (origin arbitrary; only differences matter)
    map  : mean arterial pressure (mmHg)
    eeg  : standardised, detrended alpha-band power (z-score units)
    """

    patient_id: str
    time: np.ndarray
    map: np.ndarray
    eeg: np.ndarray

    def __len__(self) -> int:
        return len(self.time)

    @classmethod
    def from_csv(cls, path: str | Path) -> "Recording":
        path = Path(path)
        df = pd.read_csv(path)
        return cls(
            patient_id=path.stem,
            time=df["time_s"].to_numpy(float),
            map=df["map_mmHg"].to_numpy(float),
            eeg=df["alpha_z"].to_numpy(float),
        )

    def to_csv(self, path: str | Path) -> None:
        pd.DataFrame({"time_s": self.time, "map_mmHg": self.map, "alpha_z": self.eeg}).to_csv(
            path, index=False
        )


def load_recordings(directory: str | Path) -> list[Recording]:
    """Load every ``*.csv`` of a directory (one file per patient)."""
    files = sorted(Path(directory).glob("*.csv"), key=lambda p: _natural_key(p.stem))
    if not files:
        raise FileNotFoundError(f"No recording (*.csv) found in {directory}")
    return [Recording.from_csv(f) for f in files]


def _natural_key(name: str):
    """Sort p2 before p10."""
    digits = "".join(c for c in name if c.isdigit())
    return (int(digits) if digits else 0, name)
