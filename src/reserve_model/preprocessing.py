"""Preprocessing of the (already filtered) MAP and alpha-power signals.

The EEG filtering, artefact removal and alpha-power extraction are described in the
manuscript (Methods) and in Turc and Lapointe et al. (2026); they precede this module.
Starting from a table with columns ``time`` (s), ``PA`` (MAP, mmHg) and the alpha power
column, ``preprocess_recording`` applies, in this order, the steps of the manuscript:

  1. downsampling to one sample every 15.87 s (every ``DOWNSAMPLE``-th row),
  2. standardisation of alpha power using only non-quasi-stationary segments,
  3. cropping to the window around the MAP challenge,
  4. subtraction of a least-squares linear trend from alpha power.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DOWNSAMPLE = 1000  # rows of the processed signals per model sample (15.87 s)
STATIONARY_DIFF = 1e-5  # |first difference| below this marks a quasi-stationary segment


def standardise(x: np.ndarray) -> np.ndarray:
    """z-score using mean and s.d. over samples whose first difference is not ~0.

    Flat (quasi-stationary) segments, typically interpolated gaps, are excluded from
    the statistics but not from the output.
    """
    diff = np.diff(x, prepend=np.nan)
    keep = np.abs(diff) >= STATIONARY_DIFF  # False for the NaN of the first sample
    values = np.where(keep, x, np.nan)
    return (x - np.nanmean(values)) / np.nanstd(values)


def linear_detrend(time: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Subtract the least-squares straight line."""
    slope, intercept = np.polyfit(time, y, 1)
    return y - (slope * time + intercept)


def preprocess_recording(
    df: pd.DataFrame,
    start: float,
    end: float,
    eeg_column: str = "alpha_power_2",
    downsample: int = DOWNSAMPLE,
) -> pd.DataFrame:
    """Raw processed signals -> model inputs (columns ``time``, ``MAP``, ``alpha``).

    ``start`` and ``end`` (seconds) delimit the analysis window around the challenge.
    """
    df = df[["time", "PA", eeg_column]].iloc[::downsample].copy()
    df[eeg_column] = standardise(df[eeg_column].to_numpy(float))
    df = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    df[eeg_column] = linear_detrend(df["time"].to_numpy(float), df[eeg_column].to_numpy(float))
    return df.rename(columns={"PA": "MAP", eeg_column: "alpha"}).reset_index(drop=True)
