import numpy as np
import pandas as pd

from reserve_model.preprocessing import linear_detrend, preprocess_recording, standardise


def test_linear_detrend_removes_a_line():
    t = np.linspace(0, 100, 50)
    y = 3 + 0.2 * t
    np.testing.assert_allclose(linear_detrend(t, y), 0, atol=1e-10)


def test_standardise_ignores_flat_segments():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 1000)
    x = x + 0.0
    x_flat = np.concatenate([x, np.full(1000, x[-1] + 50.0 * 0)])  # flat (interpolated) tail
    z = standardise(x_flat)
    assert abs(np.mean(z[1:1000])) < 0.1 and abs(np.std(z[1:1000]) - 1) < 0.1


def test_preprocess_recording_columns_and_window():
    n = 20000
    df = pd.DataFrame(
        {
            "time": np.arange(n) / 63.0,
            "PA": 90.0,
            "alpha_power_2": np.random.default_rng(1).normal(size=n).cumsum(),
        }
    )
    out = preprocess_recording(df, start=50, end=250)
    assert list(out.columns) == ["time", "MAP", "alpha"]
    assert out["time"].between(50, 250).all()
    assert abs(np.polyfit(out["time"], out["alpha"], 1)[0]) < 1e-9
