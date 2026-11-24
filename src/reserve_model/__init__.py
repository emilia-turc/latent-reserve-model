"""Latent reserve model linking mean arterial pressure (MAP) to EEG alpha power.

Package layout
--------------
models         ODE models, their Jacobians and parameter bounds
adjoint        objective function and adjoint-based gradient
fitting        delay scan and L-BFGS-B optimisation
simulation     forward simulation of a fitted model on a recording
preprocessing  signal preprocessing (raw -> model inputs)
metrics        R2, bias, RMSE, BIC
io             recordings and result tables
"""

from reserve_model.io import Recording
from reserve_model.models import MODELS, ModelSpec

__all__ = ["MODELS", "ModelSpec", "Recording"]
__version__ = "1.0.0"
