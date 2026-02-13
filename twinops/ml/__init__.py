"""Machine learning: residual correctors, neural dynamics, and training."""

from twinops.ml.dynamics import NeuralDynamicsModel, default_dynamics_net
from twinops.ml.residual import TorchResidualModel
from twinops.ml.training import (
    compute_dx_dt_central,
    prepare_ode_data_from_timeseries,
    train_dynamics,
    train_neural_ode,
    train_residual,
)

__all__ = [
    "TorchResidualModel",
    "NeuralDynamicsModel",
    "default_dynamics_net",
    "train_residual",
    "train_neural_ode",
    "train_dynamics",
    "prepare_ode_data_from_timeseries",
    "compute_dx_dt_central",
]
