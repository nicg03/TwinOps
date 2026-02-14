"""
Machine learning: apprendimento della dinamica del sistema (modello surrogato).

Struttura logica:
- surrogate: concetto di dinamica surrogata (modello da dati, usabile come physics).
- dynamics: NeuralDynamicsModel (x_{k+1} = net(x_k, u_k, dt)).
- training: train_dynamics, train_neural_ode, preparazione dati.
- learn(): entry-point dati -> modello surrogato (TwinComponent) per twin e simulazione.
"""

from twinops.ml.dynamics import NeuralDynamicsModel, default_dynamics_net
from twinops.ml.residual import TorchResidualModel
from twinops.ml.training import (
    compute_dx_dt_central,
    prepare_ode_data_from_timeseries,
    train_dynamics,
    train_neural_ode,
    train_residual,
)


def learn(*args, **kwargs):
    """
    Entry-point: dati (serie temporali o (states, inputs, next_states)) -> modello surrogato.

    Restituisce un TwinComponent (es. NeuralDynamicsModel) utilizzabile come physics
    in TwinSystem per creare/importare il twin e simulare la dinamica.

    Struttura logica: implementazione futura unificata.
    Per ora usare: default_dynamics_net + NeuralDynamicsModel + train_dynamics().
    """
    raise NotImplementedError(
        "learn() è la struttura logica per dati -> surrogato. "
        "Usare: default_dynamics_net(), NeuralDynamicsModel(), train_dynamics()."
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
    "learn",
]
