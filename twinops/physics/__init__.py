"""
Modelli fisici per il digital twin.

Gerarchia:
  - integrators: integrazione numerica (Euler, Heun, Midpoint, RK4, BackwardEuler)
  - ode: modello base ODE (ODEModel)
  - compose: composizione di modelli (SeriesModel, ParallelModel)
  - library: modelli parametrici pronti (FirstOrderLag, DoubleIntegrator, ...)
"""

# --- Integratori (livello numerico) ---
from twinops.physics.integrators import (
    BackwardEulerIntegrator,
    EulerIntegrator,
    HeunIntegrator,
    MidpointIntegrator,
    RK4Integrator,
    backward_euler_step,
    euler_step,
    heun_step,
    midpoint_step,
    rk4_step,
)

# --- Modello base ODE ---
from twinops.physics.ode import ODEModel

# --- Composizione ---
from twinops.physics.compose import ParallelModel, SeriesModel

# --- Library (modelli parametrici) ---
from twinops.physics.library import (
    DoubleIntegrator,
    FirstOrderLag,
    HarmonicOscillator,
    MassSpringDamper,
    PumpLike,
    TankLevel,
)

__all__ = [
    # Integratori
    "EulerIntegrator",
    "HeunIntegrator",
    "MidpointIntegrator",
    "RK4Integrator",
    "BackwardEulerIntegrator",
    "euler_step",
    "heun_step",
    "midpoint_step",
    "rk4_step",
    "backward_euler_step",
    # Base
    "ODEModel",
    # Composizione
    "SeriesModel",
    "ParallelModel",
    # Library
    "FirstOrderLag",
    "DoubleIntegrator",
    "MassSpringDamper",
    "TankLevel",
    "PumpLike",
    "HarmonicOscillator",
]
