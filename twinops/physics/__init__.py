"""
Physics models for the digital twin.

Hierarchy:
  - integrators: numerical integration (Euler, Heun, Midpoint, RK4, BackwardEuler)
  - ode: base ODE model (ODEModel)
  - symbolic: ODE model learned from data (SymbolicODEModel, requires gplearn)
  - neural_ode: ODE with rhs from neural network (NeuralODEModel, requires torch)
  - compose: model composition (SeriesModel, ParallelModel)
  - library: ready-made parametric models (FirstOrderLag, DoubleIntegrator, ...)
"""

# --- Integrators (numerical level) ---
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

# --- Base ODE model ---
from twinops.physics.ode import ODEModel

# --- Symbolic (physics model from data) ---
from twinops.physics.symbolic import SymbolicODEModel

# --- Neural ODE (physics from NN) ---
try:
    from twinops.physics.neural_ode import NeuralODEModel
except ImportError:
    NeuralODEModel = None  # type: ignore[misc, assignment]

# --- Composition ---
from twinops.physics.compose import ParallelModel, SeriesModel

# --- Library (parametric models) ---
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
    # Symbolic
    "SymbolicODEModel",
    # Neural ODE
    "NeuralODEModel",
    # Composition
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
