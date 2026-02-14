"""
Simulation: domain-specific complex dynamical systems.

Each submodule (_oscillators, _multibody, _electrical_networks, etc.) provides
models for that domain. Use _utils for visualization (phase portrait, time series, animation).
"""

from twinops.simulation._oscillators import CoupledHarmonicOscillators
from twinops.simulation._utils import (
    plot_phase_portrait,
    plot_state_vs_time,
)

__all__ = [
    "CoupledHarmonicOscillators",
    "plot_phase_portrait",
    "plot_state_vs_time",
]
