"""
Oscillator systems: modeling of (coupled) harmonic oscillators for simulation.

Provides composed dynamical systems that can be used as physics in TwinSystem or
as dynamics source for Learner. Builds on physics.library.HarmonicOscillator concept
for full multi-oscillator networks.
"""

from typing import Any, Optional

import numpy as np

from twinops.core.component import TwinComponent
from twinops.physics.ode import ODEModel
from twinops.physics.integrators import RK4Integrator


class CoupledHarmonicOscillators(ODEModel):
    """
    Chain of N harmonic oscillators with nearest-neighbor coupling.

    State: [x1, v1, x2, v2, ..., xN, vN] (length 2*N).
    Dynamics: for each oscillator i (position xi, velocity vi),
      dxi/dt = vi
      dvi/dt = -omega_i^2 * xi + coupling * (x_{i+1} - 2*xi + x_{i-1})
    with free endpoints (no coupling beyond boundaries). Optional external input u
    can be applied to first oscillator (e.g. u[0] as force).

    Usable as physics in TwinSystem or with Learner.learn(dynamics=...).
    """

    def __init__(
        self,
        n_oscillators: int,
        omega: float = 1.0,
        coupling: float = 0.1,
        integrator: Optional[Any] = None,
    ) -> None:
        """
        Args:
            n_oscillators: number of oscillators (N).
            omega: natural frequency (rad/s) for each oscillator (uniform).
            coupling: coupling strength between neighbors.
            integrator: ODE integrator (default: RK4Integrator).
        """
        super().__init__(integrator=integrator or RK4Integrator())
        self.n_oscillators = n_oscillators
        self.omega_sq = float(omega) ** 2
        self.coupling = float(coupling)
        self._state_dim = 2 * n_oscillators

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        n = self.n_oscillators
        x = np.asarray(x).ravel()
        if x.size != 2 * n:
            raise ValueError(f"Expected state size 2*{n}, got {x.size}")
        u = np.atleast_1d(u)
        u0 = float(u[0]) if u.size else 0.0

        dxdt = np.empty(2 * n)
        for i in range(n):
            pos_i = x[2 * i]
            vel_i = x[2 * i + 1]
            dxdt[2 * i] = vel_i
            # Coupling: (x_{i-1} - 2*x_i + x_{i+1}), zero at boundaries
            lap = 0.0
            if i > 0:
                lap += x[2 * (i - 1)] - pos_i
            if i < n - 1:
                lap += x[2 * (i + 1)] - pos_i
            force = -self.omega_sq * pos_i + self.coupling * lap
            if i == 0:
                force += u0
            dxdt[2 * i + 1] = force
        return dxdt


if __name__ == "__main__":
    # Create twin with coupled oscillators and EKF, run simulation, visualize with _utils.
    from twinops.core import TwinSystem, TwinHistory
    from twinops.estimation import EKF
    from twinops.simulation._utils import plot_phase_portrait, plot_state_vs_time

    n_osc = 2
    state_dim = 2 * n_osc
    meas_dim = 2  # observe positions x1, x2
    dt = 0.02
    n_steps = 400

    physics = CoupledHarmonicOscillators(
        n_oscillators=n_osc,
        omega=1.0,
        coupling=0.15,
    )
    physics.set_output_fn(lambda x, u: np.array([x[0], x[2]]))  # measure positions

    def f_pred(x: np.ndarray, u: np.ndarray, dtt: float) -> np.ndarray:
        out = physics.step(state=x, u=u, dt=dtt)
        return out["state"]

    ekf = EKF(state_dim=state_dim, meas_dim=meas_dim, f=f_pred)
    twin = TwinSystem(physics=physics, estimator=ekf, dt=dt)

    x0 = np.array([0.5, 0.0, -0.3, 0.0])  # (x1, v1, x2, v2)
    twin.initialize(x0=x0)

    history = TwinHistory()
    u_zero = np.array([0.0])
    for _ in range(n_steps):
        result = twin.step(u=u_zero, measurement=None)
        history.append(state=result.state, time=twin.time)

    try:
        import matplotlib.pyplot as plt
        plot_phase_portrait(history, x_idx=0, y_idx=1, xlabel="x1", ylabel="v1", title="Oscillator 1 — phase portrait")
        plt.tight_layout()
        # plt.savefig("oscillators_phase.png", dpi=120)
        plt.close()
        print("Saved oscillators_phase.png")

        plot_state_vs_time(history, state_names=["x1", "v1", "x2", "v2"], title="Coupled oscillators — state vs time")
        # plt.savefig("oscillators_state_vs_time.png", dpi=120)
        plt.close()
        print("Saved oscillators_state_vs_time.png")
    except ImportError:
        print("matplotlib not available, skip saving plots")
