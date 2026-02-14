"""
Esempio minimo: digital twin pompa con fisica ODE + EKF e health indicator.
"""

import sys
from pathlib import Path

import numpy as np

# Aggiungi root repository (TwinOps) al path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from twinops.core import TwinSystem, TwinHistory
from twinops.physics import ODEModel, RK4Integrator
from twinops.estimation import EKF
from twinops.health import HealthIndicator
from twinops.io import BatchStream


class PumpPhysics(ODEModel):
    """Pompa semplificata: stato [portata, pressione], ingresso [velocità]."""

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        q, p = x[0], x[1]
        omega = u[0] if u.size else 1.0
        # Modello semplificato
        dq = -0.1 * q + 0.5 * omega
        dp = 0.3 * q - 0.2 * p
        return np.array([dq, dp])


def main() -> None:
    dt = 0.01
    state_dim = 2
    meas_dim = 1
    input_dim = 1

    physics = PumpPhysics(integrator=RK4Integrator())
    physics.set_output_fn(lambda x, u: np.array([x[0]]))

    def f_pred(x: np.ndarray, u: np.ndarray, _dt: float) -> np.ndarray:
        return physics.step(state=x, u=u, dt=_dt)["state"]

    ekf = EKF(state_dim=state_dim, meas_dim=meas_dim, f=f_pred)
    health = HealthIndicator()

    twin = TwinSystem(
        physics=physics,
        estimator=ekf,
        residual=None,
        health=health,
        dt=dt,
    )

    x0 = np.array([0.5, 1.0])
    twin.initialize(x0=x0)

    # Dati sintetici: batch di (u, y)
    n_steps = 200
    np.random.seed(42)
    U = np.ones((n_steps, 1)) * 1.0 + np.random.randn(n_steps, 1) * 0.05
    Y = np.zeros((n_steps, 1))
    x = x0.copy()
    for i in range(n_steps):
        out = physics.step(state=x, u=U[i], dt=dt)
        x = out["state"]
        Y[i] = out["output"] + np.random.randn(1) * 0.02

    stream = BatchStream(U, Y)
    history = TwinHistory()

    print("Running twin loop...")
    for u_t, y_t in stream:
        result = twin.step(u=u_t, measurement=y_t)
        history.append(
            state=result.state,
            anomaly=result.anomaly,
            health_indicator=result.health_indicator,
        )
        if result.anomaly > 0.5:
            print(f"  [step {len(history)}] Anomalia: {result.anomaly:.4f}")

    print(f"Steps: {len(history)}, stato finale: {twin.state}")
    print("Fatto.")


if __name__ == "__main__":
    main()
