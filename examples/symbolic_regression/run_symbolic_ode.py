"""
Example: physics model learned from CSV/time series data (symbolic regression).

Generates data from a first-order system (FirstOrderLag), learns the rhs
via SymbolicODEModel and uses the model as physics in the twin (same contract
as ODEModel). Requires: pip install twinops[symbolic]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import gplearn  # noqa: F401
except ImportError:
    print("This example requires gplearn.")
    sys.exit(1)

from twinops.physics import SymbolicODEModel, FirstOrderLag, RK4Integrator


def main() -> None:
    # --- Synthetic data (in practice could come from CSV) ---
    tau = 1.0
    dt = 0.05
    n_steps = 300
    t = np.arange(n_steps + 1) * dt
    u = 0.5 + 0.2 * np.sin(0.5 * t)
    u_arr = u.reshape(-1, 1)

    physics_true = FirstOrderLag(tau=tau)
    x0 = np.array([0.0])
    x_vals = [x0.copy()]
    x = x0.copy()
    for k in range(n_steps):
        out = physics_true.step(state=x, u=np.array([u[k]]), dt=dt)
        x = out["state"]
        x_vals.append(x.copy())
    x_arr = np.array(x_vals)

    # --- Learn physics model (symbolic regression) ---
    model = SymbolicODEModel(
        n_states=1,
        n_inputs=1,
        integrator=RK4Integrator(),
        population_size=500,
        generations=25,
        random_state=42,
    )
    model.fit_from_timeseries(t, x_arr, u_arr)

    print("Learned expressions for dx/dt:")
    for i, expr in enumerate(model.get_expressions()):
        print(f"  dx{i}/dt = {expr}")

    # --- Use model as physics: step identical to ODEModel ---
    x_sym = np.array([0.0])
    t_cur = 0.0
    trajectory = [x_sym[0]]
    for k in range(n_steps):
        out = model.step(state=x_sym, u=np.array([u[k]]), dt=dt)
        x_sym = out["state"]
        t_cur += dt
        trajectory.append(x_sym[0])

    trajectory = np.array(trajectory)
    err = np.abs(trajectory - x_arr[:, 0])
    print(f"\nMean error |x_sim - x_true|: {np.mean(err):.6f}")
    print("SymbolicODEModel acts as physics model: initialize, step, rhs like ODEModel.")


if __name__ == "__main__":
    main()
