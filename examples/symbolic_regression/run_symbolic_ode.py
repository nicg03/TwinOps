"""
Esempio: modello fisico appreso da dati CSV/serie temporali (symbolic regression).

Genera dati da un sistema del primo ordine (FirstOrderLag), apprende il rhs
tramite SymbolicODEModel e usa il modello come fisica nel twin (stesso contratto
di ODEModel). Richiede: pip install twinops[symbolic]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import gplearn  # noqa: F401
except ImportError:
    print("Questo esempio richiede gplearn.")
    sys.exit(1)

from twinops.physics import SymbolicODEModel, FirstOrderLag, RK4Integrator


def main() -> None:
    # --- Dati "sintetici" (in pratica potrebbero venire da CSV) ---
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

    # --- Apprendi modello fisico (symbolic regression) ---
    model = SymbolicODEModel(
        n_states=1,
        n_inputs=1,
        integrator=RK4Integrator(),
        population_size=500,
        generations=25,
        random_state=42,
    )
    model.fit_from_timeseries(t, x_arr, u_arr)

    print("Espressioni apprese per dx/dt:")
    for i, expr in enumerate(model.get_expressions()):
        print(f"  dx{i}/dt = {expr}")

    # --- Usa il modello come fisica: step identico a ODEModel ---
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
    print(f"\nErrore medio |x_sim - x_true|: {np.mean(err):.6f}")
    print("Il SymbolicODEModel funge da modello fisico: initialize, step, rhs come ODEModel.")


if __name__ == "__main__":
    main()
