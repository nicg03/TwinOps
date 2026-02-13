"""
Full example: digital twin with physics learned from data (symbolic regression).

Flow:
  1. Generate data (t, x, u) from a "true" model (PumpLike) — in practice from CSV.
  2. Learn the physics model with SymbolicODEModel (fit_from_timeseries).
  3. Build the twin: physics=SymbolicODEModel, EKF, HealthIndicator, SimpleRUL.
  4. Run the twin loop with stream (u, y) and collect history.
  5. Print summary and export CSV/plots.

Requires: pip install twinops[symbolic]  (gplearn)
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
    print("  Install with: pip install twinops[symbolic]  (or: pip install gplearn)")
    print("  If already installed, use the same Python as pip.")
    sys.exit(1)

from twinops.core import TwinSystem, TwinHistory
from twinops.physics import SymbolicODEModel, PumpLike, RK4Integrator
from twinops.estimation import EKF
from twinops.health import HealthIndicator, SimpleRUL
from twinops.io import BatchStream


# ---------------------------------------------------------------------------
# 1) Training data (in practice: from CSV with columns t, x1, x2, u)
# ---------------------------------------------------------------------------

def generate_training_data(
    n_steps: int = 400,
    dt: float = 0.05,
    x0: np.ndarray = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (t, x, u) from PumpLike model to train SymbolicODEModel."""
    if x0 is None:
        x0 = np.array([0.0, 0.0])
    physics = PumpLike(a=0.1, b=0.5, c=0.3, d=0.2, integrator=RK4Integrator())
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps + 1) * dt
    u = 0.8 + 0.4 * np.sin(0.3 * t) + rng.standard_normal(n_steps + 1) * 0.05
    u = np.clip(u, 0.1, 1.5)
    u = u.reshape(-1, 1)

    x_vals = [np.asarray(x0, dtype=float).copy()]
    x = x_vals[0].copy()
    for k in range(n_steps):
        out = physics.step(state=x, u=u[k], dt=dt)
        x = out["state"]
        x_vals.append(x.copy())
    x_arr = np.array(x_vals)
    return t, x_arr, u


# ---------------------------------------------------------------------------
# 2) Fit symbolic physics model
# ---------------------------------------------------------------------------

def fit_physics_symbolic(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    population_size: int = 400,
    generations: int = 25,
    random_state: int = 42,
) -> SymbolicODEModel:
    """Learn rhs from (t, x, u) and return SymbolicODEModel ready for the twin."""
    n_states, n_inputs = x.shape[1], u.shape[1]
    physics = SymbolicODEModel(
        n_states=n_states,
        n_inputs=n_inputs,
        integrator=RK4Integrator(),
        population_size=population_size,
        generations=generations,
        random_state=random_state,
    )
    physics.fit_from_timeseries(t, x, u)
    return physics


# ---------------------------------------------------------------------------
# 3) Twin: symbolic physics + EKF + health + RUL
# ---------------------------------------------------------------------------

def build_twin(physics: SymbolicODEModel, dt: float) -> TwinSystem:
    """Build TwinSystem with physics (SymbolicODEModel), EKF, health, RUL."""
    state_dim = physics.n_states
    meas_dim = 1  # we only measure x[0] (flow rate)
    physics.set_output_fn(lambda x, u: np.array([x[0]]))

    def f_pred(x: np.ndarray, u: np.ndarray, _dt: float) -> np.ndarray:
        return physics.step(state=x, u=u, dt=_dt)["state"]

    ekf = EKF(state_dim=state_dim, meas_dim=meas_dim, f=f_pred)
    health = HealthIndicator()
    rul = SimpleRUL(hi_fail=0.3)

    return TwinSystem(
        physics=physics,
        estimator=ekf,
        residual=None,
        health=health,
        rul=rul,
        dt=dt,
    )


# ---------------------------------------------------------------------------
# 4) Generate stream (u, y) for the loop — "real process" simulation
# ---------------------------------------------------------------------------

def generate_replay_data(
    physics_true: PumpLike,
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    meas_noise_std: float = 0.02,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate U, Y for BatchStream: same model + measurement noise."""
    rng = np.random.default_rng(seed)
    U = np.zeros((n_steps, 1))
    Y = np.zeros((n_steps, 1))
    x = np.asarray(x0, dtype=float).copy()
    for k in range(n_steps):
        u_t = np.array([0.9 + 0.3 * np.sin(0.2 * k * dt)])
        U[k] = u_t
        out = physics_true.step(state=x, u=u_t, dt=dt)
        x = out["state"]
        y = out["output"]  # [x[0]]
        Y[k] = y + rng.standard_normal() * meas_noise_std
    return U, Y


# ---------------------------------------------------------------------------
# 5) Main: fit -> twin -> loop -> history -> export
# ---------------------------------------------------------------------------

def main() -> None:
    dt = 0.05
    state_dim = 2
    n_train = 400
    n_replay = 200
    x0 = np.array([0.5, 1.0])

    # ---- 1) Training data ----
    print("1) Generating training data (PumpLike model)...")
    t, x_train, u_train = generate_training_data(n_steps=n_train, dt=dt, x0=x0)
    print(f"   t: {len(t)} points, x: {x_train.shape}, u: {u_train.shape}")

    # ---- 2) Fit symbolic physics model ----
    print("2) Learning physics model (SymbolicODEModel)...")
    physics = fit_physics_symbolic(t, x_train, u_train)
    print("   Learned expressions:")
    for i, expr in enumerate(physics.get_expressions()):
        print(f"     dx{i}/dt = {expr}")

    # ---- 3) Build twin ----
    print("3) Building twin (physics=SymbolicODEModel, EKF, Health, RUL)...")
    twin = build_twin(physics, dt=dt)
    twin.initialize(x0=x0)

    # ---- 4) Stream (u, y) and twin loop ----
    print("4) Generating replay stream and twin loop...")
    physics_truth = PumpLike(a=0.1, b=0.5, c=0.3, d=0.2, integrator=RK4Integrator())
    physics_truth.set_output_fn(lambda x, u: np.array([x[0]]))
    U, Y = generate_replay_data(physics_truth, x0, dt, n_replay)

    stream = BatchStream(U, Y)
    history = TwinHistory()

    for step, (u_t, y_t) in enumerate(stream):
        result = twin.step(u=u_t, measurement=y_t)
        history.append(
            state=result.state,
            anomaly=result.anomaly,
            rul=result.rul if result.rul is not None else np.nan,
            health_indicator=result.health_indicator if result.health_indicator is not None else np.nan,
        )
        if result.anomaly > 0.5:
            print(f"   [step {step}] Anomaly: {result.anomaly:.4f}")

    print(f"   Steps completed: {len(history)}")

    # ---- 5) Summary and export ----
    print("5) Summary and export...")
    print(f"   Final estimated state: {twin.state}")
    print(f"   Twin time: {twin.time:.2f} s")

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "twin_symbolic_history.csv"
    history.to_csv(csv_path, delimiter=",")
    print(f"   CSV: {csv_path}")

    try:
        _plot_results(history, out_dir)
    except ImportError:
        print("   Plots skipped: matplotlib not available.")
        print("   To enable: pip install matplotlib (use the same Python you run the script with)")

    print("Done.")


def _plot_results(history: TwinHistory, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = history.to_dict()
    n = len(data["state"])
    steps = np.arange(n)
    state_0 = np.array([data["state"][i][0] for i in range(n)])
    state_1 = np.array([data["state"][i][1] for i in range(n)])

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    ax = axes[0]
    ax.plot(steps, state_0, label="x0 (estimated flow)")
    ax.plot(steps, state_1, label="x1 (estimated pressure)")
    ax.set_ylabel("State")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, data["anomaly"], label="anomaly")
    ax.set_ylabel("Anomaly")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, data["health_indicator"], label="Health Indicator")
    ax.plot(steps, data["rul"], alpha=0.9, label="RUL")
    ax.set_ylabel("HI / RUL")
    ax.set_xlabel("Step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Twin with SymbolicODEModel physics — TwinOps")
    plt.tight_layout()
    plot_path = out_dir / "twin_symbolic_plots.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"   Plot: {plot_path}")


if __name__ == "__main__":
    main()
