"""
Example: simulating a dynamical system with neural networks.

Flow:
  1. Generate data (t, x, u) from a "true" model (harmonic oscillator).
  2. Train NeuralODEModel (rhs learned by a network) and NeuralDynamicsModel (x_{k+1} = network).
  3. Simulate with both models and compare to the true trajectory.
  4. Save comparison plots.

Requires: torch (already in TwinOps dependencies).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except ImportError:
    print("This example requires PyTorch.")
    print("  Install with: pip install torch")
    sys.exit(1)

from twinops.physics import (
    NeuralODEModel,
    HarmonicOscillator,
    RK4Integrator,
)
from twinops.ml import (
    NeuralDynamicsModel,
    default_dynamics_net,
    prepare_ode_data_from_timeseries,
    train_neural_ode,
    train_dynamics,
)


# ---------------------------------------------------------------------------
# 1) Training data from the "true" system
# ---------------------------------------------------------------------------

def generate_training_data(
    n_steps: int = 500,
    dt: float = 0.05,
    x0: np.ndarray = None,
    omega: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (t, x, u) from the harmonic oscillator (no input)."""
    if x0 is None:
        x0 = np.array([1.0, 0.0])
    physics = HarmonicOscillator(omega=omega, integrator=RK4Integrator())
    t = np.arange(n_steps + 1, dtype=float) * dt
    u = np.zeros((n_steps + 1, 0))  # no input

    x_vals = [np.asarray(x0, dtype=float).copy()]
    x = x_vals[0].copy()
    for k in range(n_steps):
        out = physics.step(state=x, u=np.array([]), dt=dt)
        x = out["state"]
        x_vals.append(x.copy())
    x_arr = np.array(x_vals)
    return t, x_arr, u


# ---------------------------------------------------------------------------
# 2) Train Neural ODE (continuous dynamics)
# ---------------------------------------------------------------------------

def train_neural_ode_model(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    hidden_size: int = 64,
    epochs: int = 150,
    batch_size: int = 64,
) -> NeuralODEModel:
    """Train NeuralODEModel on (t, x, u) and return the model ready for simulation."""
    n_states = x.shape[1]
    n_inputs = u.shape[1] if u.size > 0 else 0

    X, y = prepare_ode_data_from_timeseries(t, x, u)
    model = NeuralODEModel(
        n_states=n_states,
        n_inputs=n_inputs,
        integrator=RK4Integrator(),
        hidden_size=hidden_size,
    )
    train_neural_ode(
        model,
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    return model


# ---------------------------------------------------------------------------
# 3) Train Neural Dynamics (discrete-time)
# ---------------------------------------------------------------------------

def train_neural_dynamics_model(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    hidden_size: int = 64,
    epochs: int = 150,
    batch_size: int = 64,
) -> NeuralDynamicsModel:
    """Train NeuralDynamicsModel on (x_k, u_k, x_{k+1}) and return the model."""
    n_states = x.shape[1]
    n_inputs = u.shape[1] if u.ndim > 1 and u.shape[1] > 0 else 0

    # Pairs (state_k, u_k, state_{k+1})
    states = x[:-1]
    inputs = u[:-1] if u.size > 0 and n_inputs > 0 else np.zeros((len(states), n_inputs))
    next_states = x[1:]

    net = default_dynamics_net(
        state_dim=n_states,
        input_dim=n_inputs,
        hidden=hidden_size,
    )
    model = NeuralDynamicsModel(net, state_dim=n_states, input_dim=n_inputs)
    train_dynamics(
        model,
        (states, inputs, next_states),
        dt=dt,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    return model


# ---------------------------------------------------------------------------
# 4) Simulation and comparison
# ---------------------------------------------------------------------------

def simulate_true(physics: HarmonicOscillator, x0: np.ndarray, t_vals: np.ndarray, dt: float) -> np.ndarray:
    """Simulate the true model and return trajectory (n_steps+1, n_states)."""
    x = np.asarray(x0, dtype=float).copy()
    traj = [x.copy()]
    for _ in range(len(t_vals) - 1):
        out = physics.step(state=x, u=np.array([]), dt=dt)
        x = out["state"]
        traj.append(x.copy())
    return np.array(traj)


def simulate_neural_ode(model: NeuralODEModel, x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Simulate NeuralODEModel and return trajectory."""
    model.initialize()
    x = np.asarray(x0, dtype=float).copy()
    traj = [x.copy()]
    for _ in range(n_steps):
        out = model.step(state=x, u=np.array([]), dt=dt)
        x = out["state"]
        traj.append(x.copy())
    return np.array(traj)


def simulate_neural_dynamics(model: NeuralDynamicsModel, x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Simulate NeuralDynamicsModel and return trajectory."""
    model.initialize()
    x = np.asarray(x0, dtype=float).copy()
    u = np.zeros(model.input_dim) if model.input_dim > 0 else np.array([])
    traj = [x.copy()]
    for _ in range(n_steps):
        out = model.step(state=x, u=u, dt=dt)
        x = out["state"]
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# 5) Main and plot
# ---------------------------------------------------------------------------

def main() -> None:
    dt = 0.05
    n_train = 500
    n_test = 200
    x0 = np.array([1.0, 0.0])
    omega = 1.0

    print("=" * 60)
    print("Dynamical system simulation with neural networks (TwinOps)")
    print("=" * 60)

    # ---- 1) Training data ----
    print("\n1) Generating training data (harmonic oscillator)...")
    t_train, x_train, u_train = generate_training_data(
        n_steps=n_train, dt=dt, x0=x0, omega=omega,
    )
    print(f"   t: {len(t_train)} points, x: {x_train.shape}")

    # ---- 2) Train Neural ODE ----
    print("\n2) Training NeuralODEModel (rhs = network)...")
    neural_ode = train_neural_ode_model(
        t_train, x_train, u_train,
        hidden_size=64, epochs=150, batch_size=64,
    )

    # ---- 3) Train Neural Dynamics ----
    print("\n3) Training NeuralDynamicsModel (x_{k+1} = network)...")
    neural_dynamics = train_neural_dynamics_model(
        t_train, x_train, u_train, dt,
        hidden_size=64, epochs=150, batch_size=64,
    )

    # ---- 4) Simulation on test interval ----
    print("\n4) Simulation (true model vs Neural ODE vs Neural Dynamics)...")
    physics_true = HarmonicOscillator(omega=omega, integrator=RK4Integrator())
    t_test = np.arange(n_test + 1) * dt

    traj_true = simulate_true(physics_true, x0, t_test, dt)
    traj_ode = simulate_neural_ode(neural_ode, x0, n_test, dt)
    traj_dyn = simulate_neural_dynamics(neural_dynamics, x0, n_test, dt)

    err_ode = np.mean(np.abs(traj_ode - traj_true))
    err_dyn = np.mean(np.abs(traj_dyn - traj_true))
    print(f"   Mean error |pred - true|:")
    print(f"     Neural ODE:      {err_ode:.6f}")
    print(f"     Neural Dynamics: {err_dyn:.6f}")

    # ---- 5) Plot ----
    out_dir = Path(__file__).resolve().parent
    try:
        _plot_results(t_test, traj_true, traj_ode, traj_dyn, out_dir)
        print(f"\n   Plot saved to: {out_dir / 'neural_dynamics_plot.png'}")
    except ImportError:
        print("\n   Plot skipped: matplotlib not available.")

    print("\nDone.")


def _plot_results(
    t: np.ndarray,
    traj_true: np.ndarray,
    traj_ode: np.ndarray,
    traj_dyn: np.ndarray,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # Position (x0)
    ax = axes[0]
    ax.plot(t, traj_true[:, 0], "k-", label="True", linewidth=2)
    ax.plot(t, traj_ode[:, 0], "--", label="Neural ODE", alpha=0.9)
    ax.plot(t, traj_dyn[:, 0], ":", label="Neural Dynamics", alpha=0.9)
    ax.set_ylabel("x₁ (position)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Velocity (x1)
    ax = axes[1]
    ax.plot(t, traj_true[:, 1], "k-", label="True", linewidth=2)
    ax.plot(t, traj_ode[:, 1], "--", label="Neural ODE", alpha=0.9)
    ax.plot(t, traj_dyn[:, 1], ":", label="Neural Dynamics", alpha=0.9)
    ax.set_ylabel("x₂ (velocity)")
    ax.set_xlabel("t")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Dynamical system: simulation with neural networks — TwinOps")
    plt.tight_layout()
    plt.savefig(out_dir / "neural_dynamics_plot.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
