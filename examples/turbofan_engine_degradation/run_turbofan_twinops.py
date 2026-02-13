"""
Full TwinOps pipeline for the NASA C-MAPSS turbofan engine degradation dataset.

Flow:
  1. Load train/test and RUL data from data/turbofan_engine_degradation/.
  2. State reduction: subset of sensors as state, 3 operational settings as input.
  3. Train NeuralDynamicsModel on (x_k, u_k, x_{k+1}) from training data.
  4. Build TwinSystem: physics (trained model) + EKF + Health + SimpleRUL.
  5. For each test unit: stream (u, y) → twin.step → RUL at last cycle.
  6. Compare predicted vs true RUL (RMSE, MAE) and optional plot.

Usage:
  python examples/turbofan_engine_degradation/run_turbofan_twinops.py [--fd FD001] [--plot]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except ImportError:
    print("This example requires PyTorch: pip install torch")
    sys.exit(1)

from twinops.core import TwinSystem
from twinops.estimation import EKF
from twinops.health import HealthIndicator, SimpleRUL
from twinops.io.streams import BatchStream
from twinops.ml import NeuralDynamicsModel, default_dynamics_net, train_dynamics

DATA_DIR = ROOT / "data" / "turbofan_engine_degradation"

# C-MAPSS columns: 0=unit, 1=time, 2-4=op settings, 5-25=21 sensors
N_SETTINGS = 3
N_SENSORS = 21
# Subset of sensors used as state/measurement (indices 0..20)
# Typically informative for degradation (temperature, pressure, RPM)
SENSOR_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 sensors → state_dim=8, meas_dim=8


# ---------------------------------------------------------------------------
# 1) C-MAPSS data loading
# ---------------------------------------------------------------------------


def load_cmapss_file(path: Path) -> np.ndarray:
    """Load a C-MAPSS .txt file (space-separated numbers)."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            data.append(row)
    return np.array(data) if data else np.empty((0, 26))


def split_by_unit(raw: np.ndarray):
    """
    Split data by unit number.
    raw: (N, 26) with columns [unit, time, op1, op2, op3, sensor_1..sensor_21]
    Returns: dict unit_id -> {"time": (T,), "settings": (T,3), "sensors": (T,21)}
    """
    units = {}
    for row in raw:
        uid = int(row[0])
        if uid not in units:
            units[uid] = {"time": [], "settings": [], "sensors": []}
        units[uid]["time"].append(row[1])
        units[uid]["settings"].append(row[2 : 2 + N_SETTINGS])
        units[uid]["sensors"].append(row[5 : 5 + N_SENSORS])
    for uid in units:
        for k in ("time", "settings", "sensors"):
            units[uid][k] = np.array(units[uid][k])
    return units


def extract_state_sensors(sensors: np.ndarray, indices: list) -> np.ndarray:
    """Extract subset of sensors: (T, 21) -> (T, len(indices))."""
    return np.asarray(sensors[:, indices], dtype=np.float32)


def prepare_dynamics_data(units_dict: dict, sensor_indices: list):
    """
    Build (states, inputs, next_states) for train_dynamics.
    For each unit, for each cycle t with valid t+1: state_t, u_t, state_{t+1}.
    """
    states_list, inputs_list, next_list = [], [], []
    for uid, data in units_dict.items():
        settings = np.asarray(data["settings"], dtype=np.float32)
        sensors = np.asarray(data["sensors"], dtype=np.float32)
        x = extract_state_sensors(sensors, sensor_indices)
        T = x.shape[0]
        for t in range(T - 1):
            states_list.append(x[t])
            inputs_list.append(settings[t])
            next_list.append(x[t + 1])
    return (
        np.array(states_list),
        np.array(inputs_list),
        np.array(next_list),
    )


def load_rul_true(path: Path) -> np.ndarray:
    """Load true RUL (one row per test unit, same order as in test file)."""
    with open(path) as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


# ---------------------------------------------------------------------------
# 2) NeuralDynamicsModel training
# ---------------------------------------------------------------------------


def train_physics(
    states: np.ndarray,
    inputs: np.ndarray,
    next_states: np.ndarray,
    state_dim: int,
    input_dim: int,
    hidden: int = 128,
    epochs: int = 800,
    batch_size: int = 256,
    dt: float = 1.0,
) -> NeuralDynamicsModel:
    """Train NeuralDynamicsModel and return it as the physics component."""
    net = default_dynamics_net(state_dim=state_dim, input_dim=input_dim, hidden=hidden)
    model = NeuralDynamicsModel(net, state_dim=state_dim, input_dim=input_dim)
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
# 3) Twin and RUL evaluation
# ---------------------------------------------------------------------------


def run_twin_on_unit(
    twin: TwinSystem,
    inputs: np.ndarray,
    measurements: np.ndarray,
    x0: np.ndarray,
) -> tuple:
    """
    Run the twin on a single unit (stream u, y).
    Returns (RUL at last step, list of anomalies, list of HIs).
    """
    twin.initialize(x0=x0)
    stream = BatchStream(inputs, measurements)
    rul_last = None
    anomalies, his = [], []
    for u_t, y_t in stream:
        res = twin.step(u=u_t, measurement=y_t)
        rul_last = res.rul
        anomalies.append(res.anomaly)
        his.append(res.health_indicator if res.health_indicator is not None else np.nan)
    return rul_last, anomalies, his


def main():
    parser = argparse.ArgumentParser(description="TwinOps pipeline C-MAPSS")
    parser.add_argument("--fd", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--plot", action="store_true", help="Save plot of true vs predicted RUL")
    args = parser.parse_args()

    fd = args.fd
    train_path = DATA_DIR / f"train_{fd}.txt"
    test_path = DATA_DIR / f"test_{fd}.txt"
    rul_path = DATA_DIR / f"RUL_{fd}.txt"

    if not train_path.exists() or not test_path.exists():
        print(f"Data files not found in {DATA_DIR}. Ensure train_{fd}.txt and test_{fd}.txt exist.")
        sys.exit(1)

    print("Loading C-MAPSS data...")
    train_raw = load_cmapss_file(train_path)
    test_raw = load_cmapss_file(test_path)
    train_units = split_by_unit(train_raw)
    test_units = split_by_unit(test_raw)
    rul_true = load_rul_true(rul_path) if rul_path.exists() else None

    state_dim = len(SENSOR_INDICES)
    input_dim = N_SETTINGS
    meas_dim = state_dim

    print("Preparing dynamics training data...")
    states, inputs, next_states = prepare_dynamics_data(train_units, SENSOR_INDICES)
    print(f"  Training samples: {len(states)}")

    print("Training NeuralDynamicsModel...")
    physics = train_physics(
        states, inputs, next_states,
        state_dim=state_dim,
        input_dim=input_dim,
        epochs=args.epochs,
        dt=1.0,
    )

    def f_pred(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        out = physics.step(state=x, u=u, dt=dt)
        return out["state"]

    ekf = EKF(state_dim=state_dim, meas_dim=meas_dim, f=f_pred)
    # HI from anomaly: degradation → EKF innovation grows → HI drops
    health = HealthIndicator(fn=lambda state, anomaly: float(np.clip(1.0 / (1.0 + 0.1 * anomaly), 0.0, 1.0)))
    rul_estimator = SimpleRUL(hi_fail=0.3, min_rul=0.0, max_rul=500.0)

    twin = TwinSystem(
        physics=physics,
        estimator=ekf,
        health=health,
        rul=rul_estimator,
        dt=1.0,
        state_dim=state_dim,
        input_dim=input_dim,
        meas_dim=meas_dim,
    )

    # Test unit order: same as in file (1, 2, ...)
    unit_ids = sorted(test_units.keys())
    if rul_true is not None and len(rul_true) != len(unit_ids):
        print(f"Warning: true RUL has {len(rul_true)} rows, test units {len(unit_ids)}")

    print("Running twin on test units...")
    rul_predictions = []
    for i, uid in enumerate(unit_ids):
        data = test_units[uid]
        settings = np.asarray(data["settings"], dtype=np.float32)
        sensors_sub = extract_state_sensors(
            np.asarray(data["sensors"], dtype=np.float32),
            SENSOR_INDICES,
        )
        x0 = sensors_sub[0]
        rul_last, _, _ = run_twin_on_unit(twin, settings, sensors_sub, x0)
        rul_predictions.append(rul_last if rul_last is not None else np.nan)

    rul_predictions = np.array(rul_predictions, dtype=float)

    # Metrics
    print("\n--- RUL results ---")
    if rul_true is not None and len(rul_true) == len(rul_predictions):
        valid = np.isfinite(rul_predictions)
        if np.any(valid):
            rmse = np.sqrt(np.nanmean((rul_predictions - rul_true) ** 2))
            mae = np.nanmean(np.abs(rul_predictions - rul_true))
            print(f"  RMSE RUL: {rmse:.2f}")
            print(f"  MAE  RUL: {mae:.2f}")
        else:
            print("  No valid RUL predictions (SimpleRUL requires HI trend).")
        if args.plot:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 5))
                plt.scatter(rul_true, rul_predictions, alpha=0.7, label="Units")
                mn = min(rul_true.min(), np.nanmin(rul_predictions))
                mx = max(rul_true.max(), np.nanmax(rul_predictions))
                plt.plot([mn, mx], [mn, mx], "k--", label="y=x")
                plt.xlabel("True RUL (cycles)")
                plt.ylabel("Predicted RUL (cycles)")
                plt.title(f"C-MAPSS {fd}: True vs predicted RUL (TwinOps)")
                plt.legend()
                out_path = ROOT / "examples" / "turbofan_engine_degradation" / f"rul_{fd}.png"
                plt.savefig(out_path, dpi=120)
                plt.close()
                print(f"  Plot saved: {out_path}")
            except Exception as e:
                print(f"  Plot not saved: {e}")
    else:
        print("  True RUL not available or length mismatch.")

    print("Done.")


if __name__ == "__main__":
    main()
