"""
Example: gradual degradation (efficiency decreasing over time).

Difference from run_online_degradation.py:
- Step fault: efficiency jumps from 1 to 0.72 at step 400.
- This example: efficiency decreases linearly from 1.0 to 0.70 between steps 200 and 600,
  then stays 0.70. Verifies that anomaly and HI respond to a slow trend.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from twinops.core import TwinSystem, TwinHistory
from twinops.physics import ODEModel, RK4Integrator
from twinops.estimation import EKF, AnomalyDetector
from twinops.health import HealthIndicator


# ---------------------------------------------------------------------------
# Physics model: pump [flow rate, pressure], input [speed]
# ---------------------------------------------------------------------------

class PumpPhysics(ODEModel):
    """Pump: state [flow q, pressure p], input [speed omega]."""

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        q, p = x[0], x[1]
        omega = u[0] if u.size else 1.0
        dq = -0.1 * q + 0.5 * omega
        dp = 0.3 * q - 0.2 * p
        return np.array([dq, dp])


# ---------------------------------------------------------------------------
# Online generator with gradual (linear) degradation
# ---------------------------------------------------------------------------

def generate_gradual_degradation(
    physics: ODEModel,
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    u_setpoint: float = 1.0,
    u_noise_std: float = 0.03,
    meas_noise_std: float = 0.02,
    decay_start_step: int = 200,
    decay_end_step: int = 600,
    efficiency_min: float = 0.70,
    seed: int = 43,
):
    """
    Generate (u_t, y_t) step-by-step.
    Efficiency: 1.0 until decay_start_step; then decreases linearly to
    efficiency_min at decay_end_step; then stays efficiency_min.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(n_steps):
        u_t = np.array([u_setpoint + rng.standard_normal() * u_noise_std])
        out = physics.step(state=x, u=u_t, dt=dt)
        x = out["state"]
        y_nominal = out["output"]
        # Efficiency: linear ramp from 1 to efficiency_min between decay_start and decay_end
        if k < decay_start_step:
            eff = 1.0
        elif k >= decay_end_step:
            eff = efficiency_min
        else:
            frac = (k - decay_start_step) / (decay_end_step - decay_start_step)
            eff = 1.0 - frac * (1.0 - efficiency_min)
        y_t = eff * y_nominal + rng.standard_normal(y_nominal.shape) * meas_noise_std
        yield u_t, np.atleast_1d(y_t)


def main() -> None:
    dt = 0.01
    state_dim = 2
    meas_dim = 1
    n_steps = 800
    decay_start = 200
    decay_end = 600
    efficiency_min = 0.70

    # Twin
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

    anomaly_detector = AnomalyDetector(
        alpha_ema=0.12,
        cusum_threshold=2.5,
        cusum_drift=0.0,
    )

    history = TwinHistory()
    physics_truth = PumpPhysics(integrator=RK4Integrator())
    physics_truth.set_output_fn(lambda x, u: np.array([x[0]]))

    stream = generate_gradual_degradation(
        physics=physics_truth,
        x0=x0,
        dt=dt,
        n_steps=n_steps,
        decay_start_step=decay_start,
        decay_end_step=decay_end,
        efficiency_min=efficiency_min,
    )

    print("Online pipeline: gradual degradation (linear ramp)")
    print(f"  Step 0-{decay_start}: normal regime (eff=1.0)")
    print(f"  Step {decay_start}-{decay_end}: efficiency decreases linearly -> {efficiency_min}")
    print(f"  Step {decay_end}-{n_steps}: eff={efficiency_min}")
    print("Running...")

    for step, (u_t, y_t) in enumerate(stream):
        result = twin.step(u=u_t, measurement=y_t)
        ema = anomaly_detector.update_ema(result.anomaly)
        cusum_p, cusum_n = anomaly_detector.update_cusum(result.anomaly - ema)

        t = twin.time
        history.append(
            time=t,
            state_0=float(result.state[0]),
            state_1=float(result.state[1]),
            anomaly=result.anomaly,
            health_indicator=result.health_indicator if result.health_indicator is not None else np.nan,
            ema=ema,
            cusum_pos=cusum_p,
            cusum_neg=cusum_n,
        )

        if cusum_p >= anomaly_detector.cusum_threshold or cusum_n >= anomaly_detector.cusum_threshold:
            print(f"  [step {step}] ALERT CUSUM: anomaly={result.anomaly:.4f} HI={result.health_indicator:.3f}")

        if step == decay_start:
            print(f"  [step {step}] Gradual degradation start")
        if step == decay_end:
            print(f"  [step {step}] Ramp end (gradual degradation complete).")

    print(f"Steps: {len(history)}, final state: {twin.state}")
    print("Exporting CSV and plots...")

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "gradual_degradation_history.csv"
    history.to_csv(csv_path, delimiter=",")
    print(f"  CSV: {csv_path}")

    try:
        _plot_results(history, decay_start, decay_end, out_dir)
    except ImportError:
        print("  (matplotlib not available, plots skipped)")

    print("Done.")


def _plot_results(history: TwinHistory, decay_start: int, decay_end: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = history.to_dict()
    steps = np.arange(len(data["time"]))

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

    ax = axes[0]
    ax.plot(steps, data["state_0"], label="estimated flow")
    ax.plot(steps, data["state_1"], label="estimated pressure")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2, label="degradation ramp")
    ax.set_ylabel("State")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, data["anomaly"], alpha=0.8, label="anomaly")
    ax.plot(steps, data["ema"], label="EMA(anomaly)")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2)
    ax.set_ylabel("Anomaly")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, data["health_indicator"], label="Health Indicator")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2)
    ax.axhline(0.3, color="red", linestyle=":", alpha=0.5, label="HI fail")
    ax.set_ylabel("HI")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(steps, data["cusum_pos"], label="CUSUM+")
    ax.plot(steps, data["cusum_neg"], label="CUSUM-")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2)
    ax.set_ylabel("CUSUM")
    ax.set_xlabel("Step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Gradual degradation (linear ramp) — TwinOps")
    plt.tight_layout()
    plot_path = out_dir / "gradual_degradation_plots.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
