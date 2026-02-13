"""
Physics example: thermal system with online degradation (fouling).

Model: body with temperature T, heated by power P, cooled by convection
to ambient T_amb. ODE: dT/dt = -alpha*(T - T_amb) + P/(m*c).
Degradation: "fouling" reduces heat transfer (alpha decreases) in the real
process → real system cools less → measured T higher than expected
→ anomaly rises, HI drops, RUL updates.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from twinops.core import TwinSystem, TwinHistory
from twinops.physics import ODEModel, RK4Integrator
from twinops.estimation import EKF, AnomalyDetector
from twinops.health import HealthIndicator, SimpleRUL


# ---------------------------------------------------------------------------
# Physics model: thermal system
# ---------------------------------------------------------------------------

class ThermalPhysics(ODEModel):
    """
    Thermal system: state T (temperature), input P (power).
    dT/dt = -alpha*(T - T_amb) + P/(m*c).
    alpha = heat transfer coefficient, T_amb = ambient temperature, m*c = thermal capacity.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        T_amb: float = 20.0,
        mc: float = 100.0,
        integrator: object = None,
    ) -> None:
        super().__init__(integrator=integrator)
        self.alpha = alpha
        self.T_amb = T_amb
        self.mc = mc

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        T = x[0]
        P = u[0] if u.size else 0.0
        dT = -self.alpha * (T - self.T_amb) + P / self.mc
        return np.array([dT])


# ---------------------------------------------------------------------------
# Online generator: "real" process with degrading alpha (fouling)
# ---------------------------------------------------------------------------

def generate_thermal_online(
    physics: ODEModel,
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    P_setpoint: float = 50.0,
    P_noise_std: float = 2.0,
    meas_noise_std: float = 0.3,
    decay_start_step: int = 250,
    decay_end_step: int = 650,
    alpha_nominal: float = 0.1,
    alpha_min: float = 0.04,
    seed: int = 44,
):
    """
    Generate (u_t, y_t) step-by-step.
    Real process has alpha decreasing linearly from alpha_nominal to alpha_min
    between decay_start_step and decay_end_step (fouling): worse cooling → T rises.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(n_steps):
        u_t = np.array([P_setpoint + rng.standard_normal() * P_noise_std])
        # Degraded alpha in "real process"
        if k < decay_start_step:
            alpha = alpha_nominal
        elif k >= decay_end_step:
            alpha = alpha_min
        else:
            frac = (k - decay_start_step) / (decay_end_step - decay_start_step)
            alpha = alpha_nominal - frac * (alpha_nominal - alpha_min)
        physics.alpha = alpha
        out = physics.step(state=x, u=u_t, dt=dt)
        x = out["state"]
        y_t = out["output"] + rng.standard_normal(out["output"].shape) * meas_noise_std
        yield u_t, np.atleast_1d(y_t)


def main() -> None:
    dt = 0.05
    state_dim = 1
    meas_dim = 1
    n_steps = 800
    decay_start = 250
    decay_end = 650
    alpha_nominal = 0.1
    alpha_min = 0.04
    T_amb = 20.0
    mc = 100.0

    # Twin: nominal model (fixed alpha)
    physics = ThermalPhysics(alpha=alpha_nominal, T_amb=T_amb, mc=mc, integrator=RK4Integrator())
    physics.set_output_fn(lambda x, u: np.array([x[0]]))

    def f_pred(x: np.ndarray, u: np.ndarray, _dt: float) -> np.ndarray:
        return physics.step(state=x, u=u, dt=_dt)["state"]

    ekf = EKF(state_dim=state_dim, meas_dim=meas_dim, f=f_pred)
    # HI from anomaly (temperature is not in [0,2] as in the pump)
    def hi_from_anomaly(state: np.ndarray, anomaly: float) -> float:
        return float(np.clip(1.0 / (1.0 + 0.4 * anomaly), 0.0, 1.0))
    health = HealthIndicator(fn=hi_from_anomaly)
    rul = SimpleRUL(hi_fail=0.3, min_rul=0.0, max_rul=100.0)

    twin = TwinSystem(
        physics=physics,
        estimator=ekf,
        residual=None,
        health=health,
        rul=rul,
        dt=dt,
    )

    # Initial temperature close to ambient
    x0 = np.array([22.0])
    twin.initialize(x0=x0)

    anomaly_detector = AnomalyDetector(
        alpha_ema=0.1,
        cusum_threshold=3.5,
        cusum_drift=0.0,
    )

    history = TwinHistory()
    physics_truth = ThermalPhysics(alpha=alpha_nominal, T_amb=T_amb, mc=mc, integrator=RK4Integrator())
    physics_truth.set_output_fn(lambda x, u: np.array([x[0]]))

    stream = generate_thermal_online(
        physics=physics_truth,
        x0=x0,
        dt=dt,
        n_steps=n_steps,
        decay_start_step=decay_start,
        decay_end_step=decay_end,
        alpha_nominal=alpha_nominal,
        alpha_min=alpha_min,
    )

    print("Online pipeline: thermal system with fouling (alpha decreases)")
    print(f"  Step 0-{decay_start}: alpha={alpha_nominal} (nominal)")
    print(f"  Step {decay_start}-{decay_end}: alpha decreases linearly -> {alpha_min}")
    print(f"  Step {decay_end}-{n_steps}: alpha={alpha_min}")
    print("Running...")

    for step, (u_t, y_t) in enumerate(stream):
        result = twin.step(u=u_t, measurement=y_t)
        ema = anomaly_detector.update_ema(result.anomaly)
        cusum_p, cusum_n = anomaly_detector.update_cusum(result.anomaly - ema)

        t = twin.time
        history.append(
            time=t,
            temperature=float(result.state[0]),
            anomaly=result.anomaly,
            health_indicator=result.health_indicator if result.health_indicator is not None else np.nan,
            rul=result.rul if result.rul is not None else np.nan,
            ema=ema,
            cusum_pos=cusum_p,
            cusum_neg=cusum_n,
        )

        if cusum_p >= anomaly_detector.cusum_threshold or cusum_n >= anomaly_detector.cusum_threshold:
            print(f"  [step {step}] ALERT CUSUM: T={result.state[0]:.2f} anomaly={result.anomaly:.4f} HI={result.health_indicator:.3f}")

        if step == decay_start:
            print(f"  [step {step}] Fouling start (alpha begins to decrease)")
        if step == decay_end and result.rul is not None:
            print(f"  [step {step}] Fouling ramp end; estimated RUL: {result.rul:.2f} s")

    print(f"Steps: {len(history)}, final T: {twin.state[0]:.2f}")
    print("Exporting CSV and plots...")

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "thermal_degradation_history.csv"
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
    ax.plot(steps, data["temperature"], label="estimated T (°C)")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2, label="fouling")
    ax.set_ylabel("Temperature")
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
    ax.plot(steps, data["rul"], alpha=0.9, label="RUL (s)")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2)
    ax.axhline(0.3, color="red", linestyle=":", alpha=0.5, label="HI fail")
    ax.set_ylabel("HI / RUL")
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

    plt.suptitle("Thermal system with fouling (alpha degradation) — TwinOps")
    plt.tight_layout()
    plot_path = out_dir / "thermal_degradation_plots.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
