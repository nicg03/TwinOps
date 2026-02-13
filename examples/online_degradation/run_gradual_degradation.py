"""
Esempio: degradazione graduale (efficienza che cala nel tempo).

Differenza rispetto a run_online_degradation.py:
- Guasto a step: efficienza passa da 1 a 0.72 allo step 400.
- Questo esempio: efficienza cala linearmente da 1.0 a 0.70 tra step 200 e 600,
  poi resta 0.70. Verifica che anomaly, HI e RUL reagiscano a un trend lento.
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
# Modello fisico: pompa [portata, pressione], ingresso [velocità]
# ---------------------------------------------------------------------------

class PumpPhysics(ODEModel):
    """Pompa: stato [portata q, pressione p], ingresso [velocità omega]."""

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        q, p = x[0], x[1]
        omega = u[0] if u.size else 1.0
        dq = -0.1 * q + 0.5 * omega
        dp = 0.3 * q - 0.2 * p
        return np.array([dq, dp])


# ---------------------------------------------------------------------------
# Generatore online con degradazione graduale (lineare)
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
    Genera (u_t, y_t) step-by-step.
    Efficienza: 1.0 fino a decay_start_step; poi cala linearmente fino a
    efficiency_min a decay_end_step; poi resta efficiency_min.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(n_steps):
        u_t = np.array([u_setpoint + rng.standard_normal() * u_noise_std])
        out = physics.step(state=x, u=u_t, dt=dt)
        x = out["state"]
        y_nominal = out["output"]
        # Efficienza: rampa lineare da 1 a efficiency_min tra decay_start e decay_end
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
    rul = SimpleRUL(hi_fail=0.3, min_rul=0.0, max_rul=100.0)

    twin = TwinSystem(
        physics=physics,
        estimator=ekf,
        residual=None,
        health=health,
        rul=rul,
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

    print("Pipeline online: degradazione graduale (rampa lineare)")
    print(f"  Step 0-{decay_start}: regime normale (eff=1.0)")
    print(f"  Step {decay_start}-{decay_end}: efficienza cala linearmente -> {efficiency_min}")
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
            rul=result.rul if result.rul is not None else np.nan,
            ema=ema,
            cusum_pos=cusum_p,
            cusum_neg=cusum_n,
        )

        if cusum_p >= anomaly_detector.cusum_threshold or cusum_n >= anomaly_detector.cusum_threshold:
            print(f"  [step {step}] ALERT CUSUM: anomaly={result.anomaly:.4f} HI={result.health_indicator:.3f} RUL={result.rul}")

        if step == decay_start:
            print(f"  [step {step}] Inizio degradazione graduale")
        if step == decay_end and result.rul is not None:
            print(f"  [step {step}] Fine rampa; RUL stimata: {result.rul:.2f} s")

    print(f"Steps: {len(history)}, stato finale: {twin.state}")
    print("Export CSV e grafici...")

    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "gradual_degradation_history.csv"
    history.to_csv(csv_path, delimiter=",")
    print(f"  CSV: {csv_path}")

    try:
        _plot_results(history, decay_start, decay_end, out_dir)
    except ImportError:
        print("  (matplotlib non disponibile, grafici saltati)")

    print("Fatto.")


def _plot_results(history: TwinHistory, decay_start: int, decay_end: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = history.to_dict()
    steps = np.arange(len(data["time"]))

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

    ax = axes[0]
    ax.plot(steps, data["state_0"], label="portata stimata")
    ax.plot(steps, data["state_1"], label="pressione stimata")
    ax.axvspan(decay_start, decay_end, color="gray", alpha=0.2, label="rampa degradazione")
    ax.set_ylabel("Stato")
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

    plt.suptitle("Degradazione graduale (rampa lineare) — TwinOps")
    plt.tight_layout()
    plot_path = out_dir / "gradual_degradation_plots.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
