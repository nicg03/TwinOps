"""
Esempio complesso: pipeline online con misure simulate e degradazione.

Verifica l'utilità del codice TwinOps con:
- Simulazione "online": misure generate step-by-step come da sensori reali.
- Processo con degradazione: dopo N step un guasto riduce l'efficienza,
  le misure divergono dal modello → anomaly sale, HI scende, RUL si aggiorna.
- AnomalyDetector (EMA/CUSUM) per allarmi adattivi sull'anomaly score.
- Export CSV e grafici per analisi.
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
# Generatore di misure "online" con degradazione simulata
# ---------------------------------------------------------------------------

def generate_online_measurements(
    physics: ODEModel,
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    u_setpoint: float = 1.0,
    u_noise_std: float = 0.03,
    meas_noise_std: float = 0.02,
    fault_start_step: int = 400,
    efficiency_after_fault: float = 0.72,
    seed: int = 42,
):
    """
    Genera (u_t, y_t) step-by-step come da sensori online.
    Dopo fault_start_step simula un guasto: efficienza ridotta → portata misurata minore.
    Yields (u_t, y_t) per ogni step.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    for k in range(n_steps):
        u_t = np.array([u_setpoint + rng.standard_normal() * u_noise_std])
        out = physics.step(state=x, u=u_t, dt=dt)
        x = out["state"]
        y_nominal = out["output"]
        # Degradazione: dopo fault_start_step la "rete" perde (efficienza < 1)
        eff = efficiency_after_fault if k >= fault_start_step else 1.0
        y_t = eff * y_nominal + rng.standard_normal(y_nominal.shape) * meas_noise_std
        yield u_t, np.atleast_1d(y_t)


# ---------------------------------------------------------------------------
# Main: twin + pipeline online + AnomalyDetector + export
# ---------------------------------------------------------------------------

def main() -> None:
    dt = 0.01
    state_dim = 2
    meas_dim = 1
    n_steps = 800
    fault_start = 400
    efficiency_fault = 0.72

    # Twin: fisica + EKF + health + RUL
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

    # AnomalyDetector per soglie adattive (EMA + CUSUM)
    anomaly_detector = AnomalyDetector(
        alpha_ema=0.15,
        cusum_threshold=3.0,
        cusum_drift=0.0,
    )

    history = TwinHistory()
    # Fisica separata per simulare il "processo reale" (il twin usa `physics`)
    physics_truth = PumpPhysics(integrator=RK4Integrator())
    physics_truth.set_output_fn(lambda x, u: np.array([x[0]]))

    stream = generate_online_measurements(
        physics=physics_truth,
        x0=x0,
        dt=dt,
        n_steps=n_steps,
        fault_start_step=fault_start,
        efficiency_after_fault=efficiency_fault,
    )

    print("Pipeline online: twin + misure simulate con degradazione")
    print(f"  Step 0-{fault_start}: regime normale")
    print(f"  Step {fault_start}-{n_steps}: guasto simulato (efficienza {efficiency_fault})")
    print("Running...")

    for step, (u_t, y_t) in enumerate(stream):
        result = twin.step(u=u_t, measurement=y_t)

        # AnomalyDetector: aggiorna EMA e CUSUM sull'anomaly score
        score = anomaly_detector.anomaly_score(np.array([result.anomaly]))
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

        # Allarme CUSUM
        if cusum_p >= anomaly_detector.cusum_threshold or cusum_n >= anomaly_detector.cusum_threshold:
            print(f"  [step {step}] ALERT CUSUM: anomaly={result.anomaly:.4f} ema={ema:.4f} "
                  f"cusum_pos={cusum_p:.3f} cusum_neg={cusum_n:.3f} HI={result.health_indicator:.3f} RUL={result.rul}")

        # Log degradazione attesa
        if step == fault_start:
            print(f"  [step {step}] Ingresso guasto simulato (efficienza -> {efficiency_fault})")
        if step == fault_start + 50 and result.rul is not None:
            print(f"  [step {step}] RUL stimata: {result.rul:.2f} s")

    print(f"Steps: {len(history)}, stato finale: {twin.state}")
    print("Export CSV e grafici...")

    # Export CSV
    out_dir = Path(__file__).resolve().parent
    csv_path = out_dir / "online_degradation_history.csv"
    history.to_csv(csv_path, delimiter=",")
    print(f"  CSV: {csv_path}")

    # Grafici (opzionale)
    try:
        import matplotlib.pyplot as plt
        _plot_results(history, fault_start, out_dir)
    except ImportError:
        print("  (matplotlib non disponibile, grafici saltati)")

    print("Fatto.")


def _plot_results(history: TwinHistory, fault_start: int, out_dir: Path) -> None:
    """Genera grafici di verifica della pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = history.to_dict()
    t = data["time"]
    steps = np.arange(len(t))

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

    # Stato stimato
    ax = axes[0]
    ax.plot(steps, data["state_0"], label="portata stimata")
    ax.plot(steps, data["state_1"], label="pressione stimata")
    ax.axvline(fault_start, color="gray", linestyle="--", alpha=0.7, label="inizio guasto")
    ax.set_ylabel("Stato")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Anomaly + EMA
    ax = axes[1]
    ax.plot(steps, data["anomaly"], alpha=0.8, label="anomaly (EKF innov)")
    ax.plot(steps, data["ema"], label="EMA(anomaly)")
    ax.axvline(fault_start, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("Anomaly")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Health indicator + RUL
    ax = axes[2]
    ax.plot(steps, data["health_indicator"], label="Health Indicator")
    ax.plot(steps, data["rul"], alpha=0.9, label="RUL (s)")
    ax.axvline(fault_start, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(0.3, color="red", linestyle=":", alpha=0.5, label="HI fail")
    ax.set_ylabel("HI / RUL")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # CUSUM
    ax = axes[3]
    ax.plot(steps, data["cusum_pos"], label="CUSUM+")
    ax.plot(steps, data["cusum_neg"], label="CUSUM-")
    ax.axvline(fault_start, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel("CUSUM")
    ax.set_xlabel("Step")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Pipeline online: degradazione simulata (TwinOps)")
    plt.tight_layout()
    plot_path = out_dir / "online_degradation_plots.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
