# Examples: online pipeline with simulated degradation

Three examples to verify the TwinOps pipeline with **simulated online measurements** and process **degradation** (step pump, gradual pump, thermal system with fouling).

---

## 1. `run_online_degradation.py` — step fault

- **Online simulation**: measurements `(u_t, y_t)` generated **step-by-step** (as from real sensors).
- **Step fault**: up to step 400 the process is "healthy"; from step 400 efficiency drops to 0.72. Measurements diverge from the model → **anomaly** rises, **HI** drops, **RUL** updates.
- **AnomalyDetector (EMA/CUSUM)** and **export** to CSV/plots.

**Run:**

```bash
python3 examples/online_degradation/run_online_degradation.py
```

**Output:** `online_degradation_history.csv`, `online_degradation_plots.png`.

**Parameters:** `n_steps`, `fault_start` (default 400), `efficiency_after_fault` (0.72), `cusum_threshold`.

---

## 2. `run_gradual_degradation.py` — gradual degradation

- Same pipeline (twin + EKF + health + RUL + AnomalyDetector).
- **Gradual degradation**: efficiency stays 1.0 until step 200; between steps 200 and 600 it **linearly** decreases from 1.0 to 0.70; then stays 0.70. Verifies that anomaly, HI and RUL respond to a **slow trend** rather than a step.
- Export: `gradual_degradation_history.csv`, `gradual_degradation_plots.png`.

**Run:**

```bash
python3 examples/online_degradation/run_gradual_degradation.py
```

**Parameters:** `decay_start_step` (200), `decay_end_step` (600), `efficiency_min` (0.70), `cusum_threshold` (2.5).

---

## 3. `run_thermal_degradation.py` — thermal system with fouling

- **Different physics**: thermal system (temperature T, power P). ODE: `dT/dt = -alpha*(T - T_amb) + P/(m*c)`.
- **Degradation**: in the "real" process the coefficient alpha decreases linearly (fouling → worse heat transfer → measured T higher than expected) → anomaly rises, HI drops, RUL updates.
- Custom health indicator from anomaly (state is temperature, not in [0,2] as in the pump).
- Export: `thermal_degradation_history.csv`, `thermal_degradation_plots.png`.

**Run:**

```bash
python3 examples/online_degradation/run_thermal_degradation.py
```

**Parameters:** `decay_start_step` (250), `decay_end_step` (650), `alpha_nominal` (0.1), `alpha_min` (0.04), `cusum_threshold` (3.5).
