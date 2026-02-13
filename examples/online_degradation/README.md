# Esempi: pipeline online con degradazione simulata

Tre esempi per verificare la pipeline TwinOps con **misure simulate online** e **degradazione** del processo (pompa a step, pompa graduale, sistema termico con fouling).

---

## 1. `run_online_degradation.py` — guasto a step

- **Simulazione online**: misure `(u_t, y_t)` generate **step-by-step** (come da sensori reali).
- **Guasto a step**: fino allo step 400 il processo è “sano”; da step 400 l’efficienza passa a 0.72. Le misure divergono dal modello → **anomaly** sale, **HI** scende, **RUL** si aggiorna.
- **AnomalyDetector (EMA/CUSUM)** e **export** CSV/grafici.

**Esecuzione:**

```bash
python3 examples/online_degradation/run_online_degradation.py
```

**Output:** `online_degradation_history.csv`, `online_degradation_plots.png`.

**Parametri:** `n_steps`, `fault_start` (default 400), `efficiency_after_fault` (0.72), `cusum_threshold`.

---

## 2. `run_gradual_degradation.py` — degradazione graduale

- Stessa pipeline (twin + EKF + health + RUL + AnomalyDetector).
- **Degradazione graduale**: efficienza resta 1.0 fino allo step 200; tra step 200 e 600 cala **linearmente** da 1.0 a 0.70; poi resta 0.70. Verifica che anomaly, HI e RUL reagiscano a un **trend lento** invece che a un salto.
- Export: `gradual_degradation_history.csv`, `gradual_degradation_plots.png`.

**Esecuzione:**

```bash
python3 examples/online_degradation/run_gradual_degradation.py
```

**Parametri:** `decay_start_step` (200), `decay_end_step` (600), `efficiency_min` (0.70), `cusum_threshold` (2.5).

---

## 3. `run_thermal_degradation.py` — sistema termico con fouling

- **Fisica diversa**: sistema termico (temperatura T, potenza P). ODE: `dT/dt = -alpha*(T - T_amb) + P/(m*c)`.
- **Degradazione**: nel processo “reale” il coefficiente alpha cala linearmente (fouling → peggior scambio termico → T misurata più alta del previsto) → anomaly sale, HI scende, RUL si aggiorna.
- Health indicator custom da anomaly (lo stato è temperatura, non in [0,2] come nella pompa).
- Export: `thermal_degradation_history.csv`, `thermal_degradation_plots.png`.

**Esecuzione:**

```bash
python3 examples/online_degradation/run_thermal_degradation.py
```

**Parametri:** `decay_start_step` (250), `decay_end_step` (650), `alpha_nominal` (0.1), `alpha_min` (0.04), `cusum_threshold` (3.5).
