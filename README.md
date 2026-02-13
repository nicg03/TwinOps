# TwinOps

**TwinOps** is a Python library for building **hybrid digital twins** that combine
**physics models**, **machine learning**, and **data assimilation** for monitoring,
predictive and prognostic maintenance of industrial systems.

The project aims to bridge the gap between:
- traditional physics-based simulations (accurate but rigid),
- purely data-driven models (fast but less interpretable),

providing a **modular, code-first engine** for *operational* digital twins, usable
in real time with sensor data.

---

## Project goal

TwinOps enables engineers and data scientists to:

- build a **hybrid digital twin** (physics + ML),
- continuously synchronize it with real sensor data,
- estimate **internal state** and **degrading parameters**,
- detect **anomalies** and estimate **Remaining Useful Life (RUL)**,
- do all of this **in Python**, without rewriting simulators or using monolithic tools.

TwinOps is the **computational engine** of the digital twin.

---

## Philosophy

- **Physics-first**: physics always comes first when available.
- **ML as correction**: machine learning corrects what physics does not capture.
- **Online & stateful**: the twin evolves over time and is always synchronized.
- **Modular**: every block is replaceable (physics, ML, filter, RUL).
- **Industrial-ready**: architecture designed for FMI/FMU and future integration.

---

## 📂 Repository structure

```
TwinOps/
├── pyproject.toml
├── requirements.txt
├── README.md
├── twinops/
│   ├── __init__.py
│   ├── core/           # system, component, signals, history
│   ├── physics/        # ode, fmi (stub)
│   ├── ml/             # residual, training
│   ├── estimation/     # ekf, residuals
│   ├── health/         # indicators, rul
│   └── io/             # streams, serializers
├── examples/
│   ├── pump_predictive_maintenance/
│   └── online_degradation/
└── tests/
```

---

## 📦 Module overview (file by file)

### `core/`
Core of the library.

- **system.py**
  Contains `TwinSystem`, the digital twin orchestrator.
  Handles the time loop, calls physics, ML and estimator, and produces output at each step.

- **component.py**
  Defines the base interface for all components (`initialize`, `step`, `state_dict`).

- **signals.py**
  Standardized definition of inputs/outputs (names, shape, units).

- **history.py**
  In-memory logging and utilities for data export (CSV, numpy).

---

### `physics/`
Physics models.

- **ode.py**
  Base class `ODEModel` and integrators (Euler, RK4).
  Differential equations of the physical system are defined here.

- **fmi.py** *(next phase)*
  Import/export of FMI/FMU models for industrial co-simulation.

---

### `ml/`
Machine learning.

- **residual.py**
  Wrapper for PyTorch models used as **correctors** of the physics model
  (residual learning).

- **training.py**
  Utilities for training surrogate and residual models.

---

### `estimation/`
Data assimilation and sensor synchronization.

- **ekf.py**
  Extended Kalman Filter (EKF) implementation for estimating:
  - internal state,
  - degrading parameters (friction, efficiency, etc.).

- **residuals.py**
  Anomaly score, trend, and threshold computation (EMA, CUSUM, etc.).

---

### `health/`
Health monitoring and prognostics.

- **indicators.py**
  Transforms estimated parameters or residuals into **Health Indicators (HI)**.

- **rul.py**
  Simple (or ML) models to estimate Remaining Useful Life from HIs.

---

### `io/`
Input/output and integration.

- **streams.py**
  Interface for batch or streaming data (real-time sensors).

- **serializers.py**
  Save/load configurations and twin snapshots.

---

### `examples/`
Reproducible use cases.

- **pump_predictive_maintenance/**
  Minimal example: pump digital twin with ODE physics + EKF, health and RUL.

- **online_degradation/**
  More complex example with **simulated online measurements** and **degradation**:
  - step-by-step generation of (u, y) as from sensors,
  - simulated fault (reduced efficiency) to verify anomaly, HI and RUL,
  - AnomalyDetector (EMA/CUSUM) for adaptive alarms,
  - CSV export and plots for analysis.

---

## 🚀 Minimal usage example

Simplified example of online digital twin usage.

```python
from twinops.core.system import TwinSystem
from twinops.physics.ode import PumpPhysics
from twinops.ml.residual import TorchResidualModel
from twinops.estimation.ekf import EKF

# build components
physics = PumpPhysics()
residual_model = TorchResidualModel(my_trained_torch_model)
ekf = EKF(state_dim=2, meas_dim=1)

# create twin
twin = TwinSystem(
    physics=physics,
    residual=residual_model,
    estimator=ekf,
    dt=0.01
)

twin.initialize(x0=[0.0, 0.0])

# online loop
for u_t, y_t in sensor_stream:
    result = twin.step(u=u_t, measurement=y_t)

    x_hat = result.state
    anomaly_score = result.anomaly
    rul = result.rul

    if anomaly_score > threshold:
        print("⚠️ Anomaly detected")

```
