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
- detect **anomalies** and monitor **health**,
- do all of this **in Python**, without rewriting simulators or using monolithic tools.

TwinOps is the **computational engine** of the digital twin.

---

## Philosophy

- **Physics-first**: physics always comes first when available.
- **ML as correction**: machine learning corrects what physics does not capture.
- **Online & stateful**: the twin evolves over time and is always synchronized.
- **Modular**: every block is replaceable (physics, ML, filter, health).
- **Industrial-ready**: architecture designed for FMI/FMU and future integration.

---

## 📂 Repository structure

```
TwinOps/
├── pyproject.toml
├── requirements.txt
├── README.md
├── data/               # datasets (e.g. C-MAPSS); see data/README.md
│   └── turbofan_engine_degradation/
├── twinops/
│   ├── __init__.py
│   ├── core/           # system, component, signals, history
│   ├── physics/        # ode, symbolic, neural_ode, compose, library
│   ├── ml/             # residual, dynamics, training
│   ├── estimation/     # ekf, residuals
│   ├── health/         # health indicators
│   ├── io/             # streams, serializers
│   └── simulations/   # domain-specific complex dynamical systems
├── examples/
│   ├── pump_predictive_maintenance/
│   ├── online_degradation/
│   ├── symbolic_regression/
│   └── neural_dynamics/
├── docs/               # Sphinx documentation (see docs/README.rst)
└── tests/
```

---

## 📖 Documentation

API documentation is built with **Sphinx**. To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build
```

Then open `docs/_build/html/index.html` in a browser. See `docs/README.rst` for more options (e.g. `make -C docs html`).

To publish the docs online: push the repo to GitHub (or GitLab) and follow the **Publishing on Read the Docs** steps in `docs/README.rst`. The project includes a `.readthedocs.yaml` config so Read the Docs can build the docs automatically.

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

- **symbolic.py**
  `SymbolicODEModel`: ODE whose right-hand side is learned from data via
  symbolic regression (gplearn). Optional dependency: `twinops[symbolic]`.

- **neural_ode.py**
  `NeuralODEModel`: ODE whose right-hand side is a neural network (`dx/dt = net(x, u, t)`).
  Uses the same integrators and composition (Series/Parallel) as other ODE models.
  Train with `twinops.ml.training.train_neural_ode()`.

- **compose.py**
  `SeriesModel` and `ParallelModel` to combine any `TwinComponent` (e.g. ODE + ODE).

- **library.py**
  Ready-made parametric ODE models: `FirstOrderLag`, `DoubleIntegrator`,
  `MassSpringDamper`, `TankLevel`, `PumpLike`, `HarmonicOscillator`.

- **fmi.py** *(next phase)*
  Import/export of FMI/FMU models for industrial co-simulation.

---

### `ml/`
Learning system dynamics (surrogate model).

- **surrogate.py**
  Logical structure: interface/concept of “surrogate dynamics” (model learned from data, usable as physics in the twin).

- **dynamics.py**
  `NeuralDynamicsModel`: discrete dynamics `x_{k+1} = net(x_k, u_k, dt)`.
  Usable as **physics** in `TwinSystem` or in Series/Parallel.
  Training via `Learner.learn()` or `train_dynamics()`.

- **training.py**
  Training utilities for the surrogate:
  - **train_dynamics**: discrete model (state, u, dt) → state_next.
  - **train_neural_ode**: Neural ODE rhs from (X, y) with X = [x, u, t], y = dx/dt.
  - **prepare_ode_data_from_timeseries**, **compute_dx_dt_central**: prepare data from time series.

- **Learner** (in **learner.py**)
  Class with `learn(dynamics=...)` or `learn(data=...)`: builds a surrogate (TwinComponent) from a TwinComponent or raw data, for use as physics in the twin.

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

---

### `io/`
Input/output and integration.

- **streams.py**
  Interface for batch or streaming data (real-time sensors).

- **serializers.py**
  Save/load configurations and twin snapshots.

---

### `simulations/`
Domain-specific **complex dynamical systems** for full modeling and simulation.

Each file models a class of systems (as `ODEModel` / `TwinComponent`), usable as physics in `TwinSystem` or with `Learner.learn(dynamics=...)`. Visualization helpers live in `_utils.py` (phase portrait, state vs time, animations).

- **_oscillators.py**
  Coupled harmonic oscillators (e.g. chains with nearest-neighbor coupling). State: positions and velocities; suitable for phase portraits and time-series plots.

- **_thermal.py**
  Lumped thermal systems (temperatures, heat transfer). State: node temperatures; R/C network models.

- **_(future)**  
  Planned: **_multibody.py** (rigid bodies, pendulums, robot arms), **_electrical_networks.py** (RLC circuits), **_pendulums.py**, **_vehicles.py**, etc.

- **_utils.py**
  Visualization: `plot_phase_portrait()`, `plot_state_vs_time()` from `TwinHistory` or raw (time, state) arrays. Optional matplotlib.

---

### `examples/`
Reproducible use cases.

- **pump_predictive_maintenance/**
  Minimal example: pump digital twin with ODE physics + EKF and health.

- **online_degradation/**
  More complex example with **simulated online measurements** and **degradation**:
  - step-by-step generation of (u, y) as from sensors,
  - simulated fault (reduced efficiency) to verify anomaly and HI,
  - AnomalyDetector (EMA/CUSUM) for adaptive alarms,
  - CSV export and plots for analysis.

- **symbolic_regression/**
  Learn physics from time series with `SymbolicODEModel` (gplearn), then use it
  as physics in the twin. See `run_symbolic_ode.py` and `run_twin_symbolic.py`.

- **neural_dynamics/**
  Simulate a dynamical system with **neural networks**:
  - **NeuralODEModel**: continuous dynamics, rhs = network, trained on (t, x, u) → dx/dt.
  - **NeuralDynamicsModel**: discrete-time x_{k+1} = net(x_k, u_k, dt).
  Run `python examples/neural_dynamics/run_neural_dynamics.py` to train both
  on a harmonic oscillator and compare trajectories (true vs Neural ODE vs Neural Dynamics).

- **turbofan_engine_degradation/**
  Full pipeline for **NASA C-MAPSS** (turbofan degradation): load train/test, train
  NeuralDynamicsModel on sensor + settings data, run TwinSystem (physics + EKF + health)
  on test units.
  Run `python examples/turbofan_engine_degradation/run_turbofan_twinops.py --fd FD001 [--plot]`.

---

## 🚀 Minimal usage example

Simplified example of online digital twin usage.

```python
from twinops.core import TwinSystem
from twinops.physics import PumpLike
from twinops.ml.residual import TorchResidualModel
from twinops.estimation import EKF

# build components
physics = PumpLike()
residual_model = TorchResidualModel(my_trained_torch_model, state_dim=2, input_dim=1)
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
    health_indicator = result.health_indicator

    if anomaly_score > threshold:
        print("⚠️ Anomaly detected")
```

---

## 🧠 Neural dynamics (simulate with neural networks)

You can use **neural networks** as the physics model to simulate complex dynamical systems:

- **NeuralODEModel** (continuous): `dx/dt = net(x, u, t)`, same interface as `ODEModel`, works with integrators and Series/Parallel.
- **NeuralDynamicsModel** (discrete): `x_{k+1} = net(x_k, u_k, dt)`, usable as physics in `TwinSystem`.

**Example: train and simulate with Neural ODE**

```python
from twinops.physics import NeuralODEModel, RK4Integrator
from twinops.ml import prepare_ode_data_from_timeseries, train_neural_ode

# From time series (t, x, u), build (X, y) with y = dx/dt
X, y = prepare_ode_data_from_timeseries(t, x, u)

model = NeuralODEModel(n_states=2, n_inputs=1, hidden_size=64, integrator=RK4Integrator())
train_neural_ode(model, X, y, epochs=100)

# Use as physics in the twin (same as any ODEModel)
twin = TwinSystem(physics=model, estimator=ekf, dt=0.01)
twin.initialize(x0=[0.0, 0.0])
```

**Example: discrete-time neural dynamics**

```python
from twinops.ml import NeuralDynamicsModel, default_dynamics_net, train_dynamics

net = default_dynamics_net(state_dim=2, input_dim=1)
model = NeuralDynamicsModel(net, state_dim=2, input_dim=1)
train_dynamics(model, (states, inputs, next_states), dt=0.01, epochs=100)

# Use as physics in the twin
twin = TwinSystem(physics=model, estimator=ekf, dt=0.01)
```

See **examples/neural_dynamics/run_neural_dynamics.py** for a full run (harmonic oscillator, train both models, compare trajectories).

---

## 📊 Datasets

TwinOps can be used with real-world prognostic datasets. Below are the datasets currently supported or recommended for examples and benchmarks.

### Turbofan Engine Degradation Simulation (NASA C-MAPSS)

The **Turbofan Engine Degradation Simulation** dataset (NASA C-MAPSS — *Commercial Modular Aero-Propulsion System Simulation*) is a widely used benchmark for **prognostic and health management (PHM)** of turbofan engines.

- **Content**: time series of sensor readings (temperature, pressure, RPM, etc.) on simulated engines degrading until failure. Each unit (engine) has a full life cycle.
- **Variants**: FD001–FD004 with different operating conditions and number of fault modes.
- **Use with TwinOps**: suitable for digital twin examples with state/parameter estimation, anomaly detection, and health indicators.
- **Download**: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/) (C-MAPSS).

- **Where to put files**: place datasets in **`data/`** at the repository root. For C-MAPSS: `data/turbofan_engine_degradation/` (see `data/README.md` for download and layout instructions).

To integrate C-MAPSS in a TwinOps example, load the CSVs from `data/turbofan_engine_degradation/`, define signals and history, and feed the stream into `TwinSystem` (physics + estimator + health).
