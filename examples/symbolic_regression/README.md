# Physics model from data (Symbolic Regression)

Example using **SymbolicODEModel**: an ODE model whose right-hand side `rhs(x, u, t)` is learned from time series (e.g. from CSV) via symbolic regression (genetic programming, gplearn).

- **Physics model only**: same contract as `ODEModel` (`initialize`, `step`, `rhs`), usable in the twin instead of parametric models.
- **Dependencies**: `pip install twinops[symbolic]` (installs gplearn).

## Scripts

- **run_symbolic_ode.py** — Fit from (t, x, u) and step-by-step simulation (physics only).
- **run_twin_symbolic.py** — Full twin example: fit symbolic physics → TwinSystem (physics + EKF + Health) → loop with stream (u, y) → history → CSV export and plots.

## Running

```bash
pip install twinops[symbolic]
python run_symbolic_ode.py
python run_twin_symbolic.py   # full twin example
```

## Data

Data can come from CSV: columns `t`, `x_1..x_n`, `u_1..u_m`. Use `fit_from_timeseries(t, x, u)`; derivatives `dx/dt` are estimated with central differences.
