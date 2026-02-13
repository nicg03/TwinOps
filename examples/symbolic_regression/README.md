# Modello fisico da dati (Symbolic Regression)

Esempio che usa **SymbolicODEModel**: un modello ODE il cui termine destro `rhs(x, u, t)` è appreso da serie temporali (es. da CSV) tramite symbolic regression (genetic programming, gplearn).

- **Solo modello fisico**: stesso contratto di `ODEModel` (`initialize`, `step`, `rhs`), utilizzabile nel twin al posto di modelli parametrici.
- **Dipendenze**: `pip install twinops[symbolic]` (installa gplearn).

## Script

- **run_symbolic_ode.py** — Fit da (t, x, u) e simulazione step-by-step (solo fisica).
- **run_twin_symbolic.py** — Esempio completo twin: fit fisica simbolica → TwinSystem (physics + EKF + Health + RUL) → loop con stream (u, y) → history → export CSV e grafici.

## Esecuzione

```bash
pip install twinops[symbolic]
python run_symbolic_ode.py
python run_twin_symbolic.py   # esempio completo twin
```

## Dati

I dati possono provenire da CSV: colonne `t`, `x_1..x_n`, `u_1..u_m`. Si usa `fit_from_timeseries(t, x, u)`; le derivate `dx/dt` sono stimate con differenze centrali.
