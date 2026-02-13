"""Tests for SymbolicODEModel (requires gplearn, skipped if not installed)."""

import numpy as np
import pytest

pytest.importorskip("gplearn")

from twinops.physics import SymbolicODEModel, FirstOrderLag, RK4Integrator


def test_symbolic_ode_model_fit_from_timeseries_and_step() -> None:
    """Generate data from FirstOrderLag, fit SymbolicODEModel, verify step."""
    tau = 1.0
    dt = 0.05
    n_steps = 200
    t = np.arange(n_steps + 1) * dt
    u = 0.5 + 0.3 * np.sin(t)  # input
    # Simulate FirstOrderLag: dx/dt = (u - x) / tau
    physics_true = FirstOrderLag(tau=tau)
    x0 = np.array([0.0])
    x_vals = [x0.copy()]
    x = x0.copy()
    for k in range(n_steps):
        out = physics_true.step(state=x, u=np.array([u[k]]), dt=dt)
        x = out["state"]
        x_vals.append(x.copy())
    x_arr = np.array(x_vals)
    u_arr = u.reshape(-1, 1)

    model = SymbolicODEModel(
        n_states=1,
        n_inputs=1,
        population_size=200,
        generations=10,
        random_state=42,
    )
    model.fit_from_timeseries(t, x_arr, u_arr)

    # Must have regressors and rhs must be callable
    assert len(model._regressors) == 1
    rhs = model.rhs(np.array([x_arr[100, 0]]), np.array([u[100]]), t[100])
    assert rhs.shape == (1,)
    assert np.isfinite(rhs[0])

    # step must advance the state
    out = model.step(state=np.array([0.5]), u=np.array([0.5]), dt=dt)
    assert "state" in out and "output" in out
    assert out["state"].shape == (1,)
    assert np.isfinite(out["state"][0])

    # get_expressions returns strings
    exprs = model.get_expressions()
    assert len(exprs) == 1
    assert isinstance(exprs[0], str)


def test_symbolic_ode_model_fit_direct() -> None:
    """Fit with direct X, y (without time series)."""
    n = 100
    # dx/dt = -x + u (first order)
    X = np.hstack([
        np.random.randn(n, 1),
        np.random.randn(n, 1),
        np.linspace(0, 10, n).reshape(-1, 1),
    ])
    y = (-X[:, 0] + X[:, 1]).reshape(-1, 1)  # dx/dt

    model = SymbolicODEModel(n_states=1, n_inputs=1, generations=5, random_state=1)
    model.fit(X, y)
    assert len(model._regressors) == 1
    r = model.rhs(np.array([1.0]), np.array([2.0]), 0.0)
    assert r.shape == (1,) and np.isfinite(r[0])
