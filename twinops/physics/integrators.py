"""
Numerical integrators for ODEs: time step x_{n+1} = step(f, x_n, u_n, t_n, dt).

Pure numerical level: no dependency on TwinComponent.
Interface: step(f, x, u, t, dt) -> x_next.
"""

from typing import Callable, Optional

import numpy as np

# Type for ODE right-hand side: (x, u, t) -> dx/dt
RHS = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def euler_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Explicit Euler, order 1: x_{n+1} = x_n + dt * f(x_n, u_n, t_n)."""
    return x + dt * f(x, u, t)


def heun_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Heun (improved Euler), order 2: Euler predictor + trapezoidal corrector."""
    k1 = f(x, u, t)
    k2 = f(x + dt * k1, u, t + dt)
    return x + 0.5 * dt * (k1 + k2)


def midpoint_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Midpoint (RK2): evaluation at interval center."""
    k1 = f(x, u, t)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt)
    return x + dt * k2


def rk4_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Runge-Kutta 4, order 4."""
    k1 = f(x, u, t)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, u, t + 0.5 * dt)
    k4 = f(x + dt * k3, u, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def backward_euler_step(
    f: RHS,
    x: np.ndarray,
    u: np.ndarray,
    t: float,
    dt: float,
    max_iter: int = 10,
    tol: float = 1e-10,
) -> np.ndarray:
    """Implicit Euler: x_{n+1} = x_n + dt * f(x_{n+1}, u, t+dt). Solved with Newton."""
    x_next = x + dt * f(x, u, t)
    for _ in range(max_iter):
        res = x_next - x - dt * f(x_next, u, t + dt)
        if np.linalg.norm(res) < tol:
            return x_next
        n = x.size
        J = np.eye(n)
        eps = 1e-8 * (1 + np.abs(x_next))
        for j in range(n):
            z_plus = x_next.copy()
            z_plus[j] += eps[j]
            J[:, j] = (z_plus - x - dt * f(z_plus, u, t + dt) - res) / eps[j]
        try:
            dx = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            return x_next
        x_next = x_next + dx
    return x_next


class EulerIntegrator:
    """Explicit Euler integrator, order 1."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return euler_step(f, x, u, t, dt)


class HeunIntegrator:
    """Heun integrator, order 2."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return heun_step(f, x, u, t, dt)


class MidpointIntegrator:
    """Midpoint integrator (RK2)."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return midpoint_step(f, x, u, t, dt)


class RK4Integrator:
    """Runge-Kutta 4 integrator, order 4."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return rk4_step(f, x, u, t, dt)


class BackwardEulerIntegrator:
    """Implicit Euler integrator (stable for stiff systems)."""

    def __init__(self, max_iter: int = 10, tol: float = 1e-10) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def step(self, f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return backward_euler_step(f, x, u, t, dt, max_iter=self.max_iter, tol=self.tol)
