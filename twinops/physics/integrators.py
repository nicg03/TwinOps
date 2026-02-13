"""
Integratori numerici per ODE: passo temporale x_{n+1} = step(f, x_n, u_n, t_n, dt).

Livello numerico puro: nessuna dipendenza da TwinComponent.
Interfaccia: step(f, x, u, t, dt) -> x_next.
"""

from typing import Callable, Optional

import numpy as np

# Tipo per il termine destro ODE: (x, u, t) -> dx/dt
RHS = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def euler_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Eulero esplicito, ordine 1: x_{n+1} = x_n + dt * f(x_n, u_n, t_n)."""
    return x + dt * f(x, u, t)


def heun_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Heun (Eulero migliorato), ordine 2: predittore Eulero + correttore trapezio."""
    k1 = f(x, u, t)
    k2 = f(x + dt * k1, u, t + dt)
    return x + 0.5 * dt * (k1 + k2)


def midpoint_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Midpoint (RK2): valutazione al centro dell'intervallo."""
    k1 = f(x, u, t)
    k2 = f(x + 0.5 * dt * k1, u, t + 0.5 * dt)
    return x + dt * k2


def rk4_step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Runge-Kutta 4, ordine 4."""
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
    """Eulero implicito: x_{n+1} = x_n + dt * f(x_{n+1}, u, t+dt). Risolto con Newton."""
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
    """Integratore Eulero esplicito, ordine 1."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return euler_step(f, x, u, t, dt)


class HeunIntegrator:
    """Integratore Heun, ordine 2."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return heun_step(f, x, u, t, dt)


class MidpointIntegrator:
    """Integratore Midpoint (RK2)."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return midpoint_step(f, x, u, t, dt)


class RK4Integrator:
    """Integratore Runge-Kutta 4, ordine 4."""

    @staticmethod
    def step(f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return rk4_step(f, x, u, t, dt)


class BackwardEulerIntegrator:
    """Integratore Eulero implicito (stabile per sistemi stiff)."""

    def __init__(self, max_iter: int = 10, tol: float = 1e-10) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def step(self, f: RHS, x: np.ndarray, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        return backward_euler_step(f, x, u, t, dt, max_iter=self.max_iter, tol=self.tol)
