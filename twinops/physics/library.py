"""
Ready-to-use parametric physics models.

ODEModel subclasses with rhs() already implemented; the user only sets parameters
without writing equations. Useful for composing twins or as a base to extend.
"""

from typing import Any, Optional

import numpy as np

from twinops.physics.ode import ODEModel


class FirstOrderLag(ODEModel):
    """
    First-order system: dx/dt = (u - x) / tau.
    Scalar state x; input u (uses u[0] if u is a vector).
    """

    def __init__(self, tau: float = 1.0, u_index: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tau = float(tau)
        self.u_index = u_index

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        u_val = u[self.u_index] if u.size > self.u_index else 0.0
        return np.array([(u_val - x[0]) / self.tau])


class DoubleIntegrator(ODEModel):
    """
    Double integrator: position and velocity; dv/dt = u (acceleration).
    State [pos, vel]; input u = acceleration (u[0]).
    """

    def __init__(self, u_index: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.u_index = u_index

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        pos, vel = x[0], x[1]
        acc = u[self.u_index] if u.size > self.u_index else 0.0
        return np.array([vel, acc])


class MassSpringDamper(ODEModel):
    """
    Mechanical oscillator: m*ddx + c*dx + k*x = F.
    State [pos, vel]; rhs: dx/dt = vel, dv/dt = (F - k*pos - c*vel) / m.
    F from input u[0] (default).
    """

    def __init__(
        self,
        m: float = 1.0,
        k: float = 1.0,
        c: float = 0.1,
        force_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.m = float(m)
        self.k = float(k)
        self.c = float(c)
        self.force_index = force_index

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        pos, vel = x[0], x[1]
        F = u[self.force_index] if u.size > self.force_index else 0.0
        acc = (F - self.k * pos - self.c * vel) / self.m
        return np.array([vel, acc])


class TankLevel(ODEModel):
    """
    Tank level: A * dh/dt = q_in - q_out(h).
    State [h]; q_out = k_out * sqrt(h) (orifice) or linear.
    Input u = q_in (u[0]); parameters A, k_out.
    """

    def __init__(
        self,
        A: float = 1.0,
        k_out: float = 0.1,
        linear_out: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.A = float(A)
        self.k_out = float(k_out)
        self.linear_out = linear_out

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        h = max(x[0], 0.0)
        q_in = u[0] if u.size else 0.0
        if self.linear_out:
            q_out = self.k_out * h
        else:
            q_out = self.k_out * np.sqrt(h)
        dh = (q_in - q_out) / self.A
        return np.array([dh])


class PumpLike(ODEModel):
    """
    Simplified pump: state [flow q, pressure p], input [speed omega].
    dq/dt = -a*q + b*omega, dp/dt = c*q - d*p.
    Consistent with the run_pump_twin example.
    """

    def __init__(
        self,
        a: float = 0.1,
        b: float = 0.5,
        c: float = 0.3,
        d: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        q, p = x[0], x[1]
        omega = u[0] if u.size else 1.0
        dq = -self.a * q + self.b * omega
        dp = self.c * q - self.d * p
        return np.array([dq, dp])


class HarmonicOscillator(ODEModel):
    """
    Harmonic oscillator: d²x/dt² + omega² x = 0.
    State [pos, vel]; rhs: dx/dt = vel, dv/dt = -omega²*pos.
    """

    def __init__(self, omega: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.omega_sq = float(omega) ** 2

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        x1, x2 = x[0], x[1]
        return np.array([x2, -self.omega_sq * x1])
