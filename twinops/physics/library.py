"""
Modelli fisici parametrici pronti all'uso.

Sottoclassi di ODEModel con rhs() già implementato; l'utente imposta solo i parametri
senza scrivere equazioni. Utili per comporre twin o come base da estendere.
"""

from typing import Any, Optional

import numpy as np

from twinops.physics.ode import ODEModel


class FirstOrderLag(ODEModel):
    """
    Sistema del primo ordine: dx/dt = (u - x) / tau.
    Stato scalare x; ingresso u (usa u[0] se u è vettore).
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
    Doppio integratore: posizione e velocità; dv/dt = u (accelerazione).
    Stato [pos, vel]; ingresso u = accelerazione (u[0]).
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
    Oscillatore meccanico: m*ddx + c*dx + k*x = F.
    Stato [pos, vel]; rhs: dx/dt = vel, dv/dt = (F - k*pos - c*vel) / m.
    F da ingresso u[0] (default).
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
    Livello in serbatoio: A * dh/dt = q_in - q_out(h).
    Stato [h]; q_out = k_out * sqrt(h) (orifizio) o lineare.
    Ingresso u = q_in (u[0]); parametri A, k_out.
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
    Pompa semplificata: stato [portata q, pressione p], ingresso [velocità omega].
    dq/dt = -a*q + b*omega, dp/dt = c*q - d*p.
    Coerente con l'esempio run_pump_twin.
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
    Oscillatore armonico: d²x/dt² + omega² x = 0.
    Stato [pos, vel]; rhs: dx/dt = vel, dv/dt = -omega²*pos.
    """

    def __init__(self, omega: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.omega_sq = float(omega) ** 2

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        x1, x2 = x[0], x[1]
        return np.array([x2, -self.omega_sq * x1])
