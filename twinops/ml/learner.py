"""
Learner: learn a dynamics surrogate from a TwinComponent (physics)
or from raw data (states, inputs, next_states).
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch

from twinops.core.component import TwinComponent
from twinops.ml.dynamics import NeuralDynamicsModel, default_dynamics_net
from twinops.ml.training import train_dynamics


DataTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _generate_data_from_dynamics(
    dynamics: TwinComponent,
    state_dim: int,
    input_dim: int,
    dt: float,
    n_trajectories: int,
    steps_per_trajectory: int,
    x0_low: Optional[np.ndarray] = None,
    x0_high: Optional[np.ndarray] = None,
    u_low: float = 0.0,
    u_high: float = 1.0,
    seed: Optional[int] = None,
) -> DataTuple:
    """
    Generate (states, inputs, next_states) by simulating dynamics.
    x0 sampled in [x0_low, x0_high]; u sampled in [u_low, u_high] per step.
    """
    rng = np.random.default_rng(seed)
    if x0_low is None:
        x0_low = np.zeros(state_dim)
    if x0_high is None:
        x0_high = np.ones(state_dim)
    x0_low = np.atleast_1d(np.asarray(x0_low, dtype=float))
    x0_high = np.atleast_1d(np.asarray(x0_high, dtype=float))

    states_list, inputs_list, next_list = [], [], []
    for _ in range(n_trajectories):
        x = x0_low + rng.uniform(0, 1, size=state_dim) * (x0_high - x0_low)
        dynamics.initialize(state=x.copy())
        for _ in range(steps_per_trajectory):
            u = rng.uniform(u_low, u_high, size=input_dim).astype(np.float32)
            out = dynamics.step(state=x, u=u, dt=dt)
            x_next = out.get("state", x.copy())
            x_next = np.atleast_1d(np.asarray(x_next, dtype=np.float32))
            if x_next.size != state_dim:
                x_next = np.resize(x_next, state_dim) if x_next.size >= state_dim else np.pad(x_next, (0, state_dim - x_next.size))
            states_list.append(x.copy())
            inputs_list.append(u)
            next_list.append(x_next.copy())
            x = x_next.copy()
    return (
        np.array(states_list, dtype=np.float32),
        np.array(inputs_list, dtype=np.float32),
        np.array(next_list, dtype=np.float32),
    )


class Learner:
    """
    Learn a dynamics surrogate (NeuralDynamicsModel) from either:
    - a TwinComponent (physics), by generating data via simulation, or
    - raw data (states, inputs, next_states).
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        dt: float = 0.01,
        hidden: int = 64,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Args:
            state_dim: state dimension.
            input_dim: input dimension.
            dt: time step (used for dynamics generation and training).
            hidden: hidden units for the network (default_dynamics_net).
            device: PyTorch device (default: cpu).
        """
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt
        self.hidden = hidden
        self.device = device or torch.device("cpu")
        self._model: Optional[NeuralDynamicsModel] = None

    @property
    def model(self) -> Optional[NeuralDynamicsModel]:
        """Last trained model (None if learn() has not been called yet)."""
        return self._model

    def learn(
        self,
        dynamics: Optional[TwinComponent] = None,
        data: Optional[DataTuple] = None,
        n_trajectories: int = 100,
        steps_per_trajectory: int = 50,
        x0_low: Optional[Union[np.ndarray, list]] = None,
        x0_high: Optional[Union[np.ndarray, list]] = None,
        u_low: float = 0.0,
        u_high: float = 1.0,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> NeuralDynamicsModel:
        """
        Train a surrogate (NeuralDynamicsModel).

        Provide exactly one of:
        - dynamics: TwinComponent to generate data from (simulation).
        - data: tuple (states, inputs, next_states) with shapes (N, state_dim), (N, input_dim), (N, state_dim).

        Args:
            dynamics: physics model (TwinComponent) to generate data from.
            data: (states, inputs, next_states) ready to use.
            n_trajectories: number of trajectories when dynamics is provided.
            steps_per_trajectory: steps per trajectory when dynamics is provided.
            x0_low, x0_high: bounds to sample x0 (only with dynamics).
            u_low, u_high: bounds to sample u at each step (only with dynamics).
            epochs, batch_size, verbose: training options.
            seed: random seed for reproducibility (only with dynamics).

        Returns:
            Trained NeuralDynamicsModel (TwinComponent usable as physics).
        """
        if dynamics is not None and data is not None:
            raise ValueError("Provide only one of dynamics and data.")
        if dynamics is None and data is None:
            raise ValueError("Provide either dynamics or data.")

        if data is not None:
            states, inputs, next_states = data
            states = np.asarray(states, dtype=np.float32)
            inputs = np.asarray(inputs, dtype=np.float32)
            next_states = np.asarray(next_states, dtype=np.float32)
            train_data = (states, inputs, next_states)
        else:
            train_data = _generate_data_from_dynamics(
                dynamics,
                self.state_dim,
                self.input_dim,
                self.dt,
                n_trajectories=n_trajectories,
                steps_per_trajectory=steps_per_trajectory,
                x0_low=x0_low,
                x0_high=x0_high,
                u_low=u_low,
                u_high=u_high,
                seed=seed,
            )

        net = default_dynamics_net(
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            hidden=self.hidden,
        )
        surrogate = NeuralDynamicsModel(
            net,
            state_dim=self.state_dim,
            input_dim=self.input_dim,
            device=self.device,
        )
        train_dynamics(
            surrogate,
            train_data,
            dt=self.dt,
            epochs=epochs,
            batch_size=batch_size,
            device=self.device,
            verbose=verbose,
        )
        self._model = surrogate
        return surrogate


if __name__ == "__main__":
    # Example: learn a surrogate from a simulated harmonic oscillator (TwinComponent).
    # Harmonic oscillator: d²x/dt² + omega² x = 0, state [pos, vel], no input.
    from twinops.physics import HarmonicOscillator, RK4Integrator

    omega = 1.0
    dt = 0.05
    physics = HarmonicOscillator(omega=omega, integrator=RK4Integrator())

    learner = Learner(state_dim=2, input_dim=0, dt=dt, hidden=32)
    surrogate = learner.learn(
        dynamics=physics,
        n_trajectories=80,
        steps_per_trajectory=60,
        x0_low=np.array([-1.0, -1.0]),
        x0_high=np.array([1.0, 1.0]),
        epochs=150,
        batch_size=64,
        verbose=True,
        seed=42,
    )

    # Short rollout: compare true physics vs surrogate from same initial condition.
    x0 = np.array([0.5, 0.0])
    n_steps = 100
    x_true = x0.copy()
    x_surrogate = x0.copy()
    physics.initialize(state=x0.copy())
    surrogate.initialize()

    u_empty = np.array([], dtype=np.float32)
    for _ in range(n_steps):
        out_true = physics.step(state=x_true, u=u_empty, dt=dt)
        out_surr = surrogate.step(state=x_surrogate, u=u_empty, dt=dt)
        x_true = out_true["state"]
        x_surrogate = out_surr["state"]

    print("Example complete.")
    print(f"  True oscillator final state:      {x_true}")
    print(f"  Surrogate oscillator final state: {x_surrogate}")
