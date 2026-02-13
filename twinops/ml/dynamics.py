"""
Discrete-time dynamics models based on neural networks.

NeuralDynamicsModel is a TwinComponent that simulates the system with
state_next = net(state, u, dt). Usable as physics in TwinSystem.
Training via twinops.ml.training.train_dynamics().
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from twinops.core.component import TwinComponent


def default_dynamics_net(
    state_dim: int,
    input_dim: int,
    hidden: int = 64,
) -> torch.nn.Module:
    """
    MLP mapping [state, u, dt] -> state_next.
    Input: concat(state, u, dt) of dim state_dim + input_dim + 1.
    """
    n_in = state_dim + input_dim + 1
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, state_dim),
    )


class NeuralDynamicsModel(TwinComponent):
    """
    Discrete-time dynamics model: x_{k+1} = net([x_k, u_k, dt]).

    The module receives a single tensor concat(state, u, dt) of shape
    (batch, state_dim + input_dim + 1) and returns state_next (batch, state_dim).
    Usable as physics in TwinSystem or in Series/Parallel.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        state_dim: int,
        input_dim: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """
        Args:
            model: nn.Module taking input (batch, state_dim+input_dim+1)
                   and returning (batch, state_dim).
            state_dim: state dimension.
            input_dim: input dimension.
            device: PyTorch device (default: cpu).
        """
        self.model = model
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def initialize(self, **kwargs: Any) -> None:
        self.model.eval()

    def step(
        self,
        *,
        state: np.ndarray,
        u: np.ndarray,
        dt: float,
        measurement: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            x = torch.from_numpy(np.atleast_1d(state).astype(np.float32)).unsqueeze(0).to(self.device)
            u_t = torch.from_numpy(np.atleast_1d(u).astype(np.float32)).unsqueeze(0).to(self.device)
            if x.shape[1] != self.state_dim:
                pad = torch.zeros(1, self.state_dim - x.shape[1], device=self.device)
                x = torch.cat([x, pad], dim=1)
            if u_t.shape[1] != self.input_dim:
                pad = torch.zeros(1, self.input_dim - u_t.shape[1], device=self.device)
                u_t = torch.cat([u_t, pad], dim=1)
            dt_t = torch.full((1, 1), dt, dtype=torch.float32, device=self.device)
            inp = torch.cat([x, u_t, dt_t], dim=1)
            state_next = self.model(inp)
            state_next = state_next.squeeze(0).cpu().numpy()
            if state_next.size > self.state_dim:
                state_next = state_next[: self.state_dim]
            elif state_next.size < self.state_dim:
                state_next = np.pad(state_next, (0, self.state_dim - state_next.size))
        output = np.asarray(state_next, dtype=float)
        return {"state": state_next, "output": output}

    def state_dict(self) -> Dict[str, Any]:
        return {"model_state_dict": self.model.state_dict()}
