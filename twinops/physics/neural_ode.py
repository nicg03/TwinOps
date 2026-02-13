"""
ODE physics model with right-hand side given by a neural network (Neural ODE).

NeuralODEModel is an ODEModel whose rhs(x, u, t) is computed by a PyTorch nn.Module.
Use with integrators (RK4, Euler, ...) and compose (Series/Parallel) like any ODEModel.
Train with twinops.ml.training.train_neural_ode().
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from twinops.physics.ode import ODEModel

try:
    import torch
except ImportError:
    torch = None  # type: ignore[misc, assignment]


def _default_rhs_net(n_states: int, n_inputs: int, hidden: int = 64) -> "torch.nn.Module":
    """Build a simple MLP for rhs: input [x, u, t], output dx/dt."""
    if torch is None:
        raise ImportError("NeuralODEModel requires PyTorch. Install with: pip install torch")
    n_in = n_states + n_inputs + 1
    return torch.nn.Sequential(
        torch.nn.Linear(n_in, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, n_states),
    )


class NeuralODEModel(ODEModel):
    """
    ODE model whose rhs is given by a neural network: dx/dt = net(x, u, t).

    The network must accept input of shape (batch, n_states + n_inputs + 1)
    and return shape (batch, n_states). Use train_neural_ode() to train from
    (X, y) with X = [x, u, t], y = dx/dt.
    """

    def __init__(
        self,
        n_states: int,
        n_inputs: int = 0,
        net_rhs: Optional[Any] = None,
        integrator: Optional[Any] = None,
        *,
        hidden_size: int = 64,
        device: Optional[Union[str, "torch.device"]] = None,
    ) -> None:
        """
        Args:
            n_states: number of state variables.
            n_inputs: number of inputs (default 0).
            net_rhs: PyTorch nn.Module with input (batch, n_states+n_inputs+1), output (batch, n_states).
                     If None, a default MLP is created.
            integrator: ODE integrator (default RK4).
            hidden_size: hidden size for default MLP (used only if net_rhs is None).
            device: PyTorch device (default: cpu).
        """
        super().__init__(integrator=integrator)
        if torch is None:
            raise ImportError("NeuralODEModel requires PyTorch. Install with: pip install torch")
        self.n_states = n_states
        self.n_inputs = n_inputs
        self._device = device or torch.device("cpu")
        if net_rhs is not None:
            self._net = net_rhs.to(self._device)
        else:
            self._net = _default_rhs_net(n_states, n_inputs, hidden_size).to(self._device)
        self._net.eval()

    def _rhs_tensor(self, x: "torch.Tensor", u: "torch.Tensor", t: float) -> "torch.Tensor":
        """Compute rhs in tensor form (batch dimension supported)."""
        batch = x.shape[0]
        t_scalar = torch.full((batch, 1), t, dtype=x.dtype, device=x.device)
        inp = torch.cat([x, u, t_scalar], dim=1)
        return self._net(inp)

    def rhs(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        """ODE right-hand side: dx/dt = rhs(x, u, t). Evaluates the neural network."""
        x = np.atleast_1d(x).astype(np.float32)
        u = np.atleast_1d(u).astype(np.float32)
        if u.size < self.n_inputs:
            u = np.resize(u, self.n_inputs)
        with torch.no_grad():
            x_t = torch.from_numpy(x).unsqueeze(0).to(self._device)
            u_t = torch.from_numpy(u).unsqueeze(0).to(self._device)
            out = self._rhs_tensor(x_t, u_t, t)
            return out.squeeze(0).cpu().numpy()

    def initialize(self, **kwargs: Any) -> None:
        super().initialize(**kwargs)
        self._net.eval()

    def state_dict(self) -> Dict[str, Any]:
        d = super().state_dict()
        d["n_states"] = self.n_states
        d["n_inputs"] = self.n_inputs
        d["model_state_dict"] = self._net.state_dict()
        return d

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore network weights from state_dict (e.g. after training)."""
        if "model_state_dict" in state_dict:
            self._net.load_state_dict(state_dict["model_state_dict"])
