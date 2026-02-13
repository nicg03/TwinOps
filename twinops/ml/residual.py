"""Wrapper for PyTorch models used as correctors of the physics model (residual learning)."""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from twinops.core.component import TwinComponent


class TorchResidualModel(TwinComponent):
    """
    Adapts a PyTorch nn.Module to the TwinComponent interface.
    Step receives state and input, returns a correction to add to the predicted state.
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
            model: module that takes (state, u) and returns correction (same shape as state).
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
        """No special initialization for the model."""
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
            # Ensure correct dimensions (padding if needed)
            if x.shape[1] != self.state_dim:
                pad = torch.zeros(1, self.state_dim - x.shape[1], device=self.device)
                x = torch.cat([x, pad], dim=1)
            if u_t.shape[1] != self.input_dim:
                pad = torch.zeros(1, self.input_dim - u_t.shape[1], device=self.device)
                u_t = torch.cat([u_t, pad], dim=1)
            correction = self.model(x, u_t)
            correction_np = correction.squeeze(0).cpu().numpy()
            # Return only the first state_dim components if the model returns more
            if correction_np.size > self.state_dim:
                correction_np = correction_np[: self.state_dim]
            elif correction_np.size < self.state_dim:
                correction_np = np.pad(correction_np, (0, self.state_dim - correction_np.size))
        return {"correction": correction_np}

    def state_dict(self) -> Dict[str, Any]:
        return {"model_state_dict": self.model.state_dict()}
