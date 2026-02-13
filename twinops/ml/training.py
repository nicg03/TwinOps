"""Utilities for training surrogates, residual models, and neural dynamics (Neural ODE, discrete-time)."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def compute_dx_dt_central(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Time derivatives with central differences (interior), forward/backward at endpoints.
    t: (n_steps,), x: (n_steps, n_states) -> (n_steps, n_states).
    """
    n_steps, n_states = x.shape
    dx_dt = np.empty_like(x)
    dt = np.diff(t)
    if dt.size == 0:
        return dx_dt
    for j in range(n_states):
        dx_dt[0, j] = (x[1, j] - x[0, j]) / dt[0] if n_steps > 1 else 0.0
        for i in range(1, n_steps - 1):
            dx_dt[i, j] = (x[i + 1, j] - x[i - 1, j]) / (t[i + 1] - t[i - 1])
        dx_dt[n_steps - 1, j] = (
            (x[n_steps - 1, j] - x[n_steps - 2, j]) / dt[-1] if n_steps > 1 else 0.0
        )
    return dx_dt


def prepare_ode_data_from_timeseries(
    t: np.ndarray,
    x: np.ndarray,
    u: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare (X, y) for training NeuralODEModel from time series.

    Args:
        t: times, shape (n_steps,).
        x: states, shape (n_steps, n_states).
        u: inputs, shape (n_steps,) or (n_steps, n_inputs). If None, uses zeros.

    Returns:
        X: shape (n_steps, n_states + n_inputs + 1), columns [x, u, t].
        y: shape (n_steps, n_states), estimated derivatives dx/dt.
    """
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float)
    n_steps, n_states = x.shape
    if u is None:
        u = np.zeros((n_steps, 0))
    else:
        u = np.asarray(u, dtype=float)
        if u.ndim == 1:
            u = u.reshape(-1, 1)
    n_inputs = u.shape[1]
    dx_dt = compute_dx_dt_central(t, x)
    # Design matrix [x, u, t]
    cols = [x[:, j].reshape(-1, 1) for j in range(n_states)]
    cols += [u[:, j].reshape(-1, 1) for j in range(n_inputs)]
    cols.append(t.reshape(-1, 1))
    X = np.hstack(cols)
    return X, dx_dt


def train_neural_ode(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 100,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train a NeuralODEModel on the rhs: target y = dx/dt, input X = [x, u, t].

    Args:
        model: instance of NeuralODEModel (exposes _net and _device).
        X: features (n_samples, n_states + n_inputs + 1).
        y: target derivatives (n_samples, n_states).
        loss_fn: loss(pred, target). Default: MSE.
        optimizer: default Adam lr=1e-3.
        epochs: number of epochs.
        batch_size: batch size.
        device: device (default: model._device if present).
        verbose: print loss.

    Returns:
        Dict with 'loss_history'.
    """
    net = getattr(model, "_net", model)
    dev = device or getattr(model, "_device", None) or torch.device("cpu")
    net = net.to(dev)
    net.train()
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = loss_fn or torch.nn.MSELoss()
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.6f}")
    net.eval()
    return {"loss_history": loss_history}


def train_dynamics(
    model: Any,
    train_data: Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ],
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 100,
    batch_size: int = 32,
    dt: float = 0.01,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train a NeuralDynamicsModel: target = state_next, input = [state, u, dt].

    Args:
        model: instance of NeuralDynamicsModel (exposes .model, .state_dim, .input_dim, .device).
        train_data: (states, inputs, next_states) or (states, inputs, dts, next_states).
                    states (N, state_dim), inputs (N, input_dim), next_states (N, state_dim).
                    If dts is provided, shape (N,) or (N,1).
        loss_fn: loss(pred, target). Default: MSE.
        optimizer: default Adam lr=1e-3.
        epochs: number of epochs.
        batch_size: batch size.
        dt: dt used for all samples when train_data has 3 elements (ignored when train_data has dts).
        device: device (default: model.device).
        verbose: print loss.

    Returns:
        Dict with 'loss_history'.
    """
    if len(train_data) == 4:
        states, inputs, dts, next_states = train_data
        dts = np.asarray(dts, dtype=np.float32)
        if dts.ndim == 1:
            dts = dts.reshape(-1, 1)
    else:
        states, inputs, next_states = train_data
        n = np.asarray(states).shape[0]
        dts = np.full((n, 1), dt, dtype=np.float32)
    states = np.asarray(states, dtype=np.float32)
    inputs = np.asarray(inputs, dtype=np.float32)
    next_states = np.asarray(next_states, dtype=np.float32)
    dev = device or getattr(model, "device", None) or torch.device("cpu")
    net = model.model.to(dev)
    net.train()
    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(inputs),
        torch.from_numpy(dts),
        torch.from_numpy(next_states),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = loss_fn or torch.nn.MSELoss()
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, u, dt_b, target in loader:
            x, u, dt_b, target = x.to(dev), u.to(dev), dt_b.to(dev), target.to(dev)
            optimizer.zero_grad()
            inp = torch.cat([x, u, dt_b], dim=1)
            pred = net(inp)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.6f}")
    net.eval()
    return {"loss_history": loss_history}


def train_residual(
    model: torch.nn.Module,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epochs: int = 100,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train a residual model: target = measurement - physics_output (or true state - predicted state).

    Args:
        model: nn.Module(state, u) -> correction
        train_data: (states, inputs, targets) where target is the residual to learn
        loss_fn: loss(prediction, target). Default: MSE
        optimizer: default Adam lr=1e-3
        epochs: number of epochs
        batch_size: batch size
        device: PyTorch device
        verbose: print loss each epoch

    Returns:
        Dict with 'loss_history' (and optional metrics).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    states, inputs, targets = train_data
    states = torch.from_numpy(np.asarray(states, dtype=np.float32))
    inputs = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    dataset = TensorDataset(states, inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = loss_fn or torch.nn.MSELoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, u, y in loader:
            x, u, y = x.to(device), u.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x, u)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{epochs} loss={avg_loss:.6f}")
    return {"loss_history": loss_history}
