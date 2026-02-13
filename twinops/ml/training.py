"""Utility per addestrare surrogate e modelli residuali."""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    Addestra un modello residuale: target = misura - output_fisica (o stato vero - stato predetto).

    Args:
        model: nn.Module(state, u) -> correction
        train_data: (states, inputs, targets) dove target è il residuo da apprendere
        loss_fn: loss (prediction, target). Default: MSE
        optimizer: default Adam lr=1e-3
        epochs: numero epoche
        batch_size: batch size
        device: device PyTorch
        verbose: stampa loss ogni epoca

    Returns:
        Dict con 'loss_history' (e eventuali metriche).
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
