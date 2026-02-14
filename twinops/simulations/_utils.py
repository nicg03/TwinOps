"""
Visualization utilities for simulation: phase portrait, state vs time, animations.

All functions accept either a TwinHistory (with keys 'time' and 'state' or state components)
or raw arrays (time, state). Matplotlib is optional; if not installed, functions
raise ImportError or return None.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def _get_time_and_state(
    history: Optional[Any] = None,
    time: Optional[np.ndarray] = None,
    state: Optional[np.ndarray] = None,
) -> tuple:
    """Resolve (time, state) from history or from (time, state) arrays."""
    if history is not None:
        data = getattr(history, "to_dict", lambda: {})()
        if not data:
            data = getattr(history, "_data", {})
        t = data.get("time")
        if t is None and "state" in data:
            t = np.arange(len(data["state"]))
        st = data.get("state")
        if st is not None and isinstance(st, (list, np.ndarray)):
            if isinstance(st, list) and len(st) and hasattr(st[0], "__len__"):
                state_arr = np.array(st)
            else:
                state_arr = np.asarray(st)
        else:
            parts = [data[k] for k in sorted(data) if k.startswith("state_")]
            if parts:
                state_arr = np.column_stack(parts)
            else:
                raise ValueError("History has no 'state' or 'state_*' keys.")
        time_arr = np.asarray(t, dtype=float).ravel() if t is not None else np.arange(len(state_arr))
        return time_arr, state_arr
    if time is not None and state is not None:
        return np.asarray(time).ravel(), np.asarray(state)
    raise ValueError("Provide either history= or (time=, state=).")


def plot_phase_portrait(
    history: Optional[Any] = None,
    time: Optional[np.ndarray] = None,
    state: Optional[np.ndarray] = None,
    x_idx: int = 0,
    y_idx: int = 1,
    ax: Optional[Any] = None,
    xlabel: str = "x",
    ylabel: str = "v",
    title: str = "Phase portrait",
    **kwargs: Any,
) -> Any:
    """
    Plot state[:, x_idx] vs state[:, y_idx] (phase portrait).

    Args:
        history: TwinHistory with 'state' (or state_0, state_1, ...) and optional 'time'.
        time, state: optional raw arrays if history is not used.
        x_idx, y_idx: state indices to plot (default 0, 1 for position vs velocity).
        ax: matplotlib axes (if None, creates new figure).
        xlabel, ylabel, title: axis and title labels.
        **kwargs: passed to ax.plot().

    Returns:
        matplotlib axes (or None if matplotlib not available).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_phase_portrait.")
    t, st = _get_time_and_state(history=history, time=time, state=state)
    if st.ndim == 1:
        st = st.reshape(-1, 1)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 5))
    xi = min(x_idx, st.shape[1] - 1)
    yi = min(y_idx, st.shape[1] - 1)
    ax.plot(st[:, xi], st[:, yi], **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_state_vs_time(
    history: Optional[Any] = None,
    time: Optional[np.ndarray] = None,
    state: Optional[np.ndarray] = None,
    state_names: Optional[List[str]] = None,
    ax: Optional[Any] = None,
    title: str = "State vs time",
    **kwargs: Any,
) -> Any:
    """
    Plot each state component vs time (one subplot per component or one plot with all).

    Args:
        history: TwinHistory with 'state' (or state_0, state_1, ...) and optional 'time'.
        time, state: optional raw arrays if history is not used.
        state_names: labels for each state (e.g. ["pos", "vel"]).
        ax: matplotlib axes (if None, creates new figure with subplots).
        title: figure/subplot title.
        **kwargs: passed to ax.plot().

    Returns:
        matplotlib figure or axes (or None if matplotlib not available).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_state_vs_time.")
    t, st = _get_time_and_state(history=history, time=time, state=state)
    if st.ndim == 1:
        st = st.reshape(-1, 1)
    n_states = st.shape[1]
    if state_names is None:
        state_names = [f"x{i}" for i in range(n_states)]
    while len(state_names) < n_states:
        state_names.append(f"x{len(state_names)}")

    if ax is None:
        fig, axes = plt.subplots(n_states, 1, sharex=True, figsize=(8, max(2 * n_states, 4)))
        if n_states == 1:
            axes = [axes]
        fig_ref = fig
    else:
        fig_ref = ax.figure
        axes = [ax] * n_states
    for i in range(n_states):
        a = axes[i]
        a.plot(t, st[:, i], label=state_names[i], **kwargs)
        a.set_ylabel(state_names[i])
        a.legend(loc="upper right", fontsize=8)
        a.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    if title:
        fig_ref.suptitle(title)
    try:
        fig_ref.tight_layout()
    except Exception:
        pass
    return fig_ref if ax is None else ax
