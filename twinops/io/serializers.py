"""Save and load configurations and twin snapshots."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def _numpy_encoder(obj: Any) -> Any:
    """Convert numpy arrays to lists for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a configuration (dict) to JSON.
    Numpy arrays are converted to lists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy for serialization
    def convert(d: Any) -> Any:
        if isinstance(d, np.ndarray):
            return d.tolist()
        if isinstance(d, dict):
            return {k: convert(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [convert(x) for x in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d) if isinstance(d, np.floating) else int(d)
        return d

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(config), f, indent=2, ensure_ascii=False)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a configuration from JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(state_dict: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a twin snapshot (state_dict) to .npz for arrays and JSON for metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_arrays = {}
    meta = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            np_arrays[k] = v
        else:
            meta[k] = v
    np.savez(path.with_suffix(".npz"), **np_arrays)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    save_config(meta, meta_path)


def load_snapshot(path: Union[str, Path]) -> Dict[str, Any]:
    """Load snapshot from .npz + .meta.json."""
    path = Path(path)
    data = dict(np.load(path.with_suffix(".npz"), allow_pickle=True))
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        data.update(load_config(meta_path))
    return data
