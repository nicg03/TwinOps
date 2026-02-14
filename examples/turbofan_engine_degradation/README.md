# Example: Turbofan Engine Degradation (C-MAPSS) with TwinOps

Full TwinOps pipeline for the NASA C-MAPSS turbofan engine degradation dataset: data loading, dynamics training, twin with EKF/Health, and evaluation on the test set.

## Prerequisites

- Dataset in `data/turbofan_engine_degradation/`:
  - `train_FD001.txt`, `test_FD001.txt` (and optionally FD002–FD004)
- Dependencies: `numpy`, `torch`

## Running

From the repository root:

```bash
python examples/turbofan_engine_degradation/run_turbofan_twinops.py --fd FD001 --epochs 80
```

Options:

- `--fd FD001|FD002|FD003|FD004`: C-MAPSS sub-dataset (default: FD001)
- `--epochs N`: training epochs for NeuralDynamicsModel (default: 80)

## Pipeline

1. **Loading**: parse train/test; group data by unit (engine).
2. **State and input**: subset of 8 sensors as state/measurement, 3 operational settings as input.
3. **Training**: build `(x_k, u_k, x_{k+1})` from training series; train `NeuralDynamicsModel` with `train_dynamics()`.
4. **Twin**: `TwinSystem` with physics (trained model), EKF (state + anomaly), HealthIndicator (HI from anomaly).
5. **Test**: for each test unit, stream `(u, y)` → `twin.step()`; history (anomaly, HI).

## Notes

- State is a subset of sensors; for better performance you can use PCA, autoencoders, or literature-based sensor selection.
