Examples
========

TwinOps includes several reproducible examples in the ``examples/`` folder.

.. _examples-pump:

Pump predictive maintenance
---------------------------
Minimal example: digital twin of a pump with ODE model (``PumpLike``), EKF,
health, and RUL.

**Path:** ``examples/pump_predictive_maintenance/run_pump_twin.py``

.. _examples-online:

Online degradation
------------------
Example with simulated online measurements and degradation (reduced efficiency),
AnomalyDetector (EMA/CUSUM), CSV export, and plots.

**Path:** ``examples/online_degradation/``

.. _examples-symbolic:

Symbolic regression
-------------------
Learning physics from time series with ``SymbolicODEModel`` (gplearn)
and using it in the twin.

**Path:** ``examples/symbolic_regression/``

.. _examples-neural:

Neural dynamics
---------------
Simulation with neural networks: NeuralODEModel (continuous) and NeuralDynamicsModel
(discrete), trained on a harmonic oscillator.

**Path:** ``examples/neural_dynamics/run_neural_dynamics.py``

.. _examples-turbofan:

Turbofan engine degradation (C-MAPSS)
--------------------------------------
Full pipeline on the NASA C-MAPSS dataset: load train/test, train
NeuralDynamicsModel, TwinSystem with EKF + Health + SimpleRUL on test units,
evaluate RMSE/MAE RUL.

**Path:** ``examples/turbofan_engine_degradation/run_turbofan_twinops.py``

**Run:** ``python examples/turbofan_engine_degradation/run_turbofan_twinops.py --fd FD001 [--plot]``
