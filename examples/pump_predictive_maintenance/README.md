# Example: pump predictive maintenance

Complete example (to be completed with data and training):

1. **Data generation**: pump ODE simulation + noise
2. **Training**: residual ML model (optional)
3. **Online**: TwinSystem with physics + EKF (+ residual) + health + RUL
4. **Anomaly detection and RUL**: thresholds and alarms

Run the minimal script:

```bash
python run_pump_twin.py
```
