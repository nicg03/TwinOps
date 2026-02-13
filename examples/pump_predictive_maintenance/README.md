# Esempio: manutenzione predittiva pompa

Esempio completo (da completare con dati e training):

1. **Generazione dati**: simulazione ODE pompa + rumore
2. **Training**: modello ML residuale (opzionale)
3. **Online**: TwinSystem con physics + EKF (+ residual) + health + RUL
4. **Anomaly detection e RUL**: soglie e allarmi

Esegui lo script minimo:

```bash
python run_pump_twin.py
```
