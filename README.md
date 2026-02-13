# TwinOps

**TwinOps** è una libreria Python per costruire **digital twin ibridi** che combinano  
**modelli fisici**, **machine learning** e **data assimilation** per monitoraggio,  
manutenzione predittiva e prognostica di sistemi industriali.

Il progetto nasce per colmare il gap tra:
- simulazioni fisiche tradizionali (accurate ma rigide),
- modelli puramente data-driven (veloci ma poco interpretabili),

fornendo un **motore modulare e code-first** per digital twin *operativi*, utilizzabili
in tempo reale con dati da sensori.

---

## Obiettivo del progetto

L’obiettivo di TwinOps è permettere a un ingegnere o data scientist di:

- costruire un **digital twin ibrido** (fisica + ML),
- sincronizzarlo continuamente con dati di sensori reali,
- stimare **stato interno** e **parametri degradanti**,
- individuare **anomalie** e stimare la **Remaining Useful Life (RUL)**,
- farlo **in Python**, senza dover riscrivere simulatori o usare tool monolitici.

TwinOps è il **motore computazionale** del digital twin.

---

## Filosofia

- **Physics-first**: la fisica viene sempre prima, quando disponibile.
- **ML as correction**: il machine learning corregge ciò che la fisica non cattura.
- **Online & stateful**: il twin vive nel tempo ed è sempre sincronizzato.
- **Modulare**: ogni blocco è sostituibile (fisica, ML, filtro, RUL).
- **Industrial-ready**: architettura pensata per FMI/FMU e integrazione futura.

---

## 📂 Struttura della repository



---

## 📦 Spiegazione dei moduli (file per file)

### `core/`
Cuore della libreria.

- **system.py**  
  Contiene `TwinSystem`, l’orchestratore del digital twin.  
  Gestisce il loop temporale, chiama fisica, ML ed estimatore, e produce output a ogni step.

- **component.py**  
  Definisce l’interfaccia base per tutti i componenti (`initialize`, `step`, `state_dict`).

- **signals.py**  
  Definizione standardizzata di ingressi/uscite (nomi, shape, unità).

- **history.py**  
  Logging in memoria e utility per esportare dati (CSV, numpy).

---

### `physics/`
Modelli fisici.

- **ode.py**  
  Classe base `ODEModel` e integratori (Euler, RK4).  
  Qui si definiscono le equazioni differenziali del sistema fisico.

- **fmi.py** *(fase successiva)*  
  Import/export di modelli FMI/FMU per co-simulazione industriale.

---

### `ml/`
Machine learning.

- **residual.py**  
  Wrapper per modelli PyTorch usati come **correttori** del modello fisico
  (residual learning).

- **training.py**  
  Utility per addestrare surrogate e modelli residuali.

---

### `estimation/`
Data assimilation e sincronizzazione con sensori.

- **ekf.py**  
  Implementazione di Extended Kalman Filter (EKF) per stima di:
  - stato interno,
  - parametri degradanti (attrito, efficienza, ecc.).

- **residuals.py**  
  Calcolo di anomaly score, trend, soglie (EMA, CUSUM, ecc.).

---

### `health/`
Health monitoring e prognostica.

- **indicators.py**  
  Trasforma parametri stimati o residui in **Health Indicators (HI)**.

- **rul.py**  
  Modelli semplici (o ML) per stimare la Remaining Useful Life a partire dagli HI.

---

### `io/`
Input/output e integrazione.

- **streams.py**  
  Interfaccia per dati batch o streaming (sensori real-time).

- **serializers.py**  
  Salvataggio/caricamento di configurazioni e snapshot del twin.

---

### `examples/`
Casi d’uso riproducibili.

- **pump_predictive_maintenance/**  
  Esempio completo di manutenzione predittiva per una pompa industriale:
  - generazione dati,
  - training modello ML,
  - esecuzione online del digital twin,
  - anomaly detection e RUL.

---

## 🚀 Esempio minimo di utilizzo

Esempio semplificato di utilizzo online del digital twin.

```python
from twinops.core.system import TwinSystem
from twinops.physics.ode import PumpPhysics
from twinops.ml.residual import TorchResidualModel
from twinops.estimation.ekf import EKF

# costruzione dei componenti
physics = PumpPhysics()
residual_model = TorchResidualModel(my_trained_torch_model)
ekf = EKF(state_dim=2, meas_dim=1)

# creazione del twin
twin = TwinSystem(
    physics=physics,
    residual=residual_model,
    estimator=ekf,
    dt=0.01
)

twin.initialize(x0=[0.0, 0.0])

# loop online
for u_t, y_t in sensor_stream:
    result = twin.step(u=u_t, measurement=y_t)

    x_hat = result.state
    anomaly_score = result.anomaly
    rul = result.rul

    if anomaly_score > threshold:
        print("⚠️ Anomalia rilevata")

```

