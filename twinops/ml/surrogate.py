"""
Struttura logica: dinamica surrogata.

Questo modulo rappresenta il concetto di "modello surrogato" della dinamica del sistema:
un modello appreso da dati (serie temporali o coppie (stato, ingresso, stato_next))
che espone la stessa interfaccia di un modello fisico (TwinComponent) e può essere
usato come physics in TwinSystem.

Flusso logico:
- Input: dati (es. serie temporali (t, x, u) o triple (x_k, u_k, x_{k+1})).
- learn(...): addestra un modello (es. NeuralDynamicsModel) e restituisce un
  TwinComponent pronto per creare/importare il twin e simulare la dinamica.
- Output: oggetto con initialize(), step(state, u, dt, ...) utilizzabile come
  physics in TwinSystem.

Implementazioni concrete della dinamica surrogata sono in dynamics.py
(NeuralDynamicsModel); le funzioni di training sono in training.py.
"""
