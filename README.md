<div align="center">

# RAWGRL
**Resource Allocation in Wi-Fi Networks using Graph Neural Networks and Deep Reinforcement Learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📌 Visión General / Overview
**RAWGRL** es un marco computacional avanzado de grado académico diseñado para resolver problemas de optimización estocástica no convexos en redes inalámbricas densas. Específicamente, este proyecto aborda la **asignación conjunta de canales y potencia de transmisión (JCAP)** en redes Wi-Fi empresariales utilizando inteligencia artificial.

La arquitectura combina el poder representacional de las **Graph Neural Networks (GNN)**, específicamente redes de atención espacial (GATv2), con algoritmos de vanguardia en **Deep Reinforcement Learning (DRL)** para aprender políticas de optimización de espectro altamente escalables y descentralizadas.

## 🔬 Marco Teórico y Motor Físico
A diferencia de simuladores simplificados, RAWGRL implementa un modelo de interferencia electromagnética riguroso fundamentado en el **Teorema de Shannon-Hartley**:

*   **Modelo de Propagación:** Pérdida de trayectoria Log-Distance con propagación en espacio libre (FSL).
*   **Interferencia Co-Canal (CCI) y Adyacente (ACI):** Modelado exacto de las máscaras espectrales IEEE 802.11 y superposición de frecuencias.
*   **Cálculo de SINR:** Ratio de Señal a Ruido e Interferencia calculado dinámicamente en topologías dinámicas.
*   **Sticky Client Logic:** Formalización matemática del proceso de asociación de clientes basándose en umbrales de RSSI y ganancia neta.

El entorno está formulado teóricamente como un **POMDP** (Proceso de Decisión de Markov Parcialmente Observable) en tiempo continuo, emulado a través de procesos de Poisson para arribos y distribuciones exponenciales para duración de conexiones.

## 🧠 Arquitecturas DRL Implementadas
El proyecto garantiza una **paridad arquitectónica estricta** para permitir comparaciones científicas precisas. Se evaluaron tres marcos de optimización analítica:

1.  **REINFORCE (Vanilla Policy Gradient):** Base de línea matemática. Minimiza el logaritmo negativo de la política ponderado por el retorno empírico ($G_t$).
2.  **A2C (Advantage Actor-Critic):** Integra una *Value Function* (Crítico) para reducir la varianza estructural del gradiente, empleando el Error de Diferencia Temporal (TD-Error).
3.  **PPO (Proximal Policy Optimization):** El estado del arte actual. Utiliza *Generalized Advantage Estimation (GAE)* y optimiza un *Clipped Surrogate Objective* ($\epsilon=0.2$) para prevenir divergencias por actualizaciones destructivas.

## 📂 Estructura del Repositorio
```text
RAWGRL/
├── REINFORCE/                # Implementación pura de Policy Gradient
├── A2C_Advantage Actor-Critic/ # Implementación de Arquitectura Actor-Crítico
├── PPO_Proximal Policy Optimization/ # Implementación con clipping (SOTA)
├── experiments_results/      # Resultados consolidados y métricas (CSV)
├── run_all_experiments.py    # Orquestador maestro para automatización JCAP
└── plot_comparison.py        # Generador de gráficas de convergencia (IEEE/Nature std)
```

## 🚀 Guía de Reproducibilidad (Getting Started)

### Prerrequisitos
Se recomienda el uso de un entorno virtual (`conda` o `venv`). Las dependencias centrales son: `torch`, `torch_geometric`, `pandas`, `matplotlib`, `seaborn`, `joblib`.

```bash
conda activate NetROML  # O tu entorno preferido
```

### Ejecución de Experimentos Masivos
Para garantizar reproducibilidad determinista y estricta entre modelos, se provee un script orquestador global. Este comando simulará el entorno para los 3 algoritmos y recolectará sus trayectorias:

```bash
python run_all_experiments.py --episodes 1000 --arrival_rate 3.0 --seed 42 --building_id 990
```

### Visualización Académica
Una vez finalizados los experimentos, el orquestador llamará automáticamente a `plot_comparison.py`. Este script aplica técnicas de suavizado exponencial y renderiza visualizaciones vectoriales (`.pdf` y `.png`) formateadas bajo estándares IEEE listas para anexarse en el manuscrito de tesis.

---
> *"Hacia la próxima generación de redes Wi-Fi cognitivas y autoconfigurables."*
