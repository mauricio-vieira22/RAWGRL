r"""
evaluate.py — Inferencia Empírica y Evaluación de Políticas (NetROML — PPO).

Este módulo ejecuta la política Actor $\pi_{\theta^*}$ pre-entrenada del PPO
sobre trayectorias enteras del POMDP, con el fin de cuantificar de forma imparcial
el rendimiento del modelo (G_t y Sum_rate) bajo dinámicas de tráfico estocásticas.

Mecanismos de Selección de Acción (Inferencia)
----------------------------------------------
El entorno se inicializa en $t=0$ y el agente avanza temporalmente procesando $s_t$:
- **Modo Greedy (Por Defecto)**: 
  El agente maximiza determinísticamente las salidas proyectadas del GNN.
  $$ a_t = \arg\max \pi_{\theta^*}(a_t | s_t) $$
  Esto simula la política final implementada en producción.
- **Modo Estocástico (`--stochastic`)**: 
  El agente muestrea de la distribución categórica como en la fase de exploración.
  $$ a_t \sim Categorical(\pi_{\theta^*}(a_t | s_t)) $$

Monitoreo de Variables Ocultas POMDP
------------------------------------
Durante la inferencia, extraemos sistemáticamente las variables ocultas del modelo Markoviano:
- $\bar{\epsilon}_t$: Media de los tiempos restantes de vida (clientes).
- $\delta_t$: Tiempo hasta el siguiente nacimiento en el sistema.
Esto permite la generación de analíticas para probar empíricamente si la inyección 
de este estado parcialmente observable estabiliza el entorno.
"""

from __future__ import annotations

# El patch al inspector de PyG debe aplicarse antes de cualquier importación
# de torch_geometric, incluyendo las transitivas desde los módulos del proyecto.
import torch_geometric.inspector as _pyg_inspector

try:
    _original_type_repr = _pyg_inspector.type_repr

    def _safe_type_repr(obj, _globals=None):
        try:
            return _original_type_repr(obj, _globals)
        except AttributeError as exc:
            if "'_name'" in str(exc):
                return "Union"
            raise

    _pyg_inspector.type_repr = _safe_type_repr
except Exception:
    pass

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.data_loader import load_distributions  
from model.gnn_model import GNN                             
from model.network_graph_env import NetworkGraphEnv         

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def select_device() -> torch.device:
    """Selecciona el dispositivo de cómputo disponible (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loop de evaluación
# ---------------------------------------------------------------------------

def run_episode(
    model: GNN,
    env: NetworkGraphEnv,
    device: torch.device,
    seed: int | None,
    stochastic: bool,
) -> dict:
    """Ejecuta un episodio completo y devuelve métricas.

    Parameters
    ----------
    model:
        Política GNN en modo eval (``model.eval()`` ya aplicado).
    env:
        Entorno ya configurado.
    device:
        Dispositivo donde reside el modelo.
    seed:
        Semilla para este episodio específico.
    stochastic:
        Si True, muestrea de la distribución de política (π).
        Si False, selección greedy (argmax) — modo por defecto.

    Returns
    -------
    dict con retorno total, tasa total, tasa media por step y duración.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    ep_rates: list[float] = []
    ep_delta_t: list[int] = []
    ep_eps_mean: list[float] = []
    t_start = time.time()

    with torch.no_grad():
        while not done:
            hetero_data = obs.to(device)

            # PPO GNN retorna (ch_logits, pwr_logits, state_value)
            ch_logits, pwr_logits, _ = model(
                x_dict=hetero_data.x_dict,
                edge_index_dict=hetero_data.edge_index_dict,
                edge_attr_dict=hetero_data.edge_attr_dict,
                batch_dict=hetero_data.batch_dict if hasattr(hetero_data, 'batch_dict') else None
            )

            if stochastic:
                ch_acts  = Categorical(logits=ch_logits).sample()
                pwr_acts = Categorical(logits=pwr_logits).sample()
            else:
                ch_acts  = ch_logits.argmax(dim=-1)
                pwr_acts = pwr_logits.argmax(dim=-1)

            action = torch.stack([ch_acts, pwr_acts], dim=1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            ep_rates.append(info.get("total_rate", 0.0))
            ep_delta_t.append(info.get("delta_t", 0))
            ep_eps_mean.append(info.get("epsilon_t_mean", 0.0))

    return {
        "return":         total_reward,
        "total_rate":     sum(ep_rates),
        "mean_rate":      float(np.mean(ep_rates)) if ep_rates else 0.0,
        "delta_t_mean":   float(np.mean(ep_delta_t)) if ep_delta_t else 0.0,
        "epsilon_t_mean": float(np.mean(ep_eps_mean)) if ep_eps_mean else 0.0,
        "n_steps":        len(ep_rates),
        "duration_sec":   time.time() - t_start,
    }


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    """Evalúa el modelo sobre ``args.episodes`` episodios y exporta métricas.

    Parameters
    ----------
    args:
        Espacio de nombres generado por ``parse_args()``.

    Returns
    -------
    pd.DataFrame
        Métricas por episodio para análisis posterior.
    """
    device = select_device()
    log.info("Dispositivo de cómputo: %s", device)

    # ── Parametrización del POMDP y Carga Estocástica ───────────────────────
    log.info("Inicializando Pipeline Empírico sobre subgrafo estructural '%s'", args.building_id)
    distributions = load_distributions(building_id=args.building_id, verbose=False)
    log.info("Sintetización del conjunto de arribos: %d procesos estocásticos inyectados.", len(distributions))

    available_channels: list[int]   = [1, 6, 11]
    tx_powers_dbm:      list[float] = [20.0, 17.0, 14.0, 11.0, 8.0]
    n_aps: int = len(distributions[0].blocks[0].datos)
    log.info("Nodos Access Point Detectados (N): %d", n_aps)

    # ── Construcción del Entorno de Simulación Electromagnética ─────────────
    env = NetworkGraphEnv(
        distributions=distributions,
        n_aps=n_aps,
        arrival_rate=args.arrival_rate,
        mean_duration=args.mean_dur,
        total_timesteps=args.timesteps,
        decision_period=args.decision_period,
        available_channels=available_channels,
        tx_powers_dbm=tx_powers_dbm,
        random_seed=args.seed,
        device=device,
    )

    # ── Definición de la Política Neuronal $\pi_\theta$ ─────────────────────
    model = GNN(
        hidden_channels=args.hidden,
        num_aps=n_aps,
        out_channels_ch=len(available_channels),
        out_channels_pwr=len(tx_powers_dbm),
    ).to(device)

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            f"  Verificá --model_path o corré train.py primero."
        )

    log.info("Inyectando espacio de parámetros desde: %s", model_path)
    # Usamos strict=False para permitir la carga robusta de modelos entrenados con A2C/PPO
    # ignorando silenciosamente los pesos del 'value_head' no utilizados por REINFORCE.
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # ── Bucle Central de Inferencia Algorítmica ───────────────────────────────
    mode_str = "estocástico" if args.stochastic else "greedy (argmax)"
    log.info(
        "Evaluando %d episodios x %d pasos — modo %s",
        args.episodes, args.timesteps, mode_str,
    )

    _HDR = (
        f"{'Epoch (e)':>10} | {'G_t (Return)':>12} | {'Suma(R_tau)':>11} | "
        f"{'R_tau_mean':>10} | {'Timesteps':>9} | {'delta_t':>7} | "
        f"{'eps_t_mean':>10} | {'Time(s)':>7}"
    )
    log.info("%s", _HDR)
    log.info("%s", "─" * len(_HDR))

    metrics_log: list[dict] = []
    t_total = time.time()

    for ep in range(args.episodes):
        ep_seed = args.seed + ep if args.seed is not None else None
        result  = run_episode(model, env, device, ep_seed, stochastic=args.stochastic)

        metrics_log.append({"episode": ep + 1, **result})

        log.info(
            "%10d | %10.3f | %10.1f | %10.2f | %8d | %6.1f | %8.1f | %6.1f",
            ep + 1,
            result["return"],
            result["total_rate"],
            result["mean_rate"],
            result["n_steps"],
            result["delta_t_mean"],
            result["epsilon_t_mean"],
            result["duration_sec"],
        )

    elapsed = time.time() - t_total

    # -- Estadísticas agregadas ----------------------------------------------
    df      = pd.DataFrame(metrics_log)
    returns = df["return"].values
    rates   = df["total_rate"].values

    log.info("")
    log.info("=" * len(_HDR))
    log.info("  Episodios evaluados : %d",       args.episodes)
    log.info("  Modo de selección   : %s",       mode_str)
    log.info("  Tiempo total        : %.1f min", elapsed / 60)
    log.info("")
    log.info("  Retorno  — media: %8.3f  std: %7.3f  min: %7.3f  max: %7.3f",
             returns.mean(), returns.std(), returns.min(), returns.max())
    log.info("  Rate/ep  — media: %8.1f  std: %7.1f  min: %7.1f  max: %7.1f",
             rates.mean(), rates.std(), rates.min(), rates.max())
    log.info("=" * len(_HDR))

    # -- Persistencia --------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_dir / "eval_metrics.csv"
    df.to_csv(metrics_path, index=False)
    log.info("Métricas exportadas a: %s", metrics_path)

    # Generación Automática de Gráficas de Evaluación (Thesis Standard)
    try:
        import subprocess
        plot_script = Path(__file__).parent / "plots_code" / "plot_eval.py"
        if plot_script.exists():
            log.info("Generando gráficas de evaluación...")
            subprocess.run([
                sys.executable, str(plot_script),
                "--csv", str(metrics_path),
                "--out", str(save_dir.parent / "plots")
            ], check=False)
    except Exception as e:
        log.warning("No se pudieron generar las gráficas de evaluación automáticamente: %s", e)

    return df


# ---------------------------------------------------------------------------
# Interfaz de línea de comandos
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de configuración de la evaluación."""
    p = argparse.ArgumentParser(
        description="NetROML: Evaluación de política GNN entrenada.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Identificación
    p.add_argument("--building_id",  default="814",                          type=str,
                   help="Identificador del edificio en el dataset.")
    p.add_argument("--model_path",   default="outputs/models/best_model.pt", type=str,
                   help="Ruta al archivo .pt con los pesos del modelo.")
    p.add_argument("--save_dir",     default="outputs/eval",                 type=str,
                   help="Directorio de salida para métricas de evaluación.")
    p.add_argument("--seed",
                   type=lambda x: None if str(x).lower() == "none" else int(x),
                   default=42,
                   help="Semilla base. Cada episodio usa seed+episode. 'none' para no fijarla.")

    # Hiperparámetros del entorno — deben coincidir con los usados en entrenamiento
    p.add_argument("--episodes",        type=int,   default=20,
                   help="Número de episodios de evaluación.")
    p.add_argument("--timesteps",       type=int,   default=100,
                   help="Duración máxima de cada episodio en pasos.")
    p.add_argument("--arrival_rate",    type=float, default=2.0,
                   help="Tasa de llegada de clientes (proceso de Poisson).")
    p.add_argument("--mean_dur",        type=float, default=10.0,
                   help="Duración media de sesión de cliente (pasos).")
    p.add_argument("--decision_period", type=int,   default=1,
                   help="Cada cuantos slots se toma una nueva decision.")

    # Hiperparámetros del modelo — deben coincidir con los usados en entrenamiento
    p.add_argument("--hidden",       type=int,   default=64,
                   help="Dimensión de los embeddings ocultos de la GNN.")

    # Modo de selección de acción
    p.add_argument("--stochastic",   action="store_true",
                   help="Evaluar muestreando π en lugar de argmax greedy.")

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())