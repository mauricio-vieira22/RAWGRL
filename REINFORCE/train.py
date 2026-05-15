r"""
train.py — Optimización de Políticas NetROML mediante Gradiente de Política (REINFORCE).

Este módulo constituye el núcleo del entrenamiento del sistema NetROML. Implementa 
el algoritmo REINFORCE (Monte Carlo Policy Gradient) para la optimización de una 
red neuronal de grafos (GNN) que actúa como controlador inteligente de recursos WiFi.

Fundamentación Matemática (REINFORCE):
--------------------------------------
El agente optimiza el objetivo $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t R_t]$ mediante ascenso por gradiente.
La actualización de parámetros se rige por el Policy Gradient Theorem con baseline:

$$ \theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) (G_t - b(s_t)) $$

Donde:
- $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k$: Retorno acumulado (Reward-to-go).
- $b(s_t)$: Baseline para reducción de varianza (exponencial o media móvil).
- $\pi_\theta$: Política parametrizada por la GNN heterogénea.

"""

from __future__ import annotations

# Patch crítico para compatibilidad de tipos en Torch Geometric
import torch_geometric.inspector as _pyg_inspector
try:
    _original_type_repr = _pyg_inspector.type_repr
    def _safe_type_repr(obj, _globals=None):
        try: return _original_type_repr(obj, _globals)
        except AttributeError as exc:
            if "'_name'" in str(exc): return "Union"
            raise
    _pyg_inspector.type_repr = _safe_type_repr
except Exception: pass

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import Categorical

from data.data_loader import load_distributions
from model.gnn_model import GNN
from model.network_graph_env import NetworkGraphEnv

# Configuración de Logging Académico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """
    Calcula los retornos descontados G_t (Reward-to-go) mediante retro-recursión.

    Parameters
    ----------
    rewards : list[float]
        Lista cronológica de recompensas obtenidas en el episodio.
    gamma : float
        Factor de descuento temporal.

    Returns
    -------
    torch.Tensor
        Vector de retornos calculados para cada timestep del rollout.
    """
    G = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def select_device() -> torch.device:
    """Detecta y selecciona la unidad de procesamiento óptima disponible."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def normalize_advantages(advantages: torch.Tensor, min_std: float = 1e-3) -> torch.Tensor:
    """
    Normaliza el vector de ventajas para estabilizar el aprendizaje.

    Parameters
    ----------
    advantages : torch.Tensor
        Diferencia entre retornos reales y el baseline (G_t - b).
    min_std : float, opcional
        Margen de seguridad para evitar divisiones por cero en gradientes nulos.

    Returns
    -------
    torch.Tensor
        Ventajas estandarizadas (media 0, varianza 1).
    """
    if advantages.numel() <= 1: return advantages
    std = advantages.std()
    if std > min_std:
        return (advantages - advantages.mean()) / (std + 1e-8)
    return advantages - advantages.mean()


def train(args: argparse.Namespace) -> tuple[GNN, pd.DataFrame]:
    """
    Orquesta el ciclo completo de entrenamiento del agente.

    Parameters
    ----------
    args : argparse.Namespace
        Configuración de hiperparámetros y metadatos del experimento.

    Returns
    -------
    tuple[GNN, pd.DataFrame]
        El modelo entrenado y el historial de métricas de convergencia.
    """
    device = select_device()
    log.info("Dispositivo de cómputo activo: %s", device)

    # 1. Carga de Datos y Caracterización de Topología
    log.info("Cargando distribuciones para edificio: %s", args.building_id)
    distributions = load_distributions(building_id=args.building_id, verbose=False)
    
    # Parámetros físicos estandarizados
    available_channels = [1, 6, 11]
    tx_powers_dbm      = [20.0, 14.0, 8.0]
    n_aps              = len(distributions[0].blocks[0].datos)

    # 2. Inicialización del Entorno y Modelo
    env = NetworkGraphEnv(
        distributions=distributions,
        n_aps=n_aps,
        arrival_rate=args.arrival_rate,
        mean_duration=args.mean_dur,
        total_timesteps=args.timesteps,
        decision_period=args.decision_period,
        available_channels=available_channels,
        tx_powers_dbm=tx_powers_dbm,
        sticky_mode=args.sticky_mode,
        random_seed=args.seed,
        device=device,
    )

    model = GNN(
        hidden_channels=args.hidden,
        num_aps=n_aps,
        out_channels_ch=len(available_channels),
        out_channels_pwr=len(tx_powers_dbm),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Variables de Seguimiento
    best_return = -float("inf")
    baseline    = 0.0
    metrics_log: list[dict] = []
    t_start = time.time()

    log.info("Iniciando optimización: E=%d episodios | T=%d pasos", args.episodes, args.timesteps)
    
    log.info(f"{'='*80}")
    log.info(f"{'Ep':<6} | {'Total Rate':>10} | {'G_t':>10} | {'Loss':>9} | {'Entropy':>9} | {'Grad':>9} | {'Time':>6}")
    log.info(f"-------+------------+------------+-----------+-----------+-----------+-------")

    # Bucle de Entrenamiento
    for episode in range(args.episodes):
        ep_seed = args.seed + episode if args.seed is not None else None
        obs, _ = env.reset(seed=ep_seed)

        log_probs, entropies, rewards, ep_rates = [], [], [], []
        done = False
        t_ep = time.time()

        # Fase de Rollout (Muestreo de la Política)
        while not done:
            data = obs.to(device)
            ch_logits, pwr_logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

            ch_dist  = Categorical(logits=ch_logits)
            pwr_dist = Categorical(logits=pwr_logits)

            ch_acts, pwr_acts = ch_dist.sample(), pwr_dist.sample()

            log_probs.append(ch_dist.log_prob(ch_acts).sum() + pwr_dist.log_prob(pwr_acts).sum())
            entropies.append(ch_dist.entropy().sum() + pwr_dist.entropy().sum())

            action = torch.stack([ch_acts, pwr_acts], dim=1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(reward/1000)
            ep_rates.append(info.get("total_rate", 0.0))

        # Cómputo de Pérdida Algorítmica (REINFORCE + Entropy Bonus)
        returns    = compute_returns(rewards, gamma=args.gamma).to(device)
        G_0        = returns[0].item()

        # Guard: proteger el baseline de contaminación NaN.
        # Si el episodio no tuvo clientes activos (reward=0 es válido, pero
        # una EMA con NaN infecta para siempre todos los episodios siguientes).
        if math.isfinite(G_0):
            baseline = G_0 if episode == 0 else 0.95 * baseline + 0.05 * G_0

        advantages = normalize_advantages(returns - baseline)

        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        loss_actor    = -(advantages * log_probs_t).mean()
        entropy_bonus = args.entropy_coef * entropies_t.mean()
        policy_loss   = loss_actor - entropy_bonus

        # Optimización y Estabilización de Gradientes
        # Guard: no propagar NaN al modelo si la loss no es finita.
        optimizer.zero_grad()
        if torch.isfinite(policy_loss):
            policy_loss.backward()
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            pre_clip_norm = torch.tensor(0.0)
        scheduler.step()

        # Gestión de Persistencia y Métricas
        if G_0 > best_return:
            best_return = G_0
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        metrics_log.append({
            "episode": episode + 1, "return": G_0, "baseline": baseline,
            "total_rate": sum(ep_rates), "loss": policy_loss.item(),
            "entropy": entropies_t.mean().item(),
            "grad_norm": pre_clip_norm.item(), "sec": time.time() - t_ep
        })

        if (episode + 1) % 10 == 0 or episode == 0:
            log.info(
                f"{episode+1:<6} | "
                f"{sum(ep_rates):>10.1f} | "
                f"{G_0:>10.2f} | "
                f"{policy_loss.item():>9.4f} | "
                f"{entropies_t.mean().item():>9.4f} | "
                f"{pre_clip_norm.item():>9.4f} | "
                f"{time.time() - t_ep:>6.1f}"
            )

    # Finalización del Experimento
    elapsed = time.time() - t_start
    log.info("Entrenamiento finalizado en %.1f min.", elapsed / 60)
    
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    pd.DataFrame(metrics_log).to_csv(save_dir / "training_metrics.csv", index=False)
    
    # Generación Automática de Gráficas (Thesis Standard)
    try:
        import subprocess
        plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
        if plot_script.exists():
            log.info("Generando gráficas de entrenamiento...")
            subprocess.run([
                sys.executable, str(plot_script),
                "--csv", str(save_dir / "training_metrics.csv"),
                "--out", str(Path(__file__).parent / "plots")
            ], check=False)
    except Exception as e:
        log.warning("No se pudieron generar las gráficas automáticamente: %s", e)

    return model, pd.DataFrame(metrics_log)


def parse_args() -> argparse.Namespace:
    """Configura la interfaz de línea de comandos para el entrenamiento."""
    p = argparse.ArgumentParser(description="NetROML: REINFORCE Controller")
    p.add_argument("--building_id",  default="990")
    p.add_argument("--save_dir",     default="outputs/models")
    p.add_argument("--seed",         type=int,   default=314)
    p.add_argument("--episodes",     type=int,   default=300)
    p.add_argument("--timesteps",    type=int,   default=10)
    p.add_argument("--arrival_rate", type=float, default=2.0)
    p.add_argument("--mean_dur",     type=float, default=10.0)
    p.add_argument("--decision_period", type=int, default=1)
    p.add_argument("--sticky_mode",  type=str,   default="sticky", choices=["full", "sticky", "lite"])
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=1.0)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())