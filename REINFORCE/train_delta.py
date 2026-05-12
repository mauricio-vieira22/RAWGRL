r"""
train_delta.py — REINFORCE con Inyección Tardía de delta_t (Late-Injection).

Esta versión en paralelo implementa la arquitectura donde delta_t se inyecta 
directamente en la cabeza de decisión de la GNN, permitiendo al agente 
correlacionar el estado espacial del grafo con el horizonte temporal del POMDP.

Autor: Antigravity AI (Thesis Professionalization Level 3000%)
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
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.distributions import Categorical

from data.data_loader import load_distributions
from model.gnn_delta_model import GNNDelta
from model.network_graph_env import NetworkGraphEnv

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    G = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def select_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def normalize_advantages(advantages: torch.Tensor, min_std: float = 1e-3) -> torch.Tensor:
    if advantages.numel() <= 1: return advantages
    std = advantages.std()
    if std > min_std:
        return (advantages - advantages.mean()) / (std + 1e-8)
    return advantages - advantages.mean()


def train(args: argparse.Namespace):
    device = select_device()
    log.info("Dispositivo activo: %s (Late-Injection Mode)", device)

    distributions = load_distributions(building_id=args.building_id, verbose=False)
    available_channels = [1, 6, 11]
    tx_powers_dbm      = [20.0, 17.0, 14.0, 11.0, 8.0]
    n_aps              = len(distributions[0].blocks[0].datos)

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

    # Inicializamos el modelo de Inyección Tardía
    model = GNNDelta(
        hidden_channels=args.hidden,
        num_aps=n_aps,
        out_channels_ch=len(available_channels),
        out_channels_pwr=len(tx_powers_dbm),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    save_dir = Path(args.save_dir) / "reinforce_delta"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_return = -float("inf")
    baseline    = 0.0
    metrics_log: list[dict] = []
    t_start = time.time()

    log.info("Iniciando Entrenamiento (Delta-Injected)...")
    
    for episode in range(args.episodes):
        ep_seed = args.seed + episode if args.seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        
        # Extraer delta_t inicial (normalizado)
        delta_t = float(info.get("delta_t", 0)) / args.timesteps

        log_probs, entropies, rewards, ep_rates = [], [], [], []
        done = False
        t_ep = time.time()

        while not done:
            data = obs.to(device)
            
            # Pasar delta_t al modelo junto con el grafo
            ch_logits, pwr_logits = model(data.x_dict, data.edge_index_dict, delta_t, data.edge_attr_dict)

            ch_dist  = Categorical(logits=ch_logits)
            pwr_dist = Categorical(logits=pwr_logits)
            ch_acts, pwr_acts = ch_dist.sample(), pwr_dist.sample()

            log_probs.append(ch_dist.log_prob(ch_acts).sum() + pwr_dist.log_prob(pwr_acts).sum())
            entropies.append(ch_dist.entropy().sum() + pwr_dist.entropy().sum())

            action = torch.stack([ch_acts, pwr_acts], dim=1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Actualizar delta_t para el próximo paso
            delta_t = float(info.get("delta_t", 0)) / args.timesteps

            rewards.append(reward)
            ep_rates.append(info.get("total_rate", 0.0))

        # Update loop (Standard REINFORCE)
        returns    = compute_returns(rewards, gamma=args.gamma).to(device)
        G_0        = returns[0].item()
        baseline   = G_0 if episode == 0 else 0.95 * baseline + 0.05 * G_0
        advantages = normalize_advantages(returns - baseline)

        loss_actor    = -(advantages * torch.stack(log_probs)).mean()
        policy_loss   = loss_actor - args.entropy_coef * torch.stack(entropies).mean()

        optimizer.zero_grad()
        policy_loss.backward()
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if G_0 > best_return:
            best_return = G_0
            torch.save(model.state_dict(), save_dir / "best_model.pt")

        metrics_log.append({
            "episode": episode + 1, "return": G_0, "baseline": baseline,
            "total_rate": sum(ep_rates), "loss": policy_loss.item(),
            "entropy": torch.stack(entropies).mean().item(),
            "grad_norm": pre_clip_norm.item(), "sec": time.time() - t_ep
        })

        if (episode + 1) % 10 == 0 or episode == 0:
            log.info("Ep %3d | G_t: %8.2f | Rate: %8.1f | Loss: %7.4f",
                     episode + 1, G_0, sum(ep_rates), policy_loss.item())

    # Finalizar
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    metrics_path = save_dir / "training_metrics.csv"
    pd.DataFrame(metrics_log).to_csv(metrics_path, index=False)
    
    # Graficar
    try:
        plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
        if plot_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(plot_script), "--csv", str(metrics_path), "--out", str(save_dir / "plots")], check=False)
    except: pass

    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NetROML: REINFORCE Late-Injection")
    p.add_argument("--building_id",  default="84")
    p.add_argument("--save_dir",     default="outputs/models_delta")
    p.add_argument("--seed",         type=int,   default=314)
    p.add_argument("--episodes",     type=int,   default=500)
    p.add_argument("--timesteps",    type=int,   default=100)
    p.add_argument("--arrival_rate", type=float, default=3.0)
    p.add_argument("--mean_dur",     type=float, default=15.0)
    p.add_argument("--decision_period", type=int, default=10)
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
