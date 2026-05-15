r"""
train.py – Módulo de Entrenamiento Core (Advantage Actor-Critic).

Entrena una política Actor-Critic usando la GNN parametrizada para resolver
el POMDP de asignación de recursos Wi-Fi.

Matemática del Advantage Actor-Critic (A2C)
-------------------------------------------
El modelo conjuga dos redes entrenadas simultáneamente:
1. Política Actor $\pi_\theta(a_t|s_t)$ parametrizada por pesos $\theta$.
2. Función Critic $V_\phi(s_t)$ parametrizada por pesos $\phi$.

La ventaja se estima como $\hat{A}_t = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.
La actualización del actor sigue el gradiente:
$$ \theta \leftarrow \theta + \alpha_\theta \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t $$

El crítico minimiza el error cuadrático medio del error TD:
$$ \mathcal{L}(\phi) = \mathbb{E}[(R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2] $$

La función de pérdida total implementada combina ambos objetivos y un bono de entropía:
$$ \mathcal{L}_{total} = \mathcal{L}_{actor}(\theta) + c_{vf} \mathcal{L}_{critic}(\phi) - c_{ent} \mathcal{H}(\pi_\theta) $$

Donde:
- $\mathcal{L}_{actor}(\theta) = - \mathbb{E}_t [ \log \pi_\theta(a_t|s_t) \hat{A}_t ]$
- $\mathcal{H}(\pi_\theta)$ es la Entropía para fomentar la exploración.

Validación Periódica Cruzada
----------------------------
Este script introduce un mecanismo de generalización donde cada `--eval_every` episodios,
el algoritmo interrumpe el backpropagation y evalúa la política de forma Greedy determinista
sobre un subset independiente de arribos para registrar el $\bar{G}_{val}$ (Métrica de detención).
"""

from __future__ import annotations

# IMPORTANTE: importar compat ANTES que cualquier módulo de PyG
import utils.compat  # noqa: F401 — aplica el parche de type_repr

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from data.data_loader import load_distributions
from model.network_graph_env import NetworkGraphEnv
from model.gnn_model import GNN


def compute_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Implementa cálculo de retornos descontados hacia atrás."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def select_device() -> torch.device:
    """Selecciona acelerador de hardware (CUDA, Metal o CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_episode(env: NetworkGraphEnv, model: GNN, device: torch.device, training: bool, reward_scale: float = 1.0):
    """
    Ejecuta un episodio completo y recolecta trayectorias.

    Parameters
    ----------
    training : bool
        Si True, acumula log_probs y values para backpropagation.
        Si False (eval), solo ejecuta la política sin gradiente.

    Returns
    -------
    dict con rewards, log_probs, entropies, values, ep_rates.
    """
    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values:    list[torch.Tensor] = []
    rewards:   list[float]        = []
    ep_rates:  list[float]        = []

    done = False
    while not done:
        data = obs.to(device)

        if training:
            ch_logits, pwr_logits, value = model(
                data.x_dict,
                data.edge_index_dict,
                data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None,
            )
        else:
            with torch.no_grad():
                ch_logits, pwr_logits, value = model(
                    data.x_dict,
                    data.edge_index_dict,
                    data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None,
                )

        ch_dist  = Categorical(logits=ch_logits)
        pwr_dist = Categorical(logits=pwr_logits)
        ch_acts  = ch_dist.sample()
        pwr_acts = pwr_dist.sample()

        log_prob  = ch_dist.log_prob(ch_acts).sum() + pwr_dist.log_prob(pwr_acts).sum()
        entropy_t = ch_dist.entropy().sum() + pwr_dist.entropy().sum()

        log_probs.append(log_prob)
        entropies.append(entropy_t)
        values.append(value.squeeze())

        action = torch.stack([ch_acts, pwr_acts], dim=1)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        rewards.append(reward / reward_scale)
        ep_rates.append(info.get("total_rate", 0.0))

    return {
        "rewards":   rewards,
        "log_probs": log_probs,
        "entropies": entropies,
        "values":    values,
        "ep_rates":  ep_rates,
    }


def train(args: argparse.Namespace):
    """Loop principal de inicialización y entrenamiento (A2C)."""
    print(f"\n{'='*70}")
    print(f"  Marco de Optimización Analítica: Advantage Actor-Critic (A2C)")
    print(f"  Topología Física: Edificio {args.building_id} | Horizonte Temporal E: {args.episodes} episodios")
    print(f"  Hiperparámetros de Regularización: c_vf={args.vf_coef} | c_ent={args.entropy_coef} | Cross-Validation c/{args.eval_every} ep")
    print(f"{'='*70}\n")

    device = select_device()
    print(f"[Hardware] Dispositivo de Cómputo Tensorial: {device}")

    # ── Parametrización del POMDP y Carga Estocástica ───────────────────────
    print(f"[Pipeline] Inyectando Modelos Estocásticos de Arribo (Poisson/Exponencial)...")
    distributions = load_distributions(
        building_id=args.building_id,
        verbose=True,
    )
    print(f"[Memoria]  {len(distributions)} instanciaciones cliente-canal cargadas.\n")

    available_channels = [1, 6, 11]
    tx_powers_dbm      = [20.0, 14.0, 8.0]
    n_aps              = len(distributions[0].blocks[0].datos)

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
        sticky_mode=args.sticky_mode,
        random_seed=args.seed,
        device=device,
    )

    # ── Entorno de Cross-Validation (Semilla Ortogonal) ─────────────────────
    env_eval = NetworkGraphEnv(
        distributions=distributions,
        n_aps=n_aps,
        arrival_rate=args.arrival_rate,
        mean_duration=args.mean_dur,
        total_timesteps=args.timesteps,
        decision_period=args.decision_period,
        available_channels=available_channels,
        tx_powers_dbm=tx_powers_dbm,
        sticky_mode=args.sticky_mode,
        random_seed=(args.seed + 9999) if args.seed is not None else None,
        device=device,
    )

    # ── Definición de la Política Neuronal $\pi_\theta$ y Valor $V_\phi$ ────
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

    best_G_val   = -float("inf")   # Mejor Esperanza del Retorno Empírico
    metrics      = []
    t_global_start = time.time()

    print(f"{'='*70}")
    print(f"{'Epoch (e)':<6} | {'G_t (Train)':>11} | {'G_t (Val)':>10} | {'L(theta)':>9} | {'L(phi)':>9} | {'H(pi)':>9}")
    print(f"-------+-------------+------------+-----------+-----------+-----------")

    # ── Bucle Central de Optimización Algorítmica (A2C) ───────────────────────
    for episode in range(args.episodes):
        current_seed = args.seed + episode if args.seed is not None else None
        env._adm.random_seed = current_seed

        model.train()
        traj = run_episode(env, model, device, training=True, reward_scale=args.reward_scale)

        rewards   = traj["rewards"]
        log_probs = traj["log_probs"]
        entropies = traj["entropies"]
        values    = traj["values"]
        ep_rates  = traj["ep_rates"]

        # ── Computación del Advantage y Retorno (A2C) ─────────────────────────
        # Estimación Monte Carlo del Retorno empírico: $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k}$
        returns_t  = compute_returns(rewards, gamma=args.gamma).to(device) # (T,)
        values_t   = torch.stack(values)                                   # (T,)
        
        # Función Ventaja: $\hat{A}_t = G_t - V_\phi(s_t)$
        # Aplicamos .detach() para bloquear el flujo del gradiente hacia el Critic
        # durante la actualización de la Política (Actor).
        advantages = returns_t - values_t.detach()                         # (T,)

        if len(advantages) > 1:
            # Normalización del Advantage: $\hat{A}_t \leftarrow (\hat{A}_t - \mu) / (\sigma + \epsilon)$
            # Estabiliza la alta varianza intrínseca del gradiente estocástico.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)    # (T,)
        entropies_t = torch.stack(entropies)    # (T,)

        # ── Definición de la Función Objetivo Global $\mathcal{L}(\theta, \phi)$ ─────────
        # Pérdida del Actor: $\mathcal{L}_{policy}(\theta) = - \mathbb{E}_t [ \log \pi_\theta(a_t|s_t) \hat{A}_t ]$
        loss_actor   = -(advantages * log_probs_t).mean()
        
        # Pérdida del Critic (Error Cuadrático Medio): $\mathcal{L}_{value}(\phi) = \mathbb{E}_t [ (V_\phi(s_t) - G_t)^2 ]$
        loss_critic  = F.mse_loss(values_t, returns_t)
        
        # Bono de Entropía: Fomenta la exploración penalizando la certidumbre absoluta.
        loss_entropy = -entropies_t.mean()

        # Loss Combinada: Optimizamos simultáneamente Actor y Critic compartiendo los pesos base de la GNN.
        total_loss = loss_actor + args.vf_coef * loss_critic + args.entropy_coef * loss_entropy

        # ── Backpropagation y Gradient Clipping ───────────────────────────────
        # Guard: no propagar NaN al modelo si la loss no es finita (episodio sin clientes).
        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0)
        scheduler.step()

        G_train = returns_t[0].item()

        # ── Validación Periódica ──────────────────────────────────────────────
        G_val = float("nan")
        if (episode + 1) % args.eval_every == 0 or episode == 0:
            model.eval()
            val_traj = run_episode(env_eval, model, device, training=False, reward_scale=args.reward_scale)
            G_val = compute_returns(val_traj["rewards"], gamma=args.gamma)[0].item()

            if G_val > best_G_val:
                best_G_val = G_val
                torch.save(model.state_dict(), save_dir / "best_model.pt")

        # Checkpoint periódico de entrenamiento
        if (episode + 1) % 50 == 0:
            torch.save(model.state_dict(), save_dir / f"model_ep{episode+1}.pt")

        # ── Métricas ──────────────────────────────────────────────────────────
        metrics.append({
            "episode":        episode + 1,
            "return":         G_train,
            "G_val":          G_val,
            "total_rate":     sum(ep_rates),
            "loss_actor":     loss_actor.item(),
            "loss_critic":    loss_critic.item(),
            "loss":           total_loss.item(),
            "entropy":        entropies_t.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std":  advantages.std().item() if len(advantages) > 1 else 0.0,
            "grad_norm":      grad_norm.item(),
            "sec":            time.time() - t_global_start,
        })

        if (episode + 1) % 10 == 0 or episode == 0:
            g_val_str = f"{G_val:>10.2f}" if not np.isnan(G_val) else f"{'—':>10}"
            print(
                f"{episode+1:<6} | "
                f"{G_train:>10.2f} | "
                f"{g_val_str} | "
                f"{loss_actor.item():>9.4f} | "
                f"{loss_critic.item():>9.4f} | "
                f"{entropies_t.mean().item():>9.4f}"
            )

    print(f"\n{'='*70}")
    duracion = time.time() - t_global_start
    print(f"Entrenamiento Exitoso  |  Llevó: {duracion/60:.1f} m  | Mejor G_val: {best_G_val:.2f}")

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    df_metrics = pd.DataFrame(metrics)
    metrics_path = save_dir / "training_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Modelos alojados en: {save_dir}/")
    print(f"Métricas crudas en:  {metrics_path}")
    print(f"{'='*70}\n")

    # Generación Automática de Gráficas (Thesis Standard)
    try:
        plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
        if plot_script.exists():
            import subprocess
            print(f"[Análisis] Generando gráficas de convergencia científica...")
            subprocess.run([
                sys.executable, str(plot_script),
                "--csv", str(metrics_path),
                "--out", str(Path(__file__).parent / "plots")
            ], check=False)
    except Exception as e:
        print(f"[Aviso] No se pudieron generar las gráficas automáticamente: {e}")

    return model, df_metrics


def parse_args():
    p = argparse.ArgumentParser(description="NetROML – A2C + GNN")
    p.add_argument("--building_id",   default="990",                           type=str)
    p.add_argument("--dist_joblib",   default="data/distributions_990.joblib", type=str)
    p.add_argument("--step2_csv",     default="data/dataset_990_step2.csv",    type=str)
    p.add_argument("--episodes",      type=int,   default=500)
    p.add_argument("--timesteps",     type=int,   default=100)
    p.add_argument("--arrival_rate",  type=float, default=2.0)
    p.add_argument("--mean_dur",      type=float, default=10.0)
    p.add_argument("--decision_period", type=int, default=1)
    p.add_argument("--sticky_mode",   type=str,   default="sticky", choices=["full", "sticky", "lite"])
    p.add_argument("--hidden",        type=int,   default=64)
    p.add_argument("--reward_scale",  type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--entropy_coef",  type=float, default=0.01,
                   help="Coeficiente para incentivar exploración (bono de entropía).")
    p.add_argument("--vf_coef",       type=float, default=0.5,
                   help="Peso del término Critic (L_critic) en la loss total A2C.")
    p.add_argument("--eval_every",    type=int,   default=20,
                   help="Frecuencia de episodios de validación sin gradiente.")
    p.add_argument("--seed",          type=lambda x: None if str(x).lower() == 'none' else int(x), default=314)
    p.add_argument("--save_dir",      default="outputs/models", type=str)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
