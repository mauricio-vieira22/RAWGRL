r"""
train.py — Optimización de Políticas RAWGRL mediante Gradiente de Política (REINFORCE).

Implementa el algoritmo REINFORCE (Monte Carlo Policy Gradient) con baseline
exponencialmente ponderado para la optimización de la GNN heterogénea que actúa
como controlador de recursos WiFi en entornos multi-AP.

Fundamentación Matemática (REINFORCE con Baseline)
---------------------------------------------------
El agente maximiza el objetivo:

    J(θ) = E_{π_θ}[∑_t γ^t R_t]

mediante el estimador de gradiente con varianza reducida:

    ∇_θ J(θ) ≈ ∑_{t=0}^{T} ∇_θ log π_θ(a_t | s_t) · Â_t

donde la ventaja normalizada es:

    Â_t = (G_t - b_t) / (σ(G_t - b_t) + ε)

y el baseline b_t se actualiza por media exponencial ponderada (EMA):

    b_t ← β · b_{t-1} + (1 - β) · G_t,   β = 0.99

La normalización de ventajas (centrado + escalado) desacopla la magnitud del
gradiente de la escala absoluta del reward, reduciendo la varianza del estimador
sin introducir sesgo en el gradiente de política (Williams, 1992).
"""

from __future__ import annotations

# Patch de compatibilidad de tipos para PyTorch Geometric
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

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from data.data_loader import load_distributions
from model.gnn_model import GNN
from model.gnn2_model import GNN2
from model.network_graph_env import NetworkGraphEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Modelos ───────────────────────────────────────────────────────────────────

class RandomModel(torch.nn.Module):
    """Política de referencia que selecciona acciones uniformemente al azar."""

    def __init__(self, out_channels_ch: int, out_channels_pwr: int):
        super().__init__()
        self.out_channels_ch  = out_channels_ch
        self.out_channels_pwr = out_channels_pwr
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None):
        n_aps  = x_dict['ap'].shape[0]
        device = x_dict['ap'].device
        ch_logits  = torch.zeros((n_aps, self.out_channels_ch),  device=device) + self.dummy
        pwr_logits = torch.zeros((n_aps, self.out_channels_pwr), device=device) + self.dummy
        return ch_logits, pwr_logits


# ── Funciones de Cómputo de Retornos y Ventajas ───────────────────────────────

def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """
    Calcula los retornos descontados G_t (reward-to-go) por retro-recursión.

    G_t = R_t + γ · R_{t+1} + γ² · R_{t+2} + … + γ^{T-t} · R_T

    Parameters
    ----------
    rewards : list[float]
        Secuencia cronológica de recompensas del episodio.
    gamma : float
        Factor de descuento γ ∈ (0, 1].

    Returns
    -------
    torch.Tensor
        Vector de retornos G_t, shape (T,).
    """
    G = 0.0
    returns: list[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def normalize_advantages(advantages: torch.Tensor, min_std: float = 1e-3) -> torch.Tensor:
    """
    Normaliza las ventajas por centrado y escalado (estandarización completa).

    Â_t = (A_t - mean(A)) / (std(A) + ε)

    La versión anterior de esta función solo dividía por std sin centrar, lo cual
    no eliminaba el sesgo de escala absoluta entre episodios con distinto número
    de clientes activos. El centrado es el paso que resuelve ese problema: un
    episodio con muchos clientes produce rewards altos en promedio, y al restar
    la media de las ventajas se elimina ese componente antes de propagar el
    gradiente.

    Esta operación no introduce sesgo en el estimador del gradiente de política
    (Williams, 1992), dado que la constante de normalización es independiente de
    los parámetros θ.

    Parameters
    ----------
    advantages : torch.Tensor
        Vector de ventajas A_t = G_t - b_t, shape (T,).
    min_std : float
        Umbral mínimo de desviación estándar. Si std < min_std, solo se centra
        (rollout cuasi-determinista donde todas las ventajas son casi iguales).

    Returns
    -------
    torch.Tensor
        Ventajas estandarizadas Â_t, shape (T,).
    """
    if advantages.numel() <= 1:
        return advantages
    mean = advantages.mean()
    std  = advantages.std()
    if std > min_std:
        return (advantages - mean) / (std + 1e-8)
    # std ≈ 0: el gradiente es prácticamente cero de todas formas; solo centramos.
    return advantages - mean


# ── Evaluación Determinista ────────────────────────────────────────────────────

def evaluate_policy(
    env:          NetworkGraphEnv,
    model:        GNN,
    device:       torch.device,
    gamma:        float,
    reward_scale: float = 1.0,
) -> tuple[float, float, float]:
    """
    Evalúa la política de forma determinista (argmax) con semilla fija.

    Usa un entorno separado con semilla distinta a la de entrenamiento para
    medir generalización fuera de la distribución de entrenamiento.

    Returns
    -------
    tuple[float, float, float]
        (G_val, val_rate, val_rate_per_client): retorno descontado G_0,
        throughput total acumulado y throughput medio por cliente activo.
    """
    obs, _ = env.reset()
    done   = False
    rewards, ep_rates, n_active_list = [], [], []
    model.eval()
    with torch.no_grad():
        while not done:
            data = obs.to(device)
            ch_logits, pwr_logits = model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict
            )
            action = torch.stack(
                [ch_logits.argmax(dim=-1), pwr_logits.argmax(dim=-1)], dim=1
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward / reward_scale)
            ep_rates.append(info.get("total_rate", 0.0))
            n_active = max(int(info.get("n_active_clients", 1)), 1)
            n_active_list.append(n_active)
    model.train()
    G_val    = compute_returns(rewards, gamma)[0].item() if rewards else 0.0
    val_rate = sum(ep_rates)
    val_rate_per_client = sum(ep_rates) / sum(n_active_list) if sum(n_active_list) > 0 else 0.0
    return G_val, val_rate, val_rate_per_client


import matplotlib
import matplotlib.pyplot as plt
import math

class RealTimePlotter:
    """Graficador interactivo en tiempo real para loss, reward y rate/cliente usando GUI local."""
    def __init__(self, title="RAWGRL: Real-Time Training Monitor"):
        plt.ion()  # Habilitar modo interactivo de matplotlib
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4.5))
        self.fig.suptitle(title, fontsize=12, fontweight='bold', color='#1e293b')
        
        # 1. Panel de Recompensa (Return G_t)
        self.ax_reward = self.axes[0]
        self.line_reward, = self.ax_reward.plot([], [], color='#2563eb', linewidth=1.5, label='Train Return ($G_0$)')
        self.line_val, = self.ax_reward.plot([], [], color='#f97316', linestyle='--', linewidth=1.5, label='Val Return ($G_{val}$)')
        self.ax_reward.set_title("Cumulative Reward (Throughput)", fontsize=10, fontweight='bold')
        self.ax_reward.set_xlabel("Episode", fontsize=9)
        self.ax_reward.set_ylabel("Reward", fontsize=9)
        self.ax_reward.grid(True, linestyle=':', alpha=0.6)
        self.ax_reward.legend(loc='upper left', fontsize=8)
        
        # 2. Panel de Pérdida (Policy Loss)
        self.ax_loss = self.axes[1]
        self.line_loss, = self.ax_loss.plot([], [], color='#dc2626', linewidth=1.5, label='Policy Loss')
        self.ax_loss.set_title("Optimization: Policy Loss", fontsize=10, fontweight='bold')
        self.ax_loss.set_xlabel("Episode", fontsize=9)
        self.ax_loss.set_ylabel("Loss", fontsize=9)
        self.ax_loss.grid(True, linestyle=':', alpha=0.6)
        self.ax_loss.legend(loc='upper right', fontsize=8)

        # 3. Panel de Rate por Cliente
        self.ax_rate = self.axes[2]
        self.line_rate, = self.ax_rate.plot([], [], color='#10b981', linewidth=1.5, label='Train Rate/Client')
        self.line_val_rate, = self.ax_rate.plot([], [], color='#ec4899', linestyle='--', linewidth=1.5, label='Val Rate/Client')
        self.ax_rate.set_title("Throughput per Client", fontsize=10, fontweight='bold')
        self.ax_rate.set_xlabel("Episode", fontsize=9)
        self.ax_rate.set_ylabel("Bits/s/Hz", fontsize=9)
        self.ax_rate.grid(True, linestyle=':', alpha=0.6)
        self.ax_rate.legend(loc='upper left', fontsize=8)
        
        self.fig.tight_layout()
        plt.show()
        
    def update(self, episodes, returns, val_returns, losses, rates, val_rates):
        # 1. Actualizar Recompensas
        self.line_reward.set_xdata(episodes)
        self.line_reward.set_ydata(returns)
        
        val_eps = [ep for ep, val in zip(episodes, val_returns) if not math.isnan(val)]
        val_y = [val for val in val_returns if not math.isnan(val)]
        self.line_val.set_xdata(val_eps)
        self.line_val.set_ydata(val_y)
        
        # 2. Actualizar Loss
        self.line_loss.set_xdata(episodes)
        self.line_loss.set_ydata(losses)

        # 3. Actualizar Rate/Cliente
        self.line_rate.set_xdata(episodes)
        self.line_rate.set_ydata(rates)

        val_rate_eps = [ep for ep, val in zip(episodes, val_rates) if not math.isnan(val)]
        val_rate_y = [val for val in val_rates if not math.isnan(val)]
        self.line_val_rate.set_xdata(val_rate_eps)
        self.line_val_rate.set_ydata(val_rate_y)
        
        # Re-escalar vistas de manera automática y dinámica
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()
        
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.ax_rate.relim()
        self.ax_rate.autoscale_view()
        
        # Redibujar canvas y procesar eventos GUI del sistema operativo (macOS Aqua)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# ── Bucle Principal de Entrenamiento ──────────────────────────────────────────

def train(args: argparse.Namespace) -> tuple[GNN, pd.DataFrame]:
    """
    Orquesta el ciclo completo de entrenamiento REINFORCE.

    Cambios respecto a la versión anterior
    ---------------------------------------
    1.  Baseline EMA con β = 0.99 (era 0.95). El coeficiente más alto produce
        un baseline más estable porque incorpora menos ruido de episodio a
        episodio. Con β = 0.95, el baseline tenía una vida media de ~14 episodios;
        con β = 0.99 la vida media es ~69 episodios, lo que amortigua mejor la
        alta varianza intrínseca del ambiente WiFi estocástico.

    2.  Normalización de ventajas completa (centrado + escalado). La versión
        anterior solo escalaba por std. El centrado elimina el sesgo de escala
        de reward entre episodios con distinto número de clientes activos.

    3.  Reward per-client normalizado por slot. El reward crudo es la suma de
        tasas de todos los clientes activos. Al dividir por n_active_clients en
        cada slot, el agente recibe una señal proporcional a la calidad media
        por cliente, independiente de cuántos haya llegado estocásticamente.
        Esto reduce la brecha sistemática entre Train Rate y Val Rate observada
        en el primer run, donde episodios de entrenamiento con muchos clientes
        producían gradientes artificialmente grandes.

    4.  GradNorm añadido al log para monitorear estabilidad del gradiente.

    Parameters
    ----------
    args : argparse.Namespace
        Hiperparámetros y metadatos del experimento.

    Returns
    -------
    tuple[GNN, pd.DataFrame]
        Modelo entrenado e historial de métricas.
    """
    device = torch.device("cpu")
    # log.info("Dispositivo de cómputo activo: %s", device)

    # 1. Carga de distribuciones de canal del edificio
    log.info("Cargando distribuciones para edificio: %s", args.building_id)
    distributions = load_distributions(building_id=args.building_id, verbose=False)

    available_channels = [1, 6, 11]
    # tx_powers_dbm      = [20.0, 14.0, 8.0]
    tx_powers_dbm      = [23.0, 14.0, 8.0]
    n_aps              = len(distributions[0].blocks[0].datos)

    # 2. Entornos de entrenamiento y evaluación con semillas distintas
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
        use_5g=not args.no_5g,
        device=device,
    )

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

    # 3. Inicialización del modelo
    if args.model_type == 'gnn2':
        model = GNN2(
            hidden_channels=args.hidden,
            num_aps=n_aps,
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
            num_layers=3,
            K=args.tagconv_k,
        ).to(device)
    elif args.model_type == 'random':
        model = RandomModel(
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
        ).to(device)
    else:
        model = GNN(
            hidden_channels=args.hidden,
            num_aps=n_aps,
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
            num_layers=args.num_layers,
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.episodes, eta_min=1e-5
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Estado del entrenamiento
    best_return  = -float("inf")
    # Baseline EMA escalar con β = 0.99: vida media ≈ 69 episodios.
    # Un baseline escalar (EMA de G_0) es el estándar de REINFORCE.
    # La versión anterior usaba un vector per-timestep que corrompía las ventajas.
    baseline_ema: float | None = None
    beta_ema = 0.99

    metrics_log: list[dict] = []
    t_start = time.time()

    # Inicializar el graficador interactivo en tiempo real
    try:
        plotter = RealTimePlotter()
    except Exception as e:
        log.warning("No se pudo iniciar la interfaz gráfica en tiempo real: %s. Continuando entrenamiento sin ventana GUI.", e)
        plotter = None

    print(f"\n{'='*80}")
    print(f"  Marco de Optimización Analítica: REINFORCE Policy Gradient")
    print(f"  Topología Física: Edificio {args.building_id} | Horizonte Temporal E: {args.episodes} episodios")
    print(f"{'='*80}\n")
    print(f"[Hardware] Dispositivo de Cómputo Tensorial: {device}\n")
    print(f"{'Epoch':<6} | {'G_t (Train)':>11} | {'G_t (Val)':>10} | {'L(theta)':>9} | {'L(phi)':>9} | {'Entropy':>9} | {'Time(s)':>7}")
    print(f"{'-'*7:<7}+{'-'*13:<13}+{'-'*12:<12}+{'-'*11:<11}+{'-'*11:<11}+{'-'*10:<10}+{'-'*8:<8}")

    # 5. Bucle principal REINFORCE
    for episode in range(args.episodes):
        # No pasar semilla al env de entrenamiento: torch.manual_seed() dentro
        # del env reseteaba el RNG de Categorical.sample(), destruyendo la
        # exploración estocástica de la política.
        obs, _  = env.reset(seed=None)

        log_probs_list: list[torch.Tensor] = []
        entropies_list: list[torch.Tensor] = []
        rewards_raw:    list[float]        = []
        n_clients_list: list[int]          = []
        ep_rates:       list[float]        = []
        done = False
        t_ep = time.time()

        # Fase de rollout
        while not done:
            data = obs.to(device)
            ch_logits, pwr_logits = model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict
            )

            ch_dist  = Categorical(logits=ch_logits)
            pwr_dist = Categorical(logits=pwr_logits)

            ch_acts  = ch_dist.sample()
            pwr_acts = pwr_dist.sample()

            # mean() por AP en vez de sum(): evita que la magnitud del gradiente
            # escale linealmente con N (número de APs).
            log_probs_list.append(
                ch_dist.log_prob(ch_acts).mean() + pwr_dist.log_prob(pwr_acts).mean()
            )
            entropies_list.append(
                ch_dist.entropy().mean() + pwr_dist.entropy().mean()
            )

            action = torch.stack([ch_acts, pwr_acts], dim=1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Reward per-client: normalizamos por clientes activos en el slot.
            # Desacopla la señal de la carga de tráfico estocástica del entorno.
            n_active = max(int(info.get("n_active_clients", 1)), 1)
            rewards_raw.append(reward)
            n_clients_list.append(n_active)
            ep_rates.append(info.get("total_rate", 0.0))

        # Rewards normalizados por cliente activo (sin reward_scale).
        # La normalización de ventajas posterior elimina el sesgo de escala
        # absoluta, por lo que dividir adicionalmente por reward_scale
        # solo aplastaba la señal útil.
        rewards_scaled = [
            r / n for r, n in zip(rewards_raw, n_clients_list)
        ]

        # Retornos descontados G_t (reward-to-go)
        returns = compute_returns(rewards_scaled, gamma=args.gamma).to(device)
        G_0     = returns[0].item()
        seq_len = len(returns)

        # Actualización del baseline EMA escalar (Williams, 1992)
        if baseline_ema is None:
            baseline_ema = G_0
        else:
            baseline_ema = beta_ema * baseline_ema + (1.0 - beta_ema) * G_0

        # Ventajas: A_t = G_t - b, estandarizadas (centrado + escalado)
        advantages = normalize_advantages(returns - baseline_ema)

        # Pérdida REINFORCE: L = -E[Â_t · log π(a_t|s_t)] - α_H · H[π]
        log_probs_t = torch.stack(log_probs_list)
        entropies_t = torch.stack(entropies_list)

        loss_actor    = -(advantages * log_probs_t).mean()
        entropy_bonus = args.entropy_coef * entropies_t.mean()
        policy_loss   = loss_actor - entropy_bonus

        # Actualización de parámetros con clipping de norma del gradiente
        optimizer.zero_grad()
        if torch.isfinite(policy_loss):
            policy_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0)
            log.warning(
                "Episodio %d: loss no finita (%.4f), update omitido.",
                episode + 1, policy_loss.item(),
            )
        scheduler.step()

        # Validación determinista cada 10 episodios
        G_val, val_rate, val_rate_per_client = float("nan"), float("nan"), float("nan")
        if (episode + 1) % args.eval_every == 0 or episode == 0:
            G_val, val_rate, val_rate_per_client = evaluate_policy(
                env_eval, model, device, args.gamma, args.reward_scale
            )
            if G_val > best_return:
                best_return = G_val
                torch.save(model.state_dict(), save_dir / "best_model.pt")

        rate_per_client = sum(ep_rates) / sum(n_clients_list) if sum(n_clients_list) > 0 else 0.0

        metrics_log.append({
            "episode":             episode + 1,
            "return":              G_0,
            "G_val":               G_val,
            "baseline":            baseline_ema if baseline_ema is not None else 0.0,
            "total_rate":          sum(ep_rates),
            "val_rate":            val_rate,
            "rate_per_client":     rate_per_client,
            "val_rate_per_client": val_rate_per_client,
            "loss":                policy_loss.item(),
            "entropy":             entropies_t.mean().item(),
            "grad_norm":           grad_norm.item(),
            "sec":                 time.time() - t_ep,
        })

        # Actualizar la interfaz gráfica interactiva en tiempo real (cada episodio)
        if plotter is not None:
            try:
                eps = [m["episode"] for m in metrics_log]
                returns = [m["return"] for m in metrics_log]
                val_returns = [m["G_val"] for m in metrics_log]
                losses = [m["loss"] for m in metrics_log]
                rates = [m["rate_per_client"] for m in metrics_log]
                val_rates = [m["val_rate_per_client"] for m in metrics_log]
                plotter.update(eps, returns, val_returns, losses, rates, val_rates)
            except Exception:
                pass

        if (episode + 1) % 10 == 0 or episode == 0:
            g_val_str = f"{G_val:>10.2f}" if not np.isnan(G_val) else f"{'—':>10}"
            print(
                f"{episode+1:<6} | "
                f"{G_0:>11.2f} | "
                f"{g_val_str} | "
                f"{policy_loss.item():>9.4f} | "
                f"{'—':>9} | "
                f"{entropies_t.mean().item():>9.4f} | "
                f"{time.time() - t_ep:>7.1f}"
            )
            # Guardar CSV temporal y actualizar gráficas en tiempo real
            try:
                temp_df = pd.DataFrame(metrics_log)
                temp_df.to_csv(save_dir / "training_metrics.csv", index=False)
                import subprocess
                plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
                if plot_script.exists():
                    subprocess.run(
                        [
                            sys.executable, str(plot_script),
                            "--csv", str(save_dir / "training_metrics.csv"),
                            "--out", str(Path(__file__).parent / "plots"),
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            except Exception:
                pass

    # 6. Persistencia final
    elapsed = time.time() - t_start
    log.info("Entrenamiento finalizado en %.1f min.", elapsed / 60)

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(save_dir / "training_metrics.csv", index=False)

    try:
        import subprocess
        plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
        if plot_script.exists():
            log.info("Generando gráficas de entrenamiento...")
            subprocess.run(
                [
                    sys.executable, str(plot_script),
                    "--csv", str(save_dir / "training_metrics.csv"),
                    "--out", str(Path(__file__).parent / "plots"),
                ],
                check=False,
            )
    except Exception as exc:
        log.warning("No se pudieron generar las gráficas: %s", exc)

    return model, metrics_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Configura la interfaz de línea de comandos unificada."""
    p = argparse.ArgumentParser(description="NetROML: REINFORCE WiFi Resource Allocation")
    
    # 1. Configuración de Entorno y Datos
    p.add_argument("--building_id",     default="990", type=str, help="ID del edificio objetivo.")
    p.add_argument("--dist_joblib",     default="data/distributions_990.joblib", type=str, help="Path a distributions.joblib.")
    p.add_argument("--step2_csv",       default="data/dataset_990_step2.csv", type=str, help="Path al dataset step2 CSV.")
    p.add_argument("--save_dir",        default="outputs/models", type=str, help="Directorio de guardado.")
    p.add_argument("--seed",            type=lambda x: None if str(x).lower() == 'none' else int(x), default=42, help="Semilla aleatoria.")
    p.add_argument("--eval_every",      type=int,   default=10, help="Frecuencia de evaluación (A2C/PPO/REINFORCE).")

    # 2. Parámetros del Episodio y Simulación
    p.add_argument("--episodes",        type=int,   default=500, help="Número de episodios de entrenamiento.")
    p.add_argument("--timesteps",       type=int,   default=1000, help="Horizonte temporal del episodio (slots).")
    p.add_argument("--decision_period", type=int,   default=10, help="Periodo de decisión de recursos (slots).")
    p.add_argument("--arrival_rate",    type=float, default=2.0, help="Tasa de arribo de clientes.")
    p.add_argument("--mean_dur",        type=float, default=15.0, help="Duración promedio de sesión (slots).")
    p.add_argument("--no_5g",           action="store_true", help="Deshabilita el uso de la banda de 5 GHz.")
    p.add_argument("--sticky_mode",     type=str,   default="full", choices=["full", "sticky", "lite"], help="Modo de asociación de cliente.")

    # 3. Parámetros de la Red GNN
    p.add_argument("--hidden",          type=int,   default=64, help="Dimensión latente de los embeddings.")
    p.add_argument("--model_type",      type=str,   default="gnn1", choices=["gnn1", "gnn2", "random"], help="Tipo de GNN.")
    p.add_argument("--num_layers",      type=int,   default=4, help="Número de capas convolucionales.")
    p.add_argument("--tagconv_k",       type=int,   default=5, help="Hops K para TAGConv en GNN2.")

    # 4. Hiperparámetros de RL Comunes
    p.add_argument("--reward_scale",    type=float, default=1000.0, help="Escalador de la recompensa física.")
    p.add_argument("--lr",              type=float, default=3e-4, help="Learning rate del Actor / GNN.")
    p.add_argument("--gamma",           type=float, default=0.99, help="Factor de descuento gamma.")
    p.add_argument("--entropy_coef",    type=float, default=0.005, help="Coeficiente para bono de entropía.")

    # 5. Parámetros de Critic y GAE (A2C y PPO)
    p.add_argument("--lr_critic",     type=float, default=1e-3, help="Learning rate de la cabeza Critic.")
    p.add_argument("--gae_lambda",    type=float, default=0.95, help="Lambda para GAE.")
    p.add_argument("--vf_coef",       type=float, default=0.5, help="Peso del término Critic en la loss total.")

    # 6. Parámetros Específicos de PPO
    p.add_argument("--update_epochs", type=int,   default=4, help="PPO: Épocas de optimización sobre el rollout.")
    p.add_argument("--minibatch_size",type=int,   default=64, help="PPO: Tamaño de minibatch.")
    p.add_argument("--clip_coef",     type=float, default=0.2, help="PPO: Coeficiente de clipping de la política.")
    p.add_argument("--max_grad_norm", type=float, default=0.5, help="PPO/Común: Clipping global de gradiente.")
    p.add_argument("--norm_adv",       action=argparse.BooleanOptionalAction, default=True, help="PPO: Normalizar ventajas.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())