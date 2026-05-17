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
) -> tuple[float, float]:
    """
    Evalúa la política de forma determinista (argmax) con semilla fija.

    Usa un entorno separado con semilla distinta a la de entrenamiento para
    medir generalización fuera de la distribución de entrenamiento.

    Returns
    -------
    tuple[float, float]
        (G_val, val_rate): retorno descontado G_0 del episodio de validación
        y throughput total acumulado (suma de total_rate por step).
    """
    obs, _ = env.reset()
    done   = False
    rewards, ep_rates = [], []
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
    model.train()
    G_val    = compute_returns(rewards, gamma)[0].item() if rewards else 0.0
    val_rate = sum(ep_rates)
    return G_val, val_rate


import matplotlib
import matplotlib.pyplot as plt
import math

class RealTimePlotter:
    """Graficador interactivo en tiempo real para loss y reward usando GUI local."""
    def __init__(self, title="RAWGRL: Real-Time Training Monitor"):
        plt.ion()  # Habilitar modo interactivo de matplotlib
        self.fig, self.axes = plt.subplots(1, 2, figsize=(11, 4.5))
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
        
        self.fig.tight_layout()
        plt.show()
        
    def update(self, episodes, returns, val_returns, losses):
        # 1. Actualizar Recompensas
        self.line_reward.set_xdata(episodes)
        self.line_reward.set_ydata(returns)
        
        # Filtrar valores NaN de la validación periódica
        val_eps = [ep for ep, val in zip(episodes, val_returns) if not math.isnan(val)]
        val_y = [val for val in val_returns if not math.isnan(val)]
        self.line_val.set_xdata(val_eps)
        self.line_val.set_ydata(val_y)
        
        # 2. Actualizar Loss
        self.line_loss.set_xdata(episodes)
        self.line_loss.set_ydata(losses)
        
        # Re-escalar vistas de manera automática y dinámica
        self.ax_reward.relim()
        self.ax_reward.autoscale_view()
        
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
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
    log.info("Dispositivo de cómputo activo: %s", device)

    # 1. Carga de distribuciones de canal del edificio
    log.info("Cargando distribuciones para edificio: %s", args.building_id)
    distributions = load_distributions(building_id=args.building_id, verbose=False)

    available_channels = [1, 6, 11]
    # tx_powers_dbm      = [20.0, 14.0, 8.0]
    tx_powers_dbm      = [23.0]
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 4. Estado del entrenamiento
    best_return  = -float("inf")
    # Baseline EMA con β = 0.99: vida media ≈ 69 episodios.
    # Inicializado en None; se establece con el primer retorno observado.
    baseline_ema: torch.Tensor | None = None
    beta_ema = 0.99

    metrics_log: list[dict] = []
    t_start = time.time()

    # Inicializar el graficador interactivo en tiempo real
    try:
        plotter = RealTimePlotter()
    except Exception as e:
        log.warning("No se pudo iniciar la interfaz gráfica en tiempo real: %s. Continuando entrenamiento sin ventana GUI.", e)
        plotter = None

    log.info(
        "Iniciando optimización: E=%d episodios | T=%d pasos | sticky=%s",
        args.episodes, args.timesteps, args.sticky_mode,
    )
    log.info("=" * 99)
    log.info(
        f"{'Ep':<6} | {'Val Rate':>10} | {'G_val':>10} | "
        f"{'Train Rate':>10} | {'Loss':>9} | {'Entropy':>9} | "
        f"{'GradNorm':>8} | {'Time':>6}"
    )
    log.info("-" * 99)

    # 5. Bucle principal REINFORCE
    for episode in range(args.episodes):
        ep_seed = args.seed + episode if args.seed is not None else None
        obs, _  = env.reset(seed=ep_seed)

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

            log_probs_list.append(
                ch_dist.log_prob(ch_acts).sum() + pwr_dist.log_prob(pwr_acts).sum()
            )
            entropies_list.append(
                ch_dist.entropy().sum() + pwr_dist.entropy().sum()
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

        # Rewards normalizados por cliente y por reward_scale
        rewards_scaled = [
            r / n / args.reward_scale
            for r, n in zip(rewards_raw, n_clients_list)
        ]

        # Retornos descontados G_t (reward-to-go)
        returns = compute_returns(rewards_scaled, gamma=args.gamma).to(device)
        G_0     = returns[0].item()
        seq_len = len(returns)

        # Actualización del baseline EMA
        if baseline_ema is None:
            baseline_ema = returns.clone()
        else:
            # Ajuste dinámico de longitud si el episodio cambia de tamaño
            if baseline_ema.shape[0] < seq_len:
                pad = returns[baseline_ema.shape[0]:].clone()
                baseline_ema = torch.cat([baseline_ema, pad])
            baseline_ema[:seq_len] = (
                beta_ema * baseline_ema[:seq_len] + (1.0 - beta_ema) * returns
            )

        # Ventajas: A_t = G_t - b_t, estandarizadas (centrado + escalado)
        advantages = normalize_advantages(returns - baseline_ema[:seq_len])

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
        G_val, val_rate = float("nan"), float("nan")
        if (episode + 1) % 10 == 0 or episode == 0:
            G_val, val_rate = evaluate_policy(
                env_eval, model, device, args.gamma, args.reward_scale
            )
            if G_val > best_return:
                best_return = G_val
                torch.save(model.state_dict(), save_dir / "best_model.pt")

        metrics_log.append({
            "episode":    episode + 1,
            "return":     G_0,
            "G_val":      G_val,
            "baseline":   baseline_ema[0].item() if baseline_ema is not None else 0.0,
            "total_rate": sum(ep_rates),
            "val_rate":   val_rate,
            "loss":       policy_loss.item(),
            "entropy":    entropies_t.mean().item(),
            "grad_norm":  grad_norm.item(),
            "sec":        time.time() - t_ep,
        })

        # Actualizar la interfaz gráfica interactiva en tiempo real (cada episodio)
        if plotter is not None:
            try:
                eps = [m["episode"] for m in metrics_log]
                returns = [m["return"] for m in metrics_log]
                val_returns = [m["G_val"] for m in metrics_log]
                losses = [m["loss"] for m in metrics_log]
                plotter.update(eps, returns, val_returns, losses)
            except Exception:
                pass

        if (episode + 1) % 10 == 0 or episode == 0:
            log.info(
                f"{episode+1:<6} | "
                f"{val_rate:>10.1f} | "
                f"{G_val:>10.2f} | "
                f"{sum(ep_rates):>10.1f} | "
                f"{policy_loss.item():>9.4f} | "
                f"{entropies_t.mean().item():>9.4f} | "
                f"{grad_norm.item():>8.3f} | "
                f"{time.time() - t_ep:>6.1f}"
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
    """Configura la interfaz de línea de comandos."""
    p = argparse.ArgumentParser(description="NetROML: REINFORCE WiFi Resource Allocation")
    p.add_argument("--building_id",     default="990")
    p.add_argument("--save_dir",        default="outputs/models")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--episodes",        type=int,   default=50)
    p.add_argument("--timesteps",       type=int,   default=1000)
    p.add_argument("--arrival_rate",    type=float, default=0.5)
    p.add_argument("--mean_dur",        type=float, default=50.0)
    p.add_argument("--decision_period", type=int,   default=1)
    p.add_argument("--no_5g",           action="store_true",
                   help="Deshabilita la banda de 5 GHz")
    p.add_argument("--sticky_mode",     type=str,   default="full",
                   choices=["full", "sticky", "lite"])
    p.add_argument("--hidden",          type=int,   default=64)
    p.add_argument("--model_type",      type=str,   default="gnn1",
                   choices=["gnn1", "gnn2", "random"])
    p.add_argument("--num_layers",      type=int,   default=4,
                   help="Número de capas convolucionales (GNN1)")
    p.add_argument("--tagconv_k",       type=int,   default=5,
                   help="Número de hops K para TAGConv (GNN2)")
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--reward_scale",    type=float, default=1.0)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--entropy_coef",    type=float, default=0.01)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())