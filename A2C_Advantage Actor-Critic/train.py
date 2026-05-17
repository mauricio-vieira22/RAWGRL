r"""
train.py – Módulo de Entrenamiento Core (Advantage Actor-Critic).

Entrena una política Actor-Critic usando la GNN parametrizada para resolver
el POMDP de asignación de recursos Wi-Fi.

Matemática del Advantage Actor-Critic (A2C)
-------------------------------------------
El modelo conjuga dos redes entrenadas simultáneamente:
1. Política Actor $\pi_\theta(a_t|s_t)$ parametrizada por pesos $\theta$.
2. Función Critic $V_\phi(s_t)$ parametrizada por pesos $\phi$.

La ventaja se estima con Generalized Advantage Estimation (GAE, Schulman et al. 2016):
$$ \hat{A}_t^{GAE} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad
   \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) $$

Antes de usarse en la pérdida del Actor, la ventaja se normaliza por batch:
$$ \tilde{A}_t = \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + \epsilon} $$
Esta operación elimina el sesgo constante introducido por un Crítico imperfecto
y estabiliza la magnitud del gradiente del Actor, sin alterar la dirección de mejora.

El Crítico minimiza el MSE sobre retornos normalizados por batch:
$$ \mathcal{L}(\phi) = \mathbb{E}_t \left[ \left(
    \frac{V_\phi(s_t) - \mu(G)}{\sigma(G) + \epsilon} -
    \frac{G_t - \mu(G)}{\sigma(G) + \epsilon}
   \right)^2 \right] $$
La normalización es necesaria porque $G_t = \sum_u r_{u,\tau}$ (Shannon nansum sobre
todos los clientes activos durante T slots) alcanza $\mathcal{O}(10^3)$, haciendo que
$\mathcal{L}(\phi)$ sin normalizar sea $\mathcal{O}(10^6)$ e impida la convergencia.

La función de pérdida total combina ambos objetivos y un bono de entropía:
$$ \mathcal{L}_{total} = \mathcal{L}_{actor}(\theta) + c_{vf} \mathcal{L}_{critic}(\phi) - c_{ent} \mathcal{H}(\pi_\theta) $$

Validación Periódica Cruzada
----------------------------
Cada `--eval_every` episodios el algoritmo interrumpe el backpropagation y evalúa
la política de forma determinista sobre un entorno con semilla ortogonal, registrando
$\bar{G}_{val}$ como métrica de detención y selección del mejor modelo.
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
from model.gnn2_model import GNN2


class RandomModel(torch.nn.Module):
    """Modelo baseline aleatorio compatible con la interfaz Actor-Critic."""
    def __init__(self, n_aps, n_ch, n_pwr):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.n_ch  = n_ch
        self.n_pwr = n_pwr

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None):
        device = x_dict['ap'].device
        n_ap = x_dict['ap'].size(0)
        ch_logits  = torch.zeros((n_ap, self.n_ch), device=device) + self.dummy
        pwr_logits = torch.zeros((n_ap, self.n_pwr), device=device) + self.dummy
        
        # Valor base para el critic
        if batch_dict is not None and 'ap' in batch_dict:
            n_graphs = int(batch_dict['ap'].max().item() + 1)
            value = torch.zeros((n_graphs, 1), device=device) + self.dummy
        else:
            value = torch.zeros((1, 1), device=device) + self.dummy
        return ch_logits, pwr_logits, value


def compute_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    """Implementa cálculo de retornos descontados hacia atrás."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)





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


import matplotlib
import matplotlib.pyplot as plt
import math

class RealTimePlotter:
    """Graficador interactivo en tiempo real para loss y reward usando GUI local."""
    def __init__(self, title="RAWGRL (A2C): Real-Time Training Monitor"):
        plt.ion()  # Habilitar modo interactivo de matplotlib
        self.fig, self.axes = plt.subplots(1, 2, figsize=(11, 4.5))
        self.fig.suptitle(title, fontsize=12, fontweight='bold', color='#1e293b')
        
        # 1. Panel de Recompensa (Return G_t)
        self.ax_reward = self.axes[0]
        self.line_reward, = self.ax_reward.plot([], [], color='#2563eb', linewidth=1.5, label='Train Return ($G_{train}$)')
        self.line_val, = self.ax_reward.plot([], [], color='#f97316', linestyle='--', linewidth=1.5, label='Val Return ($G_{val}$)')
        self.ax_reward.set_title("Cumulative Reward (Throughput)", fontsize=10, fontweight='bold')
        self.ax_reward.set_xlabel("Episode", fontsize=9)
        self.ax_reward.set_ylabel("Reward", fontsize=9)
        self.ax_reward.grid(True, linestyle=':', alpha=0.6)
        self.ax_reward.legend(loc='upper left', fontsize=8)
        
        # 2. Panel de Pérdida (Total Loss)
        self.ax_loss = self.axes[1]
        self.line_loss, = self.ax_loss.plot([], [], color='#dc2626', linewidth=1.5, label='Total Loss')
        self.ax_loss.set_title("Optimization: Total Loss", fontsize=10, fontweight='bold')
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


def train(args: argparse.Namespace):
    """Loop principal de inicialización y entrenamiento (A2C)."""
    print(f"\n{'='*70}")
    print(f"  Marco de Optimización Analítica: Advantage Actor-Critic (A2C)")
    print(f"  Topología Física: Edificio {args.building_id} | Horizonte Temporal E: {args.episodes} episodios")
    print(f"  Hiperparámetros de Regularización: c_vf={args.vf_coef} | c_ent={args.entropy_coef} | Cross-Validation c/{args.eval_every} ep")
    print(f"  A2C Config: GAE lambda={args.gae_lambda} | Reward Scale={args.reward_scale}")
    print(f"  Learning Rates: lr_actor={args.lr} | lr_critic={args.lr_critic} | ratio={args.lr_critic/args.lr:.1f}x")
    print(f"{'='*70}\n")

    device = torch.device("cpu")
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
        use_5g=not args.no_5g,
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
        use_5g=not args.no_5g,
        device=device,
    )

    # ── Selección de la Arquitectura de Red Neuronal ────────────────────────
    if args.model_type == "gnn2":
        print(f"[Arquitectura] Utilizando GNN2 (TAGConv) con K={args.tagconv_k}...")
        model = GNN2(
            hidden_channels=args.hidden,
            num_aps=n_aps,
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
            K=args.tagconv_k
        ).to(device)
    elif args.model_type == "random":
        print(f"[Arquitectura] Utilizando Baseline ALEATORIO (Uniforme)...")
        model = RandomModel(
            n_aps=n_aps,
            n_ch=len(available_channels),
            n_pwr=len(tx_powers_dbm)
        ).to(device)
    else:
        print(f"[Arquitectura] Utilizando GNN1 (GATv2 Heterogéneo)...")
        model = GNN(
            hidden_channels=args.hidden,
            num_aps=n_aps,
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
            num_layers=args.num_layers
        ).to(device)

    # ── Optimizadores Separados por Rol (Actor vs. Critic) ─────────────────────
    # La GNN comparte pesos base (encoders + conv1 + conv2) entre Actor y Crítico.
    # Solo las cabezas de salida son independientes. La separación de optimizadores
    # se implementa mediante grupos de parámetros dentro de un único AdamW:
    #
    #   Grupo "backbone": encoders + capas convolucionales compartidas → lr_actor
    #   Grupo "actor":    channel_head + power_head                    → lr_actor
    #   Grupo "critic":   value_head                                   → lr_critic
    #
    # Justificación del ratio lr_critic > lr_actor (Mnih et al. 2016, A3C §4):
    # El Crítico debe converger más rápido que el Actor para que la ventaja estimada
    # $\hat{A}_t$ sea una señal útil. Con lr iguales la pérdida del Crítico puede
    # dominar la dinámica de los pesos compartidos, degradando la política.
    # El ratio estándar en la literatura es lr_critic / lr_actor ∈ [2, 5].
    if hasattr(model, 'value_head'):
        actor_params = (
            list(model.ap_encoder.parameters())
            + list(model.client_encoder.parameters())
            + list(model.conv1.parameters())
            + list(model.conv2.parameters())
            + list(model.channel_head.parameters())
            + list(model.power_head.parameters())
        )
        critic_params = list(model.value_head.parameters())
        optimizer = optim.AdamW([
            {"params": actor_params,  "lr": args.lr},
            {"params": critic_params, "lr": args.lr_critic},
        ])
    else:
        # Fallback para RandomModel u otras arquitecturas sin value_head explícito.
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Scheduler independiente por grupo: cada grupo tiene su propio lr que decae.
    # T_max = episodes // 2: el lr desciende hasta eta_min en la primera mitad y
    # se mantiene en piso, evitando colapso prematuro antes de la convergencia.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.episodes // 2, 1),
        eta_min=1e-5,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_G_val   = -float("inf")   # Mejor Esperanza del Retorno Empírico
    metrics      = []
    t_global_start = time.time()

    # Inicializar el graficador interactivo en tiempo real
    try:
        plotter = RealTimePlotter()
    except Exception as e:
        print(f"[Aviso] No se pudo iniciar la interfaz gráfica en tiempo real: {e}. Continuando sin ventana GUI.")
        plotter = None

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

        # ── Computación del Advantage y Retorno (A2C con GAE) ─────────────────────
        values_t = torch.stack(values)  # (T,)
        
        # Generalized Advantage Estimation (GAE)
        advantages_list = []
        gae = 0.0
        v_preds = values_t.detach().tolist()
        v_preds.append(0.0)  # Valor terminal V(S_{T}) = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + args.gamma * v_preds[t+1] - v_preds[t]
            gae = delta + args.gamma * args.gae_lambda * gae
            advantages_list.insert(0, gae)
            
        advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)
        
        # Target del Critic: G_t = A_t + V(s_t)  (escala original, sin normalizar)
        returns_t = advantages + values_t.detach()

        if len(advantages) > 1:
            # Normalización libre de sesgo: $\hat{A}_t \leftarrow (\hat{A}_t - \mu) / (\sigma + \epsilon)$
            # Se resta la media para eliminar el sesgo constante introducido por un Crítico
            # imperfecto, y se divide por la desviación estándar para estabilizar la magnitud
            # del gradiente del Actor independientemente de la escala absoluta de la recompensa.
            # NOTA: esta operación ocurre DESPUÉS de calcular returns_t, de modo que el target
            # del Crítico conserva su escala original y no introduce sesgo en V_phi.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalización de los targets del Crítico para desacoplar L(phi) de la escala
        # absoluta de la recompensa acumulada. Con R_tau ~ sum Shannon sobre U_t clientes
        # durante T slots, returns_t puede alcanzar O(10^3), haciendo que MSE sin normalizar
        # sea O(10^6) e impida la convergencia del Crítico.
        # Se normaliza por batch (episodio): returns_norm ~ N(0,1) => L(phi) ~ O(1).
        if len(returns_t) > 1:
            returns_norm = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        else:
            returns_norm = returns_t

        log_probs_t = torch.stack(log_probs)    # (T,)
        entropies_t = torch.stack(entropies)    # (T,)

        # ── Definición de la Función Objetivo Global $\mathcal{L}(\theta, \phi)$ ─────────
        # Pérdida del Actor: $\mathcal{L}_{policy}(\theta) = - \mathbb{E}_t [ \log \pi_\theta(a_t|s_t) \hat{A}_t ]$
        loss_actor   = -(advantages * log_probs_t).mean()

        # Pérdida del Critic: el Crítico aprende a predecir retornos normalizados.
        # values_t también se normaliza para que ambos lados del MSE estén en la misma escala.
        # Esto es equivalente a entrenar el Crítico con una función objetivo reescalada,
        # sin alterar la dirección del gradiente del Actor (que usa advantages ya normalizadas).
        if len(values_t) > 1:
            values_norm = (values_t - returns_t.mean().detach()) / (returns_t.std().detach() + 1e-8)
        else:
            values_norm = values_t
        loss_critic  = F.mse_loss(values_norm, returns_norm)
        
        # Bono de Entropía: Fomenta la exploración penalizando la certidumbre absoluta.
        loss_entropy = -entropies_t.mean()

        # Loss Combinada: Optimizamos simultáneamente Actor y Critic compartiendo los pesos base de la GNN.
        total_loss = loss_actor + args.vf_coef * loss_critic + args.entropy_coef * loss_entropy

        # ── Backpropagation y Gradient Clipping ───────────────────────────────
        optimizer.zero_grad()
        if torch.isfinite(total_loss):
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            grad_norm = torch.tensor(0.0)
        scheduler.step()

        # Retorno acumulado real del episodio: suma de recompensas escaladas.
        # Nota: rewards ya está dividido por reward_scale en run_episode.
        # No se usa returns_t[0] porque ese valor es el target GAE del primer
        # timestep (mezcla de recompensa futura estimada y bootstrapping del Crítico),
        # no el retorno empírico observado del episodio.
        G_train = sum(rewards)

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

        # Actualizar la interfaz gráfica interactiva en tiempo real (cada episodio)
        if plotter is not None:
            try:
                eps = [m["episode"] for m in metrics]
                returns = [m["return"] for m in metrics]
                val_returns = [m["G_val"] for m in metrics]
                losses = [m["loss"] for m in metrics]
                plotter.update(eps, returns, val_returns, losses)
            except Exception:
                pass

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
            # Guardar CSV temporal y actualizar gráficas en tiempo real
            try:
                temp_df = pd.DataFrame(metrics)
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
    p.add_argument("--no_5g",         action="store_true", help="Deshabilita el uso de la banda de 5 GHz")
    p.add_argument("--sticky_mode",   type=str,   default="sticky", choices=["full", "sticky", "lite"])
    p.add_argument("--hidden",        type=int,   default=64)
    p.add_argument("--model_type",    type=str,   default="gnn1", choices=["gnn1", "gnn2", "random"])
    p.add_argument("--num_layers",    type=int,   default=4)
    p.add_argument("--tagconv_k",     type=int,   default=5, help="Hops K para TAGConv en GNN2")
    p.add_argument("--reward_scale",  type=float, default=1.0)
    p.add_argument("--lr",            type=float, default=3e-4,
                   help="Learning rate del Actor y del backbone GNN compartido.")
    p.add_argument("--lr_critic",     type=float, default=1e-3,
                   help="Learning rate de la cabeza Critic (value_head). "
                        "Debe ser mayor que --lr para que V_phi converja antes "
                        "que el Actor comience a explotar su señal de ventaja. "
                        "Ratio recomendado lr_critic/lr ∈ [2, 5].")
    p.add_argument("--gae_lambda",    type=float, default=0.95,
                   help="Parámetro Lambda para Generalized Advantage Estimation (GAE).")
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