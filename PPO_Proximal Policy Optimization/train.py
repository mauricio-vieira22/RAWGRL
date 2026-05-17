r"""
train.py – Módulo de Entrenamiento Core (Proximal Policy Optimization).

Entrena una política Actor-Critic usando la GNN parametrizada para resolver
el POMDP de asignación de recursos Wi-Fi, estabilizada por el algoritmo PPO.

Matemática del Proximal Policy Optimization (PPO)
-------------------------------------------------
PPO optimiza la política limitando el tamaño de las actualizaciones mediante una 
función de ventaja recortada (clipping).

Se define la razón de importancia $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.
El objetivo subrogado recortado que se maximiza es:

$$ \mathcal{L}^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right] $$

La función de pérdida total minimizada combina este objetivo con el error del crítico 
y un bono de entropía:
$$ \mathcal{L}_{total} = - \mathcal{L}^{CLIP}(\theta) + c_{vf} \mathcal{L}_{value}(\phi) - c_{ent} \mathcal{H}(\pi_\theta) $$

Técnicas de Regularización PPO Implementadas:
- Generalized Advantage Estimation (GAE): Cálculo de $\hat{A}_t$ balanceando sesgo/varianza.
- Clipping de Gradiente: Restricción de la norma para evitar divergencias.
- Memoria de Rollouts: Reutilización de trayectorias en múltiples épocas de optimización.
"""

from __future__ import annotations

# Hotfix para PyTorch Geometric en Python 3.13+/3.14 (bug en inspector de `typing.Union`)
import torch_geometric.inspector as _pyg_inspector
try:
    _original_type_repr = _pyg_inspector.type_repr
    def _safe_type_repr(obj, _globals=None):
        try:
            return _original_type_repr(obj, _globals)
        except AttributeError as e:
            if "'_name'" in str(e):
                return "Union"
            raise
    _pyg_inspector.type_repr = _safe_type_repr
except Exception:
    pass

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

# ── Importaciones locales (rutas actualizadas) ──────────────────────────
from data.data_loader import load_distributions
from model.network_graph_env import NetworkGraphEnv
from model.gnn_model import GNN
from model.gnn2_model import GNN2


class RandomModel(torch.nn.Module):
    """Modelo baseline aleatorio compatible con la interfaz PPO (Actor-Critic)."""
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


import matplotlib
import matplotlib.pyplot as plt
import math

class RealTimePlotter:
    """Graficador interactivo en tiempo real para loss y reward usando GUI local."""
    def __init__(self, title="RAWGRL (PPO): Real-Time Training Monitor"):
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
        
        # Filtrar valores NaN de la validación periódica si están presentes
        if len(val_returns) > 0 and any(not math.isnan(x) for x in val_returns):
            val_eps = [ep for ep, val in zip(episodes, val_returns) if not math.isnan(val)]
            val_y = [val for val in val_returns if not math.isnan(val)]
            self.line_val.set_xdata(val_eps)
            self.line_val.set_ydata(val_y)
        else:
            self.line_val.set_xdata([])
            self.line_val.set_ydata([])
        
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


# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    print(f"\n{'='*70}")
    print(f"  Marco de Optimización Analítica: Proximal Policy Optimization (PPO)")
    print(f"  Topología Física: Edificio {args.building_id} | Horizonte Temporal E: {args.episodes} episodios")
    print(f"  PPO Params: clip={args.clip_coef} | epochs={args.update_epochs} | GAE_lambda={args.gae_lambda}")
    print(f"{'='*70}\n")

    device = torch.device("cpu")
    print(f"[Hardware] Dispositivo de Cómputo Tensorial: {device}\n")

    # ── Parametrización del POMDP y Carga Estocástica ───────────────────────
    print("[Pipeline] Inyectando Modelos Estocásticos de Arribo (Poisson/Exponencial)...")
    distributions = load_distributions(
        building_id=args.building_id,
        verbose=True,
    )
    print(f"[Memoria]  {len(distributions)} instanciaciones cliente-canal cargadas.\n")

    # ── Configuración Numérica Base ─────────────────────────────────────────
    available_channels = [1, 6, 11]
    tx_powers_dbm      = [20.0, 14.0, 8.0]
    n_aps              = len(distributions[0].blocks[0].datos)
    print(f"[Topología] Nodos Access Point Detectados (N): {n_aps}\n")

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

    # En PPO epsilon es usualmente pequeño (1e-5) para mayor estabilidad
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Inicialización de Variables de Control de Varianza ──────────────────
    best_rate = -float("inf")
    metrics  = []
    t0       = time.time()

    # Inicializar el graficador interactivo en tiempo real
    try:
        plotter = RealTimePlotter()
    except Exception as e:
        print(f"[Aviso] No se pudo iniciar la interfaz gráfica en tiempo real: {e}. Continuando sin ventana GUI.")
        plotter = None

    print(f"{'='*85}")
    print(f"{'Ep':<6} | {'Total Rate':>10} | {'G_t':>10} | {'L_policy':>9} | {'L_value':>9} | {'Entropy':>9} | {'Time(s)':>7}")
    print(f"-------+------------+------------+-----------+-----------+-----------+--------")

    # ── Bucle Central de Optimización Algorítmica (PPO) ───────────────────────
    for episode in range(args.episodes):
        current_seed = args.seed + episode if args.seed is not None else None
        obs, _ = env.reset(seed=current_seed)

        # Buffers iterativos para PPO rollouts
        obs_list       = []
        actions_list   = []
        logprobs_list  = []
        rewards_list   = []
        values_list    = []
        dones_list     = []
        ep_rates       = []

        done    = False
        t0_ep   = time.time()

        # Fase 1: Rollout
        while not done:
            data = obs.to(device)
            obs_list.append(data.clone()) # Almacenar grafo de PyG

            with torch.no_grad():
                ch_logits, pwr_logits, state_val = model(
                    data.x_dict,
                    data.edge_index_dict,
                    data.edge_attr_dict if hasattr(data, 'edge_attr_dict') else None
                )
                
                ch_dist  = Categorical(logits=ch_logits)
                pwr_dist = Categorical(logits=pwr_logits)
                
                ch_acts  = ch_dist.sample()
                pwr_acts = pwr_dist.sample()
                
                action = torch.stack([ch_acts, pwr_acts], dim=1)
                
                # Sumamos logprob de canal y potencia en todos los APs, un escalar por grafo
                logprob_sum = ch_dist.log_prob(ch_acts).sum() + pwr_dist.log_prob(pwr_acts).sum()

            obs, float_reward, terminated, truncated, info = env.step(action)
            float_reward = float_reward / args.reward_scale
            done = terminated or truncated

            actions_list.append(action)
            logprobs_list.append(logprob_sum)
            rewards_list.append(float_reward)
            values_list.append(state_val.view(-1))
            dones_list.append(done)
            ep_rates.append(info.get("total_rate", 0.0))

        # Almacenar último estado final de manera temporal
        with torch.no_grad():
            next_data = obs.to(device)
            _, _, next_value = model(
                next_data.x_dict,
                next_data.edge_index_dict,
                next_data.edge_attr_dict if hasattr(next_data, 'edge_attr_dict') else None
            )
            next_value = next_value.view(-1)
            next_done = done

        # Fase 2: GAE (Generalized Advantage Estimation)
        num_steps = len(rewards_list)
        b_rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        b_values  = torch.stack(values_list).squeeze(-1)
        b_dones   = torch.tensor(dones_list, dtype=torch.float32, device=device)
        
        # ── Generalized Advantage Estimation (GAE) ───────────────────────────
        # Estimación recursiva para equilibrar el trade-off Sesgo-Varianza en el cálculo de la Ventaja.
        # \hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}
        # donde \delta_t = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) (Error TD)
        advantages = torch.zeros_like(b_rewards, device=device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - float(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - b_dones[t + 1]
                nextvalues = b_values[t + 1]
            delta = b_rewards[t] + args.gamma * nextvalues * nextnonterminal - b_values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        
        # Computación del Retorno empírico objetivo: $G_t = \hat{A}_t + V_\phi(s_t)$
        b_returns = advantages + b_values
        b_actions = torch.stack(actions_list)
        b_logprobs = torch.stack(logprobs_list)

        # Fase 3: PPO Optimización
        b_inds = np.arange(num_steps)
        clipfracs = []
        
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy_loss = 0.0
        updates_count = 0

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_data_list = [obs_list[i] for i in mb_inds]
                mb_batch = Batch.from_data_list(mb_data_list).to(device)
                
                # Extraer batch indices de PyG
                batch_dict = {}
                if hasattr(mb_batch['ap'], 'batch'):
                    batch_dict['ap'] = mb_batch['ap'].batch
                
                ch_logits, pwr_logits, newvalue = model(
                    mb_batch.x_dict,
                    mb_batch.edge_index_dict,
                    mb_batch.edge_attr_dict if hasattr(mb_batch, 'edge_attr_dict') else None,
                    batch_dict=batch_dict
                )
                newvalue = newvalue.view(-1)

                ch_dist = Categorical(logits=ch_logits)
                pwr_dist = Categorical(logits=pwr_logits)
                
                mb_acts_flat = b_actions[mb_inds].view(-1, 2) # [mb * n_aps, 2]
                
                newlogprob_per_ap = ch_dist.log_prob(mb_acts_flat[:, 0]) + pwr_dist.log_prob(mb_acts_flat[:, 1])
                entropy_per_ap = ch_dist.entropy() + pwr_dist.entropy()
                
                # Sumar a nivel de grafo en lugar de por AP
                if 'ap' in batch_dict:
                    newlogprob = global_add_pool(newlogprob_per_ap, batch_dict['ap'])
                    entropy_mean = global_add_pool(entropy_per_ap, batch_dict['ap']).mean()
                else:
                    newlogprob = newlogprob_per_ap.view(len(mb_inds), -1).sum(dim=1)
                    entropy_mean = entropy_per_ap.view(len(mb_inds), -1).sum(dim=1).mean()
                
                # ── Ratio de Probabilidades: r_t(\theta) ─────────────────────────
                # $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
                # Transformado a espacio lineal usando la resta de logaritmos.
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Para monitorear el KL Divergence exacto empírico de la política
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Normalización estabilizadora a nivel de Minibatch
                mb_advantages = advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # ── Surrogate Objective Clipped (PPO Policy Loss) ─────────────────
                # $L^{CLIP}(\theta) = \min( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t )$
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ── Value Loss (Error Cuadrático Medio) ───────────────────────────
                # $\mathcal{L}_{value}(\phi) = \frac{1}{2} \mathbb{E}_t [ (V_\phi(s_t) - G_t)^2 ]$
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # ── Entropy Bonus ─────────────────────────────────────────────────
                # $\mathcal{H}(\pi_\theta) = -\mathbb{E}[ \log \pi_\theta(a|s) ]$
                entropy_loss = entropy_mean

                # ── Función Objetivo Global a Minimizar ───────────────────────────
                # $\mathcal{L}_{TOTAL} = - L^{CLIP} + c_{vf} \mathcal{L}_{value} - c_{ent} \mathcal{H}$
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                if torch.isfinite(loss):
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                else:
                    grad_norm = torch.tensor(0.0, device=device)

                total_policy_loss  += pg_loss.item() if math.isfinite(pg_loss.item()) else 0.0
                total_value_loss   += v_loss.item()  if math.isfinite(v_loss.item())  else 0.0
                total_entropy_loss += entropy_loss.item() if math.isfinite(entropy_loss.item()) else 0.0
                updates_count += 1

        scheduler.step()

        # ── Monitor y Guardado ────────────────────────────────────────────────
        avg_rate_ep = sum(ep_rates)
        
        # Cómputo de Retorno Descontado (G_t) para consistencia académica
        def _get_return(rw, gam):
            g = 0.0
            for r in reversed(rw): g = r + gam * g
            return g
        
        ep_return = _get_return(rewards_list, args.gamma)

        if avg_rate_ep > best_rate:
            best_rate = avg_rate_ep
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            
        if (episode + 1) % 50 == 0:
            torch.save(model.state_dict(), save_dir / f"model_ep{episode+1}.pt")

        ep_dur = time.time() - t0_ep
        metrics.append({
            "episode":       episode + 1,
            "return":        ep_return,
            "total_rate":    avg_rate_ep,
            "mean_rate_step":np.mean(ep_rates) if ep_rates else 0.0,
            "loss":          total_policy_loss / updates_count,
            "value_loss":    total_value_loss / updates_count,
            "entropy":       total_entropy_loss / updates_count,
            "grad_norm":     grad_norm.item(),
            "lr":            optimizer.param_groups[0]["lr"],
            "sec":           ep_dur,
        })

        # Actualizar la interfaz gráfica interactiva en tiempo real (cada episodio)
        if plotter is not None:
            try:
                eps = [m["episode"] for m in metrics]
                returns = [m["return"] for m in metrics]
                val_returns = []  # No hay validación periódica registrada en métricas de PPO
                losses = [m["loss"] for m in metrics]
                plotter.update(eps, returns, val_returns, losses)
            except Exception:
                pass

        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"{episode+1:<6} | "
                f"{avg_rate_ep:>10.1f} | "
                f"{ep_return:>10.2f} | "
                f"{total_policy_loss/updates_count:>9.4f} | "
                f"{total_value_loss/updates_count:>9.4f} | "
                f"{total_entropy_loss/updates_count:>9.4f} | "
                f"{ep_dur:>7.1f}"
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

    print(f"\n{'='*65}")
    print(f"  Entrenamiento finalizado en {time.time()-t0:.1f}s  |  Mejor Total Rate: {best_rate:.2f}")
    print(f"{'='*65}\n")

    # ── Guardar Final ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), save_dir / "final_model.pt")
    df_metrics = pd.DataFrame(metrics)
    metrics_path = save_dir / "training_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Modelo final  → {save_dir / 'final_model.pt'}")
    print(f"Métricas      → {metrics_path}")

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


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NetROML – PPO Proximal Policy Optimization")
    p.add_argument("--building_id",   default="990")
    p.add_argument("--episodes",      type=int,   default=200)
    p.add_argument("--timesteps",     type=int,   default=100)
    p.add_argument("--decision_period", type=int, default=1)
    p.add_argument("--arrival_rate",  type=float, default=2.0)
    p.add_argument("--mean_dur",      type=float, default=10.0)
    p.add_argument("--no_5g",         action="store_true", help="Deshabilita el uso de la banda de 5 GHz")
    p.add_argument("--sticky_mode",   type=str,   default="sticky", choices=["full", "sticky", "lite"])
    p.add_argument("--hidden",        type=int,   default=64)
    p.add_argument("--model_type",    type=str,   default="gnn1", choices=["gnn1", "gnn2", "random"])
    p.add_argument("--num_layers",    type=int,   default=4)
    p.add_argument("--tagconv_k",     type=int,   default=5)
    p.add_argument("--reward_scale",  type=float, default=1.0)
    p.add_argument("--seed",          type=lambda x: None if str(x).lower() == 'none' else int(x), default=None)
    p.add_argument("--save_dir",      default="outputs/models")
    
    # Hiperparámetros PPO
    p.add_argument("--lr",            type=float, default=3e-4) # AdamW
    p.add_argument("--gamma",         type=float, default=0.99) # Descuento
    p.add_argument("--gae_lambda",    type=float, default=0.95) # Parámetro lambda para GAE
    p.add_argument("--update_epochs", type=int,   default=4)    # Épocas de optimización sobre el rollout
    p.add_argument("--minibatch_size",type=int,   default=64)   # Tamaño del minibatch de grafos
    p.add_argument("--clip_coef",     type=float, default=0.2)  # Surrogate clipping function (epsilon)
    p.add_argument("--ent_coef",      type=float, default=0.01) # Coeficiente de entropía (bonus)
    p.add_argument("--vf_coef",       type=float, default=0.5)  # Coeficiente del value function (reducido para estabilidad)
    p.add_argument("--max_grad_norm", type=float, default=0.5)  # Global gradient clipping
    p.add_argument("--norm_adv", action=argparse.BooleanOptionalAction, default=True)
    
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
