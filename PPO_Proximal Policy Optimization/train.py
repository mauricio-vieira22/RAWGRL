r"""
train_ppo.py — Optimización de Políticas RAWGRL mediante PPO (Proximal Policy Optimization).

Implementa el algoritmo PPO con Generalized Advantage Estimation (GAE) para la
optimización de la política Actor-Critic sobre el POMDP de asignación de recursos
WiFi. Se apoya en la misma GNN heterogénea que REINFORCE, extendida con una
cabeza de valor V_φ(s_t).

Fundamentación Matemática (PPO con GAE)
----------------------------------------
PPO maximiza un objetivo subrogado recortado que limita el tamaño de las
actualizaciones de política mediante una razón de importancia acotada:

    L^CLIP(θ) = E_t[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

donde r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) es la razón de probabilidades.

La función de pérdida total minimizada es:

    L_total = -L^CLIP(θ) + c_vf · L_value(φ) - c_ent · H[π_θ]

con L_value(φ) = (1/2) E_t[(V_φ(s_t) - G_t)^2] el error cuadrático del crítico.

Las ventajas se estiman mediante GAE (Schulman et al., 2016):

    Â_t = ∑_{l=0}^{∞} (γλ)^l δ_{t+l},    δ_t = R_t + γ V_φ(s_{t+1}) - V_φ(s_t)

donde λ ∈ [0,1] controla el trade-off sesgo-varianza: λ=0 equivale a TD(0)
(bajo varianza, alto sesgo) y λ=1 equivale a retorno Monte Carlo (bajo sesgo,
alta varianza).

Diferencias algorítmicas respecto a REINFORCE
----------------------------------------------
1. La función de valor V_φ aprendida reemplaza el baseline EMA del REINFORCE.
   El crítico produce estimaciones bootstrap del retorno esperado, reduciendo
   la varianza del estimador de gradiente sin introducir sesgo adicional.
2. El recorte PPO previene actualizaciones de política excesivamente grandes,
   estabilizando el aprendizaje en el entorno WiFi de alta varianza.
3. El rollout completo se reutiliza en múltiples épocas de optimización,
   mejorando la eficiencia de muestra respecto a REINFORCE (on-policy puro).
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
import math
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


# ── Modelo de referencia aleatoria ────────────────────────────────────────────

class RandomModel(torch.nn.Module):
    """Política de referencia uniforme. Compatible con la interfaz Actor-Critic."""

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
        n_graphs = (
            int(batch_dict['ap'].max().item() + 1)
            if (batch_dict is not None and 'ap' in batch_dict)
            else 1
        )
        value = torch.zeros((n_graphs, 1), device=device) + self.dummy
        return ch_logits, pwr_logits, value


# ── Evaluación determinista ────────────────────────────────────────────────────

def evaluate_policy(
    env:          NetworkGraphEnv,
    model:        nn.Module,
    device:       torch.device,
    gamma:        float = 0.99,
    reward_scale: float = 1.0,
) -> tuple[float, float, float]:
    """
    Evalúa la política de forma determinista (argmax) con semilla fija.

    Desactiva el cálculo de gradientes y usa torch.no_grad() para eficiencia.
    La función de valor del crítico no se usa en evaluación; solo el actor.

    Returns
    -------
    tuple[float, float, float]
        (G_val, val_rate, val_rate_per_client): retorno descontado G_0,
        throughput total acumulado y throughput medio por cliente activo.
    """
    model.eval()
    obs, _ = env.reset()
    done   = False
    rewards, ep_rates, n_clients = [], [], []

    with torch.no_grad():
        while not done:
            data = obs.to(device)
            ch_logits, pwr_logits, _ = model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict
            )
            action = torch.stack(
                [ch_logits.argmax(dim=-1), pwr_logits.argmax(dim=-1)], dim=1
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward / reward_scale)
            ep_rates.append(info.get("total_rate", 0.0))
            n_clients.append(max(int(info.get("n_active_clients", 1)), 1))

    model.train()

    # Retorno descontado G_0 por retro-recursión
    G_val = 0.0
    discount = 1.0
    for r in rewards:
        G_val    += r * discount
        discount *= gamma

    val_rate            = sum(ep_rates)
    total_clients       = sum(n_clients)
    val_rate_per_client = val_rate / total_clients if total_clients > 0 else 0.0
    return G_val, val_rate, val_rate_per_client


# ── Bucle principal PPO ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> tuple[nn.Module, pd.DataFrame]:
    """
    Orquesta el ciclo de entrenamiento PPO con GAE.

    Flujo por episodio
    ------------------
    1. Rollout: colectar (s_t, a_t, r_t, V_φ(s_t), log π(a_t|s_t)) con π_{θ_old}.
    2. GAE: calcular Â_t = ∑_l (γλ)^l δ_{t+l} y G_t = Â_t + V_φ(s_t).
    3. Optimización: iterar `update_epochs` épocas sobre minibatches del rollout,
       actualizando θ y φ conjuntamente.

    Parameters
    ----------
    args : argparse.Namespace
        Hiperparámetros y metadatos del experimento.

    Returns
    -------
    tuple[nn.Module, pd.DataFrame]
        Modelo entrenado e historial de métricas.
    """
    device = torch.device("cpu")
    log.info("Dispositivo de cómputo activo: %s", device)

    # 1. Carga de distribuciones de canal del edificio
    log.info("Cargando distribuciones para edificio: %s", args.building_id)
    distributions = load_distributions(building_id=args.building_id, verbose=False)

    available_channels = [1, 6, 11]
    tx_powers_dbm      = [20.0, 14.0, 8.0]
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
        use_5g=not args.no_5g,
        device=device,
    )

    # 3. Inicialización del modelo Actor-Critic
    if args.model_type == 'gnn2':
        # GNN2 no tiene use_critic todavía; se usa GNN1 como Actor-Critic
        log.warning(
            "GNN2 no soporta use_critic; se usa GNN1 como Actor-Critic."
        )
        model = GNN(
            hidden_channels=args.hidden,
            num_aps=n_aps,
            out_channels_ch=len(available_channels),
            out_channels_pwr=len(tx_powers_dbm),
            num_layers=args.num_layers,
            use_critic=True,
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
            use_critic=True,
        ).to(device)

    # eps=1e-5 recomendado para PPO (más conservador que el default 1e-8)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.episodes)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_return = -float("inf")
    metrics_log: list[dict] = []
    t_start = time.time()

    log.info(
        "Iniciando PPO: E=%d episodios | T=%d pasos | clip=%.2f | epochs=%d | λ_GAE=%.2f",
        args.episodes, args.timesteps, args.clip_coef, args.update_epochs, args.gae_lambda,
    )
    log.info("=" * 103)
    log.info(
        f"{'Ep':<6} | {'Val Rate':>10} | {'G_val':>10} | {'Train Rate':>10} | "
        f"{'L_clip':>8} | {'L_vf':>8} | {'Entropy':>9} | {'GradNorm':>8} | {'Time':>6}"
    )
    log.info("-" * 103)

    # 4. Bucle principal PPO
    for episode in range(args.episodes):
        ep_seed = args.seed + episode if args.seed is not None else None
        obs, _  = env.reset(seed=ep_seed)

        # Buffers del rollout
        obs_list:      list                = []
        actions_list:  list[torch.Tensor]  = []
        logprobs_list: list[torch.Tensor]  = []
        rewards_list:  list[float]         = []
        values_list:   list[torch.Tensor]  = []
        dones_list:    list[bool]          = []
        ep_rates:      list[float]         = []
        n_clients_list: list[int]          = []

        done  = False
        t_ep  = time.time()

        # Fase 1: Rollout con π_{θ_old} (sin gradiente)
        while not done:
            data = obs.to(device)
            obs_list.append(data.clone())

            with torch.no_grad():
                ch_logits, pwr_logits, state_val = model(
                    data.x_dict, data.edge_index_dict, data.edge_attr_dict
                )
                ch_dist  = Categorical(logits=ch_logits)
                pwr_dist = Categorical(logits=pwr_logits)
                ch_acts  = ch_dist.sample()
                pwr_acts = pwr_dist.sample()
                # log π(a_t|s_t): suma sobre todos los APs del grafo
                logprob = (
                    ch_dist.log_prob(ch_acts).sum()
                    + pwr_dist.log_prob(pwr_acts).sum()
                )

            action = torch.stack([ch_acts, pwr_acts], dim=1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            actions_list.append(action)
            logprobs_list.append(logprob)
            rewards_list.append(float(reward) / args.reward_scale)
            values_list.append(state_val.view(-1))
            dones_list.append(done)
            ep_rates.append(info.get("total_rate", 0.0))
            n_clients_list.append(max(int(info.get("n_active_clients", 1)), 1))

        # Valor del último estado para bootstrap GAE
        with torch.no_grad():
            next_data = obs.to(device)
            _, _, next_value = model(
                next_data.x_dict, next_data.edge_index_dict, next_data.edge_attr_dict
            )
            next_value = next_value.view(-1)

        # Fase 2: GAE — Generalized Advantage Estimation (Schulman et al., 2016)
        # Â_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + …
        # δ_t = R_t + γ V_φ(s_{t+1})(1 - done_t) - V_φ(s_t)
        num_steps = len(rewards_list)
        b_rewards  = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        b_values   = torch.stack(values_list).squeeze(-1)
        b_dones    = torch.tensor(dones_list,  dtype=torch.float32, device=device)

        advantages = torch.zeros(num_steps, device=device)
        gae        = torch.zeros(1, device=device)

        for t in reversed(range(num_steps)):
            # Bootstrap: V_φ(s_{T+1}) = next_value si no hay episodio terminado
            next_val      = next_value if t == num_steps - 1 else b_values[t + 1]
            non_terminal  = 1.0 - (b_dones[t] if t == num_steps - 1 else b_dones[t])
            delta         = b_rewards[t] + args.gamma * next_val * non_terminal - b_values[t]
            gae           = delta + args.gamma * args.gae_lambda * non_terminal * gae
            advantages[t] = gae

        # G_t = Â_t + V_φ(s_t): objetivo de regresión para el crítico
        b_returns  = advantages + b_values
        b_actions  = torch.stack(actions_list)
        b_logprobs = torch.stack(logprobs_list)

        # Fase 3: Optimización PPO en update_epochs épocas sobre el rollout
        b_inds = np.arange(num_steps)

        total_pg_loss  = 0.0
        total_vf_loss  = 0.0
        total_ent      = 0.0
        last_grad_norm = torch.tensor(0.0)
        n_updates      = 0

        for _ in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, num_steps, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                if len(mb_inds) == 0:
                    continue

                # Construir batch de grafos PyG para el minibatch
                mb_data_list = [obs_list[i] for i in mb_inds]
                mb_batch     = Batch.from_data_list(mb_data_list).to(device)

                batch_dict: dict[str, torch.Tensor] = {}
                if hasattr(mb_batch['ap'], 'batch'):
                    batch_dict['ap'] = mb_batch['ap'].batch

                ch_logits, pwr_logits, newvalue = model(
                    mb_batch.x_dict,
                    mb_batch.edge_index_dict,
                    mb_batch.edge_attr_dict if hasattr(mb_batch, 'edge_attr_dict') else None,
                    batch_dict=batch_dict,
                )
                newvalue = newvalue.view(-1)

                ch_dist  = Categorical(logits=ch_logits)
                pwr_dist = Categorical(logits=pwr_logits)

                # Acciones del minibatch: (mb * n_aps, 2)
                mb_acts = b_actions[mb_inds].view(-1, 2)

                newlogprob_per_ap = (
                    ch_dist.log_prob(mb_acts[:, 0])
                    + pwr_dist.log_prob(mb_acts[:, 1])
                )
                entropy_per_ap = ch_dist.entropy() + pwr_dist.entropy()

                # Agregar log-prob y entropía a nivel de grafo (suma sobre APs)
                if 'ap' in batch_dict:
                    newlogprob = global_add_pool(
                        newlogprob_per_ap.unsqueeze(1), batch_dict['ap']
                    ).squeeze(1)
                    entropy_mean = global_add_pool(
                        entropy_per_ap.unsqueeze(1), batch_dict['ap']
                    ).squeeze(1).mean()
                else:
                    newlogprob   = newlogprob_per_ap.sum().unsqueeze(0)
                    entropy_mean = entropy_per_ap.mean()

                # Razón de importancia: r_t(θ) = π_θ / π_{θ_old}
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio    = logratio.exp()

                # Ventajas del minibatch, normalizadas por estabilidad
                mb_adv = advantages[mb_inds]
                if args.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # L^CLIP: surrogate recortado
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # L_value: MSE normalizado por varianza del retorno para estabilidad numérica
                returns_var = b_returns.var().clamp(min=1e-6)
                vf_loss = 0.5 * (
                    ((newvalue - b_returns[mb_inds]) ** 2) / returns_var
                ).mean()

                # L_total = -L^CLIP + c_vf · L_value - c_ent · H[π]
                loss = pg_loss + args.vf_coef * vf_loss - args.entropy_coef * entropy_mean

                optimizer.zero_grad()
                if torch.isfinite(loss):
                    loss.backward()
                    last_grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    optimizer.step()
                else:
                    log.warning(
                        "Ep %d: loss no finita en minibatch, update omitido.",
                        episode + 1,
                    )

                total_pg_loss += pg_loss.item() if math.isfinite(pg_loss.item()) else 0.0
                total_vf_loss += vf_loss.item() if math.isfinite(vf_loss.item()) else 0.0
                total_ent     += entropy_mean.item() if math.isfinite(entropy_mean.item()) else 0.0
                n_updates     += 1

        scheduler.step()

        # 5. Métricas del episodio
        train_rate = sum(ep_rates)
        G_train    = sum(rewards_list)

        G_val, val_rate, val_rate_per_client = float("nan"), float("nan"), float("nan")
        if (episode + 1) % args.eval_every == 0 or episode == 0:
            G_val, val_rate, val_rate_per_client = evaluate_policy(
                env_eval, model, device, args.gamma, args.reward_scale
            )
            if G_val > best_return:
                best_return = G_val
                torch.save(model.state_dict(), save_dir / "best_model.pt")

        avg_pg   = total_pg_loss / max(n_updates, 1)
        avg_vf   = total_vf_loss / max(n_updates, 1)
        avg_ent  = total_ent     / max(n_updates, 1)
        ep_dur   = time.time() - t_ep

        metrics_log.append({
            "episode":             episode + 1,
            "G_train":             G_train,
            "G_val":               G_val,
            "total_rate":          train_rate,
            "val_rate":            val_rate,
            "val_rate_per_client": val_rate_per_client,
            "loss_clip":           avg_pg,
            "loss_vf":             avg_vf,
            "entropy":             avg_ent,
            "grad_norm":           last_grad_norm.item(),
            "lr":                  optimizer.param_groups[0]["lr"],
            "sec":                 ep_dur,
        })

        if (episode + 1) % 10 == 0 or episode == 0:
            log.info(
                f"{episode+1:<6} | "
                f"{val_rate:>10.1f} | "
                f"{G_val:>10.2f} | "
                f"{train_rate:>10.1f} | "
                f"{avg_pg:>8.4f} | "
                f"{avg_vf:>8.4f} | "
                f"{avg_ent:>9.4f} | "
                f"{last_grad_norm.item():>8.3f} | "
                f"{ep_dur:>6.1f}"
            )

    # 6. Persistencia final
    elapsed = time.time() - t_start
    log.info("Entrenamiento PPO finalizado en %.1f min.", elapsed / 60)

    torch.save(model.state_dict(), save_dir / "final_model.pt")
    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(save_dir / "training_metrics_ppo.csv", index=False)

    try:
        import subprocess
        plot_script = Path(__file__).parent / "plots_code" / "plot_training.py"
        if plot_script.exists():
            log.info("Generando gráficas de entrenamiento PPO...")
            subprocess.run(
                [
                    sys.executable, str(plot_script),
                    "--csv", str(save_dir / "training_metrics_ppo.csv"),
                    "--out", str(Path(__file__).parent / "plots"),
                ],
                check=False,
            )
    except Exception as exc:
        log.warning("No se pudieron generar las gráficas: %s", exc)

    return model, metrics_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Configura la interfaz de línea de comandos para el entrenamiento PPO."""
    p = argparse.ArgumentParser(description="RAWGRL: PPO WiFi Resource Allocation")
    # Entorno
    p.add_argument("--building_id",     default="990")
    p.add_argument("--save_dir",        default="outputs/models")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--episodes",        type=int,   default=1000)
    p.add_argument("--timesteps",       type=int,   default=1000)
    p.add_argument("--arrival_rate",    type=float, default=3.0)
    p.add_argument("--mean_dur",        type=float, default=15.0)
    p.add_argument("--decision_period", type=int,   default=5)
    p.add_argument("--no_5g",           action="store_true")
    p.add_argument("--sticky_mode",     type=str,   default="sticky",
                   choices=["full", "sticky", "lite"])
    # Arquitectura
    p.add_argument("--hidden",          type=int,   default=64)
    p.add_argument("--model_type",      type=str,   default="gnn1",
                   choices=["gnn1", "gnn2", "random"])
    p.add_argument("--num_layers",      type=int,   default=4)
    p.add_argument("--tagconv_k",       type=int,   default=5)
    # Hiperparámetros comunes
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--reward_scale",    type=float, default=1.0)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--eval_every",      type=int,   default=10)
    # Hiperparámetros PPO
    p.add_argument("--gae_lambda",      type=float, default=0.95,
                   help="λ para GAE: 0=TD(0), 1=Monte Carlo")
    p.add_argument("--clip_coef",       type=float, default=0.2,
                   help="ε del recorte PPO")
    p.add_argument("--update_epochs",   type=int,   default=4,
                   help="Épocas de optimización por rollout")
    p.add_argument("--minibatch_size",  type=int,   default=64,
                   help="Número de steps por minibatch")
    p.add_argument("--entropy_coef",    type=float, default=0.01,
                   help="c_ent: coeficiente del bono de entropía")
    p.add_argument("--vf_coef",         type=float, default=0.5,
                   help="c_vf: coeficiente de la pérdida del crítico")
    p.add_argument("--max_grad_norm",   type=float, default=0.5,
                   help="Umbral de clipping de norma del gradiente")
    p.add_argument("--norm_adv",        action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Normalizar ventajas por minibatch")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())