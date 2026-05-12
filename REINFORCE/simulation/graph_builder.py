r"""
graph_builder.py – Ensamble Topológico del Grafo Bipartito Heterogéneo para RL.

Transforma el estado matemático físico $(H_t, X_t, A_t, \epsilon_t, \delta_t)$ al 
objeto `HeteroData` de PyTorch Geometric consumido por la GNN.

Representación Gráfica (NetROML § GNN Architecture)
---------------------------------------------------
El sistema de red inalámbrica se modela formalmente como un grafo bipartito dirigido 
$\mathcal{G}_t = (\mathcal{V}_{AP} \cup \mathcal{V}_C, \mathcal{E}_{ac} \cup \mathcal{E}_{ca})$, donde:
- $\mathcal{V}_{AP}$: Conjunto de Nodos de Puntos de Acceso.
- $\mathcal{V}_C$: Conjunto de Nodos de Clientes.
- $\mathcal{E}_{ac}$: Aristas de propagación electromagnética (AP $\to$ Cliente).
- $\mathcal{E}_{ca}$: Aristas de conexión lógica (Cliente $\to$ AP).

Señales de Nodo (Espacio Observable Ampliado $\mathcal{O}_t$)
-------------------------------------------------------------
El modelo teórico base estipula un espacio de features $X_t \in \mathbb{R}^{(U_t+N)\times 2}$. 
Nuestra implementación extiende este espacio a $\mathbb{R}^{(U_t+N)\times 4}$ para inyectar 
conocimiento sobre el estado oculto del POMDP, estabilizando el aprendizaje del modelo.

Nodos AP ($v \in \mathcal{V}_{AP}$): $x_v^{(0)} \in \mathbb{R}^2$
  [0] $c_{n,t}$ : Canal seleccionado (valor real, ej. 1, 6, 11).
  [1] $P_{n,t}$ : Potencia transmitida (dBm).

  Nota: La carga $|\{u: a_{u,t}=n\}|$ es capturada implícitamente por el
  message passing GATv2 sobre las aristas $(\text{Cliente} \to \text{AP})$,
  por lo que no se incluye como feature explícita del nodo.

Nodos Cliente ($u \in \mathcal{V}_C$): $x_u^{(0)} \in \mathbb{R}^4$
  [0] $B$ : Frecuencia de banda operativa.
  [1] $\epsilon_{t,u}$ : Slots restantes de vida útil (Inyección POMDP).
  [2] $\bar{r}_{u,t}$ : Tasa promedio percibida históricamente (Running Mean).
  [3] $RSSI_{u, a_u}$ : Potencia recibida de su AP servidor.

Atributos de Arista ($e_{vu} \in \mathcal{E}_{ac}$):
  $e_{vu} \in \mathbb{R}$ — $RSSI_{u,v}$ normalizado en $[0, 1]$ usando la cota operativa $[-100, -30]$ dBm.
"""

from __future__ import annotations
import torch
from torch_geometric.data import HeteroData

# Umbral de visibilidad RF: θ_conn del modelo formal
UMBRAL_VIS = -85.0   # [dBm]


def construir_grafo_timestep(
    t: int,
    asignaciones: torch.Tensor,
    grilla_RSSI: torch.Tensor,
    decisiones_APs: torch.Tensor,
    avg_rates: torch.Tensor,
    eventos,
    available_channels: torch.Tensor,
    tx_powers_dbm: torch.Tensor,
    max_timesteps: int,
    delta_t: int = 0,
) -> HeteroData:
    """
    Construye el HeteroData de PyG para el timestep t.

    Parameters
    ----------
    t               : int       — timestep actual τ
    asignaciones    : (n_cli, T, 3) — A_t: [ap_idx, band_idx, RSSI_dBm], NaN=inactivo
    grilla_RSSI     : (n_cli, T, N, 2) — RSSI_t [dBm]
    decisiones_APs  : (N, 2) long — X_t: [canal_idx, potencia_idx]
    avg_rates       : (n_cli,) — media corrida de r_{u,τ}
    eventos         : list[ClientEvent]
    available_channels : (n_ch,) long
    tx_powers_dbm   : (n_pw,) float32
    max_timesteps   : int — horizonte T (para normalizar ε_t y δ_t)
    delta_t         : int — δ_t: slots hasta próximo arribo

    Returns
    -------
    HeteroData con:
      data['ap'].x              : (N, 2)
      data['client'].x          : (U_t, 4)
      data[ap→client].edge_index: (2, E)  + .edge_attr (E, 1)
      data[client→ap].edge_index: (2, U_t)
    """
    _, _, n_APs, _ = grilla_RSSI.shape
    n_ch = len(available_channels)
    n_pw = len(tx_powers_dbm)
    dev  = grilla_RSSI.device

    data = HeteroData()

    # ── Nodos AP: X_t = {c_{n,t}, P_{n,t}} ────────────────────────────────
    canal_idx    = decisiones_APs[:, 0]
    potencia_idx = decisiones_APs[:, 1]
    canal_raw    = available_channels[canal_idx].float()
    potencia_raw = tx_powers_dbm[potencia_idx].float()

    ap_asignado_t = asignaciones[:, t, 0]   # (n_cli,) — fila de A_t
    mask_activos  = ~torch.isnan(ap_asignado_t)

    data['ap'].x = torch.stack(
        [canal_raw, potencia_raw], dim=1
    )  # (N, 2)

    # ── Nodos Cliente: señales ε_{t,u}, r_{u,τ} y RSSI propio ─────────────────
    activos_idx = torch.where(mask_activos)[0]
    n_activos   = len(activos_idx)

    if n_activos == 0:
        data['client'].x = torch.empty((0, 4), dtype=torch.float32, device=dev)
        data['ap', 'connects', 'client'].edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
        data['ap', 'connects', 'client'].edge_attr  = torch.empty((0, 1), dtype=torch.float32, device=dev)
        data['client', 'connected_to', 'ap'].edge_index = torch.empty((2, 0), dtype=torch.long, device=dev)
        return data

    banda_t    = asignaciones[activos_idx, t, 1]
    banda_norm = banda_t / 1.0    # 0 = 2.4 GHz, 1 = 5 GHz

    # ε_{t,u} — slots restantes del cliente u en t
    epsilon_t = torch.tensor(
        [eventos[i.item()].departure_time - t for i in activos_idx],
        dtype=torch.float32, device=dev,
    )
    epsilon_t_norm = epsilon_t / max(max_timesteps, 1)   # ε_{t,u} ∈ [0,1]

    # avg_rate normalizado adaptativamente (max-norm del timestep)
    r      = avg_rates[activos_idx]
    r_max  = r.max().clamp(min=1e-6)
    rate_norm = r / r_max

    # RSSI del AP asociado — calidad actual del enlace (observable por cliente)
    ap_activos = ap_asignado_t[activos_idx].long()     # índice AP de cada cliente
    band_safe  = banda_t.nan_to_num(0).long()
    rssi_propio = grilla_RSSI[activos_idx, t, ap_activos, band_safe]  # (U_t,)
    # Normalizar al rango [0,1]: rango útil [-100, -30] dBm → shift +100 / 70
    rssi_propio_norm = ((rssi_propio + 100.0) / 70.0).clamp(0.0, 1.0)

    data['client'].x = torch.stack(
        [banda_norm, epsilon_t_norm, rate_norm, rssi_propio_norm], dim=1
    )  # (U_t, 4)

    # ── Aristas AP → Cliente: visibilidad RF (G_H ponderado) ─────────────────
    rssi_banda = grilla_RSSI[activos_idx, t, :, :].gather(
        2, band_safe.view(-1, 1, 1).expand(-1, n_APs, 1)
    ).squeeze(2)   # (U_t, N)

    vis_cli_local, vis_ap = torch.where(rssi_banda > UMBRAL_VIS)

    if len(vis_ap) > 0:
        edge_index_down = torch.stack([vis_ap, vis_cli_local], dim=0)
        edge_attr_down  = ((rssi_banda[vis_cli_local, vis_ap] + 100.0) / 70.0).unsqueeze(1)
    else:
        edge_index_down = torch.empty((2, 0), dtype=torch.long, device=dev)
        edge_attr_down  = torch.empty((0, 1), dtype=torch.float32, device=dev)

    data['ap', 'connects', 'client'].edge_index = edge_index_down
    data['ap', 'connects', 'client'].edge_attr  = edge_attr_down

    # ── Aristas Cliente → AP: asignación activa A_t (G_A binario) ────────────
    local_idx = torch.arange(n_activos, device=dev)
    data['client', 'connected_to', 'ap'].edge_index = torch.stack([local_idx, ap_activos], dim=0)

    return data
