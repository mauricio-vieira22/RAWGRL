r"""
gnn_model.py — Arquitectura de Red Neuronal de Grafos Heterogénea (GNN).

Parametriza tanto la política estocástica π_θ(a_t | s_t) del agente REINFORCE
como la política y función de valor V_φ(s_t) del agente Actor-Critic (PPO).

Arquitectura
------------
La red emplea HeteroConv con GATv2Conv (Brody et al., 2022) en L capas de
propagación de mensajes sobre el grafo bipartito heterogéneo:

    G_t = (V_AP ∪ V_C, E_ac ∪ E_ca ∪ E_aa)

donde E_ac transporta atributos de arista RSSI normalizados, E_ca codifica la
asignación activa (sin atributos), y E_aa codifica co-visibilidad de interferencia.

    Capa 0:    Encoders lineales → h_AP ∈ R^d,  h_C ∈ R^d
    Capas 1..L: HeteroConv (GATv2Conv, multi-head) con conexiones residuales
    Salida Actor:  channel_head(h_AP) → logits ∈ R^{N × |C|}
                   power_head(h_AP)   → logits ∈ R^{N × |P|}
    Salida Critic: value_head(pool(h_AP)) → V ∈ R^{n_graphs × 1}  [solo si use_critic=True]

Interfaz retrocompatible
------------------------
Con use_critic=False (defecto), forward devuelve (channel_logits, power_logits),
reproduciendo exactamente el comportamiento de la versión REINFORCE.
Con use_critic=True, devuelve (channel_logits, power_logits, value), donde
value es V_φ(s_t) escalado por grafo en el batch, requerido por PPO.

Conexión residual
-----------------
Las skip-connections se aplican desde la capa 1 en adelante (in_dim = out_dim =
hidden × heads). En la capa 0 el cambio de dimensión impide la conexión directa.
Se previene el KeyError en grafos sin clientes activos mediante filtrado explícito.

Asimetría E_ac / E_ca
----------------------
E_ac es densa (todos los APs visibles por cliente) y transporta RSSI como atributo
de arista, permitiendo que el mensaje del AP informe al cliente sobre la calidad
del enlace potencial. E_ca es escasa (solo el AP asignado) y sin atributos,
reflejando que la asociación es una variable discreta del estado.
Esta asimetría está justificada en la Sección §GNN Architecture de la tesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, global_mean_pool


class GNN(torch.nn.Module):
    """
    Política GNN heterogénea para asignación de recursos WiFi.

    Compatible con REINFORCE (use_critic=False) y PPO/Actor-Critic (use_critic=True).

    Parameters
    ----------
    hidden_channels : int
        Dimensión interna de los embeddings de nodo (d).
    num_aps : int
        Número de APs del edificio (N). Retenido para compatibilidad de interfaz.
    out_channels_ch : int
        Cardinalidad del espacio de canales |C|.
    out_channels_pwr : int
        Cardinalidad del espacio de potencias |P|.
    num_layers : int
        Número de capas de propagación de mensajes L.
    heads : int
        Número de cabezas de atención multi-head (h).
        El embedding de salida por capa tiene dimensión d · h.
    use_critic : bool
        Si True, instancia la cabeza de valor V_φ para Actor-Critic (PPO).
        Si False (defecto), la red es solo Actor, compatible con REINFORCE.
    """

    def __init__(
        self,
        hidden_channels:  int,
        num_aps:          int,
        out_channels_ch:  int  = 3,
        out_channels_pwr: int  = 3,
        num_layers:       int  = 3,
        heads:            int  = 2,
        use_critic:       bool = False,
    ):
        super().__init__()
        self.num_aps    = num_aps
        self.num_layers = num_layers
        self.heads      = heads
        self.use_critic = use_critic

        # Encoders de features crudas hacia el espacio latente d
        self.ap_encoder     = Linear(-1, hidden_channels)
        self.client_encoder = Linear(-1, hidden_channels)

        # Capas convolucionales heterogéneas
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # Capa 0: in_channels = hidden (salida del encoder)
            # Capas 1..L: in_channels = hidden * heads (salida de la capa anterior)
            in_channels = hidden_channels if i == 0 else hidden_channels * heads

            self.convs.append(HeteroConv(
                {
                    ('ap', 'connects', 'client'): GATv2Conv(
                        in_channels, hidden_channels,
                        heads=heads, add_self_loops=False, edge_dim=1,
                    ),
                    ('client', 'connected_to', 'ap'): GATv2Conv(
                        in_channels, hidden_channels,
                        heads=heads, add_self_loops=False,
                    ),
                    ('ap', 'interferes', 'ap'): GATv2Conv(
                        in_channels, hidden_channels,
                        heads=heads, add_self_loops=False, edge_dim=1,
                    ),
                },
                aggr='sum',
            ))

        # Cabezas de política Actor: leen embeddings AP de dimensión hidden * heads
        final_dim = hidden_channels * heads

        self.channel_head = nn.Sequential(
            Linear(final_dim, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_ch),
        )

        self.power_head = nn.Sequential(
            Linear(final_dim, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_pwr),
        )

        # Cabeza de valor Critic (solo instanciada si use_critic=True).
        # V_φ(s_t) ∈ R se obtiene por global_mean_pool sobre embeddings AP,
        # seguido de una MLP de dos capas. Se usa Tanh en la capa oculta para
        # evitar saturación unilateral en la estimación del retorno esperado.
        if use_critic:
            self.value_head = nn.Sequential(
                Linear(final_dim, hidden_channels),
                nn.Tanh(),
                Linear(hidden_channels, 1),
            )

    def forward(
        self,
        x_dict:          dict,
        edge_index_dict: dict,
        edge_attr_dict:  dict | None = None,
        batch_dict:      dict | None = None,
    ):
        """
        Propagación hacia adelante.

        Parameters
        ----------
        x_dict : dict
            Features de nodo por tipo: {'ap': (N, F_ap), 'client': (U, F_c)}.
        edge_index_dict : dict
            Índices de arista por relación heterogénea.
        edge_attr_dict : dict | None
            Atributos de arista (obligatorio para E_ac y E_aa).
        batch_dict : dict | None
            Índices de batch por tipo de nodo. Requerido para el critic cuando
            se procesan múltiples grafos en un minibatch. Si None, se asume
            un único grafo (todos los nodos AP pertenecen al mismo grafo).

        Returns
        -------
        Si use_critic=False:
            tuple[Tensor, Tensor]
                (channel_logits, power_logits), shape (N, |C|) y (N, |P|).
        Si use_critic=True:
            tuple[Tensor, Tensor, Tensor]
                (channel_logits, power_logits, value), donde value tiene shape
                (n_graphs, 1) con V_φ(s_t) por grafo en el batch.
        """
        # 1. Codificación inicial al espacio latente d
        x: dict[str, torch.Tensor] = {}
        x['ap'] = self.ap_encoder(x_dict['ap'])

        if 'client' in x_dict and x_dict['client'].shape[0] > 0:
            x['client'] = self.client_encoder(x_dict['client'])
        else:
            # Sin clientes activos: tensor vacío con dimensión correcta para
            # mantener la compatibilidad de forma con las capas convolucionales.
            x['client'] = x_dict['ap'].new_zeros((0, x['ap'].shape[-1]))

        # 2. Propagación de mensajes multi-capa con conexiones residuales
        ea = edge_attr_dict or {}

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index_dict, ea)

            # ELU + limpieza de NaNs numéricos (pueden surgir de grafos vacíos)
            x_new = {k: F.elu(v.nan_to_num(0.0)) for k, v in x_new.items()}

            if i > 0:
                # Conexión residual: solo cuando in_dim == out_dim == hidden * heads.
                # Se filtra por claves presentes en ambos dicts y con forma compatible
                # para evitar KeyError en timesteps sin clientes activos.
                x = {
                    k: x_new[k] + x[k]
                    for k in x_new
                    if k in x and x_new[k].shape == x[k].shape
                }
                # Preservar claves que HeteroConv omitió (ej. 'client' vacío)
                for k in x_new:
                    if k not in x:
                        x[k] = x_new[k]
            else:
                x = x_new

        # 3. Cabezas de política Actor sobre embeddings AP
        ap_emb         = x['ap']
        channel_logits = self.channel_head(ap_emb).nan_to_num(nan=0.0)
        power_logits   = self.power_head(ap_emb).nan_to_num(nan=0.0)

        if not self.use_critic:
            return channel_logits, power_logits

        # 4. Cabeza de valor Critic: pooling global de embeddings AP → V_φ(s_t).
        # global_mean_pool agrega todos los nodos AP del grafo (o batch de grafos)
        # en un único vector de representación del estado global de la red.
        ap_batch = (
            batch_dict['ap']
            if (batch_dict is not None and 'ap' in batch_dict)
            else ap_emb.new_zeros(ap_emb.shape[0], dtype=torch.long)
        )
        graph_emb = global_mean_pool(ap_emb, ap_batch)   # (n_graphs, final_dim)
        value     = self.value_head(graph_emb)            # (n_graphs, 1)

        return channel_logits, power_logits, value