r"""
gnn_model.py – Arquitectura de Red Neuronal de Grafos Heterogénea (GNN).

Esta red parametrizada actúa como la Política $\pi_\theta(a_t | s_t)$ y la Función de Valor $V_\phi(s_t)$ 
del algoritmo A2C. Emplea capas `HeteroConv` basadas en convoluciones atencionales (GATv2Conv) 
para propagar la información electromagnética a través de la topología estricta de la red Wi-Fi.

Matemática de Propagación de Mensajes (Message Passing)
-------------------------------------------------------
La actualización del embedding de cada nodo cliente $u$ en la capa $l+1$ es:
$$ h_u^{(l+1)} = \sigma \left( \sum_{v \in \mathcal{N}(u)} \alpha_{u,v} \mathbf{W} h_v^{(l)} \right) $$
Donde $\alpha_{u,v}$ es el coeficiente de atención dictado por el $RSSI_{u,v}$ normalizado (edge_attr).

Entradas (por nodo) $X_t$
-------------------------
AP     : 2 Features [canal_raw, potencia_raw (dBm)]
Cliente: 4 Features [banda_norm, epsilon_t_norm, avg_rate_norm, rssi_propio_norm]

Nota Teórica:
La carga $|\{u: a_{u,t}=n\}|$ no se incluye como feature explícita del AP
pues el mecanismo de atención GATv2 sobre las aristas $(\text{Cliente} \to \text{AP})$
ya agrega la información de todos los clientes conectados durante el message passing.
$\epsilon_t$ se expone como feature del nodo Cliente para el aprendizaje del Critic V(s).
$\delta_t$ se inyecta en la cabeza de decisión en la variante `gnn_delta_model.py`.

Salidas ($\mathcal{A}$ y $\mathbb{R}$)
--------------------------------------
channel_logits : Probabilidades de la Política Actor $\pi_\theta$ (n_APs, n_canales).
power_logits   : Probabilidades de la Política Actor $\pi_\theta$ (n_APs, n_potencias).
value          : Escalar de la Función Critic $V_\phi(s_t) \in \mathbb{R}$ (mean-pool sobre APs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


class GNN(torch.nn.Module):

    def __init__(
        self,
        hidden_channels: int,
        num_aps:         int,
        out_channels_ch:  int = 3,   # número de canales disponibles
        out_channels_pwr: int = 3,   # número de potencias disponibles
    ):
        super().__init__()
        self.num_aps = num_aps

        # Encoders de features crudas
        self.ap_encoder     = Linear(-1, hidden_channels)
        self.client_encoder = Linear(-1, hidden_channels)

        # Capa 1: AP→Cliente (con edge_attr) + Cliente→AP
        self.conv1 = HeteroConv({
            ('ap',     'connects',    'client'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False, edge_dim=1),
            ('client', 'connected_to', 'ap')  : GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
        }, aggr='sum')

        # Capa 2: razonamiento más profundo
        self.conv2 = HeteroConv({
            ('ap',     'connects',    'client'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False, edge_dim=1),
            ('client', 'connected_to', 'ap')  : GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
        }, aggr='sum')

        # Cabezas de política (sólo sobre nodos AP)
        h2 = hidden_channels * 2  # × 2 por los 2 heads de GATv2

        self.channel_head = nn.Sequential(
            Linear(h2, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_ch),
        )

        self.power_head = nn.Sequential(
            Linear(h2, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_pwr),
        )

        self.value_head = nn.Sequential(
            Linear(h2, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 1),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None):
        # 1. Encode
        x = {}
        x['ap'] = self.ap_encoder(x_dict['ap'])
        if 'client' in x_dict and x_dict['client'].shape[0] > 0:
            x['client'] = self.client_encoder(x_dict['client'])
        else:
            # Grafo vacío (sin clientes activos): AP opera sin información de carga.
            x['client'] = x_dict['ap'].new_zeros((0, x['ap'].shape[-1]))

        # 2. Convoluciones (edge_attr_dict puede ser None o dict)
        # nan_to_num garantiza que GATv2Conv con grafos vacíos no propague NaN a los APs.
        ea = edge_attr_dict or {}
        x = self.conv1(x, edge_index_dict, ea)
        x = {k: F.elu(v.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)) for k, v in x.items()}

        x = self.conv2(x, edge_index_dict, ea)
        x = {k: F.elu(v.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)) for k, v in x.items()}

        # 3. Cabezas de política — nan_to_num final como guardia de seguridad
        ap_emb = x['ap']
        channel_logits = self.channel_head(ap_emb).nan_to_num(nan=0.0)
        power_logits   = self.power_head(ap_emb).nan_to_num(nan=0.0)

        # 4. Cabeza de Valor (Critic, para A2C)
        # Pooling global de APs para obtener 1 valor por grafo
        if batch_dict is not None and 'ap' in batch_dict:
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(ap_emb, batch_dict['ap'])
        else:
            graph_emb = ap_emb.mean(dim=0, keepdim=True)

        state_value = self.value_head(graph_emb).nan_to_num(nan=0.0)

        return channel_logits, power_logits, state_value
