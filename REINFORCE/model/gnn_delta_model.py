r"""
gnn_delta_model.py – Arquitectura GNN con Inyección Tardía de delta_t (POMDP).

Este módulo implementa el concepto de "Late-Injection" para variables 
de estado globales. La GNN recibe nodos AP con solo 2 features (canal, potencia),
la carga es inferida por message passing, y $\delta_t$ se inyecta directamente
en la cabeza de decisión (FCNN) tras la convolucón, sin contaminar el grafo.

Flujo de Información:
1. GNN(Grafo: AP[c, P], Clientes[ε, r, RSSI, banda]) -> Embeddings h_i
2. h_i_combined = Concatenate(h_i, δ_t_normalized)
3. Action_Logits = MLP(h_i_combined)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


class GNNDelta(torch.nn.Module):
    """
    Arquitectura Actor con inyección de estado global delta_t en la cabeza.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_aps: int,
        out_channels_ch: int = 3,
        out_channels_pwr: int = 5,
    ):
        super().__init__()
        self.num_aps = num_aps

        # 1. Codificación Base
        self.ap_encoder     = Linear(-1, hidden_channels)
        self.client_encoder = Linear(-1, hidden_channels)

        # 2. Capas de Convolución (Extracción de Patrones Espaciales)
        self.conv1 = HeteroConv({
            ('ap',     'connects',    'client'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False, edge_dim=1),
            ('client', 'connected_to', 'ap')   : GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('ap',     'connects',    'client'): GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False, edge_dim=1),
            ('client', 'connected_to', 'ap')   : GATv2Conv((-1, -1), hidden_channels, heads=2, add_self_loops=False),
        }, aggr='sum')

        # 3. Cabeza de Decisión con Inyección Tardía
        # La dimensión de entrada es (hidden_channels * 2) de la GNN + 1 de delta_t
        h_dim_gnn = hidden_channels * 2
        h_dim_combined = h_dim_gnn + 1

        self.channel_head = nn.Sequential(
            nn.Linear(h_dim_combined, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels_ch),
        )

        self.power_head = nn.Sequential(
            nn.Linear(h_dim_combined, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels_pwr),
        )

    def forward(self, x_dict, edge_index_dict, delta_t, edge_attr_dict=None):
        """
        Inferencia de política con inyección de delta_t.

        Parameters
        ----------
        x_dict : dict
        edge_index_dict : dict
        delta_t : float or torch.Tensor
            Valor de delta_t normalizado (0 a 1).
        edge_attr_dict : dict, optional
        """
        # 1. Proyección Inicial
        x = {'ap': self.ap_encoder(x_dict['ap'])}
        if 'client' in x_dict:
            x['client'] = self.client_encoder(x_dict['client'])

        # 2. Mensajería GNN
        ea = edge_attr_dict or {}
        x = self.conv1(x, edge_index_dict, ea)
        x = {k: F.elu(v) for k, v in x.items()}

        x = self.conv2(x, edge_index_dict, ea)
        x = {k: F.elu(v) for k, v in x.items()}

        # 3. Concatenación de delta_t (Late-Injection)
        ap_embeddings = x['ap']  # (num_aps, h_dim_gnn)
        
        # Expandir delta_t para todos los APs
        if not isinstance(delta_t, torch.Tensor):
            delta_t = torch.tensor([delta_t], dtype=torch.float32, device=ap_embeddings.device)
        
        if delta_t.dim() == 0:
            delta_t = delta_t.view(1)
        
        # Si delta_t es un escalar global para el batch, lo repetimos
        # ap_embeddings shape: (N, H)
        delta_t_expanded = delta_t.expand(ap_embeddings.size(0), 1)
        
        # Unión de información espacial (GNN) y temporal (delta_t)
        combined_features = torch.cat([ap_embeddings, delta_t_expanded], dim=1)

        # 4. Decisión Final
        return self.channel_head(combined_features), self.power_head(combined_features)
