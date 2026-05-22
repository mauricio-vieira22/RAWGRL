r"""
gnn_model.py – Arquitectura de Red Neuronal de Grafos Heterogénea (GNN) para Actor-Critic.

Esta red actúa como la Política Actor y el Crítico. Implementa soporte para 
profundidad variable con GATv2 y conexiones residuales para resolver 
problemas de coordinación complejos en redes de alta densidad.
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
        out_channels_ch:  int = 3,
        out_channels_pwr: int = 3,
        num_layers:      int = 3,    # Profundidad ajustable
        heads:           int = 2     # Heads de atención
    ):
        super().__init__()
        self.num_aps = num_aps
        self.num_layers = num_layers
        self.heads = heads

        # 1. Encoders de features crudas
        self.ap_encoder     = Linear(-1, hidden_channels)
        self.client_encoder = Linear(-1, hidden_channels)

        # 2. Capas Convolucionales Heterogéneas
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_channels if i == 0 else hidden_channels * heads
            
            self.convs.append(HeteroConv({
                ('ap',     'connects',    'client'): GATv2Conv(in_channels, hidden_channels, heads=heads, add_self_loops=False, edge_dim=1),
                ('client', 'connected_to', 'ap')  : GATv2Conv(in_channels, hidden_channels, heads=heads, add_self_loops=False),
                ('ap',     'interferes',  'ap')    : GATv2Conv(in_channels, hidden_channels, heads=heads, add_self_loops=False, edge_dim=2),
            }, aggr='sum'))

        # 3. Cabezas de política y valor
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

        self.value_head = nn.Sequential(
            Linear(final_dim, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 1),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None):
        # 1. Encode inicial
        x = {}
        x['ap'] = self.ap_encoder(x_dict['ap'])
        if 'client' in x_dict and x_dict['client'].shape[0] > 0:
            x['client'] = self.client_encoder(x_dict['client'])
        else:
            x['client'] = x_dict['ap'].new_zeros((0, x['ap'].shape[-1]))

        # 2. Propagación de Mensajes Multi-capa
        ea = edge_attr_dict or {}
        
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index_dict, ea)
            x_new = {k: F.elu(v.nan_to_num(0.0)) for k, v in x_new.items()}
            
            if i > 0:
                # Conexión residual compatible
                x = {
                    k: x_new[k] + x[k]
                    for k in x_new
                    if k in x and x_new[k].shape == x[k].shape
                }
                for k in x_new:
                    if k not in x:
                        x[k] = x_new[k]
            else:
                # Conexión residual de primera capa repeat compatible
                x = {
                    k: x_new[k] + x[k].repeat(1, self.heads)
                    for k in x_new
                    if k in x and x_new[k].shape == x[k].repeat(1, self.heads).shape
                }
                for k in x_new:
                    if k not in x:
                        x[k] = x_new[k]

        # 3. Cabezas de política
        ap_emb = x['ap']
        channel_logits = self.channel_head(ap_emb).nan_to_num(nan=0.0)
        power_logits   = self.power_head(ap_emb).nan_to_num(nan=0.0)

        # 4. Cabeza de Valor (Critic) con desacoplamiento (detach) para evitar shared backbone interference
        if batch_dict is not None and 'ap' in batch_dict:
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(ap_emb, batch_dict['ap'])
        else:
            graph_emb = ap_emb.mean(dim=0, keepdim=True)

        state_value = self.value_head(graph_emb.detach()).nan_to_num(nan=0.0)

        return channel_logits, power_logits, state_value
