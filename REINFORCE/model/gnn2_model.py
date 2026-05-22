r"""
gnn2_model.py – Arquitectura GNN2 (TAGConv) Homogeneizada.

Esta arquitectura implementa la propuesta basada en TAGConv (Topology Adaptive 
Graph Convolutional Networks). Dado que TAGConv requiere grafos homogéneos 
para operar sobre polinomios de la matriz de adyacencia (A^K), esta red
unifica dinámicamente el diccionario de nodos bipartitos ('ap' y 'client')
en un único super-grafo, aplica las convoluciones, y luego extrae 
exclusivamente los nodos AP para alimentar las cabezas del Actor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, Linear

class GNN2(torch.nn.Module):
    def __init__(
        self, 
        hidden_channels: int, 
        num_aps: int,
        out_channels_ch: int = 3,
        out_channels_pwr: int = 3,
        num_layers: int = 3,
        K: int = 1
    ):
        super(GNN2, self).__init__()
        self.num_aps = num_aps
        self.hidden_dim = hidden_channels
        self.num_layers = num_layers

        # 1. Encoders para llevar AP y Client a la misma dimensión (hidden_dim)
        self.ap_encoder = Linear(-1, hidden_channels)
        self.client_encoder = Linear(-1, hidden_channels)

        # 2. Convoluciones TAGConv de la propuesta
        self.convs = torch.nn.ModuleList()
        
        # Primera capa
        self.convs.append(TAGConv(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            K=K, 
            bias=True, 
            normalize=False
        ))
        
        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(
                in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                K=K, 
                bias=True, 
                normalize=False
            ))
            
        # Última capa convolucional
        self.convs.append(TAGConv(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            K=K, 
            bias=False, 
            normalize=False
        ))

        # 3. Cabezas Actor (Policy) sobre los APs
        self.channel_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_ch),
        )

        self.power_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_pwr),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.convs.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0.0, std=0.1)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.1)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, batch_dict=None):
        device = x_dict['ap'].device
        
        # --- 1. ENCODING ---
        x_ap = self.ap_encoder(x_dict['ap'])
        n_ap = x_ap.size(0)
        
        if 'client' in x_dict and x_dict['client'].shape[0] > 0:
            x_client = self.client_encoder(x_dict['client'])
        else:
            # Si no hay clientes activos, crear tensor vacío con misma feature dim
            x_client = torch.empty((0, self.hidden_dim), device=device)
            
        # --- 2. HOMOGENEIZACIÓN ---
        # Unir todos los nodos: [ APs ... , Clients ... ]
        x_homo = torch.cat([x_ap, x_client], dim=0)
        
        # Unir y desplazar edge_index
        edges = []
        attrs = []
        
        # Aristas AP -> Cliente
        if ('ap', 'connects', 'client') in edge_index_dict:
            ei_ap_c = edge_index_dict[('ap', 'connects', 'client')]
            num_edges = ei_ap_c.size(1)
            if num_edges > 0:
                src = ei_ap_c[0]             # APs mantienen su índice (0 a N_ap - 1)
                dst = ei_ap_c[1] + n_ap      # Clientes se desplazan N_ap
                edges.append(torch.stack([src, dst], dim=0))
                
                if edge_attr_dict and ('ap', 'connects', 'client') in edge_attr_dict:
                    attr = edge_attr_dict[('ap', 'connects', 'client')]
                    if attr.dim() > 1: attr = attr.squeeze(-1)
                    attrs.append(attr)
                else:
                    attrs.append(torch.ones(num_edges, device=device))
                    
        # Aristas Cliente -> AP
        if ('client', 'connected_to', 'ap') in edge_index_dict:
            ei_c_ap = edge_index_dict[('client', 'connected_to', 'ap')]
            num_edges = ei_c_ap.size(1)
            if num_edges > 0:
                src = ei_c_ap[0] + n_ap      # Clientes desplazados
                dst = ei_c_ap[1]             # APs sin desplazar
                edges.append(torch.stack([src, dst], dim=0))
                
                if edge_attr_dict and ('client', 'connected_to', 'ap') in edge_attr_dict:
                    attr = edge_attr_dict[('client', 'connected_to', 'ap')]
                    if attr.dim() > 1: attr = attr.squeeze(-1)
                    attrs.append(attr)
                else:
                    attrs.append(torch.ones(num_edges, device=device))

        # Aristas AP ↔ AP (Interferencia por co-observación)
        if ('ap', 'interferes', 'ap') in edge_index_dict:
            ei_ap_ap = edge_index_dict[('ap', 'interferes', 'ap')]
            num_edges = ei_ap_ap.size(1)
            if num_edges > 0:
                # Los APs ocupan los primeros n_ap índices, no hace falta desplazar
                edges.append(ei_ap_ap)
                
                if edge_attr_dict and ('ap', 'interferes', 'ap') in edge_attr_dict:
                    attr = edge_attr_dict[('ap', 'interferes', 'ap')]
                    if attr.dim() > 1:
                        attr = attr[:, 0]
                    attrs.append(attr)
                else:
                    attrs.append(torch.ones(num_edges, device=device))

        if edges:
            edge_index_homo = torch.cat(edges, dim=1)
            edge_attr_homo = torch.cat(attrs, dim=0)
        else:
            edge_index_homo = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr_homo = None

        # --- 3. MESSAGE PASSING (TAGConv) ---
        for i in range(self.num_layers):
            x_homo = self.convs[i](x=x_homo, edge_index=edge_index_homo, edge_weight=edge_attr_homo)
            x_homo = x_homo.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            if i < (self.num_layers - 1):
                x_homo = F.leaky_relu(x_homo, inplace=False)

        # --- 4. EXTRACCIÓN Y CABEZAS ACTOR ---
        ap_emb = x_homo[:n_ap]
        
        channel_logits = self.channel_head(ap_emb).nan_to_num(nan=0.0)
        power_logits = self.power_head(ap_emb).nan_to_num(nan=0.0)

        return channel_logits, power_logits
