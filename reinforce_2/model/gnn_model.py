r"""
gnn_model.py — Arquitectura de Red Neuronal de Grafos Heterogénea (GNN).

Parametriza la política estocástica π_θ(a_t | s_t) del agente REINFORCE sobre
el grafo bipartito heterogéneo G_t = (V_AP ∪ V_C, E_ac ∪ E_ca ∪ E_aa).

Arquitectura
------------
La red emplea HeteroConv con GATv2Conv (Brody et al., 2022) en L capas de
propagación de mensajes, seguida de dos cabezas de política independientes que
leen exclusivamente los embeddings de nodos AP.

    Capa 0: Encoders lineales → h_AP ∈ R^d, h_C ∈ R^d
    Capas 1..L: HeteroConv con tres relaciones:
        (AP → Cliente):  E_ac, con edge_attr RSSI normalizado
        (Cliente → AP):  E_ca, sin edge_attr (asignación binaria)
        (AP → AP):       E_aa, con edge_attr co-visibilidad normalizada
    Salida: channel_head(h_AP) → logits ∈ R^{N × |C|}
            power_head(h_AP)   → logits ∈ R^{N × |P|}

Conexión residual
-----------------
Las conexiones residuales se aplican desde la capa 1 en adelante (cuando
in_dim = out_dim = hidden * heads). En la capa 0, el cambio de dimensión
(hidden → hidden*heads) impide la conexión directa. La implementación previene
explícitamente el KeyError en escenarios sin clientes activos.

Asimetría E_ac / E_ca
----------------------
E_ac es densa (todos los APs visibles por cada cliente) y transporta atributos
de arista RSSI para que el mensaje del AP informe al cliente sobre la calidad
del enlace potencial. E_ca es escasa (solo el AP asignado) y no tiene atributos
de arista, reflejando que la asociación es una variable discreta del estado.
Esta asimetría está justificada teóricamente: el agente necesita saber qué APs
puede ver cada cliente, pero solo el AP actual puede agregar información de
tráfico para la cabeza de política.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


class GNN(torch.nn.Module):
    """
    Política GNN heterogénea para asignación de recursos WiFi.

    Parameters
    ----------
    hidden_channels : int
        Dimensión interna de los embeddings de nodo (d).
    num_aps : int
        Número de APs del edificio (N). Retenido para compatibilidad de interfaz.
    out_channels_ch : int
        Cardinalidad del espacio de canales |C| = 3.
    out_channels_pwr : int
        Cardinalidad del espacio de potencias |P| = 3.
    num_layers : int
        Número de capas de propagación de mensajes L. Por defecto 3.
    heads : int
        Número de cabezas de atención multi-head (h). Por defecto 2.
        El embedding de salida por capa tiene dimensión d · h.
    """

    def __init__(
        self,
        hidden_channels:  int,
        num_aps:          int,
        out_channels_ch:  int = 3,
        out_channels_pwr: int = 1,
        num_layers:       int = 3,
        heads:            int = 2,
    ):
        super().__init__()
        self.num_aps    = num_aps
        self.num_layers = num_layers
        self.heads      = heads

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
                        heads=heads, add_self_loops=False, edge_dim=2,
                    ),
                },
                aggr='mean',
            ))

        # Cabezas de política: leen solo embeddings AP de dimensión hidden * heads
        final_dim = hidden_channels * heads

        self.policy_head = nn.Sequential(
            Linear(final_dim, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels_ch * out_channels_pwr),
        )

    def forward(
        self,
        x_dict:         dict,
        edge_index_dict: dict,
        edge_attr_dict:  dict | None = None,
        batch_dict:      dict | None = None,
    ) -> torch.Tensor:
        """
        Propagación hacia adelante.

        Parameters
        ----------
        x_dict : dict
            Features de nodo por tipo: {'ap': (N, F_ap), 'client': (U, F_c)}.
        edge_index_dict : dict
            Índices de arista por relación.
        edge_attr_dict : dict | None
            Atributos de arista por relación (obligatorio para E_ac y E_aa).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (channel_logits, power_logits), ambos de shape (N, |·|).
        """
        # 1. Codificación inicial al espacio latente d
        x: dict[str, torch.Tensor] = {}
        x['ap'] = self.ap_encoder(x_dict['ap'])

        if 'client' in x_dict and x_dict['client'].shape[0] > 0:
            x['client'] = self.client_encoder(x_dict['client'])
        else:
            # Sin clientes activos: tensor vacío con la dimensión correcta
            x['client'] = x_dict['ap'].new_zeros((0, x['ap'].shape[-1]))

        # 2. Propagación de mensajes multi-capa con conexiones residuales
        ea = edge_attr_dict or {}

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index_dict, ea)

            # ELU + limpieza de NaNs numéricos (pueden surgir de grafos vacíos)
            x_new = {k: F.elu(v.nan_to_num(0.0)) for k, v in x_new.items()}

            if i > 0:
                # Conexión residual: solo cuando in_dim == out_dim == hidden*heads.
                # Se itera sobre las claves presentes en ambos dicts para evitar
                # KeyError en timesteps sin clientes activos (x puede tener
                # 'client' con 0 filas, pero x_new podría omitirlo si HeteroConv
                # no produce salida para ese tipo de nodo).
                x = {
                    k: x_new[k] + x[k]
                    for k in x_new
                    if k in x and x_new[k].shape == x[k].shape
                }
                # Preservar claves que HeteroConv omitió (p.ej. 'client' vacío)
                for k in x_new:
                    if k not in x:
                        x[k] = x_new[k]
            else:
                # Capa 0: in_dim = hidden, out_dim = hidden * heads.
                # Proyección residual mediante repetición a lo largo de la dimensión latente.
                x = {
                    k: x_new[k] + x[k].repeat(1, self.heads)
                    for k in x_new
                    if k in x and x_new[k].shape == x[k].repeat(1, self.heads).shape
                }
                for k in x_new:
                    if k not in x:
                        x[k] = x_new[k]

        # 3. Cabeza de política sobre embeddings AP
        ap_emb = x['ap']
        logits = self.policy_head(ap_emb).nan_to_num(nan=0.0)

        return logits