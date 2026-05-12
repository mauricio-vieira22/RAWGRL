"""
step4_block_to_distribution.py – Módulo de Agregación Temporal (Distribution Objects).

Este componente finaliza el pipeline de datos mediante la consolidación de bloques
temporales individuales en trayectorias coherentes por cada cliente WiFi. Crea
instancias de la clase 'Distribution', que es la unidad mínima de información
utilizada por el cargador de datos y el entorno de entrenamiento.

Lógica de Agrupación:
    - Se agrupan los objetos 'Block' según su 'distribution_idx'.
    - Se garantiza el orden cronológico estricto de los bloques mediante el 'block_idx'.
    - Se encapsulan las trayectorias en un arreglo NumPy de alta eficiencia.
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
from ..clases import Block, Distribution


def blocks_to_distributions(
    blocks: list[Block],
    idx_to_mac: dict[int, str],
) -> list[Distribution]:
    """
    Agrupa una colección de bloques espaciales en trayectorias temporales (Distribution).

    Parameters
    ----------
    blocks : list[Block]
        Colección de bloques generados en el Step 3.
    idx_to_mac : dict[int, str]
        Diccionario de mapeo entre el índice de la trayectoria y la MAC real del cliente.

    Returns
    -------
    list[Distribution]
        Lista de trayectorias finales, ordenadas por su identificador de distribución.
    """
    # Agrupación por identificador de trayectoria
    groups: dict[int, list[Block]] = defaultdict(list)
    for block in blocks:
        groups[block.distribution_idx].append(block)

    distributions: list[Distribution] = []
    
    # Procesamiento ordenado para garantizar determinismo en la carga
    for dist_idx in sorted(groups.keys()):
        # Ordenamiento cronológico interno de la trayectoria
        client_blocks = sorted(groups[dist_idx], key=lambda b: b.block_idx)
        
        distributions.append(Distribution(
            distribution_idx = dist_idx,
            mac_client       = idx_to_mac[dist_idx],
            blocks           = np.array(client_blocks, dtype=object),
        ))
        
    return distributions
