"""
step4_block_to_distribution.py – Agrupa list[Block] en list[Distribution].
"""

from __future__ import annotations
from collections import defaultdict
import numpy as np
from data.clases import Block, Distribution


def blocks_to_distributions(
    blocks:     list[Block],
    idx_to_mac: dict[int, str],
) -> list[Distribution]:
    """
    Agrupa blocks por distribution_idx y crea objetos Distribution.

    Parameters
    ----------
    blocks     : list[Block]
    idx_to_mac : dict[int, str]  distribution_idx → mac_cliente

    Returns
    -------
    list[Distribution] ordenada por distribution_idx
    """
    groups: dict[int, list[Block]] = defaultdict(list)
    for block in blocks:
        groups[block.distribution_idx].append(block)

    distributions: list[Distribution] = []
    for dist_idx in sorted(groups.keys()):
        client_blocks = sorted(groups[dist_idx], key=lambda b: b.block_idx)
        distributions.append(Distribution(
            distribution_idx = dist_idx,
            mac_client       = idx_to_mac[dist_idx],
            blocks           = np.array(client_blocks, dtype=object),
        ))
    return distributions
