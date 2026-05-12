"""
step3_mapear_a_blocks.py – Mapea el DataFrame del Paso 2 a objetos Block.

Cada Block: (block_idx, distribution_idx, datos)
    datos = DataFrame [mac_ap, G_2_4, G_5]  (ganancia máxima entre antenas por banda)

Nota de diseño (__v0.5__)
-------------------------
`groupby.max(min_count=1)` garantiza que si todas las ganancias de una banda
son NaN (AP no visible en ninguna antena), la ganancia agregada sea NaN y no
0 ni -inf. El comportamiento de min_count cambia entre versiones de Pandas.
"""

from __future__ import annotations
import pandas as pd
from data.clases import Block


def build_datos(block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla datos de un Block:
        columnas: mac_ap, G_2_4, G_5
        valores : max(gain) entre antenas para cada banda
    """
    pivot = (
        block_df
        .groupby(['mac_ap', 'banda'])['gain']
        .max(min_count=1)   # NaN si no hay ningún valor válido (AP invisible)
        .unstack(level='banda')
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={0: 'G_2_4', 1: 'G_5'})
    for col in ['G_2_4', 'G_5']:
        if col not in pivot.columns:
            pivot[col] = None
    return pivot[['mac_ap', 'G_2_4', 'G_5']].reset_index(drop=True)


def mapear_a_blocks(df: pd.DataFrame) -> list[Block]:
    """Convierte el DataFrame completo en list[Block]."""
    blocks: list[Block] = []
    for (dist_idx, blk_idx), group in df.groupby(['distribution_idx', 'block_idx'], sort=True):
        blocks.append(Block(
            block_idx        = int(blk_idx),
            distribution_idx = int(dist_idx),
            datos            = build_datos(group),
        ))
    return blocks
