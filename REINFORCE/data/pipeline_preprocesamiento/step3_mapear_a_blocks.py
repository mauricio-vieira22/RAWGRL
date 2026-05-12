"""
step3_mapear_a_blocks.py – Módulo de Mapeo de Entidades Espaciales (Block Objects).

Este componente realiza la agregación espacial de los datos. Transforma los registros
de grano fino (antenas) en estructuras consolidadas por Access Point y Banda, creando
instancias de la clase 'Block'.

Lógica de Agregación:
    - Se agrupan los registros por 'mac_ap' y 'banda'.
    - Se selecciona la ganancia máxima (max gain) entre las antenas disponibles para
      capturar el mejor escenario de propagación.
    - Se garantiza la preservación de valores nulos (NaN) para APs invisibles.

Estructura del Block:
    - block_idx: Secuencia temporal.
    - distribution_idx: Vínculo con la trayectoria del cliente.
    - datos: DataFrame pivotado con columnas [mac_ap, G_2_4, G_5].
"""

from __future__ import annotations
import pandas as pd
from ..clases import Block


def build_datos(block_df: pd.DataFrame, ap_list: list[str]) -> pd.DataFrame:
    """
    Construye la matriz de estado (datos) para un bloque temporal específico.

    Realiza un pivoteo de la tabla de ganancias y garantiza que el orden
    de los APs coincida exactamente con la lista maestra 'ap_list'.

    Parameters
    ----------
    block_df : pd.DataFrame
        Subconjunto del dataset correspondiente a un único (cliente, tiempo).
    ap_list : list[str]
        Lista maestra ordenada de MACs de APs para el edificio.

    Returns
    -------
    pd.DataFrame
        Matriz pivotada y reindexada con columnas: ['mac_ap', 'G_2_4', 'G_5'].
    """
    # Agregación por banda: seleccionamos la mejor antena para representar al AP.
    pivot = (
        block_df
        .groupby(['mac_ap', 'banda'])['gain']
        .max(min_count=1)
        .unstack(level='banda')
    )
    
    # Asegurar presencia de columnas de banda
    for band_idx in [0, 1]:
        if band_idx not in pivot.columns:
            pivot[band_idx] = pd.NA
            
    # REINDEXACIÓN CRÍTICA: Garantiza que el orden de filas sea consistente con ap_list
    pivot = pivot.reindex(ap_list)
    pivot.index.name = 'mac_ap'
    pivot = pivot.reset_index()
    
    pivot = pivot.rename(columns={0: 'G_2_4', 1: 'G_5'})
    pivot.columns.name = None
            
    return pivot[['mac_ap', 'G_2_4', 'G_5']].reset_index(drop=True)


def mapear_a_blocks(df: pd.DataFrame, ap_list: list[str]) -> list[Block]:
    """
    Transforma el DataFrame imputado en una colección de objetos Block.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame consolidado e imputado proveniente del Step 2.
    ap_list : list[str]
        Lista maestra de MACs de APs para el edificio.

    Returns
    -------
    list[Block]
        Lista de objetos Block instanciados y validados.
    """
    blocks: list[Block] = []
    
    # Agrupación por (Trayectoria, Tiempo)
    for (dist_idx, blk_idx), group in df.groupby(['distribution_idx', 'block_idx'], sort=True):
        blocks.append(Block(
            block_idx        = int(blk_idx),
            distribution_idx = int(dist_idx),
            datos            = build_datos(group, ap_list),
        ))
        
    return blocks
