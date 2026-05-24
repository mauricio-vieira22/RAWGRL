"""
step2_gain_e_imputer.py – Módulo de Imputación Matricial y Normalización de Ganancia.

Este módulo realiza el post-procesamiento físico de los registros RSSI. Su función
principal es la transformación de la métrica de potencia (RSSI) hacia Ganancia (Gain)
y la regularización del dataset mediante la técnica de "Cross-Imputation".

Lógica de Normalización:
    Gain [dB] = RSSI [dBm] - P_TX [dBm]
    Donde P_TX es la potencia de transmisión del Access Point (Ceibalita default: 14 dBm).

Lógica de Imputación:
    Para garantizar que las Redes Neuronales de Grafos (GNN) reciban tensores de forma
    consistente, este módulo expande cada bloque temporal para incluir todas las 
    combinaciones posibles de (AP x Banda x Antena) definidas para el edificio.
    Las celdas donde no hubo señal se rellenan con un valor centinela (np.nan).
"""

from __future__ import annotations
import itertools
import pandas as pd
import numpy as np

# Constantes de espectro WiFi
ALL_BANDAS  = [0, 1]   # 0: 2.4 GHz, 1: 5 GHz
ALL_ANTENAS = [0, 1]   # Diversidad de antena


def agregar_gain(df: pd.DataFrame, P_TX: float) -> pd.DataFrame:
    """
    Calcula la ganancia física a partir del RSSI reportado.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna 'rssi'.
    P_TX : float
        Potencia de transmisión de referencia en dBm.

    Returns
    -------
    pd.DataFrame
        DataFrame con la nueva columna 'gain'.
    """
    df = df.copy()
    df['gain'] = df['rssi'] - P_TX
    return df


def imputer(
    df: pd.DataFrame,
    ap_list: list[str],
    bandas: list[int] = ALL_BANDAS,
    antenas: list[int] = ALL_ANTENAS,
    rssi_fill: float = np.nan,
    P_TX: float = 14.0,
) -> pd.DataFrame:
    """
    Realiza la imputación matricial para garantizar homogeneidad dimensional.

    Crea un producto cartesiano entre los bloques de tiempo existentes y la lista
    maestra de APs/Bandas/Antenas. Esto asegura que la grilla resultante tenga
    siempre las mismas dimensiones (N_aps * N_bandas).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset consolidado del Step 1.
    ap_list : list[str]
        Lista maestra de MACs de APs para el edificio.
    bandas : list[int], opcional
        Lista de bandas de frecuencia.
    antenas : list[int], opcional
        Lista de índices de antena.
    rssi_fill : float, opcional
        Valor de relleno para APs invisibles. Default: np.nan.
    P_TX : float, opcional
        Potencia de referencia para el cálculo de ganancia en filas imputadas.

    Returns
    -------
    pd.DataFrame
        DataFrame rectificado y ordenado por jerarquía cliente-tiempo-espacio.
    """
    # Generación de la plantilla exhaustiva de combinaciones
    expected_combos = pd.DataFrame(
        list(itertools.product(ap_list, bandas, antenas)),
        columns=['mac_ap', 'banda', 'antena'],
    )
    
    # Identificación de llaves de bloque únicas
    block_keys = ['mac_cliente', 'datetime', 'distribution_idx', 'block_idx']
    blocks     = df[block_keys].drop_duplicates()
    
    # Expansión mediante Producto Cartesiano (Cross Join)
    full_tmpl = blocks.merge(expected_combos, how='cross')

    # Fusión con datos reales y propagación de nulos
    df_merged = full_tmpl.merge(
        df[block_keys + ['mac_ap', 'banda', 'antena', 'rssi', 'gain']],
        on=block_keys + ['mac_ap', 'banda', 'antena'],
        how='left',
    )
    
    df_merged['rssi'] = df_merged['rssi'].fillna(rssi_fill)
    # Cálculo de ganancia para las filas imputadas (vectorizado)
    df_merged['gain'] = df_merged['rssi'] - P_TX

    # Selección de columnas finales y ordenamiento canónico
    cols = ['datetime', 'distribution_idx', 'mac_cliente', 'mac_ap',
            'rssi', 'gain', 'antena', 'banda', 'block_idx']
            
    return (
        df_merged[cols]
        .sort_values(['mac_cliente', 'datetime', 'mac_ap', 'banda', 'antena'])
        .reset_index(drop=True)
    )
