"""
step2_gain_e_imputer.py – Agrega columna gain e imputa combinaciones faltantes.

gain = RSSI - P_TX   (P_TX en dBm, default = 14 dBm de los ceibalitas)

El imputer garantiza que cada bloque (mac_cliente, datetime) tenga filas para
cada combinación (mac_ap × banda × antena) del building.
Las faltantes se rellenan con rssi_fill (por defecto NaN).

Nota de diseño (__v0.5__)
-------------------------
En __v0__ el valor centinela era -np.inf. Esto causaba una inconsistencia
semántica con `convertir_grilla_a_tensor` (que usaba float('nan')).
A partir de __v0.5__ el centinela unificado es np.nan en todo el pipeline.
El .joblib generado con __v0__ se convierte on-the-fly en wifi_physics.py.
"""

from __future__ import annotations
import itertools
import pandas as pd
import numpy as np

ALL_BANDAS  = [0, 1]   # 0 = 2.4 GHz, 1 = 5 GHz
ALL_ANTENAS = [0, 1]


def agregar_gain(df: pd.DataFrame, P_TX: float) -> pd.DataFrame:
    """Agrega columna gain = RSSI - P_TX."""
    df = df.copy()
    df['gain'] = df['rssi'] - P_TX
    return df


def imputer(
    df:        pd.DataFrame,
    ap_list:   list[str],
    bandas:    list[int] = ALL_BANDAS,
    antenas:   list[int] = ALL_ANTENAS,
    rssi_fill: float     = np.nan,
    P_TX:      float     = 14.0,
) -> pd.DataFrame:
    """
    Para cada bloque (mac_cliente, datetime), garantiza todas las combinaciones
    (mac_ap × banda × antena). Las faltantes se rellenan con rssi_fill.
    """
    expected_combos = pd.DataFrame(
        list(itertools.product(ap_list, bandas, antenas)),
        columns=['mac_ap', 'banda', 'antena'],
    )
    block_keys = ['mac_cliente', 'datetime', 'distribution_idx', 'block_idx']
    blocks     = df[block_keys].drop_duplicates()
    full_tmpl  = blocks.merge(expected_combos, how='cross')

    df_merged = full_tmpl.merge(
        df[block_keys + ['mac_ap', 'banda', 'antena', 'rssi', 'gain']],
        on=block_keys + ['mac_ap', 'banda', 'antena'],
        how='left',
    )
    df_merged['rssi'] = df_merged['rssi'].fillna(rssi_fill)
    # gain = NaN cuando rssi = NaN (AP no visible). Pandas propaga NaN en la resta.
    df_merged['gain'] = df_merged['rssi'] - P_TX

    cols = ['datetime', 'distribution_idx', 'mac_cliente', 'mac_ap',
            'rssi', 'gain', 'antena', 'banda', 'block_idx']
    return (
        df_merged[cols]
        .sort_values(['mac_cliente', 'datetime', 'mac_ap', 'banda', 'antena'])
        .reset_index(drop=True)
    )
