"""
preprocessing.py – Pipeline End-to-End de Preprocesamiento (Steps 1 → 4).

Este módulo actúa como orquestador del procesamiento de datos duros. 
Encadena las secuencias:
    Step 1: Filtrado, limpieza y consolidación estadística base (.tgz -> CSV).
    Step 2: Imputación masiva de datos (NaN padding) y traducción de RSSI a Ganancia (Gain).
    Step 3: Mapeo de subconjuntos espaciales a entidades lógicas `Block`.
    Step 4: Agrupación temporal en listas de entidades maestras `Distribution`.

Uso sugerido
------------
    from data.preprocessing import run_pipeline
    distributions = run_pipeline(
        building_id="990",
        step2_csv_path="data/dataset_990_step2.csv",
        distributions_path="data/distributions_990.joblib"
    )

Nota de Aceleración O(1): Si el archivo `joblib` existe directamente en el disco, 
saltea por completo el consumo de RAM/CPU del pipeline logrando parseo instantáneo.
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import joblib
from data.clases import Distribution
from data.step1_crear_dataset import crear_dataset
from data.step2_gain_e_imputer import agregar_gain, imputer, ALL_BANDAS, ALL_ANTENAS
from data.step3_mapear_a_blocks import mapear_a_blocks
from data.step4_block_to_distribution import blocks_to_distributions

ALL_MONTHS  = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
DEFAULT_P_TX = 14.0


def run_pipeline(
    building_id:    str,
    data_dir:       str | None  = None,
    mapping_file:   str | None  = None,
    months:         list[str] | None = None,
    P_TX:           float       = DEFAULT_P_TX,
    rssi_fill:      float       = -np.inf,
    step2_csv_path: str | None  = None,
    distributions_path: str | None = None,
    verbose:        bool        = True,
) -> list[Distribution]:
    """
    Ejecuta y orquesta el pipeline completo de preprocesamiento físico de los logs WiFi.

    Parameters
    ----------
    building_id : str
        ID lógico del edificio a analizar (ej. "990"). Usado para filtrar locaciones en bruto.
    data_dir : str | None
        Directorio donde se encuentran almacenados los dumps `.tgz` crudos de Splunk.
        [Requerido] si no se cuenta con un CSV acelerador (`step2_csv_path`).
    mapping_file : str | None
        Archivo con el mapeo topológico (MAC_AP -> building_id). [Requ.] si corre Step 1.
    months : list[str] | None
        Filtro escalar de meses para leer tarballs masivos. (None asume lectura global).
    P_TX : float
        Potencia de Transmisión (Transmission Power) estandarizada del Access Point 
        de Ceibalita, expresada logarítmicamente en [dBm]. Default 14.0.
    rssi_fill : float
        Valor penalizador matemático para inyectar a celdas donde un AP fue espectralmente 
        invisible al cliente (out of range/dead zone). Se emplea -inf de manera default.
    step2_csv_path : str | None
        Ruta física de aceleración (Cache CSV). Si el sistema detecta que este archivo 
        ya existe, omitirá la costosa extracción y agrupado matricial en Spark/Pandas.
    distributions_path : str | None
        Ruta física de memoria estructurada (Cache Joblib). La existencia de este archivo 
        bypassea por completo el Parseo; directamente levanta los objetos de Python.
        Si la ruta indicada no existe, entonces el final del pipeline lo guardará en disco acá.
    verbose : bool
        Variable de depuración standard output (STDOUT).

    Returns
    -------
    list[Distribution]
        Salida purificada conteniendo una lista íntegra con las historietas cliente a cliente, 
        cargadas y validadas con sus objetos Base (`Block` estáticos).
    """
    months = months or ALL_MONTHS

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"\n{'='*60}")
    log(f"  Preprocessing Pipeline  –  building_id = {building_id}")
    log(f"{'='*60}\n")

    # ── Checkpoint Joblib ─────────────────────────────────────────────────────
    if distributions_path and os.path.exists(distributions_path):
        log(f"▶ Cargando distributions desde {distributions_path} (OMITIENDO pipeline 1-4)")
        distributions = joblib.load(distributions_path)
        log(f"  ✓ Distributions: {len(distributions)}\n")
        return distributions

    # ── Checkpoint CSV ────────────────────────────────────────────────────────
    if step2_csv_path and os.path.exists(step2_csv_path):
        log(f"▶ Cargando desde {step2_csv_path} (Steps 1 y 2 omitidos)")
        df_s2 = pd.read_csv(step2_csv_path)
        log(f"  ✓ Filas: {len(df_s2):,}  |  Clientes: {df_s2['mac_cliente'].nunique()}\n")
    else:
        # Step 1
        log("▶ Step 1 – Extracción y consolidación desde .tgz")
        if data_dir is None or mapping_file is None:
            raise ValueError("data_dir y mapping_file son requeridos cuando no existe step2_csv_path")
        df_s1 = crear_dataset(
            building_id=building_id,
            data_dir=data_dir,
            mapping_file=mapping_file,
            output_file=None,
            months=months,
        )
        log(f"  ✓ Filas: {len(df_s1):,}  |  Clientes: {df_s1['mac_cliente'].nunique()}\n")

        # Step 2
        log("▶ Step 2 – Gain e Imputer")
        ap_list = df_s1['mac_ap'].unique().tolist()
        df_s2   = agregar_gain(df_s1, P_TX)
        df_s2   = imputer(df_s2, ap_list, ALL_BANDAS, ALL_ANTENAS, rssi_fill, P_TX)
        log(f"  ✓ Filas: {len(df_s2):,}\n")

        if step2_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(step2_csv_path)), exist_ok=True)
            df_s2.to_csv(step2_csv_path, index=False)
            log(f"  💾 CSV guardado en: {step2_csv_path}\n")

    # Step 3
    log("▶ Step 3 – Objetos Block")
    blocks = mapear_a_blocks(df_s2)
    log(f"  ✓ Blocks: {len(blocks):,}\n")

    # Step 4
    log("▶ Step 4 – Objetos Distribution")
    idx_to_mac    = df_s2.groupby('distribution_idx')['mac_cliente'].first().to_dict()
    distributions = blocks_to_distributions(blocks, idx_to_mac)
    log(f"  ✓ Distributions: {len(distributions)}\n")

    log(f"{'='*60}")
    log(f"  Pipeline completado  –  {len(distributions)} clientes  |  {len(blocks)} bloques")
    log(f"{'='*60}\n")

    if distributions_path:
        os.makedirs(os.path.dirname(os.path.abspath(distributions_path)), exist_ok=True)
        joblib.dump(distributions, distributions_path)
        log(f"  💾 Distributions guardadas en: {distributions_path}\n")

    return distributions
