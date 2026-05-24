"""
preprocessing.py – Orquestador de la Arquitectura de Preprocesamiento de NetROML.

Este módulo implementa el pipeline end-to-end para la transformación de logs crudos SNMP
(en formato .tgz) hacia estructuras de datos optimizadas (Distribution y Block) para el 
entrenamiento de modelos de Reinforcement Learning y Graph Neural Networks.

Arquitectura del Pipeline:
--------------------------
1. Extracción y Consolidación (Step 1): Parseo de logs, filtrado por edificio y deduplicación.
2. Imputación y Ganancia (Step 2): Cálculo de ganancia física y relleno de datos faltantes (NaN padding).
3. Agrupación Espacial (Step 3): Construcción de objetos 'Block' con matrices de estado.
4. Estructuración Temporal (Step 4): Consolidación en objetos 'Distribution' (trayectorias de clientes).
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Importaciones locales del pipeline
from .clases import Distribution
from .pipeline_preprocesamiento.step1_crear_dataset import crear_dataset, load_mapping
from .pipeline_preprocesamiento.step2_gain_e_imputer import agregar_gain, imputer, ALL_BANDAS, ALL_ANTENAS
from .pipeline_preprocesamiento.step3_mapear_a_blocks import mapear_a_blocks
from .pipeline_preprocesamiento.step4_block_to_distribution import blocks_to_distributions

# Configuración global del dataset
ALL_MONTHS   = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
DEFAULT_P_TX = 14.0  # Potencia de transmisión estándar en dBm


def run_pipeline(
    building_id: str,
    data_dir: str | None = None,
    mapping_file: str | None = None,
    months: list[str] | None = None,
    P_TX: float = DEFAULT_P_TX,
    rssi_fill: float = np.nan,
    step2_csv_path: str | None = None,
    distributions_path: str | None = None,
    verbose: bool = True,
) -> list[Distribution]:
    """
    Ejecuta el pipeline completo de preprocesamiento para un edificio específico.

    Este método implementa una lógica de caché inteligente: si se detectan archivos procesados
    (.joblib o .csv), se omiten los pasos computacionalmente costosos.

    Parameters
    ----------
    building_id : str
        Identificador del edificio objetivo (ej: "990", "1361").
    data_dir : str | None
        Directorio raíz donde se encuentran los archivos .tgz de Splunk.
    mapping_file : str | None
        Ruta al archivo CSV que mapea MAC_AP → building_id.
    months : list[str] | None
        Meses específicos a procesar (ej: ["02", "03"]). Por defecto procesa todo el año.
    P_TX : float, opcional
        Potencia de transmisión del AP para el cálculo de ganancia (Gain = RSSI - P_TX).
    rssi_fill : float, opcional
        Valor para rellenar APs no visibles (dead zones). Se recomienda np.nan para GNNs.
    step2_csv_path : str | None, opcional
        Ruta del archivo CSV de aceleración (Step 2).
    distributions_path : str | None, opcional
        Ruta del archivo Joblib final (Step 4).
    verbose : bool, opcional
        Si es True, imprime el progreso detallado en consola.

    Returns
    -------
    list[Distribution]
        Lista de trayectorias de clientes (Distribution) listas para el entorno de simulación.

    Raises
    ------
    ValueError
        Si no se proveen las rutas de origen necesarias para iniciar el procesamiento desde cero.
    """
    months = months or ALL_MONTHS

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"\n{'='*70}")
    log(f"  🚀 NetROML Preprocessing Pipeline – Building: {building_id}")
    log(f"{'='*70}\n")

    # ── Checkpoint de Caché: Joblib (O(1)) ────────────────────────────────────
    if distributions_path and os.path.exists(distributions_path):
        log(f"▶ Detección de caché (Joblib): {distributions_path}")
        log(f"  ✓ Saltando pipeline completo. Cargando objetos serializados...")
        distributions = joblib.load(distributions_path)
        log(f"  ✓ Distribuciones cargadas: {len(distributions):,}\n")
        return distributions

    # Carga de Topología Maestra (Crítico para consistencia espacial)
    ap_set  = load_mapping(mapping_file, building_id)
    ap_list = sorted(list(ap_set))

    # ── Checkpoint de Caché: CSV (O(N)) ───────────────────────────────────────
    if step2_csv_path and os.path.exists(step2_csv_path):
        log(f"▶ Detección de caché (CSV): {step2_csv_path}")
        log(f"  ✓ Saltando Steps 1 y 2. Cargando base de datos consolidada...")
        df_s2 = pd.read_csv(step2_csv_path)
        log(f"  ✓ Registros: {len(df_s2):,}  |  Clientes únicos: {df_s2['mac_cliente'].nunique()}\n")
    else:
        # Step 1: Extracción y Consolidación
        log("▶ [STEP 1/4] Extracción, Filtrado y Deduplicación")
        if data_dir is None or mapping_file is None:
            raise ValueError("Se requieren 'data_dir' y 'mapping_file' para procesar datos crudos.")
        
        df_s1 = crear_dataset(
            building_id=building_id,
            data_dir=data_dir,
            mapping_file=mapping_file,
            output_file=None,
            months=months,
        )
        log(f"  ✓ Registros extraídos: {len(df_s1):,}\n")

        # Step 2: Imputación de Malla y Cálculo de Ganancia
        log("▶ [STEP 2/4] Imputación Matricial y Cálculo de Gain (RSSI -> Gain)")
        
        df_s2 = agregar_gain(df_s1, P_TX)
        df_s2 = imputer(df_s2, ap_list, ALL_BANDAS, ALL_ANTENAS, rssi_fill, P_TX)
        log(f"  ✓ Malla completa: {len(ap_list)} APs  |  Filas totales: {len(df_s2):,}\n")

        if step2_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(step2_csv_path)), exist_ok=True)
            df_s2.to_csv(step2_csv_path, index=False)
            log(f"  💾 Caché intermedia guardada en: {step2_csv_path}\n")

    # Step 3: Construcción de Objetos Block (Espacial)
    log("▶ [STEP 3/4] Mapeo de Entidades Lógicas (Block Objects)")
    blocks = mapear_a_blocks(df_s2, ap_list)
    log(f"  ✓ Bloques espaciales generados: {len(blocks):,}\n")

    # Step 4: Construcción de Trayectorias (Temporal)
    log("▶ [STEP 4/4] Agrupación de Trayectorias (Distribution Objects)")
    idx_to_mac    = df_s2.groupby('distribution_idx')['mac_cliente'].first().to_dict()
    distributions = blocks_to_distributions(blocks, idx_to_mac)
    log(f"  ✓ Trayectorias de clientes completadas: {len(distributions):,}\n")

    log(f"{'='*70}")
    log(f"  ✅ Pipeline finalizado con éxito.")
    log(f"  📊 Resumen: {len(distributions):,} clientes | {len(blocks):,} bloques procesados.")
    log(f"{'='*70}\n")

    if distributions_path:
        os.makedirs(os.path.dirname(os.path.abspath(distributions_path)), exist_ok=True)
        joblib.dump(distributions, distributions_path)
        log(f"  💾 Dataset final persistido en: {distributions_path}\n")

    return distributions
