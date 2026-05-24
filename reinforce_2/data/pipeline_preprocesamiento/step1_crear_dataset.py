"""
step1_crear_dataset.py – Módulo de Extracción y Consolidación de Logs SNMP.

Este componente es responsable del primer nivel de ingesta de datos. Realiza el parseo
de archivos comprimidos (.tgz), filtra los registros de interés según la topología
del edificio y aplica una deduplicación crítica para garantizar la calidad de la señal.

Atributos del Dataset resultante:
    - datetime: Marca temporal del evento (minuto).
    - distribution_idx: Índice único por cliente (mac_cliente).
    - mac_cliente: Identificador único del dispositivo cliente.
    - mac_ap: Identificador único del Access Point reportador.
    - rssi: Indicador de fuerza de señal (Received Signal Strength Indicator).
    - antena: Índice de la antena que captó el evento.
    - banda: Frecuencia de operación (2.4 GHz o 5 GHz).
    - block_idx: Índice secuencial temporal por cliente.
"""

from __future__ import annotations
import os
import tarfile
import csv
import re
import glob
import pandas as pd


def load_mapping(mapping_file: str, building_id: str) -> set:
    """
    Carga el mapeo topológico oficial del edificio desde un archivo CSV.

    Parameters
    ----------
    mapping_file : str
        Ruta al archivo CSV de mapeo (MAC_AP, building_id).
    building_id : str
        Identificador del edificio a filtrar.

    Returns
    -------
    set
        Conjunto de MACs de Access Points pertenecientes al edificio especificado.
    """
    mac_aps = set()
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Archivo de mapeo no encontrado: {mapping_file}")
    
    with open(mapping_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row['building_id']).strip() == str(building_id).strip():
                mac_aps.add(row['MAC_AP'].strip())
    return mac_aps


def parse_rssi_line(line: str) -> dict | None:
    """
    Realiza el parseo sintáctico de una línea de log SNMP.

    Parameters
    ----------
    line : str
        Línea de texto cruda extraída del log.

    Returns
    -------
    dict | None
        Diccionario con los campos extraídos o None si la línea no cumple el formato esperado.
    """
    if " = INTEGER: " not in line:
        return None
    try:
        oid_part, rssi_str = line.split(" = INTEGER: ", 1)
        rssi   = int(rssi_str.strip())
        prefix = "SNMPv2-SMI::enterprises.14179.2.1.11.1.5."
        
        if not oid_part.startswith(prefix):
            return None
        
        parts = oid_part[len(prefix):].split('.')
        if len(parts) < 14:
            return None
            
        return dict(
            mac_cliente=".".join(parts[0:6]),
            mac_ap=".".join(parts[6:12]),
            banda=int(parts[12]),
            antena=int(parts[13]),
            rssi=rssi,
        )
    except Exception:
        return None


def crear_dataset(
    building_id: str,
    data_dir: str,
    mapping_file: str,
    output_file: str | None = None,
    months: list[str] | None = None,
) -> pd.DataFrame:
    """
    Orquesta el proceso de extracción masiva y consolidación de datos.

    Parameters
    ----------
    building_id : str
        ID del edificio objetivo.
    data_dir : str
        Directorio que contiene los archivos .tgz.
    mapping_file : str
        Ruta al archivo de mapeo MAC-Building.
    output_file : str | None, opcional
        Ruta para persistir el CSV resultante.
    months : list[str] | None, opcional
        Filtro de meses a procesar.

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado con la historia limpia de los clientes del edificio.
    """
    months = months or ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    print(f"  [Step 1] Cargando topología para building_id={building_id}")
    ap_set = load_mapping(mapping_file, building_id)
    if not ap_set:
        raise ValueError(f"No se detectaron APs válidos para el edificio {building_id}")

    # Búsqueda recursiva para soportar diversas estructuras de directorios en Raw_Data
    pattern = os.path.join(data_dir, "**", "RSSI_WLCs_2018-*.tgz")
    tgz_files = sorted(glob.glob(pattern, recursive=True))
    
    print(f"  [Step 1] Analizando {len(tgz_files)} archivos .tgz...")

    records = []
    for tgz_path in tgz_files:
        fname = os.path.basename(tgz_path)
        # Regex para extraer metadatos temporales del nombre del archivo
        m = re.search(r"2018-(\d{2})-(\d{2})_(\d{2})_(\d{2})", fname)
        if not m:
            continue
            
        month, day, hour, minute = m.groups()
        if month not in months:
            continue
            
        datetime_str = f"2018-{month}-{day} {hour}:{minute}"
        
        try:
            current_records_count = len(records)
            with tarfile.open(tgz_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if "datos_RSSI_WLC" in member.name and member.name.endswith(".txt"):
                        fobj = tar.extractfile(member)
                        if fobj is None: continue
                        for raw_line in fobj:
                            parsed = parse_rssi_line(raw_line.decode('utf-8', errors='ignore').strip())
                            if parsed and parsed['mac_ap'] in ap_set:
                                parsed['datetime'] = datetime_str
                                records.append(parsed)
            
            new_records = len(records) - current_records_count
            if new_records > 0:
                print(f"      [DEBUG] {fname}: +{new_records} registros")
        except Exception as exc:
            print(f"      ⚠ Error de lectura en {fname}: {exc}")
    print(f"      Archivos .tgz analizados con éxito: {len(tgz_files)}")
    print(f"      Registros extraídos (crudos): {len(records)}")
    if records:
        unique_dts = set(r['datetime'] for r in records)
        print(f"      Timestamps únicos detectados: {len(unique_dts)}")
        print(f"      Ejemplos de timestamps: {sorted(list(unique_dts))[:5]}")

    if not records:
        raise RuntimeError(f"No se encontraron registros válidos para el edificio {building_id}")

    # Construcción y Limpieza de DataFrame
    df = pd.DataFrame(records)
    
    # Deduplicación Estratégica: Seleccionamos el RSSI máximo por combinación cliente/tiempo/antena.
    # Esto evita el sesgo por retransmisiones SNMP y optimiza el uso de memoria.
    df = (
        df.groupby(['mac_cliente', 'datetime', 'mac_ap', 'banda', 'antena'], as_index=False)
          .agg({'rssi': 'max'})
    )

    df['_dt'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['mac_cliente', '_dt']).reset_index(drop=True)

    # Generación de índices estructurales para Step 3 y Step 4
    df['distribution_idx'] = df.groupby('mac_cliente').ngroup()
    df['block_idx'] = (
        df.groupby('mac_cliente')['_dt']
          .rank(method='dense', ascending=True)
          .astype(int)
    )

    final_cols = ['datetime', 'distribution_idx', 'mac_cliente', 'mac_ap',
                  'rssi', 'antena', 'banda', 'block_idx']
    df_out = df[final_cols].copy()

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        df_out.to_csv(output_file, index=False)
        
    return df_out
