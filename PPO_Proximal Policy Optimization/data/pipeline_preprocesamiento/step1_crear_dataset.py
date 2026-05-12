"""
step1_crear_dataset.py – Extrae y consolida registros RSSI desde archivos .tgz.

Columnas del CSV resultante:
    datetime, distribution_idx, mac_cliente, mac_ap, rssi, antena, banda, block_idx
"""

from __future__ import annotations
import os
import tarfile
import csv
import re
import glob
import argparse
import pandas as pd

ALL_MONTHS = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11']


def load_mapping(mapping_file: str, building_id: str) -> set:
    """Lee el CSV de mapeo MAC_AP→building_id y retorna el conjunto de MACs del building."""
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
    """Parsea una línea SNMP de datos_RSSI_WLCx.txt."""
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
    building_id:  str,
    data_dir:     str,
    mapping_file: str,
    output_file:  str | None = None,
    months:       list[str] | None = None,
) -> pd.DataFrame:
    """Extrae, filtra y transforma los registros RSSI para un building_id."""
    months = months or ALL_MONTHS

    print(f"[1/5] Cargando mapeo para building_id={building_id}...")
    ap_set = load_mapping(mapping_file, building_id)
    if not ap_set:
        raise ValueError(f"No se encontraron APs para building_id={building_id}")
    print(f"      APs encontrados: {len(ap_set)}")

    tgz_files = sorted(glob.glob(os.path.join(data_dir, "RSSI_WLCs_2018-*.tgz")))
    print(f"[2/5] Procesando {len(tgz_files)} archivos en {data_dir}...")

    records = []
    for tgz_path in tgz_files:
        fname = os.path.basename(tgz_path)
        m = re.search(r"2018-(\d{2})-(\d{2})_(\d{2})_(\d{2})", fname)
        if not m:
            continue
        month, day, hour, minute = m.groups()
        if month not in months:
            continue
        datetime_str = f"2018-{month}-{day} {hour}:{minute}"
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if "datos_RSSI_WLC" in member.name and member.name.endswith(".txt"):
                        fobj = tar.extractfile(member)
                        if fobj is None:
                            continue
                        for raw_line in fobj:
                            parsed = parse_rssi_line(raw_line.decode('utf-8', errors='ignore').strip())
                            if parsed and parsed['mac_ap'] in ap_set:
                                parsed['datetime'] = datetime_str
                                records.append(parsed)
        except Exception as exc:
            print(f"      ⚠  Error en {fname}: {exc}")

    print(f"      Registros extraídos: {len(records)}")
    if not records:
        raise RuntimeError("No se extrajeron registros.")

    print("[3/5] Construyendo DataFrame...")
    df = pd.DataFrame(records)
    df['_dt'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['mac_cliente', '_dt']).reset_index(drop=True)

    print("[4/5] Calculando índices...")
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
        print(f"[5/5] Guardando en {output_file}...")
        df_out.to_csv(output_file, index=False)
    return df_out
