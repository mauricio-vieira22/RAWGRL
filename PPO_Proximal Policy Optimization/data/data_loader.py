"""
data_loader.py – Cargador centralizado y automático de distribuciones.

Simplifica el acceso a datos mediante:
  1. Auto-detección de archivos en caché (joblib, CSV)
  2. Auto-generación de rutas basado en building_id
  3. Ejecución automática del pipeline si es necesario
  4. Interfaz simple para train.py

Uso
---
    from data.data_loader import load_distributions

    # Automático: genera rutas, detecta caché, ejecuta pipeline si falta
    distributions = load_distributions(building_id="990")
    
    # O con control completo:
    distributions = load_distributions(
        building_id="990",
        data_dir="path/to/data",
        mapping_file="path/to/mapping.csv",
        verbose=True
    )
"""

from __future__ import annotations
import os
from pathlib import Path
from data.preprocessing import run_pipeline


def load_distributions(
    building_id: str,
    data_dir: str | None = None,
    mapping_file: str | None = None,
    months: list[str] | None = None,
    verbose: bool = True,
) -> list:
    """
    Carga distribuciones automáticamente con detección inteligente de caché.

    Parameters
    ----------
    building_id : str
        ID del edificio (ej: "990", "84", "1361").
    data_dir : str | None
        Directorio con datos .tgz crudos. Si None, asume ubicación relativa
        ../preprocesamiento_de_datos/datos/
    mapping_file : str | None
        Archivo de mapeo MAC→building_id. Si None, asume 
        ../preprocesamiento_de_datos/from_mac_to_building_id.csv
    months : list[str] | None
        Meses a procesar (ej: ["02", "03"]). Si None, procesa todos.
    verbose : bool
        Imprime progreso.

    Returns
    -------
    list[Distribution]
        Lista de distribuciones cargadas/generadas.

    Examples
    --------
    >>> distributions = load_distributions("990")  # ✅ Totalmente automático
    >>> len(distributions)
    1247
    """

    # ── Rutas por defecto (relativas al proyecto) ────────────────────────────
    # Si no se proveen, se asume que los datos están en ./data o en una ruta global estándar
    base_data = Path(__file__).parent
    data_dir = data_dir or str(base_data)
    mapping_file = mapping_file or str(base_data / "from_mac_to_building_id.csv")

    # ── Rutas de caché basadas en building_id ───────────────────────────────
    data_cache_dir = Path(__file__).parent  # data/
    step2_csv_path = str(data_cache_dir / f"dataset_{building_id}_step2.csv")
    distributions_path = str(data_cache_dir / f"distributions_{building_id}.joblib")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  📦 DataLoader – Building ID: {building_id}")
        print(f"{'='*70}")
        print(f"  Step2 CSV:          {step2_csv_path}")
        print(f"  Distributions Path: {distributions_path}")

    # ── Ejecutar pipeline con caché inteligente ────────────────────────────
    distributions = run_pipeline(
        building_id=building_id,
        data_dir=data_dir,
        mapping_file=mapping_file,
        months=months,
        step2_csv_path=step2_csv_path,
        distributions_path=distributions_path,
        verbose=verbose,
    )

    if verbose:
        print(f"  ✓ Cargadas {len(distributions)} distribuciones\n")

    return distributions
