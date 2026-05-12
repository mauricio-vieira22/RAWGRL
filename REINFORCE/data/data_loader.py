"""
data_loader.py – Cargador centralizado y automático de distribuciones.

Simplifica el acceso a datos mediante:
  1. Auto-detección de archivos en caché (joblib, CSV)
  2. Auto-generación de rutas basado en building_id
  3. Ejecución automática del pipeline si es necesario
  4. Interfaz simple para train.py / evaluate.py

Estructura de directorios esperada
-----------------------------------
    proyecto/
    ├── Raw_Data/
    │   └── RSSI_WLCs_2018-*.tgz          ← datos crudos
    ├── data/
    │   ├── from_mac_to_building_id.csv   ← mapeo MAC_AP → building_id
    │   ├── clases.py
    │   ├── data_loader.py                ← este archivo
    │   ├── preprocessing.py
    │   └── pipeline_preprocesamiento/
    │       ├── __init__.py
    │       ├── step1_crear_dataset.py
    │       ├── step2_gain_e_imputer.py
    │       ├── step3_mapear_a_blocks.py
    │       └── step4_block_to_distribution.py
    ├── model/
    └── simulation/

Uso
---
    from data.data_loader import load_distributions

    # Automático: detecta caché, ejecuta pipeline si falta
    distributions = load_distributions(building_id="873")

    # Con control completo:
    distributions = load_distributions(
        building_id="873",
        raw_data_dir="otro/path/Raw_Data",
        verbose=True,
    )
"""

from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
from data.preprocessing import run_pipeline


# ---------------------------------------------------------------------------
# Utilidades de diagnóstico
# ---------------------------------------------------------------------------

def list_available_buildings(raw_data_dir: str | None = None) -> dict[str, int]:
    """
    Lee el CSV de mapeo y devuelve un dict {building_id: n_aps} ordenado
    por cantidad de APs descendente. Útil para elegir un building_id válido.

    Parameters
    ----------
    raw_data_dir : str | None
        Directorio Raw_Data. Si None, se asume ../Raw_Data relativo a data/.

    Returns
    -------
    dict[str, int]
        {building_id: cantidad_de_aps}
    """
    _data_dir    = Path(__file__).parent                          # proyecto/data/
    mapping_file = _data_dir / "from_mac_to_building_id.csv"     # siempre en data/

    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping CSV no encontrado: {mapping_file}")

    counts: dict[str, int] = {}
    with open(mapping_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bid = str(row["building_id"]).strip()
            counts[bid] = counts.get(bid, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def validate_building_id(building_id: str, raw_data_dir: str | None = None) -> int:
    """
    Verifica que building_id existe en el CSV y tiene al menos 2 APs.
    Devuelve la cantidad de APs del edificio.

    Raises
    ------
    ValueError
        Si el building_id no existe o tiene menos de 2 APs.
    """
    buildings = list_available_buildings(raw_data_dir)
    if building_id not in buildings:
        top10 = list(buildings.items())[:10]
        raise ValueError(
            f"building_id='{building_id}' no existe en el CSV de mapeo.\n"
            f"  Top-10 por cantidad de APs: {top10}\n"
            f"  Usá list_available_buildings() para ver todos."
        )
    n_aps = buildings[building_id]
    if n_aps < 2:
        raise ValueError(
            f"building_id='{building_id}' tiene solo {n_aps} AP(s). "
            f"Se necesitan al menos 2 para construir un grafo meaningful."
        )
    return n_aps


def validate_distributions(distributions: list, building_id: str) -> None:
    """
    Validaciones de integridad sobre las distribuciones cargadas:
    - No vacías
    - n_aps consistente entre todos los clientes
    - Sin valores -inf en las grillas (indica caché corrupta)

    Raises
    ------
    RuntimeError
        Si alguna validación falla.
    """
    if not distributions:
        raise RuntimeError(f"Pipeline devolvió 0 distribuciones para building_id='{building_id}'.")

    # Verificar n_aps consistente
    ap_counts = [len(d.blocks[0].datos) for d in distributions if len(d.blocks) > 0]
    unique_counts = set(ap_counts)
    if len(unique_counts) > 1:
        raise RuntimeError(
            f"n_aps inconsistente entre clientes: {unique_counts}. "
            f"El imputer no completó correctamente el cross-join. "
            f"Eliminá el caché CSV y el .joblib y regenerá."
        )

    # Verificar ausencia de -inf en grillas (bug de rssi_fill legacy)
    sample_datos = distributions[0].blocks[0].datos
    for col in ["G_2_4", "G_5"]:
        if col in sample_datos.columns:
            if sample_datos[col].isin([-np.inf]).any():
                raise RuntimeError(
                    f"Se detectaron valores -inf en la columna '{col}' del primer bloque. "
                    f"El .joblib fue generado con rssi_fill=-inf (versión legacy). "
                    f"Eliminá el archivo de caché y regenerá con rssi_fill=np.nan."
                )


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def load_distributions(
    building_id: str,
    raw_data_dir: str | None = None,
    months: list[str] | None = None,
    verbose: bool = True,
) -> list:
    """
    Carga distribuciones automáticamente con detección inteligente de caché.

    Parameters
    ----------
    building_id : str
        ID del edificio (ej: "873", "295"). Debe existir en from_mac_to_building_id.csv
        y tener al menos 2 APs. Usá list_available_buildings() para explorar opciones.
    raw_data_dir : str | None
        Directorio con los .tgz crudos y el CSV de mapeo.
        Si None, asume <project_root>/Raw_Data/.
    months : list[str] | None
        Meses a procesar (ej: ["02", "03"]). Si None, procesa todos.
    verbose : bool
        Imprime progreso.

    Returns
    -------
    list[Distribution]
        Lista de distribuciones cargadas/generadas, validadas.

    Examples
    --------
    >>> from data.data_loader import list_available_buildings, load_distributions
    >>> buildings = list_available_buildings()
    >>> print(list(buildings.items())[:5])  # top-5 por número de APs
    >>> distributions = load_distributions("873")
    """
    _data_dir     = Path(__file__).parent          # proyecto/data/
    _project_root = _data_dir.parent              # proyecto/

    # ── Rutas base ────────────────────────────────────────────────────────────
    raw_data_path = Path(raw_data_dir) if raw_data_dir else (_project_root / "Raw_Data")
    mapping_file  = str(_data_dir / "from_mac_to_building_id.csv")   # siempre en data/
    data_dir      = str(raw_data_path)

    # ── Validar building_id antes de intentar cualquier cosa ─────────────────
    n_aps_expected = validate_building_id(building_id, raw_data_dir)

    # ── Rutas de caché ────────────────────────────────────────────────────────
    step2_csv_path     = str(_data_dir / f"dataset_{building_id}_step2.csv")
    distributions_path = str(_data_dir / f"distributions_{building_id}.joblib")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  DataLoader — Building ID: {building_id}  ({n_aps_expected} APs en CSV)")
        print(f"{'='*70}")
        print(f"  Raw Data:           {raw_data_path}")
        print(f"  Mapping CSV:        {mapping_file}")
        print(f"  Caché Step2 CSV:    {step2_csv_path}")
        print(f"  Caché Joblib:       {distributions_path}")

    # ── Ejecutar pipeline ─────────────────────────────────────────────────────
    distributions = run_pipeline(
        building_id=building_id,
        data_dir=data_dir,
        mapping_file=mapping_file,
        months=months,
        step2_csv_path=step2_csv_path,
        distributions_path=distributions_path,
        verbose=verbose,
    )

    # ── Validar resultado ─────────────────────────────────────────────────────
    validate_distributions(distributions, building_id)

    if verbose:
        n_aps_real = len(distributions[0].blocks[0].datos)
        print(f"  ✓ {len(distributions)} distribuciones cargadas | n_aps={n_aps_real}\n")

    return distributions