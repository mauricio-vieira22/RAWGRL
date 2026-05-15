import subprocess
import sys
import shutil
from pathlib import Path
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ejecutar Experimentos RAWGRL para Tesis")
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios")
    parser.add_argument("--arrival_rate", type=float, default=3.0, help="Tasa de arribo de clientes")
    parser.add_argument("--seed", type=int, default=42, help="Semilla estocástica")
    parser.add_argument("--building_id", type=str, default="990", help="ID del edificio a simular")
    parser.add_argument("--sticky_modes", type=str, nargs="+", default=["full", "sticky", "lite"], help="Modos sticky a evaluar")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  INICIANDO GRID SEARCH DE EXPERIMENTOS")
    print(f"  Episodios: {args.episodes} | Arribos: {args.arrival_rate} | Edificio: {args.building_id}")
    print(f"  Modos Sticky: {args.sticky_modes}")
    print(f"{'='*70}\n")

    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "experiments_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "REINFORCE": base_dir / "REINFORCE",
        "A2C": base_dir / "A2C_Advantage Actor-Critic",
        "PPO": base_dir / "PPO_Proximal Policy Optimization"
    }

    t0_global = time.time()

    for sticky in args.sticky_modes:
        print(f"\n\n{'#'*60}")
        print(f"  REGIMEN DE MOVILIDAD: {sticky.upper()}")
        print(f"{'#'*60}\n")
        
        for name, path in models.items():
            print(f"\n[>] {name} ({sticky})")
            
            train_script = path / "train.py"
            if not train_script.exists():
                print(f"[Error] No se encontró {train_script}. Saltando...")
                continue

            cmd = [
                sys.executable, str(train_script),
                "--episodes", str(args.episodes),
                "--arrival_rate", str(args.arrival_rate),
                "--seed", str(args.seed),
                "--building_id", args.building_id,
                "--sticky_mode", sticky
            ]
            
            t0_model = time.time()
            result = subprocess.run(cmd, cwd=path)
            t1_model = time.time()
            
            if result.returncode == 0:
                csv_source = path / "outputs" / "models" / "training_metrics.csv"
                if csv_source.exists():
                    csv_dest = results_dir / f"{name.lower()}_{sticky}_metrics.csv"
                    shutil.copy2(csv_source, csv_dest)
                    print(f"  [OK] {t1_model - t0_model:.1f}s")
                else:
                    print(f"  [Warning] No se encontró CSV en {csv_source}")
            else:
                print(f"  [Error] Falló ejecución")

    t1_global = time.time()
    print(f"\n{'='*70}")
    print(f"  TODOS LOS EXPERIMENTOS COMPLETADOS EN {t1_global - t0_global:.1f} SEGUNDOS")
    print(f"  Resultados en: {results_dir}")
    print(f"{'='*70}\n")

    # Intentar graficar
    plot_script = base_dir / "plot_comparison.py"
    if plot_script.exists():
        subprocess.run([sys.executable, str(plot_script)], cwd=base_dir)

if __name__ == "__main__":
    main()
