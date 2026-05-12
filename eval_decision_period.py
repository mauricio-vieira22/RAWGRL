import subprocess
import sys
import shutil
from pathlib import Path
import time
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Sensibilidad de Periodo de Decisión (Tesis)")
    parser.add_argument("--episodes", type=int, default=50, help="Episodios de evaluación")
    parser.add_argument("--arrival_rate", type=float, default=3.0, help="Tasa de arribos")
    parser.add_argument("--seed", type=int, default=100, help="Semilla de evaluación determinista")
    parser.add_argument("--building_id", type=str, default="990", help="ID del edificio")
    args = parser.parse_args()

    periods = [1, 2, 5, 10]
    models = ["REINFORCE", "A2C", "PPO"]
    
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "experiments_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_paths = {
        "REINFORCE": base_dir / "REINFORCE",
        "A2C": base_dir / "A2C_Advantage Actor-Critic",
        "PPO": base_dir / "PPO_Proximal Policy Optimization"
    }

    all_results = []

    print(f"\n{'='*70}")
    print(f"  INICIANDO ANÁLISIS DE SENSIBILIDAD (DECISION PERIOD)")
    print(f"  Periods: {periods} | Episodes: {args.episodes} | Arribos: {args.arrival_rate} | Edificio: {args.building_id}")
    print(f"{'='*70}\n")

    t0_global = time.time()

    for name in models:
        path = model_paths[name]
        eval_script = path / "evaluate.py"
        model_pt = path / "outputs" / "models" / "best_model.pt"
        
        if not eval_script.exists():
            print(f"[Error] No se encontró {eval_script}. Saltando {name}...")
            continue
            
        if not model_pt.exists():
            print(f"[Error] No se encontró el modelo entrenado {model_pt}. Asegúrate de que el entrenamiento terminó.")
            # Seguimos porque puede ser que se llame final_model o haya algún error
            print("Se saltará esta evaluación por falta de pesos.")
            continue

        for dp in periods:
            print(f"\n[>] Evaluando {name} con decision_period = {dp} ...")
            
            cmd = [
                sys.executable, str(eval_script),
                "--episodes", str(args.episodes),
                "--arrival_rate", str(args.arrival_rate),
                "--seed", str(args.seed),
                "--building_id", args.building_id,
                "--decision_period", str(dp),
                "--model_path", str(model_pt)
            ]
            
            t0 = time.time()
            result = subprocess.run(cmd, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            t1 = time.time()
            
            if result.returncode == 0:
                eval_csv = path / "outputs" / "eval" / "eval_metrics.csv"
                if eval_csv.exists():
                    df = pd.read_csv(eval_csv)
                    mean_rate = df["total_rate"].mean()
                    std_rate = df["total_rate"].std()
                    mean_return = df["return"].mean()
                    
                    all_results.append({
                        "Model": name,
                        "Decision_Period": dp,
                        "Mean_Total_Rate": mean_rate,
                        "Std_Total_Rate": std_rate,
                        "Mean_Return": mean_return
                    })
                    print(f"    ✓ Completado en {t1-t0:.1f}s | Rate Promedio: {mean_rate:.1f} Mbps")
                else:
                    print(f"    [Error] No se encontró CSV en {eval_csv}")
            else:
                print(f"    [Error] Falló la evaluación:\n{result.stdout[-500:]}")

    t1_global = time.time()
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        out_csv = results_dir / "decision_period_sensitivity.csv"
        final_df.to_csv(out_csv, index=False)
        
        print(f"\n{'='*70}")
        print(f"  EVALUACIÓN COMPLETADA EN {t1_global - t0_global:.1f} SEGUNDOS")
        print(f"  Resultados consolidados en: {out_csv}")
        print(f"{'='*70}\n")
        
        # Llamar automáticamente al script de ploteo
        plot_script = base_dir / "plot_sensitivity.py"
        if plot_script.exists():
            print("[INFO] Generando gráficas de sensibilidad...")
            subprocess.run([sys.executable, str(plot_script)], cwd=base_dir)
    else:
        print("\n[Error] No se recopilaron resultados.")

if __name__ == "__main__":
    main()
