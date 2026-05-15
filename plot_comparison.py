import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configuración global Estilo Tesis
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

def apply_smoothing(series, weight=0.9):
    smoothed = np.zeros_like(series)
    last = series[0]
    for i, point in enumerate(series):
        smoothed_val = last * weight + (1 - weight) * point
        smoothed[i] = smoothed_val
        last = smoothed_val
    return smoothed

def plot_metric(dfs_dict, metric_col, title, ylabel, filename):
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Colores por Algoritmo, Estilos por Sticky
    colors = {'reinforce': '#D62728', 'a2c': '#1F77B4', 'ppo': '#2CA02C'}
    styles = {'full': '-', 'sticky': '--', 'lite': ':'}
    
    for key, df in dfs_dict.items():
        if metric_col not in df.columns: continue
        
        parts = key.split('_')
        model = parts[0]
        mode = parts[1]
        
        label = f"{model.upper()} ({mode})"
        c = colors.get(model, 'black')
        ls = styles.get(mode, '-')
        
        y = apply_smoothing(df[metric_col].values)
        plt.plot(df['episode'], y, label=label, color=c, linestyle=ls, linewidth=2)

    plt.title(title, fontweight='bold')
    plt.xlabel('Episodios de Entrenamiento')
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main():
    results_dir = Path("experiments_results")
    if not results_dir.exists(): return

    all_csvs = list(results_dir.glob("*_metrics.csv"))
    dfs = {}
    for cp in all_csvs:
        key = cp.stem.replace("_metrics", "")
        dfs[key] = pd.read_csv(cp)

    if not dfs: return

    plot_metric(dfs, 'return', 'Comparativa de Convergencia (Retorno)', 'Retorno ($G_t$)', results_dir / "comparison_sticky_return.png")
    plot_metric(dfs, 'total_rate', 'Capacidad del Sistema vs Movilidad', 'Suma de Tasa (bits/s/Hz)', results_dir / "comparison_sticky_rate.png")

    print(f"Gráficas generadas en {results_dir}")

if __name__ == "__main__":
    main()
