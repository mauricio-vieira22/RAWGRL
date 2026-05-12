import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Configuración global
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "axes.linewidth": 1.2,
})

def apply_smoothing(series, weight=0.85):
    """Aplica suavizado exponencial para visualización de RL"""
    smoothed = np.zeros_like(series)
    last = series[0]
    for i, point in enumerate(series):
        if not pd.isna(point):
            smoothed_val = last * weight + (1 - weight) * point
            smoothed[i] = smoothed_val
            last = smoothed_val
        else:
            smoothed[i] = last
    return smoothed

def plot_metric(dataframes, metric_col, title, ylabel, filename, weight=0.85):
    """
    Genera gráficas.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    # Paleta de colores profesional (Colorblind-friendly y alto contraste)
    # REINFORCE: Rojo oscuro, A2C: Azul acero, PPO: Verde bosque
    colors = {'REINFORCE': '#D62728', 'A2C': '#1F77B4', 'PPO': '#2CA02C'}
    
    for label, df in dataframes.items():
        if metric_col not in df.columns:
            continue
            
        x = df['episode']
        y_raw = df[metric_col].values
        y_smooth = apply_smoothing(y_raw, weight=weight)
        
        c = colors.get(label, 'black')
        
        # Plot raw data (fondo difuminado)
        ax.plot(x, y_raw, alpha=0.15, color=c, linewidth=1.0)
        # Plot smoothed data (curva principal)
        ax.plot(x, y_smooth, label=label, linewidth=2.5, color=c)
        
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('Training Episodes', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    
    # Leyenda
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
    
    # Bordes
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', format=filename.suffix[1:])
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "experiments_results"
    
    if not results_dir.exists():
        print(f"[Error] Directorio {results_dir} no encontrado.")
        return
        
    models = ['REINFORCE', 'A2C', 'PPO']
    dfs = {}
    
    for m in models:
        csv_path = results_dir / f"{m.lower()}_metrics.csv"
        if csv_path.exists():
            dfs[m] = pd.read_csv(csv_path)
            print(f"[Cargado] {csv_path.name} con {len(dfs[m])} episodios.")
        else:
            print(f"[Advertencia] {csv_path.name} no encontrado.")
            
    if not dfs:
        print("[Error] No hay datos para graficar.")
        return
        
    # Graficar Return (Gt)
    plot_metric(
        dfs, 
        metric_col='return', 
        title='Convergencia de la Política: Retorno Descontado ($G_t$)', 
        ylabel='Retorno ($G_t$)', 
        filename=results_dir / 'comparison_return.png',
        weight=0.9
    )
    
    plot_metric(
        dfs, 
        metric_col='return', 
        title='Convergencia de la Política: Retorno Descontado ($G_t$)', 
        ylabel='Retorno ($G_t$)', 
        filename=results_dir / 'comparison_return.pdf',
        weight=0.9
    )

    # Graficar Total Rate
    plot_metric(
        dfs, 
        metric_col='total_rate', 
        title='Capacidad Total del Sistema a través del Entrenamiento', 
        ylabel='Suma de Tasa (Mbps)', 
        filename=results_dir / 'comparison_total_rate.png',
        weight=0.9
    )
    
    plot_metric(
        dfs, 
        metric_col='total_rate', 
        title='Capacidad Total del Sistema a través del Entrenamiento', 
        ylabel='Suma de Tasa (Mbps)', 
        filename=results_dir / 'comparison_total_rate.pdf',
        weight=0.9
    )

    # Graficar Mean Rate Step
    plot_metric(
        dfs, 
        metric_col='mean_rate_step', 
        title='Tasa Media por Step', 
        ylabel='Tasa (Mbps)', 
        filename=results_dir / 'comparison_mean_rate_step.png',
        weight=0.9
    )
    
    plot_metric(
        dfs, 
        metric_col='mean_rate_step', 
        title='Tasa Media por Step', 
        ylabel='Tasa (Mbps)', 
        filename=results_dir / 'comparison_mean_rate_step.pdf',
        weight=0.9
    )

    print(f"\nGráficas generadas correctamente en: {results_dir}/")

if __name__ == "__main__":
    main()
