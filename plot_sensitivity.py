import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración global para nivel académico (IEEE / Tesis)
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

def plot_sensitivity(df, results_dir):
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    # Paleta de colores profesional
    colors = {'REINFORCE': '#D62728', 'A2C': '#1F77B4', 'PPO': '#2CA02C'}
    markers = {'REINFORCE': 'o', 'A2C': 's', 'PPO': '^'}
    
    # Asegurar que Decision_Period esté ordenado
    df = df.sort_values(by="Decision_Period")
    
    for model in ['REINFORCE', 'A2C', 'PPO']:
        model_data = df[df["Model"] == model]
        if model_data.empty:
            continue
            
        ax.errorbar(
            model_data["Decision_Period"], 
            model_data["Mean_Total_Rate"], 
            yerr=model_data["Std_Total_Rate"],
            label=model,
            color=colors[model],
            marker=markers[model],
            markersize=8,
            linewidth=2.5,
            capsize=5,
            capthick=1.5
        )
        
    ax.set_title('Robustez frente a la Latencia de Decisión (Ablation)', fontweight='bold', pad=15)
    ax.set_xlabel('Decision Period (Timesteps de Retraso)', fontweight='bold')
    ax.set_ylabel('Total System Rate Promedio (Mbps)', fontweight='bold')
    
    # Forzar que el eje X muestre solo los enteros que probamos (1, 2, 5, 10)
    periods = df["Decision_Period"].unique()
    ax.set_xticks(periods)
    
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')
    
    # Bordes estrictos para estilo IEEE
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        
    plt.tight_layout()
    plt.savefig(results_dir / 'sensitivity_decision_period.pdf', bbox_inches='tight')
    plt.savefig(results_dir / 'sensitivity_decision_period.png', bbox_inches='tight')
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "experiments_results"
    csv_path = results_dir / "decision_period_sensitivity.csv"
    
    if not csv_path.exists():
        print(f"[Error] Archivo {csv_path} no encontrado.")
        return
        
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("[Error] El DataFrame está vacío.")
        return
        
    print(f"[Cargado] Datos de sensibilidad con {len(df)} evaluaciones.")
    plot_sensitivity(df, results_dir)
    print(f"\n[OK] Gráficas de sensibilidad guardadas en: {results_dir}/")

if __name__ == "__main__":
    main()
