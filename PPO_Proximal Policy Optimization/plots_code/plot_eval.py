import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

def plot_eval_metrics(csv_path, output_dir=None):
    """
    Genera gráficas de nivel académico para la evaluación de NetROML.
    """
    df = pd.read_csv(csv_path)
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuración estética
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        pass
        
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "figure.dpi": 300
    })

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # 1. Distribución de Retornos (Histograma + KDE)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(df['return'], kde=True, color='skyblue', ax=ax1)
    ax1.set_title(r"Return Distribution ($G_t$)")
    ax1.set_xlabel("Cumulative Reward")
    ax1.grid(True, alpha=0.3)

    # 2. Distribución de Throughput (Histograma + KDE)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(df['total_rate'], kde=True, color='salmon', ax=ax2)
    ax2.set_title("Throughput Distribution")
    ax2.set_xlabel("Total Rate [bits/s/Hz]")
    ax2.grid(True, alpha=0.3)

    # 3. Boxplot de Métricas Clave
    ax3 = fig.add_subplot(gs[0, 2])
    metrics_to_plot = ['return', 'total_rate']
    # Normalizar para visualización comparativa
    df_norm = (df[metrics_to_plot] - df[metrics_to_plot].mean()) / df[metrics_to_plot].std()
    sns.boxplot(data=df_norm, ax=ax3, palette="Set2")
    ax3.set_title("Standardized Variance")
    ax3.set_ylabel("Z-Score")
    ax3.grid(True, alpha=0.3)

    # 4. Correlación: Delta_t vs Performance
    ax4 = fig.add_subplot(gs[1, 0])
    sns.regplot(x='delta_t_mean', y='total_rate', data=df, ax=ax4, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    ax4.set_title(r"Performance vs. $\delta_t$ (Next Arrival)")
    ax4.set_xlabel(r"Mean $\delta_t$ [slots]")
    ax4.set_ylabel("Total Rate")
    ax4.grid(True, alpha=0.3)

    # 5. Correlación: Epsilon_t vs Performance
    ax5 = fig.add_subplot(gs[1, 1])
    sns.regplot(x='epsilon_t_mean', y='total_rate', data=df, ax=ax5, scatter_kws={'alpha':0.5}, line_kws={'color':'blue'})
    ax5.set_title(r"Performance vs. $\epsilon_t$ (Remaining Life)")
    ax5.set_xlabel(r"Mean $\epsilon_t$ [slots]")
    ax5.set_ylabel("Total Rate")
    ax5.grid(True, alpha=0.3)

    # 6. Performance por Episodio (Serie Temporal)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(df['episode'], df['total_rate'], marker='o', linestyle='-', color='purple', markersize=4)
    ax6.axhline(df['total_rate'].mean(), color='black', linestyle='--', alpha=0.7, label='Mean')
    ax6.set_title("Episode-wise Consistency")
    ax6.set_xlabel("Evaluation Episode")
    ax6.set_ylabel("Total Rate")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "evaluation_summary.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfica de evaluación generada en: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", help="Output directory")
    args = parser.parse_args()
    plot_eval_metrics(args.csv, args.out)
