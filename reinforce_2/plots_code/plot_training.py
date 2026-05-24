import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def plot_metrics(csv_path, output_dir=None):
    """
    Genera gráficas de nivel académico para el entrenamiento de NetROML.
    6 paneles: Loss, Reward, Rate, Entropy, Baseline, Grad Norm.
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
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "grid.alpha": 0.2,
        "figure.dpi": 300
    })

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    axes = axes.flatten()

    window = max(1, len(df) // 20)

    # 1. Policy Loss
    ax = axes[0]
    ax.plot(df['episode'], df['loss'], color='red', alpha=0.2)
    ax.plot(df['episode'], df['loss'].rolling(window=window).mean(), color='red', linewidth=1.5, label='Policy Loss')
    ax.set_title("Optimization: Policy Loss")
    ax.set_xlabel("Episode")
    ax.grid(True)

    # 2. Cumulative Reward (G_t)
    ax = axes[1]
    ax.plot(df['episode'], df['return'], color='blue', alpha=0.2)
    ax.plot(df['episode'], df['return'].rolling(window=window).mean(), color='blue', linewidth=1.5, label=r'Return ($G_t$)')
    if 'baseline' in df.columns:
        ax.plot(df['episode'], df['baseline'], color='orange', linestyle='--', alpha=0.7, label='Baseline')
    ax.set_title(r"System Performance: Cumulative Reward ($G_t$)")
    ax.set_xlabel("Episode")
    ax.grid(True)
    ax.legend()

    # 3. Total Rate (Throughput)
    ax = axes[2]
    ax.plot(df['episode'], df['total_rate'], color='green', alpha=0.2)
    ax.plot(df['episode'], df['total_rate'].rolling(window=window).mean(), color='green', linewidth=1.5, label='Aggregate Throughput')
    ax.set_title("Network Throughput & Rate per Client")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Bits/s/Hz (Aggregate)", color='green')
    ax.tick_params(axis='y', labelcolor='green')
    ax.grid(True)

    if 'rate_per_client' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df['episode'], df['rate_per_client'], color='teal', alpha=0.2)
        ax2.plot(df['episode'], df['rate_per_client'].rolling(window=window).mean(), color='teal', linewidth=1.2, linestyle='--', label='Throughput / Client')
        ax2.set_ylabel("Bits/s/Hz (per Client)", color='teal')
        ax2.tick_params(axis='y', labelcolor='teal')

    # 4. Entropy
    ax = axes[3]
    if 'entropy' in df.columns:
        ax.plot(df['episode'], df['entropy'], color='brown', alpha=0.2)
        ax.plot(df['episode'], df['entropy'].rolling(window=window).mean(), color='brown', linewidth=1.5, label='Entropy')
        ax.set_title("Exploration: Policy Entropy")
        ax.set_xlabel("Episode")
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "Entropy not recorded", ha='center', va='center')

    # 5. Training Stability (Gradient Norm)
    ax = axes[4]
    ax.plot(df['episode'], df['grad_norm'], color='purple', alpha=0.2)
    ax.plot(df['episode'], df['grad_norm'].rolling(window=window).mean(), color='purple', linewidth=1.5, label='Grad Norm')
    ax.set_title("Stability: Gradient Norm")
    ax.set_xlabel("Episode")
    ax.grid(True)

    # 6. Step Duration (Time per Episode)
    ax = axes[5]
    ax.plot(df['episode'], df['sec'], color='gray', alpha=0.2)
    ax.plot(df['episode'], df['sec'].rolling(window=window).mean(), color='black', linewidth=1.5, label='Duration')
    ax.set_title("Computational Efficiency: Time per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Seconds")
    ax.grid(True)

    plot_path = output_dir / "training_summary.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"📊 Gráfica generada exitosamente en: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", help="Output directory")
    args = parser.parse_args()
    plot_metrics(args.csv, args.out)
