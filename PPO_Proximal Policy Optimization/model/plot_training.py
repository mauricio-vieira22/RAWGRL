"""
plot_training.py – Visualización automática durante el entrenamiento.

Módulo para generar plots en tiempo real durante el entrenamiento.
Compatible con todas las versiones (custom loops y SB3).

Uso
---
    from model.plot_training import TrainingPlotter
    
    plotter = TrainingPlotter(output_dir="outputs/logs")
    for episode in range(episodes):
        # ... entrenamiento ...
        plotter.log_episode(episode, rewards=[...], loss=0.5)
        plotter.plot_if_ready()  # Dibuja cada N episodios
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import json


class TrainingPlotter:
    """Visualizador de métricas de entrenamiento en tiempo real."""
    
    def __init__(
        self,
        output_dir: str = "outputs/logs",
        plot_every: int = 10,
        figsize: tuple = (12, 8),
    ):
        """
        Inicializa el plotter.
        
        Args:
            output_dir: Directorio para guardar plots
            plot_every: Plotear cada N episodios
            figsize: Tamaño de figura (ancho, alto)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_every = plot_every
        self.figsize = figsize
        self.episode_count = 0
        
        # Historial de métricas
        self.episodes = []
        self.rewards = []
        self.cumulative_rewards = []
        self.losses = []
        self.grad_norms = []
        self.rates = []
        self.entropies = []
        self.custom_metrics: Dict[str, List[float]] = {}
    
    def log_episode(
        self,
        episode: int,
        reward: float,
        cumulative_reward: Optional[float] = None,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        rate: Optional[float] = None,
        entropy: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Registra métricas de un episodio.
        
        Args:
            episode: Número de episodio
            reward: Reward del episodio
            cumulative_reward: Reward acumulativo (si aplica)
            loss: Loss/Policy loss
            grad_norm: Gradient norm
            rate: Rate promedio del episodio
            entropy: Entropía promedio
            **kwargs: Métricas personalizadas (se guardarán en custom_metrics)
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        
        if cumulative_reward is not None:
            self.cumulative_rewards.append(cumulative_reward)
        if loss is not None:
            self.losses.append(loss)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        if rate is not None:
            self.rates.append(rate)
        if entropy is not None:
            self.entropies.append(entropy)
        
        # Guardar métricas personalizadas
        for key, value in kwargs.items():
            if key not in self.custom_metrics:
                self.custom_metrics[key] = []
            self.custom_metrics[key].append(value)
        
        self.episode_count += 1
    
    def plot_if_ready(self) -> None:
        """Plotea si se alcanzó el número de episodios configurado."""
        if self.episode_count % self.plot_every == 0:
            self.plot()
    
    def plot(self) -> None:
        """Genera y guarda plots de todas las métricas registradas."""
        if not self.episodes:
            return
        
        episodes_arr = np.array(self.episodes)
        
        # Crear figura con subplots
        n_plots = sum([
            len(self.rewards) > 0,
            len(self.losses) > 0,
            len(self.grad_norms) > 0,
            len(self.rates) > 0,
            len(self.entropies) > 0,
        ])
        n_plots = max(2, n_plots)  # Mínimo 2 subplots
        
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        ax_idx = 0
        
        # Plot: Rewards
        if self.rewards:
            ax = axes[ax_idx]
            ax.plot(episodes_arr, self.rewards, 'b-', alpha=0.7, label="Reward")
            
            # Agregar línea de moving average
            if len(self.rewards) > 5:
                window = min(10, len(self.rewards) // 2)
                ma = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
                ma_episodes = episodes_arr[window-1:]
                ax.plot(ma_episodes, ma, 'r-', linewidth=2, label=f"MA({window})")
            
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Reward")
            ax.set_title("Reward por Episodio")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax_idx += 1
        
        # Plot: Cumulative Rewards
        if self.cumulative_rewards:
            ax = axes[ax_idx]
            ax.plot(episodes_arr[:len(self.cumulative_rewards)], 
                   self.cumulative_rewards, 'g-', alpha=0.7)
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Cumulative Reward")
            ax.set_title("Reward Acumulativo")
            ax.grid(True, alpha=0.3)
            ax_idx += 1
        
        # Plot: Loss
        if self.losses:
            ax = axes[ax_idx]
            ax.plot(episodes_arr[:len(self.losses)], self.losses, 'r-', alpha=0.7)
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Loss")
            ax.set_title("Policy Loss")
            ax.grid(True, alpha=0.3)
            ax_idx += 1
        
        # Plot: Gradient Norm
        if self.grad_norms:
            ax = axes[ax_idx]
            ax.plot(episodes_arr[:len(self.grad_norms)], self.grad_norms, 'purple', alpha=0.7)
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Gradient Norm")
            ax.set_title("Gradient Norm (Clipping)")
            ax.grid(True, alpha=0.3)
            ax_idx += 1
        
        # Plot: Rates
        if self.rates:
            ax = axes[ax_idx]
            ax.plot(episodes_arr[:len(self.rates)], self.rates, 'orange', alpha=0.7)
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Rate (bits/s)")
            ax.set_title("Average Rate por Episodio")
            ax.grid(True, alpha=0.3)
            ax_idx += 1
        
        # Plot: Entropy
        if self.entropies:
            ax = axes[ax_idx]
            ax.plot(episodes_arr[:len(self.entropies)], self.entropies, 'brown', alpha=0.7)
            ax.set_xlabel("Episodio")
            ax.set_ylabel("Entropy")
            ax.set_title("Policy Entropy (Exploración)")
            ax.grid(True, alpha=0.3)
            ax_idx += 1
        
        # Plots personalizados
        for metric_name, metric_vals in self.custom_metrics.items():
            if ax_idx >= len(axes):
                break
            ax = axes[ax_idx]
            ax.plot(episodes_arr[:len(metric_vals)], metric_vals, 'g-', alpha=0.7)
            ax.set_xlabel("Episodio")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{metric_name}")
            ax.grid(True, alpha=0.3)
            ax_idx += 1
        
        # Limpiar subplots no usados
        for i in range(ax_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        # Guardar figura
        plot_path = self.output_dir / "training_metrics.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Plot guardado: {plot_path}")
    
    def save_metrics_csv(self, csv_path: Optional[str] = None) -> None:
        """Guarda historial de métricas en CSV."""
        if csv_path is None:
            csv_path = self.output_dir / "training_metrics.csv"
        else:
            csv_path = Path(csv_path)
        
        import pandas as pd
        
        data = {"episode": self.episodes}
        
        if self.rewards:
            data["reward"] = self.rewards
        if self.cumulative_rewards:
            data["cumulative_reward"] = self.cumulative_rewards + [None] * (len(self.episodes) - len(self.cumulative_rewards))
        if self.losses:
            data["loss"] = self.losses + [None] * (len(self.episodes) - len(self.losses))
        if self.grad_norms:
            data["grad_norm"] = self.grad_norms + [None] * (len(self.episodes) - len(self.grad_norms))
        if self.rates:
            data["rate"] = self.rates + [None] * (len(self.episodes) - len(self.rates))
        if self.entropies:
            data["entropy"] = self.entropies + [None] * (len(self.episodes) - len(self.entropies))
        
        # Agregar métricas personalizadas
        for key, vals in self.custom_metrics.items():
            data[key] = vals + [None] * (len(self.episodes) - len(vals))
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"  📊 CSV guardado: {csv_path}")


class SB3TrainingCallback:
    """Callback para Stable-Baselines3 PPO con visualización automática."""
    
    def __init__(self, plotter: TrainingPlotter):
        """
        Inicializa el callback.
        
        Args:
            plotter: Instancia de TrainingPlotter
        """
        self.plotter = plotter
        self.episode_rewards = []
        self.episode_count = 0
    
    def on_step(self, step: int, reward: float) -> None:
        """Llamado cada step. SB3 usa callbacks más sofisticados, pero esto es simplicidad."""
        self.episode_rewards.append(reward)
    
    def on_episode_end(self, episode: int, total_reward: float, length: int) -> None:
        """Llamado al final de cada episodio."""
        self.plotter.log_episode(
            episode=episode,
            reward=total_reward,
            rate=total_reward / length if length > 0 else 0,
        )
        self.plotter.plot_if_ready()


if __name__ == "__main__":
    # Test básico
    plotter = TrainingPlotter(output_dir="/tmp/test_plots", plot_every=5)
    
    for ep in range(50):
        reward = np.sin(ep / 10) * 100 + np.random.normal(0, 10)
        loss = 1.0 / (1 + ep / 10) + np.random.normal(0, 0.05)
        grad_norm = np.exp(-ep / 20) + np.random.normal(0, 0.1)
        
        plotter.log_episode(
            episode=ep,
            reward=reward,
            loss=loss,
            grad_norm=grad_norm,
        )
        plotter.plot_if_ready()
    
    plotter.plot()
    plotter.save_metrics_csv()
    print("✅ Test completado")
