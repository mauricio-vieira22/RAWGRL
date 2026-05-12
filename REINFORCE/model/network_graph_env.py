r"""
network_graph_env.py – Entorno Gymnasium: Proceso de Decisión de Markov Parcialmente Observable (POMDP).

Implementa la formulación matemática estricta del MDP del problema de optimización 
de recursos en redes inalámbricas definido en la Tesis (Sección: "Reinforcement Learning").

Definición del MDP (Tupla $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$)
-------------------------------------------------------------------------------------------------------
Estado del Entorno $\mathcal{S}$:
    El estado verdadero del sistema en el timestep macro $\tau$ es:
    $$ s_\tau = (\mathbf{H}_{\tau T}, \mathbf{X}_{\tau T}, \mathbf{A}_{\tau T}, \mathbf{\epsilon}_{\tau T}, \delta_{\tau T}) $$
    donde $\mathbf{H}$ son ganancias, $\mathbf{X}$ las configuraciones (Canal/Potencia), $\mathbf{A}$ asociaciones,
    y $\epsilon, \delta$ son variables estocásticas ocultas del proceso de Poisson/Exponencial.

Acciones $\mathcal{A}$:
    El agente dictamina la configuración de la red cada $T$ micro-slots:
    $$ a_\tau = \mathbf{X}_{\tau T} = \{(c_{n, \tau T}, P_{n, \tau T})\}_{n=1}^N $$

Transición $\mathcal{P}(s'|s,a)$:
    Determinada internamente por la simulación estocástica de arribos y el modelo Sticky Client.

Función de Recompensa $\mathcal{R}(s,a)$:
    Suma acumulada de la eficiencia espectral durante el período de decisión $T$:
    $$ R_\tau(s_\tau, a_\tau) = \sum_{s=0}^{T-1} F(\mathbf{H}_{\tau T + s}, \mathbf{X}_{\tau T}, \mathbf{A}_{\tau T + s}) $$

Observabilidad Parcial (POMDP) $\mathcal{O}$:
    El entorno envuelve el estado verdadero $s_\tau$ en un objeto `HeteroData` de PyTorch Geometric.
    Inyectamos $\epsilon$ y $\delta$ como señales adicionales en este grafo para mejorar
    la aproximación markoviana del agente Actor-Critic, rompiendo la invisibilidad parcial.
"""

from __future__ import annotations
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from simulation.arrival_departure_model import ArrivalDepartureModel
from simulation.wifi_physics import (
    crear_grilla, convertir_grilla_a_tensor, obtener_grilla_RSSIs,
    asignaciones_AP, calcular_sinr, calcular_rate,
    actualizar_avg_rate, calcular_reward,
    UMBRAL_5G, UMBRAL_CONEXION, SIGMA_DBM,
    STICKY_FULL, STICKY_STD, STICKY_LITE, StickyMode,
)
from simulation.graph_builder import construir_grafo_timestep


class NetworkGraphEnv(gym.Env):
    r"""
    Entorno Computacional Simulado de Red WiFi (Wrapper Gymnasium).

    Orquesta las interacciones discretas entre el Agente (GNN) y la Física:
      1. `ArrivalDepartureModel` $\to$ Motor estocástico de línea temporal ($t_u, \Delta_u, H_u$).
      2. `wifi_physics`          $\to$ Motor electromagnético ($H_t, RSSI_t, A_t, SINR_t, r_t, R_\tau$).
      3. `graph_builder`         $\to$ Ensamble topológico del Espacio Observable (`HeteroData`).

    Parameters
    ----------
    distributions : list[Distribution]
    n_aps         : int — N: número de APs del edificio
    arrival_rate  : float — λ del proceso Poisson
    mean_duration : float — µ de la distribución Exponencial
    total_timesteps : int — horizonte T del episodio
    available_channels : list[int] — {c_1,...,c_C}, ej. [1, 6, 11]
    tx_powers_dbm : list[float] — {P_1,...,P_L} [dBm]
    sigma_dbm     : float — piso de ruido σ [dBm]
    umbral_5g     : float — θ_{5G} [dBm]
    umbral_conexion : float — θ_{conn} [dBm]
    sticky_mode   : str — 'full' | 'sticky' | 'lite'  (ver LaTeX §Dinámica)
    random_seed   : int | None
    device        : torch.device | str
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        distributions,
        n_aps:              int,
        arrival_rate:       float           = 2.0,
        mean_duration:      float           = 10.0,
        total_timesteps:    int             = 100,
        decision_period:    int             = 1,
        available_channels: list[int] | None = None,
        tx_powers_dbm:      list[float] | None = None,
        sigma_dbm:          float           = SIGMA_DBM,
        umbral_5g:          float           = UMBRAL_5G,
        umbral_conexion:    float           = UMBRAL_CONEXION,
        sticky_mode:        StickyMode      = STICKY_STD,
        # Retrocompatibilidad: sticky=True → STICKY_STD, sticky=False → STICKY_LITE
        sticky:             bool | None     = None,
        random_seed:        int | None      = None,
        device:             torch.device | str = "cpu",
    ):
        super().__init__()

        # Retrocompatibilidad booleana
        if sticky is not None:
            sticky_mode = STICKY_STD if sticky else STICKY_LITE

        self.distributions   = distributions
        self.n_aps           = n_aps
        self.arrival_rate    = arrival_rate
        self.mean_duration   = mean_duration
        self.total_timesteps = total_timesteps
        # Cada decisión RL mantiene X_t fijo por `decision_period` slots (LaTeX §RL).
        # decision_period=1 reproduce el comportamiento legacy (acción por slot).
        self.decision_period = max(int(decision_period), 1)
        self.sigma_dbm       = sigma_dbm
        self.umbral_5g       = umbral_5g
        self.umbral_conexion = umbral_conexion
        self.sticky_mode     = sticky_mode
        self.random_seed     = random_seed
        self.device          = torch.device(device)

        self.available_channels = torch.tensor(
            available_channels or [1, 6, 11],
            dtype=torch.long, device=self.device,
        )
        self.tx_powers_dbm = torch.tensor(
            tx_powers_dbm or [20.0, 17.0, 14.0, 11.0, 8.0],
            dtype=torch.float32, device=self.device,
        )
        self.n_channels = len(self.available_channels)
        self.n_powers   = len(self.tx_powers_dbm)

        # Espacio de acciones: X_t = {(c_n, P_n)}_{n=1}^N
        self.action_space = spaces.MultiDiscrete(
            [self.n_channels, self.n_powers] * self.n_aps
        )
        self.observation_space = spaces.Discrete(1)  # HeteroData, placeholder

        # Modelo de tráfico M/G/∞
        self._adm = ArrivalDepartureModel(
            distribuciones=self.distributions,
            arrival_rate=self.arrival_rate,
            mean_duration=self.mean_duration,
            total_timesteps=self.total_timesteps,
            random_seed=self.random_seed,
        )

        # Estado interno del MDP
        self.current_step     = 0
        self._delta_t: int    = 0               # δ_t — POMDP hidden var
        self._eventos         = []
        self._grilla_gan:     torch.Tensor | None = None
        self._grilla_RSSI:    torch.Tensor | None = None
        self._asignaciones:   torch.Tensor | None = None
        self._avg_rates:      torch.Tensor | None = None
        self._estado_sticky                        = None
        self._decisiones_APs: torch.Tensor | None = None
        self._n_rate_obs      = 0

    # ── reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options=None):
        """
        Inicializa un nuevo episodio.

        Ciclo
        -----
        1. Simular línea temporal (Monte Carlo prospectivo) → eventos, δ_t
        2. Construir H_t como tensor (n_cli, T, N, 2)
        3. Decisión inicial aleatoria X_0
        4. Calcular RSSI_0 y A_0 según sticky_mode
        5. Devolver obs_0 = HeteroData del estado inicial
        """
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        rng = np.random.default_rng(seed)

        # 1. Generar eventos y cache δ_t
        self._adm.random_seed = seed
        self._eventos = self._adm.simulate_all_events()

        if len(self._eventos) == 0:
            self._grilla_gan = torch.zeros(
                (0, self.total_timesteps, self.n_aps, 2),
                dtype=torch.float32, device=self.device,
            )
        else:
            grilla_obj    = crear_grilla(self._eventos, self.distributions, self.total_timesteps, rng)
            self._grilla_gan = convertir_grilla_a_tensor(grilla_obj).to(self.device)

        # 2. Decisiones iniciales aleatorias
        ch_idx  = torch.randint(0, self.n_channels, (self.n_aps,), device=self.device)
        pwr_idx = torch.randint(0, self.n_powers,   (self.n_aps,), device=self.device)
        self._decisiones_APs = torch.stack([ch_idx, pwr_idx], dim=1)

        # 3. RSSI_0 y A_0
        self._actualizar_RSSI()
        n_cli = self._grilla_RSSI.shape[0]
        self._asignaciones = torch.full(
            (n_cli, self.total_timesteps, 3), float('nan'), device=self.device
        )
        asig_t0, self._estado_sticky = asignaciones_AP(
            self._grilla_RSSI[:, 0:1, :, :],
            umbral_5g=self.umbral_5g,
            umbral_conexion=self.umbral_conexion,
            sticky_mode=self.sticky_mode,
        )
        self._asignaciones[:, 0:1, :] = asig_t0

        # 4. Inicializar avg_rates y contadores
        n_ev = len(self._eventos)
        self._avg_rates   = torch.zeros(n_ev, dtype=torch.float32, device=self.device)
        self._n_rate_obs  = 0
        self.current_step = 0
        self._delta_t     = self._adm.get_delta_t(0)

        return self._build_obs(), {}

    # ── step ───────────────────────────────────────────────────────────────────

    def step(self, action):
        """
        Ejecuta un paso del MDP.

        Ciclo
        -----
        Un step de Gym corresponde a un tiempo de decisión τ. La acción se aplica
        y se mantiene fija por `decision_period` slots (t = τT, …, τT+T-1).

        1. Parsear acción a_τ → X_{τT} = (c_n, P_n)
        2. Mantener X fijo y simular slots intermedios:
           - actualizar A_t según sticky_mode
           - calcular SINR_t → r_t
        3. Reward agregado: R_τ = Σ_{s=0}^{T-1} F(H_{τT+s}, X_{τT}, A_{τT+s})
        4. Avanzar δ_t y devolver (obs_{τ+1}, R_τ, done, False, info)
        """
        t_start = self.current_step
        if t_start >= self.total_timesteps:
            # Episodio ya finalizado: estado absorbente.
            info = {"total_rate": 0.0, "mean_rate": 0.0, "n_active_clients": 0, "timestep": t_start}
            return self._build_obs(), 0.0, True, False, info

        # 1. Aplicar acción X_{τT}
        self._decisiones_APs = self._parse_action(action)

        # 2. RSSI (constante durante este período de decisión)
        self._actualizar_RSSI()
        canales_por_AP = self.available_channels[self._decisiones_APs[:, 0]]

        # 3. Simular slots intermedios y agregar reward
        t_end = min(t_start + self.decision_period, self.total_timesteps)
        reward_sum = 0.0
        last_rate_ahora = None

        for t in range(t_start, t_end):
            asig_t, self._estado_sticky = asignaciones_AP(
                self._grilla_RSSI[:, t:t+1, :, :],
                umbral_5g=self.umbral_5g,
                umbral_conexion=self.umbral_conexion,
                sticky_mode=self.sticky_mode,
                estado_previo=self._estado_sticky,
            )
            self._asignaciones[:, t:t+1, :] = asig_t

            sinr_t = calcular_sinr(
                self._grilla_RSSI[:, t:t+1, :, :],
                self._asignaciones[:, t:t+1, :],
                canales_por_AP,
                self.sigma_dbm,
            )
            rate_t = calcular_rate(sinr_t)
            rate_ahora = rate_t[:, 0]
            last_rate_ahora = rate_ahora

            self._n_rate_obs += 1
            self._avg_rates = actualizar_avg_rate(self._avg_rates, rate_ahora, self._n_rate_obs)
            reward_sum += calcular_reward(rate_ahora)

        self.current_step = t_end
        terminated = (self.current_step >= self.total_timesteps)
        self._delta_t = self._adm.get_delta_t(self.current_step) if not terminated else 0

        # 4. Info enriquecido (variables del estado Markoviano completo)
        t_last = t_end - 1
        n_activos = (~torch.isnan(self._asignaciones[:, t_last, 0])).sum().item() if t_last >= 0 else 0
        eps_list = self._adm.get_epsilon_t(t_last) if t_last >= 0 else []
        eps_mean = float(np.mean(eps_list)) if eps_list else 0.0
        mean_rate = (
            last_rate_ahora.nanmean().item()
            if (last_rate_ahora is not None and n_activos > 0)
            else 0.0
        )

        info = {
            # Variables observadas por el agente
            "total_rate":        reward_sum,
            "mean_rate":         mean_rate,
            "n_active_clients":  n_activos,
            "timestep":          t_last,
            "timestep_start":    t_start,
            "timestep_end":      t_last,
            "decision_period":   self.decision_period,
            # Variables del estado Markoviano completo (no observadas por el agente)
            "delta_t":           self._delta_t,       # δ_t: slots hasta próximo arribo
            "epsilon_t_mean":    eps_mean,             # ε_t: slots restantes promedio
            "sticky_mode":       self.sticky_mode,
        }

        return self._build_obs(), float(reward_sum), terminated, False, info

    # ── helpers ────────────────────────────────────────────────────────────────

    def _actualizar_RSSI(self):
        """RSSI_t = H_t + P_n  (en dominio logarítmico [dBm])."""
        potencias = self.tx_powers_dbm[self._decisiones_APs[:, 1]]
        self._grilla_RSSI = obtener_grilla_RSSIs(self._grilla_gan, potencias)

    def _build_obs(self) -> 'HeteroData':
        """Construye el HeteroData que observa el agente."""
        t = min(self.current_step, self.total_timesteps - 1)
        return construir_grafo_timestep(
            t=t,
            asignaciones=self._asignaciones,
            grilla_RSSI=self._grilla_RSSI,
            decisiones_APs=self._decisiones_APs,
            avg_rates=self._avg_rates,
            eventos=self._eventos,
            available_channels=self.available_channels,
            tx_powers_dbm=self.tx_powers_dbm,
            max_timesteps=self.total_timesteps,
            delta_t=self._delta_t,
        )

    def _parse_action(self, action) -> torch.Tensor:
        """Convierte la acción del optimizador al formato interno (N, 2)."""
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action.astype(np.int64))
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        action = action.long().to(self.device)
        if action.dim() == 1:
            action = action.reshape(self.n_aps, 2)
        return action
