"""
arrival_departure_model.py – Modelo Estocástico de Tráfico WiFi.

Implementa la dinámica de llegadas y partidas del modelo formal (LaTeX):

    Proceso de Arribo  : Poisson con tasa λ (arrival_rate)
    Duración de Sesión : Exponencial con media μ (mean_duration)

Variables del Modelo (LaTeX → código)
--------------------------------------
  t_u          → ClientEvent.arrival_time
  Δ_u          → ClientEvent.duration
  ε_{t,u}      → departure_time - t     (slots restantes del usuario u en t)
  δ_t          → get_delta_t(t)         (slots hasta el próximo arribo desde t)

Observabilidad Parcial (POMDP)
-------------------------------
El proceso es Markoviano dado (H_t, X_t, A_t, ε_t, δ_t). Sin embargo,
el agente RL no observa ε_t ni δ_t directamente en la observación principal
(son variables del estado oculto). Se exponen en el dict `info` del entorno
para análisis de ablación y para el paper (§ Reinforcement Learning).
"""

from __future__ import annotations
import numpy as np
from data.clases import Distribution, ClientEvent


class ArrivalDepartureModel:
    """
    Simulador prospectivo de eventos de conexión/desconexión de usuarios WiFi.

    Genera la línea temporal completa del episodio antes de que el agente RL
    comience a interactuar (Monte Carlo prospectivo), lo cual permite:
      - Vectorización completa de H_t sobre todo el horizonte T.
      - Cálculo eficiente de ε_{t,u} y δ_t sin inferencia adicional.

    Parameters
    ----------
    distribuciones : list[Distribution]
        Pool de perfiles de ganancia de clientes reales medidos en campo.
        Cada Distribution contiene bloques de ganancias H_{u,n} ∈ ℝ^N.
    arrival_rate : float
        Parámetro λ del proceso Poisson. Esperanza de llegadas por timestep.
    mean_duration : float
        Parámetro µ de la distribución Exponencial. Duración media de sesión.
    total_timesteps : int
        Horizonte temporal del episodio T (límite superior de t).
    random_seed : int | None
        Semilla para reproducibilidad de validación y comparación entre versiones.
    """

    def __init__(
        self,
        distribuciones: list[Distribution],
        arrival_rate:    float = 2.0,
        mean_duration:   float = 10.0,
        total_timesteps: int   = 100,
        random_seed:     int | None = None,
    ):
        self.distribuciones   = distribuciones
        self.arrival_rate     = arrival_rate
        self.mean_duration    = mean_duration
        self.total_timesteps  = total_timesteps
        self.random_seed      = random_seed

        self.available_clients = list(range(len(self.distribuciones)))
        self.events: list[ClientEvent] = []

        # Cache de δ_t para acceso O(1) por timestep
        self._delta_t_array: np.ndarray | None = None

    def simulate_all_events(self) -> list[ClientEvent]:
        """
        Genera prospectivamente TODA la línea temporal del episodio.

        Proceso
        -------
        Para cada t ∈ {0, …, T-1}:
          1. Sortear n_arrivals ~ Poisson(λ)
          2. Para cada arribo: sortear Δ_u ~ Exponential(μ), truncar a [t, T]
          3. Asignar perfil de ganancia H_u desde el pool de distribuciones

        Returns
        -------
        list[ClientEvent]
            Línea de tiempo inmutable del episodio. Cada evento contiene
            t_u, Δ_u, distribution_idx. Se usa para construir H_t y ε_t.
        """
        rng = np.random.default_rng(self.random_seed)
        self.events = []

        for t in range(self.total_timesteps):
            n_arrivals = rng.poisson(self.arrival_rate)

            for _ in range(n_arrivals):
                duration   = max(1, int(np.round(rng.exponential(self.mean_duration))))
                departure  = min(t + duration, self.total_timesteps)
                dist_idx   = rng.choice(self.available_clients)

                self.events.append(ClientEvent(
                    distribution_idx = int(dist_idx),
                    arrival_time     = t,
                    departure_time   = departure,
                    duration         = departure - t,
                ))

        # Construir cache de δ_t tras generar todos los eventos
        self._build_delta_t_array()

        return self.events

    # ── Variables del Estado Markoviano ───────────────────────────────────────

    def get_active_clients(self, timestep: int) -> list[ClientEvent]:
        """
        Clientes activos en t: {u : t_u ≤ t < t_u + Δ_u}.

        Returns
        -------
        list[ClientEvent]
            Subconjunto de self.events activo en el timestep dado.
        """
        return [e for e in self.events if e.arrival_time <= timestep < e.departure_time]

    def get_epsilon_t(self, timestep: int) -> list[int]:
        """
        Slots restantes ε_{t,u} para cada cliente activo en t.

        Formalmente: ε_{t,u} = (t_u + Δ_u) - t  para todo u activo.

        Nota de Observabilidad
        ----------------------
        En el sistema real, el AP no conoce cuándo se desconectará cada cliente.
        Esta variable forma parte del estado oculto del POMDP; se incluye como
        feature de nodo cliente en el grafo (aproximación del agente al estado).

        Returns
        -------
        list[int]  — Un valor por cliente activo, en el mismo orden que get_active_clients().
        """
        return [e.departure_time - timestep for e in self.get_active_clients(timestep)]

    def get_delta_t(self, timestep: int) -> int:
        """
        Tiempo hasta el próximo arribo desde el timestep actual: δ_t.

        Formalmente:
            δ_t = min{ t' > t : ∃ u con t_u = t' } − t
        Si no hay más arribos, δ_t = 0 (horizonte agotado).

        Nota de Observabilidad
        ----------------------
        δ_t no es observable por el agente en un sistema real (los APs no saben
        cuándo llegará el próximo usuario). Se expone en `info` del entorno para
        estudios de ablación sobre el impacto de la observabilidad parcial.

        Returns
        -------
        int — Slots hasta el próximo arribo. 0 si no hay más arribos en [t+1, T).
        """
        if self._delta_t_array is not None:
            idx = min(timestep, len(self._delta_t_array) - 1)
            return int(self._delta_t_array[idx])
        # Fallback si se llama antes de simulate_all_events
        nexts = [e.arrival_time for e in self.events if e.arrival_time > timestep]
        return (min(nexts) - timestep) if nexts else 0

    # ── Cache Interno ─────────────────────────────────────────────────────────

    def _build_delta_t_array(self):
        """
        Pre-computa δ_t para todo t ∈ {0, …, T-1} en O(U_total + T).

        La entrada en la posición t es el número de slots hasta el siguiente
        arribo: 0 si hay un arribo en t+1, 1 si es en t+2, etc.
        Si no hay más arribos, el valor es 0.
        """
        T = self.total_timesteps
        delta = np.zeros(T, dtype=np.int32)

        # Instantes de arribo únicos, ordenados
        arrivals = sorted({e.arrival_time for e in self.events})
        if not arrivals:
            self._delta_t_array = delta
            return

        # Para cada t, δ_t = (próximo arribo estrictamente mayor que t) - t, o 0 si no existe.
        j = 0
        for t in range(T):
            while j < len(arrivals) and arrivals[j] <= t:
                j += 1
            delta[t] = (arrivals[j] - t) if j < len(arrivals) else 0

        self._delta_t_array = delta
