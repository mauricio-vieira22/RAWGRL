r"""
wifi_physics.py – Funciones Físicas y Ecuaciones de Propagación WiFi.

Implementa el modelo matemático formal definido en el paper de NetROML
(Sección: "Modelo Wi-Fi con Sticky Clients" y "Formulación del MDP"). 
Las variables siguen estrictamente la notación del modelo teórico.

Notación Matemática (POMDP)
--------------------------
  $H_t  \in \mathbb{R}^{U_t \times N \times 2}$ : Matriz de ganancias (Banda 2.4G y 5G). [dB]
  $RSSI_{u,n} = H_{u,n} + P_n$                 : Potencia de recepción proyectada. [dBm]
  $A_t  \in \{0,1\}^{U_t \times N}$            : Matriz de asignaciones. $a_{u,n}=1$ si $u$ conecta a $n$.
  $\epsilon_t  \in \mathbb{N}^{U_t}$           : departure_time - t (Slots restantes por usuario).
  $\delta_t  \in \mathbb{N}$                   : Slots hasta el próximo arribo al sistema.

Modelos de Transición de Handover (Sticky Client)
-------------------------------------------------
En la dinámica real, los clientes no saltan de AP en cada micro-slot.
  STICKY_FULL  ('full')   : $a_{u,t+1} \leftarrow a_{u,t}$. Nunca cambia tras el nacimiento.
  STICKY_STD   ('sticky') : Cambia si y solo si $RSSI_{u, a_u} \leq \theta_{conn}$.
  STICKY_LITE  ('lite')   : Handover instantáneo si existe un $n^*$ tal que $RSSI_{u, n^*} > RSSI_{u, a_u}$.

Flujo de Ecuaciones
-------------------
  1. `crear_grilla`           → Genera $H_t$ para toda la historia del episodio.
  2. `obtener_grilla_RSSIs`   → $RSSI_{u,n} = H_{u,n} + P_n$.
  3. `asignaciones_AP`        → Resuelve el problema de asociación basado en Sticky mode.
  4. `calcular_sinr`          → Calcula Ecuación de Interferencia Co-Canal (CCI).
  5. `calcular_rate`          → Evalúa Ecuación de Shannon-Hartley.
  6. `calcular_reward`        → $R_\tau = \sum_{u} r_{u,\tau}$.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Literal

# ── Constantes Físicas del Entorno ────────────────────────────────────────────
UMBRAL_5G:       float = -70.0   # Cota para preferir banda 5 GHz  [dBm]
UMBRAL_CONEXION: float = -85.0   # Sensibilidad mínima del chip WiFi [dBm]
SIGMA_DBM:       float = -90.0   # Piso de ruido térmico estándar   [dBm]

# ── Constantes de Modo Sticky ─────────────────────────────────────────────────
STICKY_FULL = 'full'    # Full Sticky   : nunca cambia de AP tras el primer handshake
STICKY_STD  = 'sticky'  # Sticky        : cambia sólo si la señal cae bajo θ_conn
STICKY_LITE = 'lite'    # Sticky "Lite" : cambia si cualquier otro AP es mejor

StickyMode = Literal['full', 'sticky', 'lite']


# ── 1. Inicialización de Tensores Base ────────────────────────────────────────

def crear_grilla(eventos, distribuciones, total_timesteps: int, rng=None) -> np.ndarray:
    """
    Materializa la línea temporal de ganancias H_t en memoria matricial
    heterogénea (array de objetos Block).

    Cada fila u corresponde a un cliente; cada columna t a un slot.
    El valor es None cuando el cliente u no está activo en t.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(eventos)
    grilla = np.full((n, total_timesteps), None, dtype=object)
    for fila, ev in enumerate(eventos):
        bloques = distribuciones[ev.distribution_idx].blocks
        sorteados = rng.choice(bloques, size=ev.duration, replace=True)
        grilla[fila, ev.arrival_time:ev.departure_time] = sorteados
    return grilla


def convertir_grilla_a_tensor(grilla: np.ndarray) -> torch.Tensor:
    """
    Convierte la grilla de objetos Block a Tensor float32.

    Garantiza que la dimensión de APs (N) sea consistente en toda la grilla.
    Si un bloque tiene una dimensión distinta, se ignoran sus valores para
    evitar errores de broadcasting, manteniendo el valor NaN.
    """
    primer = next((b for b in grilla.flat if b is not None), None)
    if primer is None:
        return torch.full((grilla.shape[0], grilla.shape[1], 0, 2), float('nan'))
    
    cant_APs = len(primer.datos)
    n_ev, T = grilla.shape
    out = torch.full((n_ev, T, cant_APs, 2), float('nan'), dtype=torch.float32)
    
    for (i, j), bloque in np.ndenumerate(grilla):
        if bloque is not None:
            # Validación de seguridad: el bloque debe tener la dimensión esperada
            if len(bloque.datos) == cant_APs:
                vals = torch.tensor(
                    bloque.datos[['G_2_4', 'G_5']].values.astype(np.float32),
                    dtype=torch.float32,
                )
                out[i, j] = vals
                
    # Normalizar: -inf → NaN (compatibilidad con .joblib legacy)
    out = torch.nan_to_num(out, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
    return out


# ── 2. Cálculo de RSSI ────────────────────────────────────────────────────────

def obtener_grilla_RSSIs(
    grilla_ganancias: torch.Tensor,
    potencias_APs_dBm: torch.Tensor,
) -> torch.Tensor:
    """
    Modelo de propagación de enlace (log-lineal):

        RSSI_{u,n,t} [dBm] = H_{u,n,t} [dB] + P_{n,t} [dBm]

    Parameters
    ----------
    grilla_ganancias  : (n_cli, T, N, 2)  — ganancias espaciales H_t  [dB]
    potencias_APs_dBm : (N,)              — decisión X_t del agente    [dBm]

    Returns
    -------
    Tensor (n_cli, T, N, 2)  — RSSI_t                                  [dBm]
    """
    p = potencias_APs_dBm.view(1, 1, -1, 1).to(grilla_ganancias.dtype)
    return grilla_ganancias + p


# ── 3. Motor de Handover (Tres modos Sticky) ──────────────────────────────────

def asignaciones_AP(
    grilla_RSSI: torch.Tensor,
    umbral_5g: float = UMBRAL_5G,
    umbral_conexion: float = UMBRAL_CONEXION,
    sticky_mode: StickyMode = STICKY_STD,
    estado_previo=None,
    # Retrocompatibilidad: sticky=True → STICKY_STD, sticky=False → STICKY_LITE
    sticky: bool | None = None,
):
    """
    Implementa los tres modelos de conexión del LaTeX (§ Dinámica).

    Modelos
    -------
    'full'   (Full Sticky)  : a_{u,t+1} ← a_{u,t} tras nacimiento.  Sin roaming.
    'sticky' (Sticky)       : Cambia de AP si RSSI_cur ≤ θ_conn.    [por defecto]
    'lite'   (Sticky Lite)  : Cambia de AP si algún otro supera al actual.

    Parameters
    ----------
    grilla_RSSI   : (n_cli, T, N, 2)  — RSSI_t [dBm]
    umbral_5g     : float             — θ_{5G}, umbral para preferir 5 GHz
    umbral_conexion: float            — θ_{conn}, sensibilidad mínima
    sticky_mode   : str               — 'full' | 'sticky' | 'lite'
    estado_previo : tuple | None      — (cur_ap, cur_band) del timestep anterior

    Returns
    -------
    asignaciones  : Tensor (n_cli, T, 3)  — [ap_idx, band_idx, RSSI_dBm], NaN=inactivo
    ultimo_estado : tuple(Tensor, Tensor) — estado sticky para el siguiente step
    """
    # Retrocompatibilidad booleana
    if sticky is not None:
        sticky_mode = STICKY_STD if sticky else STICKY_LITE

    n_cli, T, n_APs, _ = grilla_RSSI.shape
    dev = grilla_RSSI.device

    # ── AP ideal por timestep: preferir 5 GHz si supera θ_{5G} ──────────────
    rssi_5g = grilla_RSSI[:, :, :, 1]
    # BUGFIX: torch.max propaga NaNs. Usamos un valor centinela muy bajo para el cálculo.
    rssi_5g_safe = rssi_5g.nan_to_num(-200.0)
    max_5g, argmax_5g = rssi_5g_safe.max(dim=2)

    flat = grilla_RSSI.reshape(n_cli, T, n_APs * 2)
    flat_safe = flat.nan_to_num(-200.0)
    max_all, argmax_flat = flat_safe.max(dim=2)
    
    argmax_all_ap   = argmax_flat // 2
    argmax_all_band = argmax_flat % 2

    # Una posición es válida si existe al menos un AP por encima del umbral de conexión
    mask_active = (max_all >= umbral_conexion)

    argmax_5g_f       = argmax_5g.float().masked_fill(~mask_active, float('nan'))
    argmax_all_ap_f   = argmax_all_ap.float().masked_fill(~mask_active, float('nan'))
    argmax_all_band_f = argmax_all_band.float().masked_fill(~mask_active, float('nan'))

    cumple_5g  = (max_5g >= umbral_5g) & mask_active
    ideal_ap   = torch.where(cumple_5g, argmax_5g_f,   argmax_all_ap_f)
    ideal_band = torch.where(cumple_5g, torch.ones_like(argmax_5g_f), argmax_all_band_f)
    ideal_rssi = torch.where(cumple_5g, max_5g, max_all).masked_fill(~mask_active, float('nan'))

    # ── Modo LITE: siempre el mejor AP disponible ─────────────────────────────
    if sticky_mode == STICKY_LITE:
        asig = torch.stack([ideal_ap, ideal_band, ideal_rssi], dim=-1)
        return asig, (ideal_ap[:, -1].clone(), ideal_band[:, -1].clone())

    # ── Modos FULL y STICKY: requieren estado ─────────────────────────────────
    assigned_ap   = torch.full((n_cli, T), float('nan'), device=dev)
    assigned_band = torch.full((n_cli, T), float('nan'), device=dev)
    assigned_rssi = torch.full((n_cli, T), float('nan'), device=dev)

    if estado_previo is not None:
        cur_ap, cur_band = estado_previo[0].clone(), estado_previo[1].clone()
    else:
        cur_ap   = torch.full((n_cli,), float('nan'), device=dev)
        cur_band = torch.full((n_cli,), float('nan'), device=dev)

    idx_cli = torch.arange(n_cli, device=dev)

    for t in range(T):
        active_now = ~torch.isnan(ideal_rssi[:, t])
        ap_safe    = cur_ap.nan_to_num(0).long()
        band_safe  = cur_band.nan_to_num(0).long()
        rssi_cur   = grilla_RSSI[idx_cli, t, ap_safe, band_safe]
        rssi_cur[torch.isnan(cur_ap)] = float('nan')

        was_connected = ~torch.isnan(cur_ap)

        if sticky_mode == STICKY_FULL:
            # Full Sticky: solo cambia AP en el momento del nacimiento (primer slot activo)
            # Una vez conectado, NUNCA cambia → change_mask solo para recién llegados
            change_mask = ~was_connected & active_now
        else:
            # Sticky estándar: cambia si RSSI < umbral_conexion O si es nuevo
            change_mask = (was_connected & (rssi_cur <= umbral_conexion)) | (~was_connected & active_now)

        cur_ap   = torch.where(change_mask, ideal_ap[:, t],   cur_ap)
        cur_band = torch.where(change_mask, ideal_band[:, t], cur_band)
        cur_rssi = torch.where(change_mask, ideal_rssi[:, t],
                               rssi_cur.nan_to_num(nan=float('nan')))

        cur_ap[~active_now]   = float('nan')
        cur_band[~active_now] = float('nan')
        cur_rssi[~active_now] = float('nan')

        assigned_ap[:, t]   = cur_ap
        assigned_band[:, t] = cur_band
        assigned_rssi[:, t] = cur_rssi

    asig = torch.stack([assigned_ap, assigned_band, assigned_rssi], dim=-1)
    return asig, (cur_ap.clone(), cur_band.clone())


# ── 4. Conversión Log → Lineal ────────────────────────────────────────────────

def db_to_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Conversión dBm → mW. 
    
    Maneja NaNs y valores extremos para evitar inestabilidad numérica.
    """
    # Filtramos nans para evitar warnings en pow
    mask_nan = torch.isnan(x)
    x_safe = x.nan_to_num(-200.0) # RSSI extremadamente bajo -> ~0 mW
    
    out = torch.pow(10.0, x_safe / 10.0)
    out[mask_nan] = float('nan')
    return out


# ── 5. SINR con Interferencia Co-Canal (CCI) ──────────────────────────────────

def calcular_sinr(
    grilla_RSSI: torch.Tensor,
    asignaciones: torch.Tensor,
    canales_por_AP: torch.Tensor,
    sigma_dbm: float = SIGMA_DBM,
) -> torch.Tensor:
    r"""
    Calcula la relación Señal a Ruido más Interferencia (SINR) en el dominio lineal.

    La interferencia Co-Canal (CCI) proviene de aquellos puntos de acceso que 
    irradian en el mismo espectro físico que el AP servidor.

    Fórmula Matemática (NetROML § Physical Model):
    $$ SINR_{u,t} = \frac{P_{a_u,t} G_{u, a_u}}{N_0 + \sum_{j \neq a_u, c_j = c_{a_u}} P_{j,t} G_{u,j}} $$

    Donde:
        - $P_{a_u,t} G_{u, a_u}$: Potencia recibida de la señal deseada (mW).
        - $N_0$: Piso de ruido ambiental (mW) dado por `sigma_dbm`.
        - $\sum P_{j,t} G_{u,j}$: Suma de potencias recibidas de APs en el mismo canal $c_{a_u}$.

    Returns
    -------
    Tensor (n_cli, T)  — Tensor lineal del SINR (adimensional). NaN indica inactividad.
    """
    n_cli, T, n_APs, _ = grilla_RSSI.shape
    dev = grilla_RSSI.device

    ap_idx   = asignaciones[:, :, 0]
    band_idx = asignaciones[:, :, 1]
    rssi_dBm = asignaciones[:, :, 2]
    mask_nan = torch.isnan(ap_idx)

    signal_lin = db_to_linear(rssi_dBm)

    ap_safe   = ap_idx.nan_to_num(0).long()
    band_safe = band_idx.nan_to_num(0).long()

    # RSSI de todos los APs en la banda del cliente
    band_exp      = band_safe.unsqueeze(2).expand(n_cli, T, n_APs)
    rssi_en_banda = grilla_RSSI.gather(3, band_exp.unsqueeze(3)).squeeze(3)
    rssi_en_banda_lin = db_to_linear(rssi_en_banda)

    # APs activos en cada timestep (tienen al menos un cliente asociado)
    A = torch.arange(n_APs, device=dev)
    aps_activos = (ap_idx.unsqueeze(2) == A.view(1, 1, n_APs)).any(dim=0)  # (T, N)

    # Máscara de interferencia CCI: mismo canal, AP distinto, ambos activos
    canal_cli   = canales_por_AP[ap_safe]                                  # (n_cli, T)
    mismo_canal = (canales_por_AP.view(1, 1, n_APs) == canal_cli.unsqueeze(2))
    no_soy_yo   = (A.view(1, 1, n_APs) != ap_safe.unsqueeze(2))
    activo      = (~mask_nan).unsqueeze(2).expand(n_cli, T, n_APs)

    interf_mask = mismo_canal & aps_activos.unsqueeze(0) & no_soy_yo & activo
    interf_lin  = (rssi_en_banda_lin.nan_to_num(0.0) * interf_mask.float()).sum(dim=2).clamp(min=0.0)

    sigma_lin = db_to_linear(torch.tensor(sigma_dbm, dtype=torch.float32, device=dev))

    sinr = signal_lin / (interf_lin + sigma_lin + 1e-30)
    sinr = sinr.masked_fill(mask_nan, float('nan'))
    return sinr


# ── 6. Capacidad (Teorema de Shannon-Hartley) ─────────────────────────────────

def calcular_rate(sinr: torch.Tensor) -> torch.Tensor:
    r"""
    Implementa el Teorema de Shannon-Hartley para la Capacidad del Canal.

    Evalúa la cota teórica superior de la tasa de transferencia de información 
    para un ancho de banda normalizado de 1 Hz.

    Fórmula Matemática:
    $$ r_{u,t} = B \cdot \log_2(1 + SINR_{u,t}) $$
    (Consideramos $B=1$ de forma genérica, devolviendo la eficiencia espectral en bits/s/Hz).

    Parameters
    ----------
    sinr : Tensor de SINR en el dominio lineal (NO en decibelios).

    Returns
    -------
    Tensor (n_cli, T) — Capacidad de canal instantánea (bits/s/Hz).
    """
    rate = torch.log2(1.0 + sinr.nan_to_num(nan=0.0))
    return rate.masked_fill(torch.isnan(sinr), float('nan'))


def actualizar_avg_rate(
    avg_prev: torch.Tensor,
    nuevo_rate: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Media aritmética recursiva (running mean) de la tasa percibida:

        avg_n = ((n-1) · avg_{n-1} + r_n) / n

    Clientes inactivos (NaN) conservan su valor previo sin actualización.
    """
    new_val = nuevo_rate.nan_to_num(nan=0.0)
    avg_new = ((n - 1) * avg_prev + new_val) / n
    return torch.where(torch.isnan(nuevo_rate), avg_prev, avg_new)


def calcular_reward(rate_t: torch.Tensor) -> float:
    r"""
    Calcula la Señal de Recompensa (Reward) agregada del MDP para un timestep $\tau$.

    De acuerdo a la formulación del problema (Optimización de Utilidad Total del Sistema),
    el reward escalar que el entorno devuelve al agente es la sumatoria de las tasas
    de transferencia de todos los usuarios activos en el instante $t$.

    Fórmula Matemática (NetROML § Reinforcement Learning):
    $$ R_\tau = \sum_{u=1}^{U_\tau} r_{u,\tau} $$

    El objetivo del algoritmo PPO/A2C/REINFORCE es maximizar el valor esperado del retorno
    descontado: $\mathbb{E} \left[ \sum_{\tau=0}^\infty \gamma^\tau R_\tau \right]$.

    Returns
    -------
    float — Reward escalar $R_\tau$ (suma de tasas del slot $t$).
    """
    if rate_t.numel() == 0 or torch.all(torch.isnan(rate_t)):
        return 0.0
    # Average Rate: Optimiza la calidad de servicio promedio, independizando
    # el reward de la cantidad absoluta de clientes (promueve fairness y aísla la política).
    return rate_t.nanmean().item()