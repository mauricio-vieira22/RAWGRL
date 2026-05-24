r"""
wifi_physics.py — Funciones Físicas y Ecuaciones de Propagación WiFi.

Implementa el modelo matemático formal del sistema RAWGRL (Sección: "Modelo
Wi-Fi con Sticky Clients" y "Formulación del MDP"). Las variables siguen
estrictamente la notación del modelo teórico.

Notación Matemática (POMDP)
---------------------------
  H_t  ∈ R^{U_t × N × 2}  : Matriz de ganancias de canal (2.4 GHz y 5 GHz). [dB]
  RSSI_{u,n} = H_{u,n} + P_n : Potencia de recepción proyectada.             [dBm]
  A_t  ∈ {0,1}^{U_t × N}   : Matriz de asignaciones. a_{u,n}=1 si u→n.
  ε_t  ∈ N^{U_t}            : departure_time - t (slots restantes por usuario).
  δ_t  ∈ N                  : Slots hasta el próximo arribo al sistema.

Modelos de Handover (Sticky Client)
------------------------------------
Los tres modos modelan distintos grados de inercia de reconexión del cliente:

  STICKY_FULL  ('full')
      a_{u,t+1} ← a_{u,0}  para todo t > t_u.
      El cliente nunca cambia de AP tras el handshake inicial.
      Modelo de mayor inercia; representa hardware legacy o configuración manual.

  STICKY_STD   ('sticky')
      Cambia de AP si y solo si RSSI_{u, a_u, t} ≤ θ_conn.
      Modelo intermedio; el cliente tolera degradación hasta caer bajo el umbral.

  STICKY_LITE  ('lite')
      Cambia de AP si existe n* tal que RSSI_{u,n*,t} > RSSI_{u,a_u,t}.
      Modelo de menor inercia; el cliente siempre busca el AP de mayor señal.
      Equivalente a asociación óptima greedy por timestep.

Flujo de Ecuaciones
-------------------
  1. crear_grilla          → H_t para todo el horizonte T.
  2. obtener_grilla_RSSIs  → RSSI_{u,n} = H_{u,n} + P_n.
  3. asignaciones_AP       → A_t según modo sticky.
  4. calcular_sinr         → SINR con interferencia co-canal (CCI).
  5. calcular_rate         → Eficiencia espectral Shannon-Hartley.
  6. calcular_reward       → R_τ = ∑_u r_{u,τ} (throughput agregado).
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Literal

# ── Constantes Físicas del Entorno ────────────────────────────────────────────
UMBRAL_5G:       float = -70.0   # θ_{5G}: umbral para preferir banda 5 GHz  [dBm]
UMBRAL_CONEXION: float = -85.0   # θ_{conn}: sensibilidad mínima del chip WiFi [dBm]
SIGMA_DBM:       float = -90.0   # N_0: piso de ruido térmico estándar        [dBm]

# ── Identificadores de Modo Sticky ────────────────────────────────────────────
STICKY_FULL = 'full'    # Full Sticky: sin roaming tras handshake inicial
STICKY_STD  = 'sticky'  # Sticky:      roaming si RSSI < θ_conn
STICKY_LITE = 'lite'    # Sticky Lite: roaming greedy (mejor AP disponible)

StickyMode = Literal['full', 'sticky', 'lite']


# ── 1. Inicialización de Tensores de Ganancia ─────────────────────────────────

def crear_grilla(eventos, distribuciones, total_timesteps: int, rng=None) -> np.ndarray:
    """
    Materializa H_t como array de objetos Block para todo el horizonte T.

    Cada fila u corresponde a un cliente (ClientEvent); cada columna t a un
    slot. El valor es None cuando el cliente u no está activo en t.

    Parameters
    ----------
    eventos : list[ClientEvent]
        Línea temporal del episodio generada por ArrivalDepartureModel.
    distribuciones : list[Distribution]
        Pool de perfiles de ganancia medidos en campo.
    total_timesteps : int
        Horizonte T del episodio.
    rng : np.random.Generator | None
        Generador aleatorio para reproducibilidad. Si None, se crea uno nuevo.

    Returns
    -------
    np.ndarray
        Array de objetos shape (n_eventos, T).
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
    Convierte la grilla de objetos Block a Tensor float32 (n_cli, T, N, 2).

    Garantiza consistencia de la dimensión N (número de APs) validando cada
    bloque antes de insertar. Bloques con dimensión distinta se ignoran y el
    valor permanece NaN.

    Returns
    -------
    torch.Tensor
        Shape (n_cli, T, N, 2) con ganancias H en dB. NaN donde inactivo.
    """
    primer = next((b for b in grilla.flat if b is not None), None)
    if primer is None:
        return torch.full((grilla.shape[0], grilla.shape[1], 0, 2), float('nan'))

    cant_APs = len(primer.datos)
    n_ev, T  = grilla.shape
    out = torch.full((n_ev, T, cant_APs, 2), float('nan'), dtype=torch.float32)

    for (i, j), bloque in np.ndenumerate(grilla):
        if bloque is not None and len(bloque.datos) == cant_APs:
            vals = torch.tensor(
                bloque.datos[['G_2_4', 'G_5']].values.astype(np.float32),
                dtype=torch.float32,
            )
            out[i, j] = vals

    out = torch.nan_to_num(out, nan=float('nan'), posinf=float('nan'), neginf=float('nan'))
    return out


# ── 2. Cálculo de RSSI ────────────────────────────────────────────────────────

def obtener_grilla_RSSIs(
    grilla_ganancias:  torch.Tensor,
    potencias_APs_dBm: torch.Tensor,
) -> torch.Tensor:
    """
    Modelo de propagación log-lineal:

        RSSI_{u,n,t} [dBm] = H_{u,n,t} [dB] + P_{n,t} [dBm]

    Parameters
    ----------
    grilla_ganancias  : (n_cli, T, N, 2) — ganancias espaciales H_t [dB]
    potencias_APs_dBm : (N,)             — decisión X_t del agente   [dBm]

    Returns
    -------
    torch.Tensor
        Shape (n_cli, T, N, 2) — RSSI_t [dBm].
    """
    p = potencias_APs_dBm.view(1, 1, -1, 1).to(grilla_ganancias.dtype)
    return grilla_ganancias + p


# ── 3. Motor de Handover (Tres Modos Sticky) ──────────────────────────────────

def asignaciones_AP(
    grilla_RSSI:     torch.Tensor,
    umbral_5g:       float    = UMBRAL_5G,
    umbral_conexion: float    = UMBRAL_CONEXION,
    sticky_mode:     StickyMode = STICKY_STD,
    estado_previo                = None,
    sticky:          bool | None = None,
    use_5g:          bool    = True,
):
    """
    Resuelve el problema de asociación cliente-AP para los tres modos sticky.

    Modos
    -----
    'full'   (Full Sticky)
        Una vez conectado, el cliente no cambia de AP. Solo el primer slot
        activo realiza la selección inicial de AP.

    'sticky' (Sticky estándar)
        Cambia de AP si RSSI_{u,a_u,t} ≤ θ_conn, o si es el primer slot activo.
        Modela la reconexión por señal débil típica de drivers WiFi estándar.

    'lite'   (Sticky Lite)
        Selección greedy por timestep: siempre el AP de mayor RSSI disponible.
        Equivale a asociación óptima sin inercia. Implementado sin bucle temporal
        vectorizado sobre toda la dimensión T.

    Parameters
    ----------
    grilla_RSSI    : (n_cli, T, N, 2) — RSSI_t [dBm]
    umbral_5g      : float — θ_{5G}, umbral para preferir 5 GHz
    umbral_conexion: float — θ_{conn}, sensibilidad mínima
    sticky_mode    : str — 'full' | 'sticky' | 'lite'
    estado_previo  : tuple(Tensor, Tensor) | None — (cur_ap, cur_band) del step anterior
    use_5g         : bool — si False, solo opera en 2.4 GHz

    Returns
    -------
    asignaciones  : Tensor (n_cli, T, 3) — [ap_idx, band_idx, RSSI_dBm], NaN=inactivo
    ultimo_estado : tuple(Tensor, Tensor) — (cur_ap, cur_band) para propagar al siguiente step
    """
    # Compatibilidad con parámetro legacy 'sticky' (bool)
    if sticky is not None:
        sticky_mode = STICKY_STD if sticky else STICKY_LITE

    n_cli, T, n_APs, _ = grilla_RSSI.shape
    dev = grilla_RSSI.device

    # ── AP ideal por timestep (vectorizado sobre T) ──────────────────────────
    if not use_5g:
        rssi_24      = grilla_RSSI[:, :, :, 0]
        max_all, argmax_all_ap = rssi_24.nan_to_num(-200.0).max(dim=2)
        mask_active  = max_all >= umbral_conexion
        ideal_ap     = argmax_all_ap.float().masked_fill(~mask_active, float('nan'))
        ideal_band   = torch.zeros_like(ideal_ap)
        ideal_rssi   = max_all.masked_fill(~mask_active, float('nan'))
    else:
        rssi_5g      = grilla_RSSI[:, :, :, 1]
        rssi_5g_safe = rssi_5g.nan_to_num(-200.0)
        max_5g, argmax_5g = rssi_5g_safe.max(dim=2)

        flat         = grilla_RSSI.reshape(n_cli, T, n_APs * 2)
        flat_safe    = flat.nan_to_num(-200.0)
        max_all, argmax_flat = flat_safe.max(dim=2)

        argmax_all_ap   = argmax_flat // 2
        argmax_all_band = argmax_flat % 2
        mask_active     = max_all >= umbral_conexion

        cumple_5g    = (max_5g >= umbral_5g) & mask_active
        ideal_ap     = torch.where(
            cumple_5g,
            argmax_5g.float().masked_fill(~mask_active, float('nan')),
            argmax_all_ap.float().masked_fill(~mask_active, float('nan')),
        )
        ideal_band   = torch.where(
            cumple_5g,
            torch.ones_like(argmax_5g, dtype=torch.float32).masked_fill(~mask_active, float('nan')),
            argmax_all_band.float().masked_fill(~mask_active, float('nan')),
        )
        ideal_rssi   = torch.where(
            cumple_5g, max_5g, max_all
        ).masked_fill(~mask_active, float('nan'))

    # ── Modo LITE: asociación greedy sin estado ───────────────────────────────
    # No requiere bucle temporal: para cada (cliente, timestep) se asigna
    # directamente el AP óptimo calculado arriba. El último estado se extrae
    # de la última columna temporal para mantener interfaz uniforme con los
    # otros modos (aunque en el siguiente step se ignorará).
    if sticky_mode == STICKY_LITE:
        asig = torch.stack([ideal_ap, ideal_band, ideal_rssi], dim=-1)
        ultimo_estado = (ideal_ap[:, -1].clone(), ideal_band[:, -1].clone())
        return asig, ultimo_estado

    # ── Modos FULL y STICKY: requieren estado temporal ────────────────────────
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
        active_now    = ~torch.isnan(ideal_rssi[:, t])
        ap_safe       = cur_ap.nan_to_num(0).long()
        band_safe     = cur_band.nan_to_num(0).long()
        rssi_cur      = grilla_RSSI[idx_cli, t, ap_safe, band_safe]
        rssi_cur[torch.isnan(cur_ap)] = float('nan')

        was_connected = ~torch.isnan(cur_ap)

        if sticky_mode == STICKY_FULL:
            # Solo cambia en el primer slot activo (nacimiento del cliente)
            change_mask = ~was_connected & active_now
        else:
            # STICKY_STD: cambia si señal < θ_conn o si es cliente nuevo
            change_mask = (
                (was_connected & (rssi_cur <= umbral_conexion))
                | (~was_connected & active_now)
            )

        cur_ap   = torch.where(change_mask, ideal_ap[:, t],   cur_ap)
        cur_band = torch.where(change_mask, ideal_band[:, t], cur_band)
        cur_rssi = torch.where(
            change_mask,
            ideal_rssi[:, t],
            rssi_cur.nan_to_num(nan=float('nan')),
        )

        # Clientes inactivos → NaN en todos los campos
        cur_ap[~active_now]   = float('nan')
        cur_band[~active_now] = float('nan')
        cur_rssi[~active_now] = float('nan')

        assigned_ap[:, t]   = cur_ap
        assigned_band[:, t] = cur_band
        assigned_rssi[:, t] = cur_rssi

    asig = torch.stack([assigned_ap, assigned_band, assigned_rssi], dim=-1)
    return asig, (cur_ap.clone(), cur_band.clone())


# ── 4. Conversión Logarítmica a Lineal ────────────────────────────────────────

def db_to_linear(x: torch.Tensor) -> torch.Tensor:
    """
    Convierte de dBm a mW: P_mW = 10^(P_dBm / 10).

    Los valores NaN se propagan correctamente; valores extremadamente bajos
    (< -200 dBm) se mapean a ≈ 0 mW mediante clampeo previo.

    Parameters
    ----------
    x : torch.Tensor — Potencia en dBm.

    Returns
    -------
    torch.Tensor — Potencia en mW, con NaN donde x era NaN.
    """
    mask_nan = torch.isnan(x)
    x_safe   = x.nan_to_num(-200.0)
    out      = torch.pow(10.0, x_safe / 10.0)
    out[mask_nan] = float('nan')
    return out


# ── 5. SINR con Interferencia Co-Canal (CCI) ──────────────────────────────────

def calcular_sinr(
    grilla_RSSI:   torch.Tensor,
    asignaciones:  torch.Tensor,
    canales_por_AP: torch.Tensor,
    sigma_dbm:     float = SIGMA_DBM,
) -> torch.Tensor:
    r"""
    Calcula la Relación Señal a Ruido más Interferencia (SINR) en dominio lineal.

    La interferencia Co-Canal (CCI) proviene de APs que transmiten en el mismo
    canal que el AP servidor del cliente.

    Fórmula (RAWGRL § Physical Model):

        SINR_{u,t} = P_{a_u} · G_{u,a_u} / (N_0 + ∑_{j≠a_u, c_j=c_{a_u}} P_j · G_{u,j})

    donde:
        - P_{a_u} · G_{u,a_u}: Potencia de señal deseada (mW).
        - N_0: Piso de ruido térmico (mW) dado por sigma_dbm.
        - ∑ P_j · G_{u,j}: Suma de interferentes co-canal activos.

    Parameters
    ----------
    grilla_RSSI   : (n_cli, T, N, 2) — RSSI_t [dBm]
    asignaciones  : (n_cli, T, 3)    — [ap_idx, band_idx, RSSI_dBm]
    canales_por_AP: (N,)             — canal físico de cada AP en el step actual
    sigma_dbm     : float            — N_0 [dBm]

    Returns
    -------
    torch.Tensor
        Shape (n_cli, T) — SINR lineal. NaN donde el cliente está inactivo.
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

    # RSSI de todos los APs en la banda del cliente (interferentes potenciales)
    band_exp      = band_safe.unsqueeze(2).expand(n_cli, T, n_APs)
    rssi_en_banda = grilla_RSSI.gather(3, band_exp.unsqueeze(3)).squeeze(3)
    rssi_en_banda_lin = db_to_linear(rssi_en_banda)

    # APs con al menos un cliente asociado en cada timestep
    A         = torch.arange(n_APs, device=dev)
    aps_activos = (ap_idx.unsqueeze(2) == A.view(1, 1, n_APs)).any(dim=0)  # (T, N)

    # Máscara CCI: mismo canal, AP distinto, cliente activo
    canal_cli   = canales_por_AP[ap_safe]                                    # (n_cli, T)
    mismo_canal = (canales_por_AP.view(1, 1, n_APs) == canal_cli.unsqueeze(2))
    no_soy_yo   = (A.view(1, 1, n_APs) != ap_safe.unsqueeze(2))
    activo      = (~mask_nan).unsqueeze(2).expand(n_cli, T, n_APs)

    interf_mask = mismo_canal & aps_activos.unsqueeze(0) & no_soy_yo & activo
    interf_lin  = (
        rssi_en_banda_lin.nan_to_num(0.0) * interf_mask.float()
    ).sum(dim=2).clamp(min=0.0)

    sigma_lin = db_to_linear(
        torch.tensor(sigma_dbm, dtype=torch.float32, device=dev)
    )

    sinr = signal_lin / (interf_lin + sigma_lin + 1e-30)
    sinr = sinr.masked_fill(mask_nan, float('nan'))
    return sinr


# ── 6. Eficiencia Espectral (Shannon-Hartley) ─────────────────────────────────

def calcular_rate(sinr: torch.Tensor) -> torch.Tensor:
    r"""
    Cota superior de eficiencia espectral (Teorema de Shannon-Hartley).

        r_{u,t} = B · log₂(1 + SINR_{u,t})

    Se asume B = 1 Hz (eficiencia espectral normalizada en bits/s/Hz).

    Parameters
    ----------
    sinr : torch.Tensor
        SINR en dominio lineal (no dB). Shape (n_cli, T).

    Returns
    -------
    torch.Tensor
        Eficiencia espectral r_{u,t}, shape (n_cli, T). NaN donde inactivo.
    """
    rate = torch.log2(1.0 + sinr.nan_to_num(nan=0.0))
    return rate.masked_fill(torch.isnan(sinr), float('nan'))


def actualizar_avg_rate(
    avg_prev:   torch.Tensor,
    nuevo_rate: torch.Tensor,
    n:          int,
) -> torch.Tensor:
    """
    Media aritmética recursiva (running mean) de la tasa percibida:

        avg_n = ((n-1) · avg_{n-1} + r_n) / n

    Clientes inactivos (NaN en nuevo_rate) conservan su valor previo sin
    actualización, preservando la historia de tasa promedio del cliente.

    Parameters
    ----------
    avg_prev   : (n_cli,) — media acumulada hasta el paso anterior.
    nuevo_rate : (n_cli,) — tasa instantánea del paso actual.
    n          : int      — número de observaciones acumuladas (incluye la actual).

    Returns
    -------
    torch.Tensor
        Media actualizada, shape (n_cli,).
    """
    new_val = nuevo_rate.nan_to_num(nan=0.0)
    avg_new = ((n - 1) * avg_prev + new_val) / n
    return torch.where(torch.isnan(nuevo_rate), avg_prev, avg_new)


def calcular_reward(rate_t: torch.Tensor) -> float:
    r"""
    Señal de recompensa agregada del MDP para un slot τ.

        R_τ = ∑_{u=1}^{U_τ} r_{u,τ}

    Maximiza el throughput total del sistema (suma de tasas de todos los clientes
    activos). La normalización per-client se realiza en train.py antes de
    acumular los retornos, para no alterar la semántica física de esta función.

    Parameters
    ----------
    rate_t : torch.Tensor
        Tasas instantáneas del slot, shape (n_cli,). NaN = cliente inactivo.

    Returns
    -------
    float
        Reward escalar R_τ.
    """
    if rate_t.numel() == 0 or torch.all(torch.isnan(rate_t)):
        return 0.0
    return rate_t.nansum().item()