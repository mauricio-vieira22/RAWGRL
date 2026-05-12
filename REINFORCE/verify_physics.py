import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Añadir el raíz del proyecto al sys.path
sys.path.append(str(Path(__file__).parent))

from data.preprocessing import run_pipeline
from simulation.wifi_physics import (
    crear_grilla, convertir_grilla_a_tensor, obtener_grilla_RSSIs,
    asignaciones_AP, calcular_sinr, calcular_rate,
    db_to_linear, SIGMA_DBM
)
from simulation.arrival_departure_model import ArrivalDepartureModel

def run_test():
    print("Iniciando Verificación Visual Matemática...\n")
    device = torch.device('cpu')
    
    # 1. Cargar subconjunto de datos para que sea legible
    import joblib
    dists = joblib.load("data/distributions_84.joblib")
    print(f"\nCargados {len(dists)} clientes.\n")

    # 2. Generar simulación muy corta
    adm = ArrivalDepartureModel(dists, arrival_rate=10.0, mean_duration=10.0, total_timesteps=10, random_seed=42)
    eventos = adm.simulate_all_events()
    
    if len(eventos) > 5:
        eventos = eventos[:5]
    
    # Armamos la física
    n_aps = len(dists[0].blocks[0].datos)
    grilla_obj = crear_grilla(eventos, dists, total_timesteps=10)
    grilla_gan = convertir_grilla_a_tensor(grilla_obj).to(device)

    # 3. Forzar Política Manual Simplificada
    # Todos los APs transmiten a 20 dBm (índice 0)
    tx_powers_dbm = torch.tensor([20.0, 17.0, 14.0], dtype=torch.float32)
    potencias_APs = torch.full((n_aps,), 20.0, device=device) 
    
    # Mitad de APs en canal 1, mitad en canal 6
    canales_disp = torch.tensor([1, 6, 11])
    canales_APs = torch.tensor([canales_disp[i % 2] for i in range(n_aps)], device=device)

    # 4. Cálculo Paso a Paso
    grilla_rssi = obtener_grilla_RSSIs(grilla_gan, potencias_APs)

    asignaciones, _ = asignaciones_AP(
        grilla_rssi, umbral_5g=-70.0, umbral_conexion=-85.0, sticky=False
    )
    
    sinr_lin = calcular_sinr(grilla_rssi, asignaciones, canales_APs, sigma_dbm=SIGMA_DBM)
    rate     = calcular_rate(sinr_lin)
    
    print(f"DEBUG: Max RSSI en grilla_rssi: {grilla_rssi.nan_to_num(-200).max().item():.2f}")
    print(f"DEBUG: Conexiones detectadas en t=0: {(~torch.isnan(asignaciones[:, 0, 0])).sum().item()}")

    # 5. Volcar Resultados del primer timestep con actividad
    t_act = 0
    for t in range(10):
        if (~torch.isnan(asignaciones[:, t, 0])).any():
            t_act = t
            break
            
    print(f"\n--- TIMESTEP {t_act} --- (Sigma [Ruido Térmico] = {SIGMA_DBM} dBm, es decir {db_to_linear(torch.tensor(SIGMA_DBM)).item():.2e} mW)")
    print(f"{'Cli_ID':<8} | {'AP Asociado':<12} | {'Canal':<6} | {'RSSI Útil [dBm]':<16} | {'SINR Lineal [mW]':<16} | {'Rate [bits/s/Hz]':<16}")
    print("-" * 88)

    for cli_idx in range(len(eventos)):
        ap_idx    = asignaciones[cli_idx, t_act, 0].item()
        band_idx  = asignaciones[cli_idx, t_act, 1].item()
        rssi_val  = asignaciones[cli_idx, t_act, 2].item()
        sinr_val  = sinr_lin[cli_idx, t_act].item()
        rate_val  = rate[cli_idx, t_act].item()
        
        if np.isnan(ap_idx):
            continue
            
        canal_elegido = canales_APs[int(ap_idx)].item()
        
        print(f"Cli {cli_idx:<4} | AP {int(ap_idx):<9} | {canal_elegido:<6} | {rssi_val:<15.2f} | {sinr_val:<16.4f} | {rate_val:<15.4f}")
        
    print("\nAuditoría Completa. Matemáticamente Rate == log2(1 + SINR_Lineal)")


if __name__ == "__main__":
    run_test()
