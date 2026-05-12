import joblib
import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--building_id", default="990")
    args = parser.parse_args()
    
    path = f"data/distributions_{args.building_id}.joblib"
    if not os.path.exists(path):
        print(f"Error: {path} no existe")
        return
        
    dist = joblib.load(path)
    print(f"--- Inspección de {path} ---")
    print(f"Total clientes: {len(dist)}")
    
    if len(dist) > 0:
        first_client = dist[0]
        print(f"Cliente 0: {first_client.mac_client}")
        print(f"Bloques: {len(first_client.blocks)}")
        
        if len(first_client.blocks) > 0:
            df = first_client.blocks[0].datos
            print("\nPrimer Bloque - Datos (AP order):")
            print(df)
            print("\nEstadísticas de Gain:")
            print(df[['G_2_4', 'G_5']].describe())
            
            nan_count = df[['G_2_4', 'G_5']].isna().sum().sum()
            total_cells = df[['G_2_4', 'G_5']].size
            print(f"\nNaN ratio: {nan_count/total_cells:.2%}")

if __name__ == "__main__":
    main()
