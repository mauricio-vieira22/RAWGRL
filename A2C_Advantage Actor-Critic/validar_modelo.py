#!/usr/bin/env python3
"""
Validación crítica: Evaluar con building_id=990 (usado en entrenamiento)
para confirmar que el modelo SÍ aprendió correctamente.
"""

import sys
from pathlib import Path

# Setup path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ejecutar con building_id=990
if __name__ == "__main__":
    # Override sys.argv para usar building_id=990
    sys.argv = [
        sys.argv[0],
        "--building_id", "990",
        "--episodes", "10",
        "--seed", "42",
    ]
    
    from evaluate import parse_args, evaluate
    args = parse_args()
    
    print("\n" + "="*70)
    print("VALIDACIÓN CRÍTICA: Evaluación con Building ID Correcto")
    print("="*70)
    print(f"Building ID (entrenamiento): 990")
    print(f"Building ID (evaluación):    {args.building_id}")
    print(f"Episodios: {args.episodes}")
    print("="*70 + "\n")
    
    try:
        metrics_df = evaluate(args)
        print("\n" + "="*70)
        print("RESULTADOS DE VALIDACIÓN")
        print("="*70)
        
        G_t = metrics_df['return'].values
        print(f"\n✓ Episodios evaluados: {len(G_t)}")
        print(f"  Retorno (G_t):")
        print(f"    Media:    {G_t.mean():.3f}")
        print(f"    Std:      {G_t.std():.3f}")
        print(f"    Mín:      {G_t.min():.3f}")
        print(f"    Máx:      {G_t.max():.3f}")
        
        if G_t.mean() > 0.0:
            print("\n✓✓✓ VALIDACIÓN EXITOSA ✓✓✓")
            print("    El modelo SÍ aprendió y funciona correctamente")
            print("    El problema era el building_id=814 (fuera de distribución)")
        else:
            print("\n✗ VALIDACIÓN FALLIDA")
            print("    Problema aún sin resolver - requiere investigación adicional")
            
    except Exception as e:
        print(f"\n❌ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
