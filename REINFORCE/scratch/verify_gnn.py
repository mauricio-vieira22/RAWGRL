import torch
from model.gnn_model import GNN
from simulation.graph_builder import construir_grafo_timestep
import joblib

def test_gnn_compatibility():
    print("Testing GNN compatibility with new feature dimensions...")
    device = torch.device('cpu')
    
    # 1. Mock Data
    n_aps = 3
    n_cli = 2
    T = 10
    
    available_channels = torch.tensor([1, 6, 11])
    tx_powers_dbm = torch.tensor([20.0, 17.0, 14.0, 11.0, 8.0])
    
    asignaciones = torch.full((n_cli, T, 3), float('nan'))
    asignaciones[0, 0, 0] = 0 # AP 0
    asignaciones[0, 0, 1] = 1 # Band 5G
    
    grilla_RSSI = torch.full((n_cli, T, n_aps, 2), -60.0)
    decisiones_APs = torch.tensor([[0, 0], [1, 1], [2, 2]]) # [ch_idx, pwr_idx]
    avg_rates = torch.zeros(n_cli)
    
    class MockEvent:
        def __init__(self, departure_time):
            self.departure_time = departure_time
    
    eventos = [MockEvent(5), MockEvent(8)]
    
    # 2. Build Graph
    data = construir_grafo_timestep(
        t=0,
        asignaciones=asignaciones,
        grilla_RSSI=grilla_RSSI,
        decisiones_APs=decisiones_APs,
        avg_rates=avg_rates,
        eventos=eventos,
        available_channels=available_channels,
        tx_powers_dbm=tx_powers_dbm,
        max_timesteps=T,
        delta_t=2
    )
    
    print(f"AP features shape: {data['ap'].x.shape}")
    print(f"AP features (first row): {data['ap'].x[0]}")
    
    assert data['ap'].x.shape[1] == 3, f"Expected 3 AP features, got {data['ap'].x.shape[1]}"
    
    # 3. Initialize GNN
    model = GNN(hidden_channels=16, num_aps=n_aps, out_channels_ch=3, out_channels_pwr=5)
    
    # 4. Forward Pass
    try:
        ch_logits, pwr_logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        print("GNN forward pass successful!")
        print(f"Output shapes: {ch_logits.shape}, {pwr_logits.shape}")
    except Exception as e:
        print(f"GNN forward pass FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_gnn_compatibility()
