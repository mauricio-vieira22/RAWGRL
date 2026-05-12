import torch
from model.gnn_delta_model import GNNDelta
from simulation.graph_builder import construir_grafo_timestep

def test_delta_injection():
    print("Testing GNNDelta compatibility and Late-Injection...")
    device = torch.device('cpu')
    
    # 1. Mock Data (matching new 3-feature AP nodes)
    n_aps = 3
    n_cli = 2
    T = 10
    
    available_channels = torch.tensor([1, 6, 11])
    tx_powers_dbm = torch.tensor([20.0, 17.0, 14.0, 11.0, 8.0])
    
    # Graph without delta_t in node features
    asignaciones = torch.full((n_cli, T, 3), float('nan'))
    grilla_RSSI = torch.full((n_cli, T, n_aps, 2), -60.0)
    decisiones_APs = torch.tensor([[0, 0], [1, 1], [2, 2]])
    avg_rates = torch.zeros(n_cli)
    
    class MockEvent:
        def __init__(self, departure_time):
            self.departure_time = departure_time
    eventos = [MockEvent(5), MockEvent(8)]
    
    # Build Graph (delta_t is passed but not used for features anymore)
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
        delta_t=2 # Scaled later in train loop
    )
    
    print(f"AP features shape (should be [N, 3]): {data['ap'].x.shape}")
    
    # 2. Initialize Model
    hidden = 16
    model = GNNDelta(hidden_channels=hidden, num_aps=n_aps)
    
    # 3. Forward Pass with delta_t injection
    delta_t_norm = 0.5
    try:
        ch_logits, pwr_logits = model(data.x_dict, data.edge_index_dict, delta_t_norm, data.edge_attr_dict)
        print("GNNDelta forward pass successful!")
        print(f"Output shapes: {ch_logits.shape}, {pwr_logits.shape}")
        
        # Verify MLP dimensions internally
        # Combined dim: (hidden * 2) + 1
        expected_combined = (hidden * 2) + 1
        actual_in_features = model.channel_head[0].in_features
        print(f"MLP Input Features: {actual_in_features} (Expected: {expected_combined})")
        assert actual_in_features == expected_combined
        
    except Exception as e:
        print(f"GNNDelta forward pass FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_delta_injection()
